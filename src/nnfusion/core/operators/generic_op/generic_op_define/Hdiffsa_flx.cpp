// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(hdiffsa_flx)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        nnfusion::Shape output_shape_0(3);
        output_shape_0[0] = 68;
        output_shape_0[1] = 68;
        output_shape_0[2] = 60;
        gnode->set_output_type_and_shape(0, element::f32, output_shape_0);
    })
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        std::stringstream expression;
        expression << "- einstein_v2(\" output0[N, M, K] = (input0[N + 1, M, K] - input0[N, M, K]).when([(input0[N + 1, M, K] - input0[N, M, K])*(input1[N+1,M,K]-input1[N,M,K])<=0], 0.0) where N in 66, M in 66, K in 60; \", "
            <<"input_dict={ "
            <<"\"input0\" : { \"dtype\" : \"float32\", \"shape\" : [68, 68, 60]}, "
            <<"\"input1\" : { \"dtype\" : \"float32\", \"shape\" : [68, 68, 60]} })";
        return expression.str();
    });
