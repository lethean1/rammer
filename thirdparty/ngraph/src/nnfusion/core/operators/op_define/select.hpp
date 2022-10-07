//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "../op.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise selection operation.
        ///
        /// ## Inputs
        ///
        /// |        | Type                                          | Description                                                  |
        /// | ------ | --------------------------------------------- | ------------------------------------------------------------ |
        /// | `arg0` | \f$\texttt{bool}[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape, with element `bool`.                  |
        /// | `arg1` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$             | A tensor of the same shape as `arg0`, with any element type. |
        /// | `arg2` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$             | A tensor of the same shape and element type as `arg1`.       |
        ///
        /// ## Output
        ///
        /// | Type                   | Description                                                                                                                                                             |
        /// | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
        /// | \f$E[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \texttt{arg1}[i_1,\dots,i_n]\text{ if }\texttt{arg0}[i_1,\dots,i_n] \neq 0\text{, else }\texttt{arg2}[i_1,\dots,i_n]\f$ |
        class Select : public Op
        {
        public:
            /// \brief Constructs a selection operation.
            Select();

        protected:
            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;
        };
    }
}
