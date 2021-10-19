#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# These are the backing OpView classes generated from the linalg tablegen
# definitions following these steps:
#   DSL -> YAML -> tblgen -> pytblgen -> build/.../_linalg_ops_gen.py.
from .._linalg_ops_gen import *

# These are the ground truth functions defined as:
# ```
#    @linalg_structured_op
#    def matmul(A=TensorDef(T1, S.M, S.K),
#               B=TensorDef(T2, S.K, S.N),
#               C=TensorDef(U, S.M, S.N, output=True)):
# ```
# using the linalg-py eDSL.
# The linalg-py eDSL builds a python representation (PyRepr) that is 
# used in following ways:
#  1. PyRepr -> YAML to generate the C++ and Python .td files. These
#     then turn into the core C++ Op classes and Python OpView classes
#     respectively (made available in _linalg_ops_gen). The generic OpView class 
#     mechanism makes the C++ classes available to python through the CAPI.
#     PyRepr -> YAML currently occurs before compiler compile time.
#     The other steps in this category occur at compiler compile time.
#  2. PyRepr -> linalg.core_named_ops calls: piggybacks on the 
#     _linalg_ops_gen classes and the OpView mechanism to build IR at
#     runtime in python:
#       a. by default, the Named Op Form is emitted, e.g.:
#          `linalg.matmul(lhs, rhs, outs=[out])` creates the following IR:
#          ```
#             %1 = linalg.matmul ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<16x8xf32>) 
#                               outs(%0 : tensor<4x8xf32>)
#                  -> tensor<4x8xf32>   
#          ```
#       b. by setting emit_generic=True, the Generic Op Form is emitted, e.g.:
#           `linalg.matmul(lhs, rhs, outs=[out], emit_generic=True)` creates the following IR:
#          ```
#             %1 = linalg.generic {indexing_maps = [...], iterator_types = [...]} 
#               ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<16x8xf32>) 
#              outs(%0 : tensor<4x8xf32>) {
#               ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  
#                  ...
#                  linalg.yield %3 : f32
#             } -> tensor<4x8xf32>  
#          ```
#  3. PyRepr -> Runtime Custom Op definitions: directly generates a
#     linalg.generic form like in 2.b.
#     !!!WARNING!!!: if one creates a runtime custom op with the same name 
#     as an existing core named op, step 2. will likely take precedence.
#     TODO: guard against surprises and fail create Runtime Custom Ops with 
#     the same name as existing Core Named Ops.
from .opdsl.ops.core_named_ops import *
