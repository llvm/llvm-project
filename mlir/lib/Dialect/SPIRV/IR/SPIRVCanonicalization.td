//==- SPIRVCanonicalization.td - Canonicalization Patterns ---*- tablegen -*==//

// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines SPIR-V canonicalization patterns.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/SPIRV/IR/SPIRVOps.td"

//===----------------------------------------------------------------------===//
// spirv.Bitcast
//===----------------------------------------------------------------------===//

def ConvertChainedBitcast : Pat<(SPIRV_BitcastOp (SPIRV_BitcastOp $operand)),
                                (SPIRV_BitcastOp $operand)>;

//===----------------------------------------------------------------------===//
// spirv.LogicalNot
//===----------------------------------------------------------------------===//

def ConvertLogicalNotOfIEqual : Pat<
    (SPIRV_LogicalNotOp (SPIRV_IEqualOp $lhs, $rhs)),
    (SPIRV_INotEqualOp $lhs, $rhs)>;

def ConvertLogicalNotOfINotEqual : Pat<
    (SPIRV_LogicalNotOp (SPIRV_INotEqualOp $lhs, $rhs)),
    (SPIRV_IEqualOp $lhs, $rhs)>;

def ConvertLogicalNotOfLogicalEqual : Pat<
    (SPIRV_LogicalNotOp (SPIRV_LogicalEqualOp $lhs, $rhs)),
    (SPIRV_LogicalNotEqualOp $lhs, $rhs)>;

def ConvertLogicalNotOfLogicalNotEqual : Pat<
    (SPIRV_LogicalNotOp (SPIRV_LogicalNotEqualOp $lhs, $rhs)),
    (SPIRV_LogicalEqualOp $lhs, $rhs)>;

//===----------------------------------------------------------------------===//
// spirv.Select -> spirv.GL.*Clamp
//===----------------------------------------------------------------------===//

def ValuesAreEqual : Constraint<CPred<"$0 == $1">>;

foreach CmpClampPair = [
    [SPIRV_FOrdLessThanOp, SPIRV_GLFClampOp],
    [SPIRV_FOrdLessThanEqualOp, SPIRV_GLFClampOp],
    [SPIRV_SLessThanOp, SPIRV_GLSClampOp],
    [SPIRV_SLessThanEqualOp, SPIRV_GLSClampOp],
    [SPIRV_ULessThanOp, SPIRV_GLUClampOp],
    [SPIRV_ULessThanEqualOp, SPIRV_GLUClampOp]] in {

// Detect: $min < $input, $input < $max
def ConvertComparisonIntoClamp1_#CmpClampPair[0] : Pat<
    (SPIRV_SelectOp
        (CmpClampPair[0]
            (SPIRV_SelectOp:$middle0
                (CmpClampPair[0] $min, $input),
                $input,
                $min
            ),
            $max
        ),
        $middle1,
        $max),
    (CmpClampPair[1] $input, $min, $max),
    [(ValuesAreEqual $middle0, $middle1)]>;

// Detect: $input < $min, $max < $input
def ConvertComparisonIntoClamp2_#CmpClampPair[0] : Pat<
    (SPIRV_SelectOp
        (CmpClampPair[0] $max, $input),
        $max,
        (SPIRV_SelectOp
            (CmpClampPair[0] $input, $min),
            $min,
            $input
        )),
    (CmpClampPair[1] $input, $min, $max)>;
}
