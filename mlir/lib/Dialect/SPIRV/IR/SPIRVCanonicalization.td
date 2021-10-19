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

include "mlir/Dialect/SPIRV/IR/SPIRVOps.td"

//===----------------------------------------------------------------------===//
// spv.Bitcast
//===----------------------------------------------------------------------===//

def ConvertChainedBitcast : Pat<(SPV_BitcastOp (SPV_BitcastOp $operand)),
                                (SPV_BitcastOp $operand)>;

//===----------------------------------------------------------------------===//
// spv.LogicalNot
//===----------------------------------------------------------------------===//

def ConvertLogicalNotOfIEqual : Pat<
    (SPV_LogicalNotOp (SPV_IEqualOp $lhs, $rhs)),
    (SPV_INotEqualOp $lhs, $rhs)>;

def ConvertLogicalNotOfINotEqual : Pat<
    (SPV_LogicalNotOp (SPV_INotEqualOp $lhs, $rhs)),
    (SPV_IEqualOp $lhs, $rhs)>;

def ConvertLogicalNotOfLogicalEqual : Pat<
    (SPV_LogicalNotOp (SPV_LogicalEqualOp $lhs, $rhs)),
    (SPV_LogicalNotEqualOp $lhs, $rhs)>;

def ConvertLogicalNotOfLogicalNotEqual : Pat<
    (SPV_LogicalNotOp (SPV_LogicalNotEqualOp $lhs, $rhs)),
    (SPV_LogicalEqualOp $lhs, $rhs)>;

//===----------------------------------------------------------------------===//
// Re-write spv.Select + spv.<less_than_op> to a suitable variant of
// spv.<glsl_clamp_op>
//===----------------------------------------------------------------------===//

def ValuesAreEqual : Constraint<CPred<"$0 == $1">>;

foreach CmpClampPair = [
    [SPV_FOrdLessThanOp, SPV_GLSLFClampOp],
    [SPV_FOrdLessThanEqualOp, SPV_GLSLFClampOp],
    [SPV_SLessThanOp, SPV_GLSLSClampOp],
    [SPV_SLessThanEqualOp, SPV_GLSLSClampOp],
    [SPV_ULessThanOp, SPV_GLSLUClampOp],
    [SPV_ULessThanEqualOp, SPV_GLSLUClampOp]] in {
def ConvertComparisonIntoClamp#CmpClampPair[0] : Pat<
    (SPV_SelectOp
        (CmpClampPair[0]
            (SPV_SelectOp:$middle0
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
}
