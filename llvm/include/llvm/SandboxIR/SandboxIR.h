//===- SandboxIR.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sandbox IR is a lightweight overlay transactional IR on top of LLVM IR.
// Features:
// - You can save/rollback the state of the IR at any time.
// - Any changes made to Sandbox IR will automatically update the underlying
//   LLVM IR so both IRs are always in sync.
// - Feels like LLVM IR, similar API.
//
// SandboxIR forms a class hierarchy that resembles that of LLVM IR
// but is in the `sandboxir` namespace:
//
// namespace sandboxir {
//
// Value -+- Argument
//        |
//        +- BasicBlock
//        |
//        +- User ------+- Constant ------ Function
//                      |
//                      +- Instruction -+- BinaryOperator
//                                      |
//                                      +- BranchInst
//                                      |
//                                      +- CastInst --------+- AddrSpaceCastInst
//                                      |                   |
//                                      |                   +- BitCastInst
//                                      |                   |
//                                      |                   +- FPExtInst
//                                      |                   |
//                                      |                   +- FPToSIInst
//                                      |                   |
//                                      |                   +- FPToUIInst
//                                      |                   |
//                                      |                   +- FPTruncInst
//                                      |                   |
//                                      |                   +- IntToPtrInst
//                                      |                   |
//                                      |                   +- PtrToIntInst
//                                      |                   |
//                                      |                   +- SExtInst
//                                      |                   |
//                                      |                   +- SIToFPInst
//                                      |                   |
//                                      |                   +- TruncInst
//                                      |                   |
//                                      |                   +- UIToFPInst
//                                      |                   |
//                                      |                   +- ZExtInst
//                                      |
//                                      +- CallBase --------+- CallBrInst
//                                      |                   |
//                                      |                   +- CallInst
//                                      |                   |
//                                      |                   +- InvokeInst
//                                      |
//                                      +- CmpInst ---------+- ICmpInst
//                                      |                   |
//                                      |                   +- FCmpInst
//                                      |
//                                      +- ExtractElementInst
//                                      |
//                                      +- GetElementPtrInst
//                                      |
//                                      +- InsertElementInst
//                                      |
//                                      +- OpaqueInst
//                                      |
//                                      +- PHINode
//                                      |
//                                      +- ReturnInst
//                                      |
//                                      +- SelectInst
//                                      |
//                                      +- ShuffleVectorInst
//                                      |
//                                      +- ExtractValueInst
//                                      |
//                                      +- InsertValueInst
//                                      |
//                                      +- StoreInst
//                                      |
//                                      +- UnaryInstruction -+- LoadInst
//                                      |                    |
//                                      |                    +- CastInst
//                                      |
//                                      +- UnaryOperator
//                                      |
//                                      +- UnreachableInst
//
// Use
//
// } // namespace sandboxir
//

#ifndef LLVM_SANDBOXIR_SANDBOXIR_H
#define LLVM_SANDBOXIR_SANDBOXIR_H

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/SandboxIR/Argument.h"
#include "llvm/SandboxIR/BasicBlock.h"
#include "llvm/SandboxIR/Constant.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/Module.h"
#include "llvm/SandboxIR/Tracker.h"
#include "llvm/SandboxIR/Type.h"
#include "llvm/SandboxIR/Use.h"
#include "llvm/SandboxIR/User.h"
#include "llvm/SandboxIR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>

namespace llvm {

namespace sandboxir {

class BasicBlock;
class ConstantInt;
class ConstantFP;
class ConstantAggregateZero;
class ConstantPointerNull;
class PoisonValue;
class BlockAddress;
class DSOLocalEquivalent;
class ConstantTokenNone;
class GlobalValue;
class GlobalObject;
class GlobalIFunc;
class GlobalVariable;
class GlobalAlias;
class NoCFIValue;
class ConstantPtrAuth;
class ConstantExpr;
class Context;
class Function;
class Module;
class Instruction;
class VAArgInst;
class FreezeInst;
class FenceInst;
class SelectInst;
class ExtractElementInst;
class InsertElementInst;
class ShuffleVectorInst;
class ExtractValueInst;
class InsertValueInst;
class BranchInst;
class UnaryInstruction;
class LoadInst;
class ReturnInst;
class StoreInst;
class User;
class UnreachableInst;
class Value;
class CallBase;
class CallInst;
class InvokeInst;
class CallBrInst;
class LandingPadInst;
class FuncletPadInst;
class CatchPadInst;
class CleanupPadInst;
class CatchReturnInst;
class CleanupReturnInst;
class GetElementPtrInst;
class CastInst;
class PossiblyNonNegInst;
class PtrToIntInst;
class BitCastInst;
class AllocaInst;
class ResumeInst;
class CatchSwitchInst;
class SwitchInst;
class UnaryOperator;
class BinaryOperator;
class PossiblyDisjointInst;
class AtomicRMWInst;
class AtomicCmpXchgInst;
class CmpInst;
class ICmpInst;
class FCmpInst;

} // namespace sandboxir
} // namespace llvm

#endif // LLVM_SANDBOXIR_SANDBOXIR_H
