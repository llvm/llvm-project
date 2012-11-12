//===- CodeGen/Analysis.h - CodeGen LLVM IR Analysis Utilities --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares several CodeGen-specific LLVM IR analysis utilties.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_ANALYSIS_H
#define LLVM_CODEGEN_ANALYSIS_H

#include "llvm/Instructions.h"
#include "llvm/InlineAsm.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/Support/CallSite.h"

namespace llvm {

class GlobalVariable;
class TargetLowering;
class SDNode;
class SDValue;
class SelectionDAG;

/// ComputeLinearIndex - Given an LLVM IR aggregate type and a sequence
/// of insertvalue or extractvalue indices that identify a member, return
/// the linearized index of the start of the member.
///
unsigned ComputeLinearIndex(Type *Ty,
                            const unsigned *Indices,
                            const unsigned *IndicesEnd,
                            unsigned CurIndex = 0);

inline unsigned ComputeLinearIndex(Type *Ty,
                                   ArrayRef<unsigned> Indices,
                                   unsigned CurIndex = 0) {
  return ComputeLinearIndex(Ty, Indices.begin(), Indices.end(), CurIndex);
}

/// ComputeValueVTs - Given an LLVM IR type, compute a sequence of
/// EVTs that represent all the individual underlying
/// non-aggregate types that comprise it.
///
/// If Offsets is non-null, it points to a vector to be filled in
/// with the in-memory offsets of each of the individual values.
///
void ComputeValueVTs(const TargetLowering &TLI, Type *Ty,
                     SmallVectorImpl<EVT> &ValueVTs,
                     SmallVectorImpl<uint64_t> *Offsets = 0,
                     uint64_t StartingOffset = 0);

/// ExtractTypeInfo - Returns the type info, possibly bitcast, encoded in V.
GlobalVariable *ExtractTypeInfo(Value *V);

/// hasInlineAsmMemConstraint - Return true if the inline asm instruction being
/// processed uses a memory 'm' constraint.
bool hasInlineAsmMemConstraint(InlineAsm::ConstraintInfoVector &CInfos,
                               const TargetLowering &TLI);

/// getFCmpCondCode - Return the ISD condition code corresponding to
/// the given LLVM IR floating-point condition code.  This includes
/// consideration of global floating-point math flags.
///
ISD::CondCode getFCmpCondCode(FCmpInst::Predicate Pred);

/// getFCmpCodeWithoutNaN - Given an ISD condition code comparing floats,
/// return the equivalent code if we're allowed to assume that NaNs won't occur.
ISD::CondCode getFCmpCodeWithoutNaN(ISD::CondCode CC);

/// getICmpCondCode - Return the ISD condition code corresponding to
/// the given LLVM IR integer condition code.
///
ISD::CondCode getICmpCondCode(ICmpInst::Predicate Pred);

/// Test if the given instruction is in a position to be optimized
/// with a tail-call. This roughly means that it's in a block with
/// a return and there's nothing that needs to be scheduled
/// between it and the return.
///
/// This function only tests target-independent requirements.
bool isInTailCallPosition(ImmutableCallSite CS, Attributes CalleeRetAttr,
                          const TargetLowering &TLI);

bool isInTailCallPosition(SelectionDAG &DAG, SDNode *Node,
                          SDValue &Chain, const TargetLowering &TLI);

} // End llvm namespace

#endif
