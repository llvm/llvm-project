//===-- VerifierInternal.h - Internal verifier infrastructure --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared definitions used by the verifier implementation files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_IR_VERIFIERINTERNAL_H
#define LLVM_LIB_IR_VERIFIERINTERNAL_H

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Printable.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {

struct VerifierSupport {
  raw_ostream *OS;
  const Module &M;
  ModuleSlotTracker MST;
  const Triple &TT;
  const DataLayout &DL;
  LLVMContext &Context;

  /// Track the brokenness of the module while recursively visiting.
  bool Broken = false;
  /// Broken debug info can be "recovered" from by stripping the debug info.
  bool BrokenDebugInfo = false;
  /// Whether to treat broken debug info as an error.
  bool TreatBrokenDebugInfoAsError = true;

  explicit VerifierSupport(raw_ostream *OS, const Module &M)
      : OS(OS), M(M), MST(&M), TT(M.getTargetTriple()), DL(M.getDataLayout()),
        Context(M.getContext()) {}

private:
  void Write(const Module *M) {
    *OS << "; ModuleID = '" << M->getModuleIdentifier() << "'\n";
  }

  void Write(const Value *V) {
    if (V)
      Write(*V);
  }

  void Write(const Value &V) {
    if (isa<Instruction>(V)) {
      V.print(*OS, MST);
      *OS << '\n';
    } else {
      V.printAsOperand(*OS, true, MST);
      *OS << '\n';
    }
  }

  void Write(const DbgRecord *DR) {
    if (DR) {
      DR->print(*OS, MST, false);
      *OS << '\n';
    }
  }

  void Write(DbgVariableRecord::LocationType Type) {
    switch (Type) {
    case DbgVariableRecord::LocationType::Value:
      *OS << "value";
      break;
    case DbgVariableRecord::LocationType::Declare:
      *OS << "declare";
      break;
    case DbgVariableRecord::LocationType::DeclareValue:
      *OS << "declare_value";
      break;
    case DbgVariableRecord::LocationType::Assign:
      *OS << "assign";
      break;
    case DbgVariableRecord::LocationType::End:
      *OS << "end";
      break;
    case DbgVariableRecord::LocationType::Any:
      *OS << "any";
      break;
    };
  }

  void Write(const Metadata *MD) {
    if (!MD)
      return;
    MD->print(*OS, MST, &M);
    *OS << '\n';
  }

  template <class T> void Write(const MDTupleTypedArrayWrapper<T> &MD) {
    Write(MD.get());
  }

  void Write(const NamedMDNode *NMD) {
    if (!NMD)
      return;
    NMD->print(*OS, MST);
    *OS << '\n';
  }

  void Write(Type *T) {
    if (!T)
      return;
    *OS << ' ' << *T;
  }

  void Write(const Comdat *C) {
    if (!C)
      return;
    *OS << *C;
  }

  void Write(const APInt *AI) {
    if (!AI)
      return;
    *OS << *AI << '\n';
  }

  void Write(const unsigned i) { *OS << i << '\n'; }

  // NOLINTNEXTLINE(readability-identifier-naming)
  void Write(const Attribute *A) {
    if (!A)
      return;
    *OS << A->getAsString() << '\n';
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  void Write(const AttributeSet *AS) {
    if (!AS)
      return;
    *OS << AS->getAsString() << '\n';
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  void Write(const AttributeList *AL) {
    if (!AL)
      return;
    AL->print(*OS);
  }

  void Write(Printable P) { *OS << P; }

  template <typename T> void Write(ArrayRef<T> Vs) {
    for (const T &V : Vs)
      Write(V);
  }

  template <typename T1, typename... Ts>
  void WriteTs(const T1 &V1, const Ts &...Vs) {
    Write(V1);
    WriteTs(Vs...);
  }

  template <typename... Ts> void WriteTs() {}

public:
  /// A check failed, so printout out the condition and the message.
  ///
  /// This provides a nice place to put a breakpoint if you want to see why
  /// something is not correct.
  void CheckFailed(const Twine &Message) {
    if (OS)
      *OS << Message << '\n';
    Broken = true;
  }

  /// A check failed (with values to print).
  ///
  /// This calls the Message-only version so that the above is easier to set a
  /// breakpoint on.
  template <typename T1, typename... Ts>
  void CheckFailed(const Twine &Message, const T1 &V1, const Ts &...Vs) {
    CheckFailed(Message);
    if (OS)
      WriteTs(V1, Vs...);
  }

  /// A debug info check failed.
  void DebugInfoCheckFailed(const Twine &Message) {
    if (OS)
      *OS << Message << '\n';
    Broken |= TreatBrokenDebugInfoAsError;
    BrokenDebugInfo = true;
  }

  /// A debug info check failed (with values to print).
  template <typename T1, typename... Ts>
  void DebugInfoCheckFailed(const Twine &Message, const T1 &V1,
                            const Ts &...Vs) {
    DebugInfoCheckFailed(Message);
    if (OS)
      WriteTs(V1, Vs...);
  }
};

//==============================================================================
// AMDGPU-specific verification functions

void verifyAMDGPUModuleFlag(VerifierSupport &VS, const MDString *ID,
                            Module::ModFlagBehavior MFB, const MDNode *Op);

void verifyAMDGPUFunctionMetadata(VerifierSupport &VS, const Function &F);

void verifyAMDGPUAlloca(VerifierSupport &VS, const AllocaInst &AI);

void verifyAMDGPUIntrinsicCall(VerifierSupport &VS, Intrinsic::ID ID,
                               CallBase &Call);

bool isAMDGPUCallBrIntrinsic(Intrinsic::ID ID);

//==============================================================================

} // namespace llvm

#endif // LLVM_LIB_IR_VERIFIERINTERNAL_H
