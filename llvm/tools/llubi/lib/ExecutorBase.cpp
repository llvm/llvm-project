//===- ExecutorBase.cpp - Non-visitor methods of InstExecutor -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements non-visitor methods of InstExecutor for code reuse.
//
//===----------------------------------------------------------------------===//

#include "ExecutorBase.h"

namespace llvm::ubi {
Frame::Frame(Function &F, CallBase *CallSite, Frame *LastFrame,
             ArrayRef<AnyValue> Args, AnyValue &RetVal,
             const TargetLibraryInfoImpl &TLIImpl)
    : Func(F), LastFrame(LastFrame), CallSite(CallSite), Args(Args),
      RetVal(RetVal), TLI(TLIImpl, &F) {
  assert((Args.size() == F.arg_size() ||
          (F.isVarArg() && Args.size() >= F.arg_size())) &&
         "Expected enough arguments to call the function.");
  BB = &Func.getEntryBlock();
  PC = BB->begin();
  for (Argument &Arg : F.args())
    ValueMap[&Arg] = Args[Arg.getArgNo()];
}

DiagnosticReporter ExecutorBase::reportImmediateUB() {
  return DiagnosticReporter(*this, DiagnosticKind::ImmediateUB);
}

DiagnosticReporter ExecutorBase::reportError() {
  return DiagnosticReporter(*this, DiagnosticKind::Error);
}

void ExecutorBase::reportImmediateUBString(StringRef Msg) {
  // Check if we have already reported an immediate UB.
  if (hasProgramExited())
    return;
  dumpStackTrace();
  requestProgramExit(ProgramExitInfo::ProgramExitKind::Failed);
  Handler.onImmediateUB(Msg);
}

void ExecutorBase::reportErrorString(StringRef Msg) {
  // Check if we have already reported an error message.
  if (hasProgramExited())
    return;
  dumpStackTrace();
  requestProgramExit(ProgramExitInfo::ProgramExitKind::Failed);
  Handler.onError(Msg);
}

std::optional<uint64_t> ExecutorBase::verifyMemAccess(const MemoryObject &MO,
                                                      const APInt &Address,
                                                      uint64_t AccessSize,
                                                      Align Alignment,
                                                      bool IsStore) {
  // Loading from a stack object outside its lifetime is not undefined
  // behavior and returns a poison value instead. Storing to it is still
  // undefined behavior.
  if (IsStore ? MO.getState() != MemoryObjectState::Alive
              : MO.getState() == MemoryObjectState::Freed) {
    reportImmediateUB() << "Try to access a dead memory object at address 0x"
                        << Twine::utohexstr(Address.getZExtValue()) << ".";
    return std::nullopt;
  }

  if (Address.countr_zero() < Log2(Alignment)) {
    reportImmediateUB() << "Misaligned memory access. Address: 0x"
                        << Twine::utohexstr(Address.getZExtValue())
                        << ", Required alignment: " << Alignment.value() << ".";
    return std::nullopt;
  }

  if (AccessSize > MO.getSize() || Address.ult(MO.getAddress())) {
    reportImmediateUB() << "Memory access is out of bounds. Accessed size: "
                        << AccessSize << ", Address: 0x"
                        << Twine::utohexstr(Address.getZExtValue())
                        << ", Object base: 0x"
                        << Twine::utohexstr(MO.getAddress())
                        << ", Object size: " << MO.getSize() << ".";
    return std::nullopt;
  }

  APInt Offset = Address - MO.getAddress();

  if (Offset.ugt(MO.getSize() - AccessSize)) {
    reportImmediateUB() << "Memory access is out of bounds. Accessed size: "
                        << AccessSize << ", Address: 0x"
                        << Twine::utohexstr(Address.getZExtValue())
                        << ", Object base: 0x"
                        << Twine::utohexstr(MO.getAddress())
                        << ", Object size: " << MO.getSize() << ".";
    return std::nullopt;
  }

  return Offset.getZExtValue();
}

AnyValue ExecutorBase::load(const AnyValue &Ptr, Align Alignment, Type *ValTy) {
  if (Ptr.isPoison()) {
    reportImmediateUB() << "Invalid memory access with a poison pointer.";
    return AnyValue::getPoisonValue(Ctx, ValTy);
  }
  auto &PtrVal = Ptr.asPointer();
  auto *MO = PtrVal.getMemoryObject();
  if (!MO) {
    reportImmediateUB()
        << "Invalid memory access via a pointer with nullary provenance.";
    return AnyValue::getPoisonValue(Ctx, ValTy);
  }
  // TODO: pointer capability check
  if (auto Offset =
          verifyMemAccess(*MO, PtrVal.address(),
                          Ctx.getEffectiveTypeStoreSize(ValTy), Alignment,
                          /*IsStore=*/false)) {
    // Load from a dead stack object yields poison value.
    if (MO->getState() == MemoryObjectState::Dead)
      return AnyValue::getPoisonValue(Ctx, ValTy);

    return Ctx.load(*MO, *Offset, ValTy);
  }
  return AnyValue::getPoisonValue(Ctx, ValTy);
}

void ExecutorBase::store(const AnyValue &Ptr, Align Alignment,
                         const AnyValue &Val, Type *ValTy) {
  if (Ptr.isPoison()) {
    reportImmediateUB() << "Invalid memory access with a poison pointer.";
    return;
  }
  auto &PtrVal = Ptr.asPointer();
  auto *MO = PtrVal.getMemoryObject();
  if (!MO) {
    reportImmediateUB()
        << "Invalid memory access via a pointer with nullary provenance.";
    return;
  }
  // TODO: pointer capability check
  if (auto Offset =
          verifyMemAccess(*MO, PtrVal.address(),
                          Ctx.getEffectiveTypeStoreSize(ValTy), Alignment,
                          /*IsStore=*/true))
    Ctx.store(*MO, *Offset, Val, ValTy);
}

void ExecutorBase::requestProgramExit(ProgramExitInfo::ProgramExitKind Kind,
                                      uint64_t ExitCode) {
  ExitInfo.emplace(Kind, ExitCode);
  Handler.onProgramExit(*ExitInfo);
}

void ExecutorBase::setFailed() {
  requestProgramExit(ProgramExitInfo::ProgramExitKind::Failed);
}

bool ExecutorBase::hasProgramExited() const { return ExitInfo.has_value(); }

std::optional<ProgramExitInfo> ExecutorBase::getExitInfo() const {
  return ExitInfo;
}

unsigned ExecutorBase::getIntSize() const {
  return CurrentFrame->TLI.getIntSize();
}

void ExecutorBase::dumpStackTrace() const {
  errs() << "Stacktrace:\n";
  const Frame *TheFrame = CurrentFrame;
  unsigned Index = 0;
  while (TheFrame != nullptr) {
    if (TheFrame->BB) {
      Instruction &Inst = *TheFrame->PC;
      errs() << "#" << Index++ << " " << Inst << " at ";
      Inst.getFunction()->printAsOperand(errs(), /*PrintType=*/false);
      errs() << "\n";
    }
    TheFrame = TheFrame->LastFrame;
  }
}
} // namespace llvm::ubi
