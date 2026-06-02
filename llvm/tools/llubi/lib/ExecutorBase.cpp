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

std::pair<MemoryObject *, uint64_t>
ExecutorBase::verifyMemAccess(const Pointer &Ptr, uint64_t AccessSize,
                              Align Alignment, bool IsStore, unsigned AS) {
  auto *MO = Ctx.checkProvenance(
      Ptr,
      [](const Provenance &) {
        // TODO: check provenance
        // TODO: check inrange(S, E)
        return true;
      },
      AS);
  if (!MO) {
    reportImmediateUB()
        << "Invalid memory access via a pointer with nullary provenance.";
    return {};
  }
  const APInt &Address = Ptr.address();
  // Loading from a stack object outside its lifetime is not undefined
  // behavior and returns a poison value instead. Storing to it is still
  // undefined behavior.
  if (IsStore ? MO->getState() != MemoryObjectState::Alive
              : MO->getState() == MemoryObjectState::Freed) {
    reportImmediateUB() << "Try to access a dead memory object at address 0x"
                        << Twine::utohexstr(Address.getZExtValue()) << ".";
    return {};
  }

  if (Address.countr_zero() < Log2(Alignment)) {
    reportImmediateUB() << "Misaligned memory access. Address: 0x"
                        << Twine::utohexstr(Address.getZExtValue())
                        << ", Required alignment: " << Alignment.value() << ".";
    return {};
  }

  if (AccessSize > MO->getSize() || Address.ult(MO->getAddress())) {
    reportImmediateUB() << "Memory access is out of bounds. Accessed size: "
                        << AccessSize << ", Address: 0x"
                        << Twine::utohexstr(Address.getZExtValue())
                        << ", Object base: 0x"
                        << Twine::utohexstr(MO->getAddress())
                        << ", Object size: " << MO->getSize() << ".";
    return {};
  }

  APInt Offset = Address - MO->getAddress();

  if (Offset.ugt(MO->getSize() - AccessSize)) {
    reportImmediateUB() << "Memory access is out of bounds. Accessed size: "
                        << AccessSize << ", Address: 0x"
                        << Twine::utohexstr(Address.getZExtValue())
                        << ", Object base: 0x"
                        << Twine::utohexstr(MO->getAddress())
                        << ", Object size: " << MO->getSize() << ".";
    return {};
  }

  return {MO, Offset.getZExtValue()};
}

AnyValue ExecutorBase::load(const AnyValue &Ptr, Align Alignment, Type *ValTy,
                            bool NoUndef, unsigned AS) {
  if (Ptr.isPoison()) {
    reportImmediateUB() << "Invalid memory access with a poison pointer.";
    return AnyValue::getPoisonValue(Ctx, ValTy);
  }
  auto &PtrVal = Ptr.asPointer();
  if (auto [MO, Offset] = verifyMemAccess(
          PtrVal, Ctx.getEffectiveTypeStoreSize(ValTy), Alignment,
          /*IsStore=*/false, AS);
      MO) {
    // Load from a dead stack object yields poison value.
    if (MO->getState() == MemoryObjectState::Dead)
      return AnyValue::getPoisonValue(Ctx, ValTy);

    bool ContainsUndefinedBits = false;
    AnyValue Res = Ctx.load(*MO, Offset, ValTy,
                            NoUndef ? &ContainsUndefinedBits : nullptr);
    if (NoUndef && ContainsUndefinedBits)
      reportImmediateUB() << "The value loaded contains undefined bits.";
    return Res;
  }
  return AnyValue::getPoisonValue(Ctx, ValTy);
}

void ExecutorBase::store(const AnyValue &Ptr, Align Alignment,
                         const AnyValue &Val, Type *ValTy, unsigned AS) {
  if (Ptr.isPoison()) {
    reportImmediateUB() << "Invalid memory access with a poison pointer.";
    return;
  }
  auto &PtrVal = Ptr.asPointer();
  if (auto [MO, Offset] = verifyMemAccess(
          PtrVal, Ctx.getEffectiveTypeStoreSize(ValTy), Alignment,
          /*IsStore=*/true, AS);
      MO) {
    if (MO->isConstant()) {
      reportImmediateUB() << "Try to write to a constant memory object: "
                          << PtrVal << ".";
      return;
    }
    Ctx.store(*MO, Offset, Val, ValTy);
  }
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
  const AsmParserContext *ParserContext = Ctx.getParserContext();
  StringRef ModuleFileName = Ctx.getModule().getModuleIdentifier();
  while (TheFrame != nullptr) {
    if (TheFrame->BB) {
      Instruction &Inst = *TheFrame->PC;
      errs() << "#" << Index++ << " " << Inst << " at ";
      Inst.getFunction()->printAsOperand(errs(), /*PrintType=*/false);
      if (ParserContext) {
        if (auto Loc = ParserContext->getInstructionOrArgumentLocation(&Inst))
          errs() << ' ' << ModuleFileName << ':' << Loc->Start.Line + 1;
      }
      errs() << "\n";
    }
    TheFrame = TheFrame->LastFrame;
  }
}
} // namespace llvm::ubi
