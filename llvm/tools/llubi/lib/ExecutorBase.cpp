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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cstring>

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
                              Align Alignment, bool IsStore) {
  auto *MO = Ctx.checkProvenance(Ptr, [](const Provenance &) {
    // TODO: check provenance
    // TODO: check inrange(S, E)
    return true;
  });
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
    SmallString<32> AddrStr;
    Address.toStringUnsigned(AddrStr, 16);
    reportImmediateUB() << "Try to access a dead memory object at address 0x"
                        << AddrStr << ".";
    return {};
  }

  if (IsStore && MO->isConstant()) {
    reportImmediateUB() << "Try to write to a constant memory object: " << Ptr
                        << ".";
    return {};
  }

  if (Address.countr_zero() < Log2(Alignment)) {
    SmallString<32> AddrStr;
    Address.toStringUnsigned(AddrStr, 16);
    reportImmediateUB() << "Misaligned memory access. Address: 0x" << AddrStr
                        << ", Required alignment: " << Alignment.value() << ".";
    return {};
  }

  if (AccessSize > MO->getSize() || Address.ult(MO->getAddress())) {
    SmallString<32> AddrStr;
    Address.toStringUnsigned(AddrStr, 16);
    reportImmediateUB() << "Memory access is out of bounds. Accessed size: "
                        << AccessSize << ", Address: 0x" << AddrStr
                        << ", Object base: 0x"
                        << Twine::utohexstr(MO->getAddress())
                        << ", Object size: " << MO->getSize() << ".";
    return {};
  }

  APInt Offset = Address - MO->getAddress();

  if (Offset.ugt(MO->getSize() - AccessSize)) {
    SmallString<32> AddrStr;
    Address.toStringUnsigned(AddrStr, 16);
    reportImmediateUB() << "Memory access is out of bounds. Accessed size: "
                        << AccessSize << ", Address: 0x" << AddrStr
                        << ", Object base: 0x"
                        << Twine::utohexstr(MO->getAddress())
                        << ", Object size: " << MO->getSize() << ".";
    return {};
  }

  return {MO, Offset.getZExtValue()};
}

AnyValue ExecutorBase::load(const AnyValue &Ptr, Align Alignment, Type *ValTy,
                            bool NoUndef) {
  if (Ptr.isPoison()) {
    reportImmediateUB() << "Invalid memory access with a poison pointer.";
    return AnyValue::getPoisonValue(Ctx, ValTy);
  }
  auto &PtrVal = Ptr.asPointer();
  if (auto [MO, Offset] = verifyMemAccess(
          PtrVal, Ctx.getEffectiveTypeStoreSize(ValTy), Alignment,
          /*IsStore=*/false);
      MO) {
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
                         const AnyValue &Val, Type *ValTy) {
  if (Ptr.isPoison()) {
    reportImmediateUB() << "Invalid memory access with a poison pointer.";
    return;
  }
  auto &PtrVal = Ptr.asPointer();
  if (auto [MO, Offset] = verifyMemAccess(
          PtrVal, Ctx.getEffectiveTypeStoreSize(ValTy), Alignment,
          /*IsStore=*/true);
      MO)
    Ctx.store(*MO, Offset, Val, ValTy);
}

void ExecutorBase::executeMemTransfer(const AnyValue &Dest, const AnyValue &Src,
                                      const AnyValue &Length, Intrinsic::ID IID,
                                      bool IsLibCall) {
  assert((IID == Intrinsic::memcpy || IID == Intrinsic::memcpy_inline ||
          IID == Intrinsic::memmove) &&
         "Unexpected memory transfer intrinsic");
  const bool IsMemMove = IID == Intrinsic::memmove;
  const StringRef Name = IsMemMove ? "memmove" : "memcpy";

  if (Length.isPoison()) {
    reportImmediateUB() << Name << " called with poison length.";
    return;
  }

  const std::optional<uint64_t> LenOpt = Length.asInteger().tryZExtValue();
  if (!LenOpt) {
    reportImmediateUB() << Name << " called with length overflows uint64_t.";
    return;
  }
  const uint64_t Len = *LenOpt;
  if (Len == 0)
    return;

  if (Dest.isPoison()) {
    reportImmediateUB() << Name << " called with poison destination pointer.";
    return;
  }

  if (Src.isPoison()) {
    reportImmediateUB() << Name << " called with poison source pointer.";
    return;
  }

  const Pointer &DstPtr = Dest.asPointer();
  const Pointer &SrcPtr = Src.asPointer();

  // Alignment attributes have already been handled at the call site.
  auto [SrcMO, SrcOffset] =
      verifyMemAccess(SrcPtr, Len, Align(1), /*IsStore=*/false);
  if (!SrcMO)
    return;

  auto [DstMO, DstOffset] =
      verifyMemAccess(DstPtr, Len, Align(1), /*IsStore=*/true);
  if (!DstMO)
    return;

  if (!IsMemMove && SrcMO == DstMO) {
    const uint64_t SrcEnd = SrcOffset + Len;
    const uint64_t DstEnd = DstOffset + Len;
    // Unlike the LLVM intrinsic, the C memcpy function also considers equal
    // source and destination ranges to overlap.
    if (SrcOffset < DstEnd && DstOffset < SrcEnd &&
        (IsLibCall || SrcOffset != DstOffset)) {
      reportImmediateUB() << "memcpy with overlapping source and destination.";
      return;
    }
  }

  MutableArrayRef<Byte> DstBytes = DstMO->getBytes().slice(DstOffset, Len);
  ArrayRef<Byte> SrcBytes = SrcMO->getBytes().slice(SrcOffset, Len);
  std::memmove(DstBytes.data(), SrcBytes.data(), Len * sizeof(Byte));
}

void ExecutorBase::executeMemSet(const AnyValue &Dest, const AnyValue &Val,
                                 const AnyValue &Length) {
  if (Length.isPoison()) {
    reportImmediateUB() << "memset called with poison length.";
    return;
  }

  const std::optional<uint64_t> LenOpt = Length.asInteger().tryZExtValue();
  if (!LenOpt) {
    reportImmediateUB() << "memset called with length overflows uint64_t.";
    return;
  }
  const uint64_t Len = *LenOpt;
  if (Len == 0)
    return;

  if (Dest.isPoison()) {
    reportImmediateUB() << "memset called with poison destination pointer.";
    return;
  }

  const Pointer &DstPtr = Dest.asPointer();

  // Alignment attributes have already been handled at the call site.
  auto [DstMO, DstOffset] =
      verifyMemAccess(DstPtr, Len, Align(1), /*IsStore=*/true);
  if (!DstMO)
    return;

  Byte FillByte = Val.isPoison()
                      ? Byte::poison()
                      : Byte::concrete(Val.asInteger().getZExtValue());
  fill(DstMO->getBytes().slice(DstOffset, Len), FillByte);
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
