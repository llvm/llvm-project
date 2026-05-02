//===- Library.cpp - Library calls for llubi ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements common libcalls for llubi.
//
//===----------------------------------------------------------------------===//

#include "Library.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm::ubi {

static uint64_t getMaxAlign(const DataLayout &DL) {
  // Return an alignment of 16 for 64-bit platforms, and 8 for 32-bit ones.
  return DL.getPointerABIAlignment(0).value() >= 8 ? 16 : 8;
}

Library::Library(Context &Ctx, EventHandler &Handler, const DataLayout &DL,
                 ExecutorBase &Executor)
    : Ctx(Ctx), Handler(Handler), DL(DL), Executor(Executor) {}

std::optional<std::string> Library::readStringFromMemory(const Pointer &Ptr) {
  auto *MO = Ptr.getMemoryObject();
  if (!MO) {
    Executor.reportImmediateUB(
        "Invalid memory access via a pointer with nullary "
        "provenance.");
    return std::nullopt;
  }

  std::string Result;
  const APInt &Address = Ptr.address();
  uint64_t Offset = 0;

  while (true) {
    auto ValidOffset =
        Executor.verifyMemAccess(*MO, Address + Offset, 1, Align(1), false);
    if (!ValidOffset)
      return std::nullopt;

    Byte B = (*MO)[*ValidOffset];
    if (B.ConcreteMask != 0xFF) {
      Executor.reportImmediateUB("Read uninitialized or poison memory while "
                                 "parsing C-string.");
      return std::nullopt;
    }

    if (B.Value == 0)
      break;

    Result.push_back(static_cast<char>(B.Value));
    ++Offset;
  }

  return Result;
}

AnyValue Library::executeMalloc(StringRef Name, Type *Type,
                                ArrayRef<AnyValue> Args,
                                MemAllocKind AllocKind) {
  assert((AllocKind == MemAllocKind::Malloc || AllocKind == MemAllocKind::New ||
          AllocKind == MemAllocKind::NewArray) &&
         "Unexpected MemAllocKind for malloc()/new/new[]");

  const auto &SizeVal = Args[0];

  const uint64_t AllocSize = SizeVal.asInteger().getZExtValue();

  const IntrusiveRefCntPtr<MemoryObject> Obj =
      Ctx.allocate(AllocSize, getMaxAlign(DL), Name, 0,
                   MemInitKind::Uninitialized, AllocKind);

  if (!Obj) {
    if (AllocKind == MemAllocKind::New || AllocKind == MemAllocKind::NewArray) {
      // FIXME: As llubi doesn't support stack unwinding yet, we report an error
      // when new/new[] fails.
      Executor.reportError("Insufficient heap space.");
      return AnyValue::poison();
    }
    return AnyValue::getNullValue(Ctx, Type);
  }

  return Ctx.deriveFromMemoryObject(Obj);
}

AnyValue Library::executeCalloc(StringRef Name, Type *Type,
                                ArrayRef<AnyValue> Args,
                                MemAllocKind AllocKind) {
  assert(AllocKind == MemAllocKind::Malloc &&
         "Unexpected MemAllocKind for calloc()");

  const auto &CountVal = Args[0];
  const auto &SizeVal = Args[1];

  const APInt &Count = CountVal.asInteger();
  const APInt &Size = SizeVal.asInteger();

  bool Overflow = false;
  const APInt AllocSize = Count.umul_ov(Size, Overflow);
  if (Overflow)
    return AnyValue::getNullValue(Ctx, Type);

  const IntrusiveRefCntPtr<MemoryObject> Obj =
      Ctx.allocate(AllocSize.getLimitedValue(), getMaxAlign(DL), Name, 0,
                   MemInitKind::Zeroed, AllocKind);

  if (!Obj)
    return AnyValue::getNullValue(Ctx, Type);

  return Ctx.deriveFromMemoryObject(Obj);
}

AnyValue Library::executeFree(ArrayRef<AnyValue> Args) {
  const auto &PtrVal = Args[0];

  auto &Ptr = PtrVal.asPointer();
  // no-op when free is called with a null pointer.
  if (Ptr.address().isZero())
    return AnyValue();

  MemoryObject *Obj = Ptr.getMemoryObject();
  if (!Obj) {
    Executor.reportImmediateUB("freeing a pointer with nullary provenance.");
    return AnyValue::poison();
  }

  if (const uint64_t Address = Ptr.address().getZExtValue();
      Address != Obj->getAddress()) {
    Executor.reportImmediateUB(
        "freeing a pointer that does not point to the start of an allocation.");
    return AnyValue::poison();
  }

  if (Obj->getState() == MemoryObjectState::Freed) {
    Executor.reportImmediateUB("double-freeing a memory object.");
    return AnyValue::poison();
  }

  if (!Obj->isHeapAllocated()) {
    Executor.reportImmediateUB("freeing a non-heap allocation.");
    return AnyValue::poison();
  }

  // Currently we don't check for cases where a memory allocated with C
  // allocation family (malloc, calloc, etc.) is freed with a different free
  // function comes from a different family (C++ delete, etc.)

  if (!Ctx.free(*Obj)) {
    Executor.reportImmediateUB("freeing an invalid pointer.");
    return AnyValue::poison();
  }

  return AnyValue();
}

AnyValue Library::executePuts(ArrayRef<AnyValue> Args) {
  const auto &PtrVal = Args[0];

  const auto StrOpt = readStringFromMemory(PtrVal.asPointer());
  if (!StrOpt)
    return AnyValue::poison();

  Handler.onPrint(*StrOpt + "\n");
  return AnyValue(APInt(Executor.getIntSize(), 1));
}

AnyValue Library::executePrintf(ArrayRef<AnyValue> Args) {
  const auto &FormatPtrVal = Args[0];

  const auto FormatStrOpt = readStringFromMemory(FormatPtrVal.asPointer());
  if (!FormatStrOpt)
    return AnyValue::poison();

  const std::string &FormatStr = *FormatStrOpt;
  std::string Output;
  raw_string_ostream OS(Output);
  unsigned ArgIndex = 1; // Start from 1 since 0 is the format string.

  for (unsigned I = 0; I < FormatStr.size();) {
    if (FormatStr[I] != '%') {
      OS << FormatStr[I++];
      continue;
    }

    const size_t Start = I++;
    if (I < FormatStr.size() && FormatStr[I] == '%') {
      OS << '%';
      ++I;
      continue;
    }

    while (I < FormatStr.size() &&
           StringRef("-= #0123456789").contains(FormatStr[I]))
      ++I;

    while (I < FormatStr.size() && StringRef("hljzt").contains(FormatStr[I]))
      ++I;

    if (I >= FormatStr.size()) {
      Executor.reportImmediateUB(
          "Invalid format string in printf: missing conversion "
          "specifier.");
      return AnyValue::poison();
    }

    char Specifier = FormatStr[I++];
    std::string CleanChunk = FormatStr.substr(Start, I - Start - 1);
    CleanChunk.erase(
        llvm::remove_if(CleanChunk,
                        [](char C) { return StringRef("hljzt").contains(C); }),
        CleanChunk.end());

    if (ArgIndex >= Args.size()) {
      Executor.reportImmediateUB(
          "Not enough arguments provided for the format string.");
      return AnyValue::poison();
    }

    const auto &Arg = Args[ArgIndex++];
    if (Arg.isPoison()) {
      Executor.reportImmediateUB("Poison argument passed to printf.");
      return AnyValue::poison();
    }

    switch (Specifier) {
    case 'd':
    case 'i': {
      std::string HostFmt = CleanChunk + "ll" + Specifier;
      OS << format(HostFmt.c_str(),
                   static_cast<long long>(Arg.asInteger().getSExtValue()));
      break;
    }
    case 'u':
    case 'o':
    case 'x':
    case 'X': {
      // FIXME: The format specifiers "b" and "B" are not implemented here
      // since currently MSVC doesn't support it.
      std::string HostFmt = CleanChunk + "ll" + Specifier;
      OS << format(HostFmt.c_str(), static_cast<unsigned long long>(
                                        Arg.asInteger().getZExtValue()));
      break;
    }
    case 'c': {
      std::string HostFmt = CleanChunk + Specifier;
      OS << format(HostFmt.c_str(),
                   static_cast<int>(Arg.asInteger().getZExtValue()));
      break;
    }
    case 'f':
    case 'e':
    case 'E':
    case 'g':
    case 'G':
    case 'a':
    case 'A': {
      std::string HostFmt = CleanChunk + Specifier;
      OS << format(HostFmt.c_str(), Arg.asFloat().convertToDouble());
      break;
    }
    case 'n': {
      OS.flush();
      Executor.store(Arg, Align(4), AnyValue(APInt(32, Output.size())),
                     Type::getInt32Ty(Ctx.getContext()));
      break;
    }
    case 'p': {
      std::string HostFmt = CleanChunk + "llx";
      OS << "0x"
         << format(HostFmt.c_str(),
                   static_cast<unsigned long long>(
                       Arg.asPointer().address().getZExtValue()));
      break;
    }
    case 's': {
      auto StrOpt = readStringFromMemory(Arg.asPointer());
      if (!StrOpt)
        return AnyValue::poison();
      std::string HostFmt = CleanChunk + "s";
      OS << format(HostFmt.c_str(), StrOpt->c_str());
      break;
    }
    default:
      Executor.reportImmediateUB(
          "Unknown or unsupported format specifier in printf.");
      return AnyValue::poison();
    }
  }

  OS.flush();
  Handler.onPrint(Output);
  return AnyValue(APInt(Executor.getIntSize(), Output.size()));
}

AnyValue Library::executeExit(ArrayRef<AnyValue> Args) {
  const auto &RetCodeVal = Args[0];

  Executor.requestProgramExit(ProgramExitInfo::ProgramExitKind::Exited,
                              RetCodeVal.asInteger().getZExtValue());
  return AnyValue();
}

AnyValue Library::executeAbort() {
  Executor.requestProgramExit(ProgramExitInfo::ProgramExitKind::Aborted);
  return AnyValue();
}

AnyValue Library::executeTerminate() {
  Executor.requestProgramExit(ProgramExitInfo::ProgramExitKind::Terminated);
  return AnyValue();
}

std::optional<AnyValue> Library::executeLibcall(LibFunc LF, StringRef Name,
                                                Type *Type,
                                                ArrayRef<AnyValue> Args) {
  for (const AnyValue &Arg : Args) {
    if (Arg.isPoison()) {
      Executor.reportImmediateUB("Poison argument passed to a library call.");
      return AnyValue::poison();
    }
  }

  switch (LF) {
  case LibFunc_malloc:
    return executeMalloc(Name, Type, Args, MemAllocKind::Malloc);
  case LibFunc_Znwm:
    return executeMalloc(Name, Type, Args, MemAllocKind::New);
  case LibFunc_Znam:
    return executeMalloc(Name, Type, Args, MemAllocKind::NewArray);

  case LibFunc_calloc:
    return executeCalloc(Name, Type, Args, MemAllocKind::Malloc);

  case LibFunc_free:
  case LibFunc_ZdaPv:
  case LibFunc_ZdlPv:
    return executeFree(Args);

  case LibFunc_puts:
    return executePuts(Args);

  case LibFunc_printf:
    return executePrintf(Args);

  case LibFunc_exit:
    return executeExit(Args);

  case LibFunc_abort:
    return executeAbort();

  case LibFunc_terminate:
    return executeTerminate();

  default:
    return std::nullopt;
  }
}
} // namespace llvm::ubi
