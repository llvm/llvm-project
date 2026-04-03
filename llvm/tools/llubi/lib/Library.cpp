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
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/InstrTypes.h"

namespace llvm::ubi {

static uint64_t getMaxAlignT(const DataLayout &DL) {
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
  const uint64_t Address = Ptr.address().getZExtValue();
  uint64_t Offset = 0;

  while (true) {
    auto ValidOffset = Executor.verifyMemAccess(
        *MO, APInt(DL.getPointerSizeInBits(0), Address + Offset), 1, Align(1),
        false);
    if (!ValidOffset) {
      return std::nullopt;
    }

    Byte B = (*MO)[*ValidOffset];
    if (B.ConcreteMask != 0xFF) {
      Executor.reportImmediateUB("Read uninitialized or poison memory while "
                                 "parsing C-string.");
      return std::nullopt;
    }

    if (B.Value == 0) {
      break;
    }

    Result.push_back(static_cast<char>(B.Value));
    ++Offset;
  }

  return Result;
}

AnyValue Library::executeMalloc(StringRef Name, Type *Type,
                                ArrayRef<AnyValue> Args) {
  const auto &SizeVal = Args[0];
  if (SizeVal.isPoison()) {
    Executor.reportImmediateUB("malloc() called with a poison size.");
    return AnyValue::poison();
  }

  const uint64_t AllocSize = SizeVal.asInteger().getZExtValue();
  const uint64_t MaxAlign = getMaxAlignT(DL);

  const auto Obj =
      Ctx.allocate(AllocSize, MaxAlign, Name, 0, MemInitKind::Uninitialized);

  if (!Obj)
    return AnyValue::getNullValue(Ctx, Type);

  return Ctx.deriveFromMemoryObject(Obj);
}

AnyValue Library::executeCalloc(StringRef Name, Type *Type,
                                ArrayRef<AnyValue> Args) {
  const auto &CountVal = Args[0];
  const auto &SizeVal = Args[1];

  if (CountVal.isPoison()) {
    Executor.reportImmediateUB("calloc() called with a poison count.");
    return AnyValue::poison();
  }
  if (SizeVal.isPoison()) {
    Executor.reportImmediateUB("calloc() called with a poison size.");
    return AnyValue::poison();
  }

  const uint64_t Count = CountVal.asInteger().getZExtValue();
  const uint64_t Size = SizeVal.asInteger().getZExtValue();

  bool Overflow;
  const uint64_t AllocSize = SaturatingMultiply(Count, Size, &Overflow);
  if (Overflow) {
    return AnyValue::getNullValue(Ctx, Type);
  }

  const uint64_t MaxAlign = getMaxAlignT(DL);

  // TODO: Figure out how to name the allocation
  const auto Obj =
      Ctx.allocate(AllocSize, MaxAlign, Name, 0, MemInitKind::Zeroed);

  if (!Obj) {
    return AnyValue::getNullValue(Ctx, Type);
  }

  return Ctx.deriveFromMemoryObject(Obj);
}

AnyValue Library::executeFree(StringRef Name, Type *Type,
                              ArrayRef<AnyValue> Args) {
  const auto &PtrVal = Args[0];
  if (PtrVal.isPoison()) {
    Executor.reportImmediateUB("free() called with a poison pointer.");
    return AnyValue::poison();
  }

  auto &Ptr = PtrVal.asPointer();
  if (Ptr.address().isZero()) {
    // no-op when free is called with a null pointer.
    return AnyValue();
  }

  if (!Ctx.free(Ptr.address().getZExtValue())) {
    Executor.reportImmediateUB(
        "freeing an invalid, unallocated, or already freed pointer.");
    return AnyValue::poison();
  }

  return AnyValue();
}

AnyValue Library::executePuts(StringRef Name, Type *Type,
                              ArrayRef<AnyValue> Args) {
  const auto &PtrVal = Args[0];
  if (PtrVal.isPoison()) {
    Executor.reportImmediateUB("puts called with a poison pointer.");
    return AnyValue::poison();
  }

  const auto StrOpt = readStringFromMemory(PtrVal.asPointer());
  if (!StrOpt) {
    return AnyValue::poison();
  }

  Handler.onPrint(*StrOpt + "\n");
  return AnyValue(APInt(32, 1));
}

AnyValue Library::executePrintf(StringRef Name, Type *Type,
                                ArrayRef<AnyValue> Args) {
  const auto &FormatPtrVal = Args[0];
  if (FormatPtrVal.isPoison()) {
    Executor.reportImmediateUB(
        "printf called with a poison format string pointer.");
    return AnyValue::poison();
  }

  const auto FormatStrOpt = readStringFromMemory(FormatPtrVal.asPointer());
  if (!FormatStrOpt) {
    return AnyValue::poison();
  }

  const std::string FormatStr = *FormatStrOpt;
  std::string Output;
  unsigned ArgIndex = 1; // Start from 1 since 0 is the format string.

  for (size_t i = 0; i < FormatStr.size();) {
    if (FormatStr[i] != '%') {
      Output.push_back(FormatStr[i++]);
      continue;
    }

    const size_t Start = i++;
    if (i < FormatStr.size() && FormatStr[i] == '%') {
      Output.push_back('%');
      ++i;
      continue;
    }

    while (i < FormatStr.size() && strchr("-= #0123456789", FormatStr[i])) {
      ++i;
    }

    while (i < FormatStr.size() && strchr("hljzt", FormatStr[i])) {
      ++i;
    }

    if (i >= FormatStr.size()) {
      Executor.reportImmediateUB(
          "Invalid format string in printf: missing conversion "
          "specifier.");
      return AnyValue::poison();
    }

    char Specifier = FormatStr[i++];
    std::string CleanChunk = FormatStr.substr(Start, i - Start - 1);
    CleanChunk.erase(std::remove_if(CleanChunk.begin(), CleanChunk.end(),
                                    [](char c) { return strchr("hljzt", c); }),
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

    char Buf[1024];
    switch (Specifier) {
    case 'd':
    case 'i': {
      std::string HostFmt = CleanChunk + "ll" + Specifier;
      snprintf(Buf, sizeof(Buf), HostFmt.c_str(),
               static_cast<long long>(Arg.asInteger().getSExtValue()));
      Output += Buf;
      break;
    }
    case 'u':
    case 'o':
    case 'x':
    case 'X':
    case 'c': {
      std::string HostFmt = CleanChunk + "ll" + Specifier;
      snprintf(Buf, sizeof(Buf), HostFmt.c_str(),
               static_cast<unsigned long long>(Arg.asInteger().getZExtValue()));
      Output += Buf;
      break;
    }
    case 'f':
    case 'e':
    case 'E':
    case 'g':
    case 'G': {
      std::string HostFmt = CleanChunk + Specifier;
      snprintf(Buf, sizeof(Buf), HostFmt.c_str(),
               Arg.asFloat().convertToDouble());
      Output += Buf;
      break;
    }
    case 'p': {
      std::string HostFmt = CleanChunk + "llx";
      snprintf(Buf, sizeof(Buf), HostFmt.c_str(),
               static_cast<unsigned long long>(
                   Arg.asPointer().address().getZExtValue()));
      Output += "0x";
      Output += Buf;
      break;
    }
    case 's': {
      auto StrOpt = readStringFromMemory(Arg.asPointer());
      if (!StrOpt)
        return AnyValue::poison();
      std::string HostFmt = CleanChunk + "s";
      snprintf(Buf, sizeof(Buf), HostFmt.c_str(), StrOpt->c_str());
      Output += Buf;
      break;
    }
    default:
      Executor.reportImmediateUB("Unknown format specifier in printf.");
      return AnyValue::poison();
    }
  }

  Handler.onPrint(Output);
  return AnyValue(APInt(32, Output.size()));
}

AnyValue Library::executeExit(StringRef Name, Type *Type,
                              ArrayRef<AnyValue> Args) {
  const auto &RetCodeVal = Args[0];

  if (RetCodeVal.isPoison()) {
    Executor.reportImmediateUB("exit() called with a poison exit code.");
    return AnyValue::poison();
  }

  Executor.requestProgramExit(ProgramExitInfo::ProgramExitKind::Exited,
                              RetCodeVal.asInteger().getZExtValue());
  return AnyValue();
}

AnyValue Library::executeAbort(StringRef Name, Type *Type,
                               ArrayRef<AnyValue> Args) {
  Executor.requestProgramExit(ProgramExitInfo::ProgramExitKind::Aborted);
  return AnyValue();
}

AnyValue Library::executeTerminate(StringRef Name, Type *Type,
                                   ArrayRef<AnyValue> Args) {
  Executor.requestProgramExit(ProgramExitInfo::ProgramExitKind::Terminated);
  return AnyValue();
}

std::optional<AnyValue> Library::executeLibcall(LibFunc LF, StringRef Name,
                                                Type *Type,
                                                ArrayRef<AnyValue> Args) {
  switch (LF) {
  case LibFunc_malloc:
  case LibFunc_Znwm:
  case LibFunc_Znam:
    return executeMalloc(Name, Type, Args);

  case LibFunc_calloc:
    return executeCalloc(Name, Type, Args);

  case LibFunc_free:
  case LibFunc_ZdaPv:
  case LibFunc_ZdlPv:
    return executeFree(Name, Type, Args);

  case LibFunc_puts:
    return executePuts(Name, Type, Args);

  case LibFunc_printf:
    return executePrintf(Name, Type, Args);

  case LibFunc_exit:
    return executeExit(Name, Type, Args);

  case LibFunc_abort:
    return executeAbort(Name, Type, Args);

  case LibFunc_terminate:
    return executeTerminate(Name, Type, Args);

  default:
    return std::nullopt;
  }
}
} // namespace llvm::ubi
