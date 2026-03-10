//===- Library.cpp - Library Function Simulator for llubi -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Library.h"
#include <cstdarg>

namespace llvm::ubi {
static uint64_t getMaxAlignT(const DataLayout &DL) {
  return DL.getPointerABIAlignment(0).value() >= 8 ? 16 : 8;
}

std::optional<std::string>
LibraryEnvironment::readStringFromMemory(const Pointer &Ptr) {
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
std::optional<AnyValue> LibraryEnvironment::executeMalloc(CallBase &CB) {
  const auto SizeVal = Executor.getValue(CB.getArgOperand(0));
  if (SizeVal.isPoison()) {
    Executor.reportImmediateUB("malloc called with a poison size.");
    return std::nullopt;
  }

  const uint64_t AllocSize = SizeVal.asInteger().getZExtValue();
  const uint64_t MaxAlign = getMaxAlignT(DL);

  const auto Obj = Ctx.allocate(AllocSize, MaxAlign, CB.getName(), 0,
                                MemInitKind::Uninitialized);

  if (!Obj)
    return AnyValue::getNullValue(Ctx, CB.getType());

  return Ctx.deriveFromMemoryObject(Obj);
}
std::optional<AnyValue> LibraryEnvironment::executeCalloc(CallBase &CB) {
  const auto CountVal = Executor.getValue(CB.getArgOperand(0));
  const auto SizeVal = Executor.getValue(CB.getArgOperand(1));

  if (CountVal.isPoison()) {
    Executor.reportImmediateUB("calloc called with a poison count.");
    return std::nullopt;
  }
  if (SizeVal.isPoison()) {
    Executor.reportImmediateUB("calloc called with a poison size.");
    return std::nullopt;
  }

  const uint64_t Count = CountVal.asInteger().getZExtValue();
  const uint64_t Size = SizeVal.asInteger().getZExtValue();

  bool Overflow;
  const uint64_t AllocSize = SaturatingMultiply(Count, Size, &Overflow);
  if (Overflow) {
    return AnyValue::getNullValue(Ctx, CB.getType());
  }

  const uint64_t MaxAlign = getMaxAlignT(DL);

  auto Obj =
      Ctx.allocate(AllocSize, MaxAlign, CB.getName(), 0, MemInitKind::Zeroed);

  if (!Obj) {
    return AnyValue::getNullValue(Ctx, CB.getType());
  }

  return Ctx.deriveFromMemoryObject(Obj);
}
std::optional<AnyValue> LibraryEnvironment::executeFree(CallBase &CB) {
  const auto PtrVal = Executor.getValue(CB.getArgOperand(0));
  if (PtrVal.isPoison()) {
    Executor.reportImmediateUB("free called with a poison pointer.");
    return std::nullopt;
  }

  auto &Ptr = PtrVal.asPointer();
  if (Ptr.address().isZero()) {
    // no-op when free is called with a null pointer.
    return AnyValue();
  }

  if (!Ctx.free(Ptr.address().getZExtValue())) {
    Executor.reportImmediateUB(
        "freeing an invalid, unallocated, or already freed pointer.");
    return std::nullopt;
  }

  return AnyValue();
}
std::optional<AnyValue> LibraryEnvironment::executePuts(CallBase &CB) {
  const auto PtrVal = Executor.getValue(CB.getArgOperand(0));
  if (PtrVal.isPoison()) {
    Executor.reportImmediateUB("puts called with a poison pointer.");
    return std::nullopt;
  }

  const auto StrOpt = readStringFromMemory(PtrVal.asPointer());
  if (!StrOpt) {
    return std::nullopt;
  }

  Handler.onPrint(*StrOpt + "\n");
  return AnyValue(APInt(32, 1));
}
std::optional<AnyValue> LibraryEnvironment::executePrintf(CallBase &CB) {
  auto FormatPtrVal = Executor.getValue(CB.getArgOperand(0));
  if (FormatPtrVal.isPoison()) {
    Executor.reportImmediateUB(
        "printf called with a poison format string pointer.");
    return std::nullopt;
  }

  auto FormatStrOpt = readStringFromMemory(FormatPtrVal.asPointer());
  if (!FormatStrOpt) {
    return std::nullopt;
  }

  std::string FormatStr = *FormatStrOpt;
  std::string Output;
  unsigned ArgIndex = 1; // Start from 1 since 0 is the format string.

  for (size_t i = 0; i < FormatStr.size();) {
    if (FormatStr[i] != '%') {
      Output.push_back(FormatStr[i++]);
      continue;
    }

    size_t Start = i++;
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
      return std::nullopt;
    }

    char Specifier = FormatStr[i++];
    std::string CleanChunk = FormatStr.substr(Start, i - Start - 1);
    CleanChunk.erase(std::remove_if(CleanChunk.begin(), CleanChunk.end(),
                                    [](char c) { return strchr("hljzt", c); }),
                     CleanChunk.end());

    if (ArgIndex >= CB.arg_size()) {
      Executor.reportImmediateUB(
          "Not enough arguments provided for the format string.");
      return std::nullopt;
    }

    AnyValue Arg = Executor.getValue(CB.getArgOperand(ArgIndex++));
    if (Arg.isPoison()) {
      Executor.reportImmediateUB("Poison argument passed to printf.");
      return std::nullopt;
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
        return std::nullopt;
      std::string HostFmt = CleanChunk + "s";
      snprintf(Buf, sizeof(Buf), HostFmt.c_str(), StrOpt->c_str());
      Output += Buf;
      break;
    }
    default:
      Executor.reportImmediateUB("Unknown format specifier in printf.");
      return std::nullopt;
    }
  }

  Handler.onPrint(Output);
  return AnyValue(APInt(32, Output.size()));
}
std::optional<AnyValue> LibraryEnvironment::executeExit(CallBase &CB) {
  const auto CodeVal = Executor.getValue(CB.getArgOperand(0));
  if (CodeVal.isPoison()) {
    Executor.reportImmediateUB("exit called with a poison code.");
    return std::nullopt;
  }
  Executor.requestProgramExit(ProgramExitKind::Exit,
                              CodeVal.asInteger().getZExtValue());
  return std::nullopt;
}
std::optional<AnyValue> LibraryEnvironment::executeAbort(CallBase &CB) {
  Executor.requestProgramExit(ProgramExitKind::Abort);
  return std::nullopt;
}
std::optional<AnyValue> LibraryEnvironment::executeTerminate(CallBase &CB) {
  Executor.requestProgramExit(ProgramExitKind::Terminate);
  return std::nullopt;
}
std::optional<AnyValue> LibraryEnvironment::call(LibFunc LF, CallBase &CB) {
  switch (LF) {
  case LibFunc_malloc:
  case LibFunc_Znwm:
  case LibFunc_Znam: {
    return executeMalloc(CB);
  }

  case LibFunc_calloc: {
    return executeCalloc(CB);
  }

  case LibFunc_free:
  case LibFunc_ZdaPv:
  case LibFunc_ZdlPv: {
    return executeFree(CB);
  }

  case LibFunc_puts: {
    return executePuts(CB);
  }

  case LibFunc_printf: {
    return executePrintf(CB);
  }

  case LibFunc_exit: {
    return executeExit(CB);
  }

  case LibFunc_abort: {
    return executeAbort(CB);
  }

  case LibFunc_terminate: {
    return executeTerminate(CB);
  }

  default: {
    return std::nullopt;
  }
  }
}
} // namespace llvm::ubi

