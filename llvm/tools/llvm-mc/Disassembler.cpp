//===- Disassembler.cpp - Disassembler for hex strings --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class implements the disassembler of strings of bytes written in
// hexadecimal, from standard input or from a file.
//
//===----------------------------------------------------------------------===//

#include "Disassembler.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

typedef std::pair<std::vector<unsigned char>, std::vector<const char *>>
    ByteArrayTy;

static MCDisassembler::DecodeStatus getInstruction(const MCDisassembler &DisAsm,
                                                   const MCSubtargetInfo &STI,
                                                   MCInst &Inst, uint64_t &Size,
                                                   ArrayRef<uint8_t> Bytes,
                                                   uint64_t Address) {
  if (STI.getTargetTriple().getArch() == Triple::hexagon)
    return DisAsm.getInstructionBundle(Inst, Size, Bytes, Address, nulls());
  return DisAsm.getInstruction(Inst, Size, Bytes, Address, nulls());
}

static bool printInsts(const MCDisassembler &DisAsm, const ByteArrayTy &Bytes,
                       SourceMgr &SM, MCStreamer &Streamer, bool InAtomicBlock,
                       const MCSubtargetInfo &STI, unsigned NumBenchmarkRuns) {
  ArrayRef<uint8_t> Data(Bytes.first);

  // Disassemble it to strings.
  uint64_t Size;

  for (uint64_t Index = 0; Index < Bytes.first.size(); Index += Size) {

    MCInst Inst;
    MCDisassembler::DecodeStatus S =
        getInstruction(DisAsm, STI, Inst, Size, Data.slice(Index), Index);
    switch (S) {
    case MCDisassembler::Fail:
      SM.PrintMessage(SMLoc::getFromPointer(Bytes.second[Index]),
                      SourceMgr::DK_Warning,
                      "invalid instruction encoding");
      // Don't try to resynchronise the stream in a block
      if (InAtomicBlock)
        return true;

      if (Size == 0)
        Size = 1; // skip illegible bytes

      break;

    case MCDisassembler::SoftFail:
      SM.PrintMessage(SMLoc::getFromPointer(Bytes.second[Index]),
                      SourceMgr::DK_Warning,
                      "potentially undefined instruction encoding");
      [[fallthrough]];

    case MCDisassembler::Success:
      Streamer.emitInstruction(Inst, STI);
      break;
    }

    if (S == MCDisassembler::Success && NumBenchmarkRuns != 0) {
      // Benchmark mode, collect timing for decoding the instruction several
      // times.
      MCInst BMInst;
      TimeTraceScope timeScope("getInstruction");
      for (unsigned I = 0; I < NumBenchmarkRuns; ++I) {
        BMInst.clear();
        BMInst.setOpcode(0);
        S = getInstruction(DisAsm, STI, BMInst, Size, Data.slice(Index), Index);
      }
    }
  }

  return false;
}

static bool SkipToToken(StringRef &Str) {
  for (;;) {
    if (Str.empty())
      return false;

    // Strip horizontal whitespace and commas.
    if (size_t Pos = Str.find_first_not_of(" \t\r\n,")) {
      Str = Str.substr(Pos);
      continue;
    }

    // If this is the start of a comment, remove the rest of the line.
    if (Str[0] == '#') {
        Str = Str.substr(Str.find_first_of('\n'));
      continue;
    }
    return true;
  }
}

static bool byteArrayFromString(ByteArrayTy &ByteArray, StringRef &Str,
                                SourceMgr &SM, bool HexBytes) {
  while (SkipToToken(Str)) {
    // Handled by higher level
    if (Str[0] == '[' || Str[0] == ']')
      return false;

    // Get the current token.
    size_t Next = Str.find_first_of(" \t\n\r,#[]");
    StringRef Value = Str.substr(0, Next);

    // Convert to a byte and add to the byte vector.
    unsigned ByteVal;
    if (HexBytes) {
      if (Next < 2) {
        SM.PrintMessage(SMLoc::getFromPointer(Value.data()),
                        SourceMgr::DK_Error, "expected two hex digits");
        Str = Str.substr(Next);
        return true;
      }
      Next = 2;
      unsigned C0 = hexDigitValue(Value[0]);
      unsigned C1 = hexDigitValue(Value[1]);
      if (C0 == -1u || C1 == -1u) {
        SM.PrintMessage(SMLoc::getFromPointer(Value.data()),
                        SourceMgr::DK_Error, "invalid input token");
        Str = Str.substr(Next);
        return true;
      }
      ByteVal = C0 * 16 + C1;
    } else if (Value.getAsInteger(0, ByteVal) || ByteVal > 255) {
      // If we have an error, print it and skip to the end of line.
      SM.PrintMessage(SMLoc::getFromPointer(Value.data()), SourceMgr::DK_Error,
                      "invalid input token");
      Str = Str.substr(Str.find('\n'));
      ByteArray.first.clear();
      ByteArray.second.clear();
      continue;
    }

    ByteArray.first.push_back(ByteVal);
    ByteArray.second.push_back(Value.data());
    Str = Str.substr(Next);
  }

  return false;
}

int Disassembler::disassemble(const Target &T, const std::string &Triple,
                              MCSubtargetInfo &STI, MCStreamer &Streamer,
                              MemoryBuffer &Buffer, SourceMgr &SM,
                              MCContext &Ctx, const MCTargetOptions &MCOptions,
                              bool HexBytes, unsigned NumBenchmarkRuns) {
  std::unique_ptr<const MCRegisterInfo> MRI(T.createMCRegInfo(Triple));
  if (!MRI) {
    errs() << "error: no register info for target " << Triple << "\n";
    return -1;
  }

  std::unique_ptr<const MCAsmInfo> MAI(
      T.createMCAsmInfo(*MRI, Triple, MCOptions));
  if (!MAI) {
    errs() << "error: no assembly info for target " << Triple << "\n";
    return -1;
  }

  std::unique_ptr<const MCDisassembler> DisAsm(
    T.createMCDisassembler(STI, Ctx));
  if (!DisAsm) {
    errs() << "error: no disassembler for target " << Triple << "\n";
    return -1;
  }

  bool ErrorOccurred = false;

  // Convert the input to a vector for disassembly.
  ByteArrayTy ByteArray;
  StringRef Str = Buffer.getBuffer();
  bool InAtomicBlock = false;

  while (SkipToToken(Str)) {
    ByteArray.first.clear();
    ByteArray.second.clear();

    if (Str[0] == '[') {
      if (InAtomicBlock) {
        SM.PrintMessage(SMLoc::getFromPointer(Str.data()), SourceMgr::DK_Error,
                        "nested atomic blocks make no sense");
        ErrorOccurred = true;
      }
      InAtomicBlock = true;
      Str = Str.drop_front();
      continue;
    } else if (Str[0] == ']') {
      if (!InAtomicBlock) {
        SM.PrintMessage(SMLoc::getFromPointer(Str.data()), SourceMgr::DK_Error,
                        "attempt to close atomic block without opening");
        ErrorOccurred = true;
      }
      InAtomicBlock = false;
      Str = Str.drop_front();
      continue;
    }

    // It's a real token, get the bytes and emit them
    ErrorOccurred |= byteArrayFromString(ByteArray, Str, SM, HexBytes);

    if (!ByteArray.first.empty())
      ErrorOccurred |= printInsts(*DisAsm, ByteArray, SM, Streamer,
                                  InAtomicBlock, STI, NumBenchmarkRuns);
  }

  if (InAtomicBlock) {
    SM.PrintMessage(SMLoc::getFromPointer(Str.data()), SourceMgr::DK_Error,
                    "unclosed atomic block");
    ErrorOccurred = true;
  }

  return ErrorOccurred;
}
