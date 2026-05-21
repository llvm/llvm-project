//===-- Win64EH.cpp - Win64 EH V3 Support -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements decoding helpers for V3 unwind information on Win64.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Win64EH.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

using namespace llvm;
using namespace llvm::Win64EH;

StringRef Win64EH::getRegisterNameV3(unsigned Reg) {
  static const char *const Names[] = {
      "RAX", "RCX", "RDX", "RBX", "RSP", "RBP", "RSI", "RDI",
      "R8",  "R9",  "R10", "R11", "R12", "R13", "R14", "R15",
      "R16", "R17", "R18", "R19", "R20", "R21", "R22", "R23",
      "R24", "R25", "R26", "R27", "R28", "R29", "R30", "R31",
  };
  if (Reg >= std::size(Names))
    return "<invalid>";
  return Names[Reg];
}

Expected<DecodedWOD> Win64EH::decodeWOD(ArrayRef<uint8_t> Pool,
                                        unsigned Offset) {
  if (Offset >= Pool.size())
    return createStringError("WOD pool overflow at offset %u", Offset);

  uint8_t FirstByte = Pool[Offset];
  DecodedWOD W = {};

  // Determine opcode from variable-width prefix encoding.
  // The dispatch order matters: check shorter prefixes first since they
  // occupy the lowest bits, then fall through to longer prefixes.
  //   3-bit prefix (bits [2:0] >= 4): opcodes 4-7
  //   4-bit prefix (bits [3:0] >= 8): opcodes 8-10
  //   6-bit prefix (bits [5:0] == 0x20): opcode 32 (PUSH2)
  //   8-bit prefix (full byte 0-3): opcodes 0-3
  uint8_t Low3 = FirstByte & 0x07;

  // 3-bit opcode: bits [2:0] in {4, 5, 6, 7}
  if (Low3 >= 4) {
    switch (Low3) {
    case WOD_PUSH: {
      W.Opcode = WOD_PUSH;
      W.ByteSize = 1;
      W.Register = (FirstByte >> 3) & 0x1F; // 5-bit register
      return W;
    }
    case WOD_SAVE_NONVOL_FAR: {
      W.Opcode = WOD_SAVE_NONVOL_FAR;
      W.ByteSize = 5;
      if (Offset + 5 > Pool.size())
        return createStringError("WOD_SAVE_NONVOL_FAR truncated at offset %u",
                                 Offset);
      W.Register = (FirstByte >> 3) & 0x1F;
      W.Displacement = support::endian::read32le(&Pool[Offset + 1]);
      return W;
    }
    case WOD_SAVE_NONVOL: {
      W.Opcode = WOD_SAVE_NONVOL;
      W.ByteSize = 3;
      if (Offset + 3 > Pool.size())
        return createStringError("WOD_SAVE_NONVOL truncated at offset %u",
                                 Offset);
      W.Register = (FirstByte >> 3) & 0x1F;
      W.Displacement =
          (uint32_t)support::endian::read16le(&Pool[Offset + 1]) * 8;
      return W;
    }
    case WOD_PUSH_CONSECUTIVE_2: {
      W.Opcode = WOD_PUSH_CONSECUTIVE_2;
      W.ByteSize = 1;
      W.Register = (FirstByte >> 3) & 0x1F;
      return W;
    }
    default:
      return createStringError("unknown WOD opcode 0x%02X at pool offset %u",
                               FirstByte, Offset);
    }
  }

  // 4-bit opcode: bits [3:0] in {8, 9, 10, ...}
  uint8_t Low4 = FirstByte & 0x0F;
  if (Low4 >= 8) {
    switch (Low4) {
    case WOD_ALLOC_SMALL: {
      W.Opcode = WOD_ALLOC_SMALL;
      W.ByteSize = 1;
      W.Size = (unsigned)(((FirstByte >> 4) & 0x0F) + 1) * 8;
      return W;
    }
    case WOD_SAVE_XMM128_FAR: {
      W.Opcode = WOD_SAVE_XMM128_FAR;
      W.ByteSize = 5;
      if (Offset + 5 > Pool.size())
        return createStringError("WOD_SAVE_XMM128_FAR truncated at offset %u",
                                 Offset);
      W.Register = (FirstByte >> 4) & 0x0F;
      W.Displacement = support::endian::read32le(&Pool[Offset + 1]);
      return W;
    }
    case WOD_SAVE_XMM128: {
      W.Opcode = WOD_SAVE_XMM128;
      W.ByteSize = 3;
      if (Offset + 3 > Pool.size())
        return createStringError("WOD_SAVE_XMM128 truncated at offset %u",
                                 Offset);
      W.Register = (FirstByte >> 4) & 0x0F;
      W.Displacement =
          (uint32_t)support::endian::read16le(&Pool[Offset + 1]) * 16;
      return W;
    }
    default:
      return createStringError("unknown WOD opcode 0x%02X at pool offset %u",
                               FirstByte, Offset);
    }
  }

  // 6-bit opcode: bits [5:0] == 0x20 (WOD_PUSH2)
  uint8_t Low6 = FirstByte & 0x3F;
  if (Low6 == WOD_PUSH2) {
    W.Opcode = WOD_PUSH2;
    W.ByteSize = 2;
    if (Offset + 2 > Pool.size())
      return createStringError("WOD_PUSH2 truncated at offset %u", Offset);
    uint8_t SecondByte = Pool[Offset + 1];
    // First reg from bits [7:6] of first byte (2 bits) and bits [2:0] of second
    // (3 bits)
    W.Register = ((FirstByte >> 6) & 0x03) | ((SecondByte & 0x07) << 2);
    W.Register2 = (SecondByte >> 3) & 0x1F;
    return W;
  }

  // 8-bit opcode: full byte is opcode (values 0-3)
  switch (FirstByte) {
  case WOD_SET_FPREG: {
    W.Opcode = WOD_SET_FPREG;
    W.ByteSize = 2;
    if (Offset + 2 > Pool.size())
      return createStringError("WOD_SET_FPREG truncated at offset %u", Offset);
    uint8_t SecondByte = Pool[Offset + 1];
    W.Register = SecondByte & 0x0F; // 4-bit register
    W.Displacement = (unsigned)((SecondByte >> 4) & 0x0F) * 16;
    return W;
  }
  case WOD_ALLOC_HUGE: {
    W.Opcode = WOD_ALLOC_HUGE;
    W.ByteSize = 5;
    if (Offset + 5 > Pool.size())
      return createStringError("WOD_ALLOC_HUGE truncated at offset %u", Offset);
    W.Size = support::endian::read32le(&Pool[Offset + 1]);
    return W;
  }
  case WOD_ALLOC_LARGE: {
    W.Opcode = WOD_ALLOC_LARGE;
    W.ByteSize = 3;
    if (Offset + 3 > Pool.size())
      return createStringError("WOD_ALLOC_LARGE truncated at offset %u",
                               Offset);
    W.Size = (uint32_t)support::endian::read16le(&Pool[Offset + 1]) * 8;
    return W;
  }
  case WOD_PUSH_CANONICAL_FRAME: {
    W.Opcode = WOD_PUSH_CANONICAL_FRAME;
    W.ByteSize = 2;
    if (Offset + 2 > Pool.size())
      return createStringError(
          "WOD_PUSH_CANONICAL_FRAME truncated at offset %u", Offset);
    W.Type = Pool[Offset + 1];
    return W;
  }
  default:
    return createStringError("unknown WOD opcode 0x%02X at pool offset %u",
                             FirstByte, Offset);
  }
}

Expected<DecodedUnwindInfoV3>
Win64EH::decodeUnwindInfoV3(ArrayRef<uint8_t> Data) {
  if (Data.size() < 4)
    return createStringError("V3 unwind info too short: %zu bytes",
                             Data.size());

  DecodedUnwindInfoV3 Info;
  Info.Version = Data[0] & 0x07;
  Info.Flags = (Data[0] >> 3) & 0x1F;
  Info.SizeOfProlog = Data[1];
  Info.CountOfCodes = Data[2];
  Info.NumberOfOps = Data[3] & 0x1F;
  Info.NumberOfEpilogs = (Data[3] >> 5) & 0x07;

  unsigned Offset = 4; // Start of payload

  // Read prolog IP offsets (one byte each)
  for (unsigned I = 0; I < Info.NumberOfOps; ++I) {
    if (Offset >= Data.size())
      return createStringError(
          "V3 payload truncated reading prolog IP offset %u", I);
    Info.PrologIpOffsets.push_back(Data[Offset++]);
  }

  // Read epilog descriptors
  for (unsigned I = 0; I < Info.NumberOfEpilogs; ++I) {
    DecodedEpilogV3 Epi;
    if (Offset >= Data.size())
      return createStringError(
          "V3 payload truncated reading epilog %u FlagsAndNumOps", I);
    uint8_t FlagsAndNumOps = Data[Offset++];
    Epi.Flags = FlagsAndNumOps & 0x07;
    Epi.NumberOfOps = (FlagsAndNumOps >> 3) & 0x1F;

    if (Offset + 2 > Data.size())
      return createStringError(
          "V3 payload truncated reading epilog %u EpilogOffset", I);
    Epi.EpilogOffset =
        static_cast<int16_t>(support::endian::read16le(&Data[Offset]));
    Offset += 2;

    // Inherited descriptors (NumberOfOps == 0) are only 3 bytes:
    // FlagsAndNumOps(1) + EpilogOffset(2). They have no FirstOp,
    // IpOffsetOfLastInstruction, or IP offset fields.
    if (Epi.NumberOfOps == 0) {
      Epi.FirstOp = 0;
      Epi.IpOffsetOfLastInstruction = 0;
      Info.Epilogs.push_back(std::move(Epi));
      continue;
    }

    if (Offset + 2 > Data.size())
      return createStringError("V3 payload truncated reading epilog %u FirstOp",
                               I);
    Epi.FirstOp = support::endian::read16le(&Data[Offset]);
    Offset += 2;

    if (Offset >= Data.size())
      return createStringError(
          "V3 payload truncated reading epilog %u IpOffsetOfLastInstruction",
          I);
    Epi.IpOffsetOfLastInstruction = Data[Offset++];

    // Read epilog IP offsets (one byte each)
    for (unsigned J = 0; J < Epi.NumberOfOps; ++J) {
      if (Offset >= Data.size())
        return createStringError(
            "V3 payload truncated reading epilog %u IP offset %u", I, J);
      Epi.IpOffsets.push_back(Data[Offset++]);
    }

    Info.Epilogs.push_back(std::move(Epi));
  }

  // Identify WOD pool: everything from current offset until end of
  // CountOfCodes * 2 bytes (payload area)
  unsigned PayloadEnd = 4 + Info.CountOfCodes * 2;
  if (PayloadEnd > Data.size())
    PayloadEnd = Data.size();
  unsigned WODPoolStart = Offset;
  unsigned WODPoolEnd = PayloadEnd;
  if (WODPoolStart < WODPoolEnd)
    Info.WODPool = Data.slice(WODPoolStart, WODPoolEnd - WODPoolStart);
  else
    Info.WODPool = ArrayRef<uint8_t>();

  Info.PayloadSize = PayloadEnd;

  return Info;
}
