//===- PerfJITUnwindInfo.cpp - Prepare CFI for perf ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PerfJITUnwindInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::dwarf;

namespace {

Error invalidCFI(StringRef Message) {
  return make_error<StringError>(Message, inconvertibleErrorCode());
}

// Each reader is bounded by its record or augmentation, so truncated fields
// cannot consume bytes belonging to the next record.
struct Reader {
  StringRef Data;
  uint64_t Pos, End;
  std::string Failure;

  void fail(StringRef Message) {
    if (Failure.empty())
      Failure = Message.str();
  }
  StringRef readBytes(uint64_t Size) {
    if (!Failure.empty())
      return {};
    if (Size > End - Pos) {
      fail("truncated CFI record");
      return {};
    }
    StringRef Result = Data.substr(Pos, Size);
    Pos += Size;
    return Result;
  }
  uint64_t readUnsigned(unsigned Size) {
    StringRef B = readBytes(Size);
    if (B.size() != Size)
      return 0;
    uint64_t Value = 0;
    for (unsigned I = 0; I != Size; ++I)
      Value |= uint64_t(uint8_t(B[I])) << (8 * I);
    return Value;
  }
  uint64_t readULEB() {
    if (!Failure.empty())
      return 0;
    unsigned Size = 0;
    const char *Err = nullptr;
    auto *P = reinterpret_cast<const uint8_t *>(Data.data());
    uint64_t Value = decodeULEB128(P + Pos, &Size, P + End, &Err);
    if (Err)
      fail(Err);
    else
      Pos += Size;
    return Value;
  }
  void readSLEB() {
    if (!Failure.empty())
      return;
    unsigned Size = 0;
    const char *Err = nullptr;
    auto *P = reinterpret_cast<const uint8_t *>(Data.data());
    decodeSLEB128(P + Pos, &Size, P + End, &Err);
    if (Err)
      fail(Err);
    else
      Pos += Size;
  }
  StringRef readString() {
    if (!Failure.empty())
      return {};
    size_t Size = Data.substr(Pos, End - Pos).find('\0');
    if (Size == StringRef::npos) {
      fail("unterminated CIE augmentation");
      return {};
    }
    return readBytes(Size + 1).drop_back();
  }
};

uint64_t readPCRelative32(Reader &R, uint64_t FrameAddr) {
  uint64_t Base = FrameAddr + R.Pos;
  int64_t Delta = int32_t(R.readUnsigned(4));
  if ((Delta < 0 && uint64_t(-Delta) > Base) ||
      (Delta >= 0 && uint64_t(Delta) > UINT64_MAX - Base)) {
    R.fail("CFI PC-relative address overflows");
    return 0;
  }
  return Delta < 0 ? Base - uint64_t(-Delta) : Base + uint64_t(Delta);
}

// The first implementation deliberately supports ordinary compiler-generated
// register rules. Decline set_loc and expressions rather than copying embedded
// addresses into a differently laid out ELF.
Error checkProgram(StringRef Program) {
  Reader R{Program, 0, Program.size(), {}};
  while (R.Pos < R.End && R.Failure.empty()) {
    uint8_t Op = R.readUnsigned(1);
    if ((Op & 0xc0) == DW_CFA_offset) {
      R.readULEB();
      continue;
    }
    if (Op & 0xc0) // advance_loc or restore.
      continue;
    switch (Op) {
    case DW_CFA_nop:
    case DW_CFA_remember_state:
    case DW_CFA_restore_state:
      break;
    case DW_CFA_advance_loc1:
      R.readBytes(1);
      break;
    case DW_CFA_advance_loc2:
      R.readBytes(2);
      break;
    case DW_CFA_advance_loc4:
      R.readBytes(4);
      break;
    case DW_CFA_restore_extended:
    case DW_CFA_undefined:
    case DW_CFA_same_value:
    case DW_CFA_def_cfa_register:
    case DW_CFA_def_cfa_offset:
    case DW_CFA_GNU_args_size:
      R.readULEB();
      break;
    case DW_CFA_def_cfa_offset_sf:
      R.readSLEB();
      break;
    case DW_CFA_offset_extended:
    case DW_CFA_register:
    case DW_CFA_def_cfa:
    case DW_CFA_val_offset:
      R.readULEB();
      R.readULEB();
      break;
    case DW_CFA_offset_extended_sf:
    case DW_CFA_def_cfa_sf:
    case DW_CFA_val_offset_sf:
      R.readULEB();
      R.readSLEB();
      break;
    default:
      return invalidCFI("unsupported perf CFI instruction");
    }
  }
  return R.Failure.empty() ? Error::success() : invalidCFI(R.Failure);
}

void append32(std::string &Out, uint32_t Value) {
  char Bytes[4];
  support::endian::write32le(Bytes, Value);
  Out.append(Bytes, sizeof(Bytes));
}

Error appendRecord(std::string &Out, std::string Body) {
  // The following header points back to .eh_frame using pcrel+sdata4.
  // Reserve four bytes for the zero-length terminator.
  constexpr uint64_t MaxFrameSize = INT32_MAX - 4;
  if (Out.size() > MaxFrameSize - 4 || Body.size() > MaxFrameSize - 8 ||
      alignTo(Body.size() + 4, uint64_t(8)) > MaxFrameSize - 4 - Out.size())
    return invalidCFI("perf CFI is too large");
  Body.resize(alignTo(Body.size() + 4, uint64_t(8)) - 4, char(DW_CFA_nop));
  append32(Out, Body.size());
  Out += Body;
  return Error::success();
}

} // namespace

Expected<PerfJITUnwindInfo> PerfJITUnwindInfo::create(StringRef Frame,
                                                      uint64_t FrameAddr,
                                                      uint8_t AddressSize,
                                                      bool IsLittleEndian) {
  if (AddressSize != 8 || !IsLittleEndian)
    return invalidCFI("unsupported perf CFI target");
  if (Frame.size() > UINT64_MAX - FrameAddr)
    return invalidCFI("CFI section address overflows");
  PerfJITUnwindInfo Info;
  DenseMap<uint64_t, size_t> CIEIndices;
  Reader R{Frame, 0, Frame.size(), {}};
  while (R.Pos < R.End) {
    uint64_t Offset = R.Pos, Length = R.readUnsigned(4);
    if (!R.Failure.empty())
      return invalidCFI(R.Failure);
    if (!Length)
      break;
    if (Length == UINT32_MAX)
      return invalidCFI("DWARF64 perf CFI is not supported");
    if (Length < 4 || Length > R.End - R.Pos)
      return invalidCFI("truncated CFI record length");
    uint64_t End = R.Pos + Length;
    Reader E{Frame, R.Pos, End, {}};
    uint64_t IDOffset = E.Pos, ID = E.readUnsigned(4);
    if (!ID) {
      CIE C{};
      C.Version = E.readUnsigned(1);
      StringRef Augmentation = E.readString();
      if (C.Version != 1 && C.Version != 3)
        return invalidCFI("unsupported CIE version");
      if (Augmentation != "zR" && Augmentation != "zPLR")
        return invalidCFI("unsupported CIE augmentation");
      uint64_t Parameters = E.Pos;
      uint64_t CodeAlignment = E.readULEB();
      E.readSLEB(); // Data alignment.
      if (C.Version == 1)
        E.readUnsigned(1); // Return register.
      else
        E.readULEB();
      C.Parameters = Frame.substr(Parameters, E.Pos - Parameters);
      uint64_t Size = E.readULEB();
      if (!E.Failure.empty() || !CodeAlignment || Size > End - E.Pos)
        return invalidCFI("invalid CIE parameters or augmentation size");
      Reader A{Frame, E.Pos, E.Pos + Size, {}};
      if (Augmentation == "zPLR") {
        // Personality and LSDA are not needed by the profiler. Consume their
        // common encodings without ever dereferencing the personality pointer.
        uint8_t P = A.readUnsigned(1);
        if (P == 0 || P == 0x80)
          A.readBytes(8);
        else if (P == 0x03 || P == 0x83 || P == 0x1b || P == 0x9b)
          A.readBytes(4);
        else
          return invalidCFI("unsupported personality encoding");
        uint8_t L = A.readUnsigned(1);
        if (L != DW_EH_PE_omit && L != (DW_EH_PE_pcrel | DW_EH_PE_sdata4))
          return invalidCFI("unsupported LSDA encoding");
        C.HasLSDA = L != DW_EH_PE_omit;
      }
      uint8_t Encoding = A.readUnsigned(1);
      if (!A.Failure.empty() || A.Pos != A.End ||
          Encoding != (DW_EH_PE_pcrel | DW_EH_PE_sdata4))
        return invalidCFI("unsupported or malformed FDE encoding");
      C.Instructions = Frame.substr(A.End, End - A.End);
      CIEIndices[Offset] = Info.CIEs.size();
      Info.CIEs.push_back(C);
    } else {
      if (ID > IDOffset)
        return invalidCFI("FDE references a missing CIE");
      auto It = CIEIndices.find(IDOffset - ID);
      if (It == CIEIndices.end())
        return invalidCFI("FDE references a missing CIE");
      uint64_t PC = readPCRelative32(E, FrameAddr);
      uint64_t Range = E.readUnsigned(4);
      uint64_t Size = E.readULEB();
      if (!E.Failure.empty() || Range > INT32_MAX || Range > UINT64_MAX - PC ||
          Size != (Info.CIEs[It->second].HasLSDA ? 4u : 0u))
        return invalidCFI("invalid FDE range or augmentation");
      E.readBytes(Size);
      if (!E.Failure.empty())
        return invalidCFI(E.Failure);
      Info.FDEs.push_back(
          {PC, Range, Frame.substr(E.Pos, End - E.Pos), It->second});
    }
    R.Pos = End;
  }
  return Info;
}

Expected<std::string> PerfJITUnwindInfo::emitEHFrame(uint64_t CodeAddr,
                                                     uint64_t CodeSize) const {
  if (!CodeSize || CodeSize > INT32_MAX || CodeSize > UINT64_MAX - CodeAddr)
    return invalidCFI("empty or oversized perf code range");
  // Native ELF64: .text follows the ELF and program headers at 0x80.
  // perf places .eh_frame after the function's 8-byte-aligned .text.
  constexpr uint64_t TextAddress = 0x80;
  uint64_t FrameAddress = TextAddress + alignTo(CodeSize, uint64_t(8));
  std::string Result;
  DenseMap<size_t, uint64_t> EmittedCIEs;
  for (const FDE &F : FDEs) {
    if (!F.Range || F.InitialLocation >= CodeAddr + CodeSize ||
        CodeAddr >= F.InitialLocation + F.Range)
      continue;
    uint64_t NewPC;
    if (F.InitialLocation < CodeAddr) {
      if (CodeAddr - F.InitialLocation > TextAddress)
        return invalidCFI("perf FDE starts before the synthetic ELF");
      NewPC = TextAddress - (CodeAddr - F.InitialLocation);
    } else {
      NewPC = TextAddress + (F.InitialLocation - CodeAddr);
    }
    const CIE &C = CIEs[F.CIEIndex];
    if (!EmittedCIEs.count(F.CIEIndex)) {
      if (auto Err = checkProgram(C.Instructions))
        return std::move(Err);
      EmittedCIEs[F.CIEIndex] = Result.size();
      std::string Body(4, '\0');
      Body += char(C.Version);
      Body.append("zR\0", 3);
      Body += C.Parameters;
      Body.append("\x01\x1b", 2); // Augmentation length and FDE encoding.
      Body += C.Instructions;
      if (auto Err = appendRecord(Result, std::move(Body)))
        return std::move(Err);
    }
    if (auto Err = checkProgram(F.Instructions))
      return std::move(Err);
    uint64_t Offset = Result.size();
    uint64_t Magnitude = FrameAddress + Offset + 8 - NewPC;
    if (Magnitude > uint64_t(INT32_MAX) + 1)
      return invalidCFI("perf FDE address is out of range");
    std::string Body;
    append32(Body, Offset + 4 - EmittedCIEs.lookup(F.CIEIndex));
    append32(Body, uint32_t(-Magnitude));
    append32(Body, F.Range);
    Body += '\0'; // No LSDA in profiler-only CFI.
    Body += F.Instructions;
    if (auto Err = appendRecord(Result, std::move(Body)))
      return std::move(Err);
  }
  if (Result.empty())
    return invalidCFI("no FDE covers this function");
  append32(Result, 0);
  return Result;
}
