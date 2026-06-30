//===- Win64EHDumper.cpp - Win64 EH Printer ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Win64EHDumper.h"
#include "llvm-readobj.h"
#include "llvm/ADT/Enum.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::Win64EH;

// clang-format off
constexpr EnumStringDef<unsigned> UnwindFlagDefs[] = {
  {{"ExceptionHandler"}, UNW_ExceptionHandler},
  {{"TerminateHandler"}, UNW_TerminateHandler},
  {{"ChainInfo"}       , UNW_ChainInfo       },
  {{"Large"}           , UNW_FlagLarge       }
};
constexpr auto UnwindFlags = BUILD_ENUM_STRINGS(UnwindFlagDefs);

constexpr EnumStringDef<unsigned> EpilogFlagDefs[] = {
  {{"ParentFragmentTransfer"}, EPILOG_PARENT_FRAGMENT_TRANSFER},
  {{"Large"}                 , EPILOG_INFO_LARGE              }
};
constexpr auto EpilogFlags = BUILD_ENUM_STRINGS(EpilogFlagDefs);

constexpr EnumStringDef<unsigned> UnwindOpInfoDefs[] = {
  {{"RAX"},  0},
  {{"RCX"},  1},
  {{"RDX"},  2},
  {{"RBX"},  3},
  {{"RSP"},  4},
  {{"RBP"},  5},
  {{"RSI"},  6},
  {{"RDI"},  7},
  {{"R8"},   8},
  {{"R9"},   9},
  {{"R10"}, 10},
  {{"R11"}, 11},
  {{"R12"}, 12},
  {{"R13"}, 13},
  {{"R14"}, 14},
  {{"R15"}, 15}
};
constexpr auto UnwindOpInfo = BUILD_ENUM_STRINGS(UnwindOpInfoDefs);
// clang-format on

static uint64_t getOffsetOfLSDA(const UnwindInfo& UI) {
  return static_cast<const char*>(UI.getLanguageSpecificData())
         - reinterpret_cast<const char*>(&UI);
}

static uint32_t getLargeSlotValue(ArrayRef<UnwindCode> UC) {
  if (UC.size() < 3)
    return 0;
  return UC[1].FrameOffset + (static_cast<uint32_t>(UC[2].FrameOffset) << 16);
}

// Returns the name of the unwind code.
static StringRef getUnwindCodeTypeName(uint8_t Code) {
  switch (Code) {
  default: llvm_unreachable("Invalid unwind code");
  case UOP_PushNonVol: return "PUSH_NONVOL";
  case UOP_AllocLarge: return "ALLOC_LARGE";
  case UOP_AllocSmall: return "ALLOC_SMALL";
  case UOP_SetFPReg: return "SET_FPREG";
  case UOP_SaveNonVol: return "SAVE_NONVOL";
  case UOP_SaveNonVolBig: return "SAVE_NONVOL_FAR";
  case UOP_SaveXMM128: return "SAVE_XMM128";
  case UOP_SaveXMM128Big: return "SAVE_XMM128_FAR";
  case UOP_PushMachFrame: return "PUSH_MACHFRAME";
  case UOP_Epilog:
    return "EPILOG";
  }
}

// Returns the name of a referenced register.
static StringRef getUnwindRegisterName(uint8_t Reg) {
  switch (Reg) {
  default: llvm_unreachable("Invalid register");
  case 0: return "RAX";
  case 1: return "RCX";
  case 2: return "RDX";
  case 3: return "RBX";
  case 4: return "RSP";
  case 5: return "RBP";
  case 6: return "RSI";
  case 7: return "RDI";
  case 8: return "R8";
  case 9: return "R9";
  case 10: return "R10";
  case 11: return "R11";
  case 12: return "R12";
  case 13: return "R13";
  case 14: return "R14";
  case 15: return "R15";
  }
}

// Calculates the number of array slots required for the unwind code.
static unsigned getNumUsedSlots(const UnwindCode &UnwindCode) {
  switch (UnwindCode.getUnwindOp()) {
  default: llvm_unreachable("Invalid unwind code");
  case UOP_PushNonVol:
  case UOP_AllocSmall:
  case UOP_SetFPReg:
  case UOP_PushMachFrame:
  case UOP_Epilog:
    return 1;
  case UOP_SaveNonVol:
  case UOP_SaveXMM128:
    return 2;
  case UOP_SaveNonVolBig:
  case UOP_SaveXMM128Big:
    return 3;
  case UOP_AllocLarge:
    return (UnwindCode.getOpInfo() == 0) ? 2 : 3;
  }
}

static std::error_code getSymbol(const COFFObjectFile &COFF, uint64_t VA,
                                 object::SymbolRef &Sym) {
  for (const auto &Symbol : COFF.symbols()) {
    Expected<uint64_t> Address = Symbol.getAddress();
    if (!Address)
      return errorToErrorCode(Address.takeError());
    if (*Address == VA) {
      Sym = Symbol;
      return std::error_code();
    }
  }
  return inconvertibleErrorCode();
}

static object::SymbolRef getPreferredSymbol(const COFFObjectFile &COFF,
                                            object::SymbolRef Sym,
                                            uint32_t &SymbolOffset,
                                            bool IsRangeEnd) {
  // The symbol resolved by ResolveSymbol can be any internal
  // nondescriptive symbol; try to resolve a more descriptive one.
  COFFSymbolRef CoffSym = COFF.getCOFFSymbol(Sym);
  if (CoffSym.getStorageClass() != COFF::IMAGE_SYM_CLASS_LABEL &&
      CoffSym.getSectionDefinition() == nullptr)
    return Sym;
  for (const auto &S : COFF.symbols()) {
    COFFSymbolRef CS = COFF.getCOFFSymbol(S);
    if (CS.getSectionNumber() == CoffSym.getSectionNumber() &&
        CS.getValue() <= CoffSym.getValue() + SymbolOffset &&
        CS.getStorageClass() != COFF::IMAGE_SYM_CLASS_LABEL &&
        CS.getSectionDefinition() == nullptr) {
      uint32_t Offset = CoffSym.getValue() + SymbolOffset - CS.getValue();
      // For the end of a range, don't pick a symbol with a zero offset;
      // prefer a symbol with a small positive offset.
      if (Offset <= SymbolOffset && (!IsRangeEnd || Offset > 0)) {
        SymbolOffset = Offset;
        Sym = S;
        CoffSym = CS;
        if (CS.isExternal() && SymbolOffset == 0)
          return Sym;
      }
    }
  }
  return Sym;
}

static std::string formatSymbol(const Dumper::Context &Ctx,
                                const coff_section *Section, uint64_t Offset,
                                uint32_t Displacement,
                                bool IsRangeEnd = false) {
  std::string Buffer;
  raw_string_ostream OS(Buffer);

  SymbolRef Symbol;
  if (!Ctx.ResolveSymbol(Section, Offset, Symbol, Ctx.UserData)) {
    // We found a relocation at the given offset in the section, pointing
    // at a symbol.

    // Try to resolve label/section symbols into function names.
    Symbol = getPreferredSymbol(Ctx.COFF, Symbol, Displacement, IsRangeEnd);

    Expected<StringRef> Name = Symbol.getName();
    if (Name) {
      OS << *Name;
      if (Displacement > 0)
        OS << format(" +0x%X (0x%" PRIX64 ")", Displacement, Offset);
      else
        OS << format(" (0x%" PRIX64 ")", Offset);
      return OS.str();
    } else {
      // TODO: Actually report errors helpfully.
      consumeError(Name.takeError());
    }
  } else if (!getSymbol(Ctx.COFF, Ctx.COFF.getImageBase() + Displacement,
                        Symbol)) {
    Expected<StringRef> Name = Symbol.getName();
    if (Name) {
      OS << *Name;
      OS << format(" (0x%" PRIX64 ")", Ctx.COFF.getImageBase() + Displacement);
      return OS.str();
    } else {
      consumeError(Name.takeError());
    }
  }

  if (Displacement > 0)
    OS << format("(0x%" PRIX64 ")", Ctx.COFF.getImageBase() + Displacement);
  else
    OS << format("(0x%" PRIX64 ")", Offset);
  return OS.str();
}

static std::error_code resolveRelocation(const Dumper::Context &Ctx,
                                         const coff_section *Section,
                                         uint64_t Offset,
                                         const coff_section *&ResolvedSection,
                                         uint64_t &ResolvedAddress) {
  SymbolRef Symbol;
  if (std::error_code EC =
          Ctx.ResolveSymbol(Section, Offset, Symbol, Ctx.UserData))
    return EC;

  Expected<uint64_t> ResolvedAddressOrErr = Symbol.getAddress();
  if (!ResolvedAddressOrErr)
    return errorToErrorCode(ResolvedAddressOrErr.takeError());
  ResolvedAddress = *ResolvedAddressOrErr;

  Expected<section_iterator> SI = Symbol.getSection();
  if (!SI)
    return errorToErrorCode(SI.takeError());
  ResolvedSection = Ctx.COFF.getCOFFSection(**SI);
  return std::error_code();
}

static const object::coff_section *
getSectionContaining(const COFFObjectFile &COFF, uint64_t VA) {
  for (const auto &Section : COFF.sections()) {
    uint64_t Address = Section.getAddress();
    uint64_t Size = Section.getSize();

    if (VA >= Address && (VA - Address) <= Size)
      return COFF.getCOFFSection(Section);
  }
  return nullptr;
}

namespace llvm {
namespace Win64EH {
void Dumper::printRuntimeFunctionEntry(const Context &Ctx,
                                       const coff_section *Section,
                                       uint64_t Offset,
                                       const RuntimeFunction &RF) {
  SW.printString("StartAddress",
                 formatSymbol(Ctx, Section, Offset + 0, RF.StartAddress));
  SW.printString("EndAddress",
                 formatSymbol(Ctx, Section, Offset + 4, RF.EndAddress,
                              /*IsRangeEnd=*/true));
  SW.printString("UnwindInfoAddress",
                 formatSymbol(Ctx, Section, Offset + 8, RF.UnwindInfoOffset));
}

// Prints one unwind code. Because an unwind code can occupy up to 3 slots in
// the unwind codes array, this function requires that the correct number of
// slots is provided.
void Dumper::printUnwindCode(const UnwindInfo &UI, ArrayRef<UnwindCode> UC,
                             bool &SeenFirstEpilog) {
  assert(UC.size() >= getNumUsedSlots(UC[0]));

  SW.startLine() << format("0x%02X: ", unsigned(UC[0].u.CodeOffset))
                 << getUnwindCodeTypeName(UC[0].getUnwindOp());

  switch (UC[0].getUnwindOp()) {
  case UOP_PushNonVol:
    OS << " reg=" << getUnwindRegisterName(UC[0].getOpInfo());
    break;

  case UOP_AllocLarge:
    OS << " size="
       << ((UC[0].getOpInfo() == 0) ? UC[1].FrameOffset * 8
                                    : getLargeSlotValue(UC));
    break;

  case UOP_AllocSmall:
    OS << " size=" << (UC[0].getOpInfo() + 1) * 8;
    break;

  case UOP_SetFPReg:
    if (UI.getFrameRegister() == 0)
      OS << " reg=<invalid>";
    else
      OS << " reg=" << getUnwindRegisterName(UI.getFrameRegister())
         << format(", offset=0x%X", UI.getFrameOffset() * 16);
    break;

  case UOP_SaveNonVol:
    OS << " reg=" << getUnwindRegisterName(UC[0].getOpInfo())
       << format(", offset=0x%X", UC[1].FrameOffset * 8);
    break;

  case UOP_SaveNonVolBig:
    OS << " reg=" << getUnwindRegisterName(UC[0].getOpInfo())
       << format(", offset=0x%X", getLargeSlotValue(UC));
    break;

  case UOP_SaveXMM128:
    OS << " reg=XMM" << static_cast<uint32_t>(UC[0].getOpInfo())
       << format(", offset=0x%X", UC[1].FrameOffset * 16);
    break;

  case UOP_SaveXMM128Big:
    OS << " reg=XMM" << static_cast<uint32_t>(UC[0].getOpInfo())
       << format(", offset=0x%X", getLargeSlotValue(UC));
    break;

  case UOP_PushMachFrame:
    OS << " errcode=" << (UC[0].getOpInfo() == 0 ? "no" : "yes");
    break;

  case UOP_Epilog:
    if (SeenFirstEpilog) {
      uint32_t Offset = UC[0].getEpilogOffset();
      if (Offset == 0) {
        OS << " padding";
      } else {
        OS << " offset=" << format("0x%X", Offset);
      }
    } else {
      SeenFirstEpilog = true;
      bool AtEnd = (UC[0].getOpInfo() & 0x1) != 0;
      uint32_t Length = UC[0].u.CodeOffset;
      OS << " atend=" << (AtEnd ? "yes" : "no")
         << ", length=" << format("0x%X", Length);
    }
    break;
  }

  OS << "\n";
}

void Dumper::printUnwindInfo(const Context &Ctx, const coff_section *Section,
                             off_t Offset, const UnwindInfo &UI) {
  DictScope UIS(SW, "UnwindInfo");
  SW.printNumber("Version", UI.getVersion());
  SW.printFlags("Flags", UI.getFlags(), EnumStrings(UnwindFlags));
  SW.printNumber("PrologSize", UI.PrologSize);
  if (UI.getFrameRegister()) {
    SW.printEnum("FrameRegister", UI.getFrameRegister(),
                 EnumStrings(UnwindOpInfo));
    SW.printHex("FrameOffset", UI.getFrameOffset());
  } else {
    SW.printString("FrameRegister", StringRef("-"));
    SW.printString("FrameOffset", StringRef("-"));
  }

  SW.printNumber("UnwindCodeCount", UI.NumCodes);
  {
    ListScope UCS(SW, "UnwindCodes");
    ArrayRef<UnwindCode> UC(&UI.UnwindCodes[0], UI.NumCodes);
    bool SeenFirstEpilog = false;
    for (const UnwindCode *UCI = UC.begin(), *UCE = UC.end(); UCI < UCE; ++UCI) {
      unsigned UsedSlots = getNumUsedSlots(*UCI);
      if (UsedSlots > UC.size()) {
        errs() << "corrupt unwind data";
        return;
      }

      printUnwindCode(UI, ArrayRef(UCI, UCE), SeenFirstEpilog);
      UCI = UCI + UsedSlots - 1;
    }
  }

  uint64_t LSDAOffset = Offset + getOffsetOfLSDA(UI);
  if (UI.getFlags() & (UNW_ExceptionHandler | UNW_TerminateHandler)) {
    SW.printString("Handler",
                   formatSymbol(Ctx, Section, LSDAOffset,
                                UI.getLanguageSpecificHandlerOffset()));
  } else if (UI.getFlags() & UNW_ChainInfo) {
    if (const RuntimeFunction *Chained = UI.getChainedFunctionEntry()) {
      DictScope CS(SW, "Chained");
      printRuntimeFunctionEntry(Ctx, Section, LSDAOffset, *Chained);
    }
  }
}

void Dumper::printRuntimeFunction(const Context &Ctx,
                                  const coff_section *Section,
                                  uint64_t SectionOffset,
                                  const RuntimeFunction &RF) {
  DictScope RFS(SW, "RuntimeFunction");
  printRuntimeFunctionEntry(Ctx, Section, SectionOffset, RF);

  const coff_section *XData = nullptr;
  uint64_t Offset;
  resolveRelocation(Ctx, Section, SectionOffset + 8, XData, Offset);
  Offset = Offset + RF.UnwindInfoOffset;

  if (!XData) {
    uint64_t Address = Ctx.COFF.getImageBase() + RF.UnwindInfoOffset;
    XData = getSectionContaining(Ctx.COFF, Address);
    if (!XData)
      return;
    Offset = RF.UnwindInfoOffset - XData->VirtualAddress;
  }

  ArrayRef<uint8_t> Contents;
  if (Error E = Ctx.COFF.getSectionContents(XData, Contents))
    reportError(std::move(E), Ctx.COFF.getFileName());

  if (Contents.empty())
    return;

  if (Offset > Contents.size())
    return;

  // Check version before casting to UnwindInfo struct.
  // Only byte 0 (VersionAndFlags) is layout-compatible between V1/V2 and V3.
  uint8_t VersionByte = Contents[Offset];
  uint8_t Version = VersionByte & 0x07;

  if (Version == 3) {
    ArrayRef<uint8_t> RawData = Contents.slice(Offset);
    printUnwindInfoV3(Ctx, XData, Offset, RawData);
  } else {
    const auto UI =
        reinterpret_cast<const UnwindInfo *>(Contents.data() + Offset);
    printUnwindInfo(Ctx, XData, Offset, *UI);
  }
}

static void printDecodedWOD(ScopedPrinter &SW, raw_ostream &OS,
                            const DecodedWOD &W) {
  switch (W.Opcode) {
  case WOD_PUSH:
    OS << "PUSH Reg=" << getRegisterNameV3(W.Register);
    break;
  case WOD_PUSH2:
    OS << "PUSH2 Reg1=" << getRegisterNameV3(W.Register)
       << ", Reg2=" << getRegisterNameV3(W.Register2);
    break;
  case WOD_PUSH_CONSECUTIVE_2:
    OS << "PUSH_CONSECUTIVE_2 Reg=" << getRegisterNameV3(W.Register) << " (+"
       << getRegisterNameV3(W.Register + 1) << ")";
    break;
  case WOD_ALLOC_SMALL:
    OS << format("ALLOC_SMALL Size=0x%X", W.Size);
    break;
  case WOD_ALLOC_LARGE:
    OS << format("ALLOC_LARGE Size=0x%X", W.Size);
    break;
  case WOD_ALLOC_HUGE:
    OS << format("ALLOC_HUGE Size=0x%X", W.Size);
    break;
  case WOD_SET_FPREG:
    OS << "SET_FPREG Reg=" << getRegisterNameV3(W.Register)
       << format(", Offset=0x%X", W.Displacement);
    break;
  case WOD_SAVE_NONVOL:
    OS << "SAVE_NONVOL Reg=" << getRegisterNameV3(W.Register)
       << format(", Disp=0x%X", W.Displacement);
    break;
  case WOD_SAVE_NONVOL_FAR:
    OS << "SAVE_NONVOL_FAR Reg=" << getRegisterNameV3(W.Register)
       << format(", Disp=0x%X", W.Displacement);
    break;
  case WOD_SAVE_XMM128:
    OS << "SAVE_XMM128 Reg=XMM" << static_cast<unsigned>(W.Register)
       << format(", Disp=0x%X", W.Displacement);
    break;
  case WOD_SAVE_XMM128_FAR:
    OS << "SAVE_XMM128_FAR Reg=XMM" << static_cast<unsigned>(W.Register)
       << format(", Disp=0x%X", W.Displacement);
    break;
  case WOD_PUSH_CANONICAL_FRAME:
    // TODO: When the Windows x64 Unwind V3 spec is finalized, replace this
    // raw Type value with a descriptive name. Type values are defined by the
    // OS (see the Windows SDK headers) but the set is not yet stable.
    OS << "PUSH_CANONICAL_FRAME Type=" << static_cast<unsigned>(W.Type);
    break;
  }
}

/// Decode and print N WODs from the pool starting at byte offset PoolOffset,
/// pairing each with the corresponding IP offset from IpOffsets.
static void printWODSequence(ScopedPrinter &SW, raw_ostream &OS,
                             ArrayRef<uint8_t> WODPool, unsigned PoolOffset,
                             ArrayRef<uint16_t> IpOffsets, unsigned Count) {
  unsigned CurrentOffset = PoolOffset;
  for (unsigned I = 0; I < Count; ++I) {
    Expected<DecodedWOD> WOrErr = decodeWOD(WODPool, CurrentOffset);
    if (!WOrErr) {
      WithColor::warning(errs()) << toString(WOrErr.takeError()) << "\n";
      return;
    }
    const DecodedWOD &W = *WOrErr;
    SW.startLine() << format("[%u] IP +0x%04X: ", I,
                             I < IpOffsets.size() ? IpOffsets[I] : 0);
    printDecodedWOD(SW, OS, W);
    OS << "\n";
    CurrentOffset += W.ByteSize;
  }
}

void Dumper::printUnwindInfoV3(const Context &Ctx,
                               const object::coff_section *Section,
                               off_t Offset, ArrayRef<uint8_t> Data) {
  DictScope UIS(SW, "UnwindInfo");

  Expected<DecodedUnwindInfoV3> InfoOrErr = decodeUnwindInfoV3(Data);
  if (!InfoOrErr) {
    WithColor::warning(errs()) << toString(InfoOrErr.takeError()) << "\n";
    return;
  }
  const DecodedUnwindInfoV3 &Info = *InfoOrErr;

  SW.printNumber("Version", Info.Version);
  SW.printFlags("Flags", Info.Flags, EnumStrings(UnwindFlags));
  SW.printHex("SizeOfProlog", Info.SizeOfProlog);
  SW.printNumber("PayloadWords", Info.PayloadWords);
  SW.printNumber("NumberOfOps", Info.NumberOfOps);
  SW.printNumber("NumberOfEpilogs", Info.NumberOfEpilogs);

  // Validation: SizeOfProlog must be >= first (largest) prolog IP offset.
  // SizeOfProlog is the total prolog size in bytes, while the first IP offset
  // is the start of the last unwind-affecting instruction within the prolog.
  if (Info.NumberOfOps > 0 && Info.SizeOfProlog < Info.PrologIpOffsets[0]) {
    WithColor::warning(errs())
        << format("SizeOfProlog (%u) is smaller than first prolog IP offset "
                  "(%u)\n",
                  Info.SizeOfProlog, Info.PrologIpOffsets[0]);
  }

  // Per the V3 spec, Flags bit 4 (0x10) is reserved and must be zero. Warn
  // (rather than error) so we stay forward-compatible if Microsoft later
  // defines this bit.
  if (Info.Flags & 0x10)
    WithColor::warning(errs())
        << "V3 unwind info has reserved Flags bit 4 set\n";

  // Print prolog ops
  {
    SW.startLine() << format("Prolog [%u ops]:\n", Info.NumberOfOps);
    SW.indent();
    printWODSequence(SW, OS, Info.WODPool, 0, ArrayRef(Info.PrologIpOffsets),
                     Info.NumberOfOps);
    SW.unindent();
  }

  // Print epilog descriptors
  uint8_t BaseEpilogFlags = 0;
  bool HaveBaseEpilog = false;
  for (unsigned I = 0; I < Info.NumberOfEpilogs; ++I) {
    const DecodedEpilogV3 &Epi = Info.Epilogs[I];

    DictScope ES(SW, formatv("Epilog [{0}]", I).str());
    SW.printFlags("Flags", Epi.Flags, EnumStrings(EpilogFlags));
    // Format the signed EpilogOffset as hex with explicit sign so negative
    // tail-relative offsets remain readable (e.g. "-0x14" rather than
    // "0xFFFFFFEC").
    {
      int32_t SignedOff = static_cast<int32_t>(Epi.EpilogOffset);
      uint32_t AbsOff =
          SignedOff < 0
              ? static_cast<uint32_t>(-static_cast<int64_t>(SignedOff))
              : static_cast<uint32_t>(SignedOff);
      SW.printString(
          "EpilogOffset",
          formatv("{0}0x{1:X-}", SignedOff < 0 ? "-" : "+", AbsOff).str());
    }
    SW.printNumber("NumberOfOps", Epi.NumberOfOps);

    if (Epi.NumberOfOps == 0) {
      if (I == 0) {
        WithColor::warning(errs())
            << "first epilog cannot inherit (NumberOfOps=0)\n";
      } else {
        // Per the V3 spec, Flags bits 0 and 1 are producer-replicated into an
        // inherited descriptor, so they must match the base epilog. Warn if a
        // non-compliant producer left them inconsistent.
        if (HaveBaseEpilog && (Epi.Flags & 0x03) != (BaseEpilogFlags & 0x03))
          WithColor::warning(errs())
              << format("inherited epilog flags (0x%X) do not match base "
                        "epilog flags (0x%X)\n",
                        Epi.Flags & 0x03, BaseEpilogFlags & 0x03);
        // Surface the values inherited from the base epilog so a
        // reader can see what the unwinder will actually execute.
        SW.startLine() << format(
            "(inherits from base epilog: FirstOp=0x%X, "
            "IpOffsetOfLastInstruction=0x%X, %u ops)\n",
            Epi.FirstOp, static_cast<unsigned>(Epi.IpOffsetOfLastInstruction),
            static_cast<unsigned>(Epi.IpOffsets.size()));
      }
    } else {
      SW.printHex("FirstOp", Epi.FirstOp);
      SW.printHex("IpOffsetOfLastInstruction", Epi.IpOffsetOfLastInstruction);
      printWODSequence(SW, OS, Info.WODPool, Epi.FirstOp,
                       ArrayRef(Epi.IpOffsets), Epi.NumberOfOps);
      // This is a full descriptor; it becomes the base that subsequent
      // inherited descriptors replicate their flags from.
      BaseEpilogFlags = Epi.Flags;
      HaveBaseEpilog = true;
    }
  }

  // Optionally dump the WOD pool with byte offsets. This is useful for
  // understanding how WODs are shared between the prolog and epilogs but is
  // normally redundant with the per-prolog / per-epilog decoded output, so
  // it's gated behind --unwind-show-wod-pool.
  if (opts::UnwindShowWODPool) {
    ListScope WS(SW, formatv("WODPool [{0} bytes]", Info.WODPool.size()).str());
    unsigned PoolOffset = 0;
    while (PoolOffset < Info.WODPool.size()) {
      // PayloadWords counts 2-byte words, so the pool may have a single
      // trailing zero padding byte to round up to a word boundary. A bare
      // 0x00 byte is never a valid 1-byte WOD (WOD_ALLOC_SMALL requires the
      // low nibble to be 8), so treat a final zero byte as padding rather
      // than trying to decode it.
      if (PoolOffset + 1 == Info.WODPool.size() &&
          Info.WODPool[PoolOffset] == 0) {
        SW.startLine() << format("+0x%04X: (padding)\n", PoolOffset);
        break;
      }
      Expected<DecodedWOD> WOrErr = decodeWOD(Info.WODPool, PoolOffset);
      if (!WOrErr) {
        WithColor::warning(errs()) << toString(WOrErr.takeError()) << "\n";
        break;
      }
      const DecodedWOD &W = *WOrErr;
      SW.startLine() << format("+0x%04X: ", PoolOffset);
      printDecodedWOD(SW, OS, W);
      OS << "\n";
      PoolOffset += W.ByteSize;
    }
  }

  // Handle exception handler / chain info
  uint64_t LSDAOffset = Offset + Info.PayloadSize;
  if (Info.Flags & (UNW_ExceptionHandler | UNW_TerminateHandler)) {
    if (LSDAOffset + 4 <= Data.size() + Offset) {
      uint32_t HandlerRVA = support::endian::read32le(&Data[Info.PayloadSize]);
      SW.printString("Handler",
                     formatSymbol(Ctx, Section, LSDAOffset, HandlerRVA));
    }
  } else if (Info.Flags & UNW_ChainInfo) {
    if (LSDAOffset + sizeof(RuntimeFunction) <= Data.size() + Offset) {
      const auto *Chained =
          reinterpret_cast<const RuntimeFunction *>(&Data[Info.PayloadSize]);
      DictScope CS(SW, "Chained");
      printRuntimeFunctionEntry(Ctx, Section, LSDAOffset, *Chained);
    }
  }
}

void Dumper::printData(const Context &Ctx) {
  for (const auto &Section : Ctx.COFF.sections()) {
    StringRef Name;
    if (Expected<StringRef> NameOrErr = Section.getName())
      Name = *NameOrErr;
    else
      consumeError(NameOrErr.takeError());

    if (Name != ".pdata" && !Name.starts_with(".pdata$"))
      continue;

    const coff_section *PData = Ctx.COFF.getCOFFSection(Section);
    ArrayRef<uint8_t> Contents;

    if (Error E = Ctx.COFF.getSectionContents(PData, Contents))
      reportError(std::move(E), Ctx.COFF.getFileName());
    if (Contents.empty())
      continue;

    const RuntimeFunction *Entries =
      reinterpret_cast<const RuntimeFunction *>(Contents.data());
    const size_t Count = Contents.size() / sizeof(RuntimeFunction);
    ArrayRef<RuntimeFunction> RuntimeFunctions(Entries, Count);

    size_t Index = 0;
    for (const auto &RF : RuntimeFunctions) {
      printRuntimeFunction(Ctx, Ctx.COFF.getCOFFSection(Section),
                           Index * sizeof(RuntimeFunction), RF);
      ++Index;
    }
  }
}
}
}

