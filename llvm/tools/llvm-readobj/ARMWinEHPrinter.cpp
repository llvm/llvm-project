//===-- ARMWinEHPrinter.cpp - Windows on ARM EH Data Printer ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Windows on ARM uses a series of serialised data structures (RuntimeFunction)
// to create a table of information for unwinding.  In order to conserve space,
// there are two different ways that this data is represented.
//
// For functions with canonical forms for the prologue and epilogue, the data
// can be stored in a "packed" form.  In this case, the data is packed into the
// RuntimeFunction's remaining 30-bits and can fully describe the entire frame.
//
//        +---------------------------------------+
//        |         Function Entry Address        |
//        +---------------------------------------+
//        |           Packed Form Data            |
//        +---------------------------------------+
//
// This layout is parsed by Decoder::dumpPackedEntry.  No unwind bytecode is
// associated with such a frame as they can be derived from the provided data.
// The decoder does not synthesize this data as it is unnecessary for the
// purposes of validation, with the synthesis being required only by a proper
// unwinder.
//
// For functions that are large or do not match canonical forms, the data is
// split up into two portions, with the actual data residing in the "exception
// data" table (.xdata) with a reference to the entry from the "procedure data"
// (.pdata) entry.
//
// The exception data contains information about the frame setup, all of the
// epilogue scopes (for functions for which there are multiple exit points) and
// the associated exception handler.  Additionally, the entry contains byte-code
// describing how to unwind the function (c.f. Decoder::decodeOpcodes).
//
//        +---------------------------------------+
//        |         Function Entry Address        |
//        +---------------------------------------+
//        |      Exception Data Entry Address     |
//        +---------------------------------------+
//
// This layout is parsed by Decoder::dumpUnpackedEntry.  Such an entry must
// first resolve the exception data entry address.  This structure
// (ExceptionDataRecord) has a variable sized header
// (c.f. ARM::WinEH::HeaderWords) and encodes most of the same information as
// the packed form.  However, because this information is insufficient to
// synthesize the unwinding, there are associated unwinding bytecode which make
// up the bulk of the Decoder.
//
// The decoder itself is table-driven, using the first byte to determine the
// opcode and dispatching to the associated printing routine.  The bytecode
// itself is a variable length instruction encoding that can fully describe the
// state of the stack and the necessary operations for unwinding to the
// beginning of the frame.
//
// The byte-code maintains a 1-1 instruction mapping, indicating both the width
// of the instruction (Thumb2 instructions are variable length, 16 or 32 bits
// wide) allowing the program to unwind from any point in the prologue, body, or
// epilogue of the function.

#include "ARMWinEHPrinter.h"
#include "Error.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ARMWinEH.h"
#include "llvm/Support/Format.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support;

namespace llvm {
raw_ostream &operator<<(raw_ostream &OS, const ARM::WinEH::ReturnType &RT) {
  switch (RT) {
  case ARM::WinEH::ReturnType::RT_POP:
    OS << "pop {pc}";
    break;
  case ARM::WinEH::ReturnType::RT_B:
    OS << "b target";
    break;
  case ARM::WinEH::ReturnType::RT_BW:
    OS << "b.w target";
    break;
  case ARM::WinEH::ReturnType::RT_NoEpilogue:
    OS << "(no epilogue)";
    break;
  }
  return OS;
}
}

static std::string formatSymbol(StringRef Name, uint64_t Address,
                                uint64_t Offset = 0) {
  std::string Buffer;
  raw_string_ostream OS(Buffer);

  if (!Name.empty())
    OS << Name << " ";

  if (Offset)
    OS << format("+0x%X (0x%" PRIX64 ")", Offset, Address);
  else if (!Name.empty())
    OS << format("(0x%" PRIX64 ")", Address);
  else
    OS << format("0x%" PRIX64, Address);

  return OS.str();
}

namespace llvm {
namespace ARM {
namespace WinEH {
const size_t Decoder::PDataEntrySize = sizeof(RuntimeFunction);

// TODO name the uops more appropriately
const Decoder::RingEntry Decoder::Ring[] = {
  { 0x80, 0x00, &Decoder::opcode_0xxxxxxx },  // UOP_STACK_FREE (16-bit)
  { 0xc0, 0x80, &Decoder::opcode_10Lxxxxx },  // UOP_POP (32-bit)
  { 0xf0, 0xc0, &Decoder::opcode_1100xxxx },  // UOP_STACK_SAVE (16-bit)
  { 0xf8, 0xd0, &Decoder::opcode_11010Lxx },  // UOP_POP (16-bit)
  { 0xf8, 0xd8, &Decoder::opcode_11011Lxx },  // UOP_POP (32-bit)
  { 0xf8, 0xe0, &Decoder::opcode_11100xxx },  // UOP_VPOP (32-bit)
  { 0xfc, 0xe8, &Decoder::opcode_111010xx },  // UOP_STACK_FREE (32-bit)
  { 0xfe, 0xec, &Decoder::opcode_1110110L },  // UOP_POP (16-bit)
  { 0xff, 0xee, &Decoder::opcode_11101110 },  // UOP_MICROSOFT_SPECIFIC (16-bit)
                                              // UOP_PUSH_MACHINE_FRAME
                                              // UOP_PUSH_CONTEXT
                                              // UOP_PUSH_TRAP_FRAME
                                              // UOP_REDZONE_RESTORE_LR
  { 0xff, 0xef, &Decoder::opcode_11101111 },  // UOP_LDRPC_POSTINC (32-bit)
  { 0xff, 0xf5, &Decoder::opcode_11110101 },  // UOP_VPOP (32-bit)
  { 0xff, 0xf6, &Decoder::opcode_11110110 },  // UOP_VPOP (32-bit)
  { 0xff, 0xf7, &Decoder::opcode_11110111 },  // UOP_STACK_RESTORE (16-bit)
  { 0xff, 0xf8, &Decoder::opcode_11111000 },  // UOP_STACK_RESTORE (16-bit)
  { 0xff, 0xf9, &Decoder::opcode_11111001 },  // UOP_STACK_RESTORE (32-bit)
  { 0xff, 0xfa, &Decoder::opcode_11111010 },  // UOP_STACK_RESTORE (32-bit)
  { 0xff, 0xfb, &Decoder::opcode_11111011 },  // UOP_NOP (16-bit)
  { 0xff, 0xfc, &Decoder::opcode_11111100 },  // UOP_NOP (32-bit)
  { 0xff, 0xfd, &Decoder::opcode_11111101 },  // UOP_NOP (16-bit) / END
  { 0xff, 0xfe, &Decoder::opcode_11111110 },  // UOP_NOP (32-bit) / END
  { 0xff, 0xff, &Decoder::opcode_11111111 },  // UOP_END
};

void Decoder::printRegisters(const std::pair<uint16_t, uint32_t> &RegisterMask) {
  static const char * const GPRRegisterNames[16] = {
    "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10",
    "r11", "ip", "sp", "lr", "pc",
  };

  const uint16_t GPRMask = std::get<0>(RegisterMask);
  const uint16_t VFPMask = std::get<1>(RegisterMask);

  OS << '{';
  bool Comma = false;
  for (unsigned RI = 0, RE = 11; RI < RE; ++RI) {
    if (GPRMask & (1 << RI)) {
      if (Comma)
        OS << ", ";
      OS << GPRRegisterNames[RI];
      Comma = true;
    }
  }
  for (unsigned RI = 0, RE = 32; RI < RE; ++RI) {
    if (VFPMask & (1 << RI)) {
      if (Comma)
        OS << ", ";
      OS << "d" << unsigned(RI);
      Comma = true;
    }
  }
  for (unsigned RI = 11, RE = 16; RI < RE; ++RI) {
    if (GPRMask & (1 << RI)) {
      if (Comma)
        OS << ", ";
      OS << GPRRegisterNames[RI];
      Comma = true;
    }
  }
  OS << '}';
}

ErrorOr<object::SectionRef>
Decoder::getSectionContaining(const COFFObjectFile &COFF, uint64_t VA) {
  for (const auto &Section : COFF.sections()) {
    uint64_t Address = Section.getAddress();
    uint64_t Size = Section.getSize();

    if (VA >= Address && (VA - Address) <= Size)
      return Section;
  }
  return readobj_error::unknown_symbol;
}

ErrorOr<object::SymbolRef> Decoder::getSymbol(const COFFObjectFile &COFF,
                                              uint64_t VA, bool FunctionOnly) {
  for (const auto &Symbol : COFF.symbols()) {
    Expected<SymbolRef::Type> Type = Symbol.getType();
    if (!Type)
      return errorToErrorCode(Type.takeError());
    if (FunctionOnly && *Type != SymbolRef::ST_Function)
      continue;

    Expected<uint64_t> Address = Symbol.getAddress();
    if (!Address)
      return errorToErrorCode(Address.takeError());
    if (*Address == VA)
      return Symbol;
  }
  return readobj_error::unknown_symbol;
}

ErrorOr<SymbolRef> Decoder::getRelocatedSymbol(const COFFObjectFile &,
                                               const SectionRef &Section,
                                               uint64_t Offset) {
  for (const auto &Relocation : Section.relocations()) {
    uint64_t RelocationOffset = Relocation.getOffset();
    if (RelocationOffset == Offset)
      return *Relocation.getSymbol();
  }
  return readobj_error::unknown_symbol;
}

bool Decoder::opcode_0xxxxxxx(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  uint8_t Imm = OC[Offset] & 0x7f;
  SW.startLine() << format("0x%02x                ; %s sp, #(%u * 4)\n",
                           OC[Offset],
                           static_cast<const char *>(Prologue ? "sub" : "add"),
                           Imm);
  ++Offset;
  return false;
}

bool Decoder::opcode_10Lxxxxx(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  unsigned Link = (OC[Offset] & 0x20) >> 5;
  uint16_t RegisterMask = (Link << (Prologue ? 14 : 15))
                        | ((OC[Offset + 0] & 0x1f) << 8)
                        | ((OC[Offset + 1] & 0xff) << 0);
  assert((~RegisterMask & (1 << 13)) && "sp must not be set");
  assert((~RegisterMask & (1 << (Prologue ? 15 : 14))) && "pc must not be set");

  SW.startLine() << format("0x%02x 0x%02x           ; %s.w ",
                           OC[Offset + 0], OC[Offset + 1],
                           Prologue ? "push" : "pop");
  printRegisters(std::make_pair(RegisterMask, 0));
  OS << '\n';

  Offset += 2;
  return false;
}

bool Decoder::opcode_1100xxxx(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  if (Prologue)
    SW.startLine() << format("0x%02x                ; mov r%u, sp\n",
                             OC[Offset], OC[Offset] & 0xf);
  else
    SW.startLine() << format("0x%02x                ; mov sp, r%u\n",
                             OC[Offset], OC[Offset] & 0xf);
  ++Offset;
  return false;
}

bool Decoder::opcode_11010Lxx(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  unsigned Link = (OC[Offset] & 0x4) >> 3;
  unsigned Count = (OC[Offset] & 0x3);

  uint16_t GPRMask = (Link << (Prologue ? 14 : 15))
                   | (((1 << (Count + 1)) - 1) << 4);

  SW.startLine() << format("0x%02x                ; %s ", OC[Offset],
                           Prologue ? "push" : "pop");
  printRegisters(std::make_pair(GPRMask, 0));
  OS << '\n';

  ++Offset;
  return false;
}

bool Decoder::opcode_11011Lxx(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  unsigned Link = (OC[Offset] & 0x4) >> 2;
  unsigned Count = (OC[Offset] & 0x3) + 4;

  uint16_t GPRMask = (Link << (Prologue ? 14 : 15))
                   | (((1 << (Count + 1)) - 1) << 4);

  SW.startLine() << format("0x%02x                ; %s.w ", OC[Offset],
                           Prologue ? "push" : "pop");
  printRegisters(std::make_pair(GPRMask, 0));
  OS << '\n';

  ++Offset;
  return false;
}

bool Decoder::opcode_11100xxx(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  unsigned High = (OC[Offset] & 0x7);
  uint32_t VFPMask = (((1 << (High + 1)) - 1) << 8);

  SW.startLine() << format("0x%02x                ; %s ", OC[Offset],
                           Prologue ? "vpush" : "vpop");
  printRegisters(std::make_pair(0, VFPMask));
  OS << '\n';

  ++Offset;
  return false;
}

bool Decoder::opcode_111010xx(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  uint16_t Imm = ((OC[Offset + 0] & 0x03) << 8) | ((OC[Offset + 1] & 0xff) << 0);

  SW.startLine() << format("0x%02x 0x%02x           ; %s.w sp, #(%u * 4)\n",
                           OC[Offset + 0], OC[Offset + 1],
                           static_cast<const char *>(Prologue ? "sub" : "add"),
                           Imm);

  Offset += 2;
  return false;
}

bool Decoder::opcode_1110110L(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  uint8_t GPRMask = ((OC[Offset + 0] & 0x01) << (Prologue ? 14 : 15))
                  | ((OC[Offset + 1] & 0xff) << 0);

  SW.startLine() << format("0x%02x 0x%02x           ; %s ", OC[Offset + 0],
                           OC[Offset + 1], Prologue ? "push" : "pop");
  printRegisters(std::make_pair(GPRMask, 0));
  OS << '\n';

  Offset += 2;
  return false;
}

bool Decoder::opcode_11101110(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  assert(!Prologue && "may not be used in prologue");

  if (OC[Offset + 1] & 0xf0)
    SW.startLine() << format("0x%02x 0x%02x           ; reserved\n",
                             OC[Offset + 0], OC[Offset +  1]);
  else
    SW.startLine()
      << format("0x%02x 0x%02x           ; microsoft-specific (type: %u)\n",
                OC[Offset + 0], OC[Offset + 1], OC[Offset + 1] & 0x0f);

  Offset += 2;
  return false;
}

bool Decoder::opcode_11101111(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  assert(!Prologue && "may not be used in prologue");

  if (OC[Offset + 1] & 0xf0)
    SW.startLine() << format("0x%02x 0x%02x           ; reserved\n",
                             OC[Offset + 0], OC[Offset +  1]);
  else
    SW.startLine()
      << format("0x%02x 0x%02x           ; ldr.w lr, [sp], #%u\n",
                OC[Offset + 0], OC[Offset + 1], OC[Offset + 1] << 2);

  Offset += 2;
  return false;
}

bool Decoder::opcode_11110101(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  unsigned Start = (OC[Offset + 1] & 0xf0) >> 4;
  unsigned End = (OC[Offset + 1] & 0x0f) >> 0;
  uint32_t VFPMask = ((1 << (End - Start)) - 1) << Start;

  SW.startLine() << format("0x%02x 0x%02x           ; %s ", OC[Offset + 0],
                           OC[Offset + 1], Prologue ? "vpush" : "vpop");
  printRegisters(std::make_pair(0, VFPMask));
  OS << '\n';

  Offset += 2;
  return false;
}

bool Decoder::opcode_11110110(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  unsigned Start = (OC[Offset + 1] & 0xf0) >> 4;
  unsigned End = (OC[Offset + 1] & 0x0f) >> 0;
  uint32_t VFPMask = ((1 << (End - Start)) - 1) << 16;

  SW.startLine() << format("0x%02x 0x%02x           ; %s ", OC[Offset + 0],
                           OC[Offset + 1], Prologue ? "vpush" : "vpop");
  printRegisters(std::make_pair(0, VFPMask));
  OS << '\n';

  Offset += 2;
  return false;
}

bool Decoder::opcode_11110111(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  uint32_t Imm = (OC[Offset + 1] << 8) | (OC[Offset + 2] << 0);

  SW.startLine() << format("0x%02x 0x%02x 0x%02x      ; %s sp, sp, #(%u * 4)\n",
                           OC[Offset + 0], OC[Offset + 1], OC[Offset + 2],
                           static_cast<const char *>(Prologue ? "sub" : "add"),
                           Imm);

  Offset += 3;
  return false;
}

bool Decoder::opcode_11111000(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  uint32_t Imm = (OC[Offset + 1] << 16)
               | (OC[Offset + 2] << 8)
               | (OC[Offset + 3] << 0);

  SW.startLine()
    << format("0x%02x 0x%02x 0x%02x 0x%02x ; %s sp, sp, #(%u * 4)\n",
              OC[Offset + 0], OC[Offset + 1], OC[Offset + 2], OC[Offset + 3],
              static_cast<const char *>(Prologue ? "sub" : "add"), Imm);

  Offset += 4;
  return false;
}

bool Decoder::opcode_11111001(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  uint32_t Imm = (OC[Offset + 1] << 8) | (OC[Offset + 2] << 0);

  SW.startLine()
    << format("0x%02x 0x%02x 0x%02x      ; %s.w sp, sp, #(%u * 4)\n",
              OC[Offset + 0], OC[Offset + 1], OC[Offset + 2],
              static_cast<const char *>(Prologue ? "sub" : "add"), Imm);

  Offset += 3;
  return false;
}

bool Decoder::opcode_11111010(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  uint32_t Imm = (OC[Offset + 1] << 16)
               | (OC[Offset + 2] << 8)
               | (OC[Offset + 3] << 0);

  SW.startLine()
    << format("0x%02x 0x%02x 0x%02x 0x%02x ; %s.w sp, sp, #(%u * 4)\n",
              OC[Offset + 0], OC[Offset + 1], OC[Offset + 2], OC[Offset + 3],
              static_cast<const char *>(Prologue ? "sub" : "add"), Imm);

  Offset += 4;
  return false;
}

bool Decoder::opcode_11111011(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  SW.startLine() << format("0x%02x                ; nop\n", OC[Offset]);
  ++Offset;
  return false;
}

bool Decoder::opcode_11111100(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  SW.startLine() << format("0x%02x                ; nop.w\n", OC[Offset]);
  ++Offset;
  return false;
}

bool Decoder::opcode_11111101(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  SW.startLine() << format("0x%02x                ; b\n", OC[Offset]);
  ++Offset;
  return true;
}

bool Decoder::opcode_11111110(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  SW.startLine() << format("0x%02x                ; b.w\n", OC[Offset]);
  ++Offset;
  return true;
}

bool Decoder::opcode_11111111(const uint8_t *OC, unsigned &Offset,
                              unsigned Length, bool Prologue) {
  ++Offset;
  return true;
}

void Decoder::decodeOpcodes(ArrayRef<uint8_t> Opcodes, unsigned Offset,
                            bool Prologue) {
  assert((!Prologue || Offset == 0) && "prologue should always use offset 0");

  bool Terminated = false;
  for (unsigned OI = Offset, OE = Opcodes.size(); !Terminated && OI < OE; ) {
    for (unsigned DI = 0;; ++DI) {
      if ((Opcodes[OI] & Ring[DI].Mask) == Ring[DI].Value) {
        Terminated = (this->*Ring[DI].Routine)(Opcodes.data(), OI, 0, Prologue);
        break;
      }
      assert(DI < array_lengthof(Ring) && "unhandled opcode");
    }
  }
}

bool Decoder::dumpXDataRecord(const COFFObjectFile &COFF,
                              const SectionRef &Section,
                              uint64_t FunctionAddress, uint64_t VA) {
  ArrayRef<uint8_t> Contents;
  if (COFF.getSectionContents(COFF.getCOFFSection(Section), Contents))
    return false;

  uint64_t SectionVA = Section.getAddress();
  uint64_t Offset = VA - SectionVA;
  const ulittle32_t *Data =
    reinterpret_cast<const ulittle32_t *>(Contents.data() + Offset);
  const ExceptionDataRecord XData(Data);

  DictScope XRS(SW, "ExceptionData");
  SW.printNumber("FunctionLength", XData.FunctionLength() << 1);
  SW.printNumber("Version", XData.Vers());
  SW.printBoolean("ExceptionData", XData.X());
  SW.printBoolean("EpiloguePacked", XData.E());
  SW.printBoolean("Fragment", XData.F());
  SW.printNumber(XData.E() ? "EpilogueOffset" : "EpilogueScopes",
                 XData.EpilogueCount());
  SW.printNumber("ByteCodeLength",
                 static_cast<uint64_t>(XData.CodeWords() * sizeof(uint32_t)));

  if (XData.E()) {
    ArrayRef<uint8_t> UC = XData.UnwindByteCode();
    if (!XData.F()) {
      ListScope PS(SW, "Prologue");
      decodeOpcodes(UC, 0, /*Prologue=*/true);
    }
    if (XData.EpilogueCount()) {
      ListScope ES(SW, "Epilogue");
      decodeOpcodes(UC, XData.EpilogueCount(), /*Prologue=*/false);
    }
  } else {
    ArrayRef<ulittle32_t> EpilogueScopes = XData.EpilogueScopes();
    ListScope ESS(SW, "EpilogueScopes");
    for (const EpilogueScope ES : EpilogueScopes) {
      DictScope ESES(SW, "EpilogueScope");
      SW.printNumber("StartOffset", ES.EpilogueStartOffset());
      SW.printNumber("Condition", ES.Condition());
      SW.printNumber("EpilogueStartIndex", ES.EpilogueStartIndex());

      ListScope Opcodes(SW, "Opcodes");
      decodeOpcodes(XData.UnwindByteCode(), ES.EpilogueStartIndex(),
                    /*Prologue=*/false);
    }
  }

  if (XData.X()) {
    const uint32_t Address = XData.ExceptionHandlerRVA();
    const uint32_t Parameter = XData.ExceptionHandlerParameter();
    const size_t HandlerOffset = HeaderWords(XData)
                               + (XData.E() ? 0 : XData.EpilogueCount())
                               + XData.CodeWords();

    ErrorOr<SymbolRef> Symbol =
      getRelocatedSymbol(COFF, Section, HandlerOffset * sizeof(uint32_t));
    if (!Symbol)
      Symbol = getSymbol(COFF, Address, /*FunctionOnly=*/true);

    Expected<StringRef> Name = Symbol->getName();
    if (!Name) {
      std::string Buf;
      llvm::raw_string_ostream OS(Buf);
      logAllUnhandledErrors(Name.takeError(), OS, "");
      OS.flush();
      report_fatal_error(Buf);
    }

    ListScope EHS(SW, "ExceptionHandler");
    SW.printString("Routine", formatSymbol(*Name, Address));
    SW.printHex("Parameter", Parameter);
  }

  return true;
}

bool Decoder::dumpUnpackedEntry(const COFFObjectFile &COFF,
                                const SectionRef Section, uint64_t Offset,
                                unsigned Index, const RuntimeFunction &RF) {
  assert(RF.Flag() == RuntimeFunctionFlag::RFF_Unpacked &&
         "packed entry cannot be treated as an unpacked entry");

  ErrorOr<SymbolRef> Function = getRelocatedSymbol(COFF, Section, Offset);
  if (!Function)
    Function = getSymbol(COFF, RF.BeginAddress, /*FunctionOnly=*/true);

  ErrorOr<SymbolRef> XDataRecord = getRelocatedSymbol(COFF, Section, Offset + 4);
  if (!XDataRecord)
    XDataRecord = getSymbol(COFF, RF.ExceptionInformationRVA());

  if (!RF.BeginAddress && !Function)
    return false;
  if (!RF.UnwindData && !XDataRecord)
    return false;

  StringRef FunctionName;
  uint64_t FunctionAddress;
  if (Function) {
    Expected<StringRef> FunctionNameOrErr = Function->getName();
    if (!FunctionNameOrErr) {
      std::string Buf;
      llvm::raw_string_ostream OS(Buf);
      logAllUnhandledErrors(FunctionNameOrErr.takeError(), OS, "");
      OS.flush();
      report_fatal_error(Buf);
    }
    FunctionName = *FunctionNameOrErr;
    Expected<uint64_t> FunctionAddressOrErr = Function->getAddress();
    if (!FunctionAddressOrErr) {
      std::string Buf;
      llvm::raw_string_ostream OS(Buf);
      logAllUnhandledErrors(FunctionAddressOrErr.takeError(), OS, "");
      OS.flush();
      report_fatal_error(Buf);
    }
    FunctionAddress = *FunctionAddressOrErr;
  } else {
    const pe32_header *PEHeader;
    if (COFF.getPE32Header(PEHeader))
      return false;
    FunctionAddress = PEHeader->ImageBase + RF.BeginAddress;
  }

  SW.printString("Function", formatSymbol(FunctionName, FunctionAddress));

  if (XDataRecord) {
    Expected<StringRef> Name = XDataRecord->getName();
    if (!Name) {
      std::string Buf;
      llvm::raw_string_ostream OS(Buf);
      logAllUnhandledErrors(Name.takeError(), OS, "");
      OS.flush();
      report_fatal_error(Buf);
    }

    Expected<uint64_t> AddressOrErr = XDataRecord->getAddress();
    if (!AddressOrErr) {
      std::string Buf;
      llvm::raw_string_ostream OS(Buf);
      logAllUnhandledErrors(AddressOrErr.takeError(), OS, "");
      OS.flush();
      report_fatal_error(Buf);
    }
    uint64_t Address = *AddressOrErr;

    SW.printString("ExceptionRecord", formatSymbol(*Name, Address));

    Expected<section_iterator> SIOrErr = XDataRecord->getSection();
    if (!SIOrErr) {
      // TODO: Actually report errors helpfully.
      consumeError(SIOrErr.takeError());
      return false;
    }
    section_iterator SI = *SIOrErr;

    return dumpXDataRecord(COFF, *SI, FunctionAddress, Address);
  } else {
    const pe32_header *PEHeader;
    if (COFF.getPE32Header(PEHeader))
      return false;

    uint64_t Address = PEHeader->ImageBase + RF.ExceptionInformationRVA();
    SW.printString("ExceptionRecord", formatSymbol("", Address));

    ErrorOr<SectionRef> Section =
      getSectionContaining(COFF, RF.ExceptionInformationRVA());
    if (!Section)
      return false;

    return dumpXDataRecord(COFF, *Section, FunctionAddress,
                           RF.ExceptionInformationRVA());
  }
}

bool Decoder::dumpPackedEntry(const object::COFFObjectFile &COFF,
                              const SectionRef Section, uint64_t Offset,
                              unsigned Index, const RuntimeFunction &RF) {
  assert((RF.Flag() == RuntimeFunctionFlag::RFF_Packed ||
          RF.Flag() == RuntimeFunctionFlag::RFF_PackedFragment) &&
         "unpacked entry cannot be treated as a packed entry");

  ErrorOr<SymbolRef> Function = getRelocatedSymbol(COFF, Section, Offset);
  if (!Function)
    Function = getSymbol(COFF, RF.BeginAddress, /*FunctionOnly=*/true);

  StringRef FunctionName;
  uint64_t FunctionAddress;
  if (Function) {
    Expected<StringRef> FunctionNameOrErr = Function->getName();
    if (!FunctionNameOrErr) {
      std::string Buf;
      llvm::raw_string_ostream OS(Buf);
      logAllUnhandledErrors(FunctionNameOrErr.takeError(), OS, "");
      OS.flush();
      report_fatal_error(Buf);
    }
    FunctionName = *FunctionNameOrErr;
    Expected<uint64_t> FunctionAddressOrErr = Function->getAddress();
    if (!FunctionAddressOrErr) {
      std::string Buf;
      llvm::raw_string_ostream OS(Buf);
      logAllUnhandledErrors(FunctionAddressOrErr.takeError(), OS, "");
      OS.flush();
      report_fatal_error(Buf);
    }
    FunctionAddress = *FunctionAddressOrErr;
  } else {
    const pe32_header *PEHeader;
    if (COFF.getPE32Header(PEHeader))
      return false;
    FunctionAddress = PEHeader->ImageBase + RF.BeginAddress;
  }

  SW.printString("Function", formatSymbol(FunctionName, FunctionAddress));
  SW.printBoolean("Fragment",
                  RF.Flag() == RuntimeFunctionFlag::RFF_PackedFragment);
  SW.printNumber("FunctionLength", RF.FunctionLength());
  SW.startLine() << "ReturnType: " << RF.Ret() << '\n';
  SW.printBoolean("HomedParameters", RF.H());
  SW.startLine() << "SavedRegisters: ";
                 printRegisters(SavedRegisterMask(RF));
  OS << '\n';
  SW.printNumber("StackAdjustment", StackAdjustment(RF) << 2);

  return true;
}

bool Decoder::dumpProcedureDataEntry(const COFFObjectFile &COFF,
                                     const SectionRef Section, unsigned Index,
                                     ArrayRef<uint8_t> Contents) {
  uint64_t Offset = PDataEntrySize * Index;
  const ulittle32_t *Data =
    reinterpret_cast<const ulittle32_t *>(Contents.data() + Offset);

  const RuntimeFunction Entry(Data);
  DictScope RFS(SW, "RuntimeFunction");
  if (Entry.Flag() == RuntimeFunctionFlag::RFF_Unpacked)
    return dumpUnpackedEntry(COFF, Section, Offset, Index, Entry);
  return dumpPackedEntry(COFF, Section, Offset, Index, Entry);
}

void Decoder::dumpProcedureData(const COFFObjectFile &COFF,
                                const SectionRef Section) {
  ArrayRef<uint8_t> Contents;
  if (COFF.getSectionContents(COFF.getCOFFSection(Section), Contents))
    return;

  if (Contents.size() % PDataEntrySize) {
    errs() << ".pdata content is not " << PDataEntrySize << "-byte aligned\n";
    return;
  }

  for (unsigned EI = 0, EE = Contents.size() / PDataEntrySize; EI < EE; ++EI)
    if (!dumpProcedureDataEntry(COFF, Section, EI, Contents))
      break;
}

std::error_code Decoder::dumpProcedureData(const COFFObjectFile &COFF) {
  for (const auto &Section : COFF.sections()) {
    StringRef SectionName;
    if (std::error_code EC =
            COFF.getSectionName(COFF.getCOFFSection(Section), SectionName))
      return EC;

    if (SectionName.startswith(".pdata"))
      dumpProcedureData(COFF, Section);
  }
  return std::error_code();
}
}
}
}
