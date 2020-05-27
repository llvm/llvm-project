//===- DWARFDebugFrame.h - Parsing of .debug_frame ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <string>
#include <vector>

using namespace llvm;
using namespace dwarf;


// See DWARF standard v3, section 7.23
const uint8_t DWARF_CFI_PRIMARY_OPCODE_MASK = 0xc0;
const uint8_t DWARF_CFI_PRIMARY_OPERAND_MASK = 0x3f;

Error CFIProgram::parse(DWARFDataExtractor Data, uint64_t *Offset,
                        uint64_t EndOffset) {
  while (*Offset < EndOffset) {
    uint8_t Opcode = Data.getRelocatedValue(1, Offset);
    // Some instructions have a primary opcode encoded in the top bits.
    uint8_t Primary = Opcode & DWARF_CFI_PRIMARY_OPCODE_MASK;

    if (Primary) {
      // If it's a primary opcode, the first operand is encoded in the bottom
      // bits of the opcode itself.
      uint64_t Op1 = Opcode & DWARF_CFI_PRIMARY_OPERAND_MASK;
      switch (Primary) {
      default:
        return createStringError(errc::illegal_byte_sequence,
                                 "Invalid primary CFI opcode 0x%" PRIx8,
                                 Primary);
      case DW_CFA_advance_loc:
      case DW_CFA_restore:
        addInstruction(Primary, Op1);
        break;
      case DW_CFA_offset:
        addInstruction(Primary, Op1, Data.getULEB128(Offset));
        break;
      }
    } else {
      // Extended opcode - its value is Opcode itself.
      switch (Opcode) {
      default:
        return createStringError(errc::illegal_byte_sequence,
                                 "Invalid extended CFI opcode 0x%" PRIx8,
                                 Opcode);
      case DW_CFA_nop:
      case DW_CFA_remember_state:
      case DW_CFA_restore_state:
      case DW_CFA_GNU_window_save:
        // No operands
        addInstruction(Opcode);
        break;
      case DW_CFA_set_loc:
        // Operands: Address
        addInstruction(Opcode, Data.getRelocatedAddress(Offset));
        break;
      case DW_CFA_advance_loc1:
        // Operands: 1-byte delta
        addInstruction(Opcode, Data.getRelocatedValue(1, Offset));
        break;
      case DW_CFA_advance_loc2:
        // Operands: 2-byte delta
        addInstruction(Opcode, Data.getRelocatedValue(2, Offset));
        break;
      case DW_CFA_advance_loc4:
        // Operands: 4-byte delta
        addInstruction(Opcode, Data.getRelocatedValue(4, Offset));
        break;
      case DW_CFA_restore_extended:
      case DW_CFA_undefined:
      case DW_CFA_same_value:
      case DW_CFA_def_cfa_register:
      case DW_CFA_def_cfa_offset:
      case DW_CFA_GNU_args_size:
        // Operands: ULEB128
        addInstruction(Opcode, Data.getULEB128(Offset));
        break;
      case DW_CFA_def_cfa_offset_sf:
        // Operands: SLEB128
        addInstruction(Opcode, Data.getSLEB128(Offset));
        break;
      case DW_CFA_offset_extended:
      case DW_CFA_register:
      case DW_CFA_def_cfa:
      case DW_CFA_val_offset: {
        // Operands: ULEB128, ULEB128
        // Note: We can not embed getULEB128 directly into function
        // argument list. getULEB128 changes Offset and order of evaluation
        // for arguments is unspecified.
        auto op1 = Data.getULEB128(Offset);
        auto op2 = Data.getULEB128(Offset);
        addInstruction(Opcode, op1, op2);
        break;
        }
        case DW_CFA_offset_extended_sf:
        case DW_CFA_def_cfa_sf:
        case DW_CFA_val_offset_sf: {
          // Operands: ULEB128, SLEB128
          // Note: see comment for the previous case
          auto op1 = Data.getULEB128(Offset);
          auto op2 = (uint64_t)Data.getSLEB128(Offset);
          addInstruction(Opcode, op1, op2);
          break;
        }
        case DW_CFA_def_cfa_expression: {
          uint32_t ExprLength = Data.getULEB128(Offset);
          addInstruction(Opcode, 0);
          DataExtractor Extractor(
              Data.getData().slice(*Offset, *Offset + ExprLength),
              Data.isLittleEndian(), Data.getAddressSize());
          // Note. We do not pass the DWARF format to DWARFExpression, because
          // DW_OP_call_ref, the only operation which depends on the format, is
          // prohibited in call frame instructions, see sec. 6.4.2 in DWARFv5.
          Instructions.back().Expression =
              DWARFExpression(Extractor, Data.getAddressSize());
          *Offset += ExprLength;
          break;
        }
        case DW_CFA_expression:
        case DW_CFA_val_expression: {
          auto RegNum = Data.getULEB128(Offset);
          auto BlockLength = Data.getULEB128(Offset);
          addInstruction(Opcode, RegNum, 0);
          DataExtractor Extractor(
              Data.getData().slice(*Offset, *Offset + BlockLength),
              Data.isLittleEndian(), Data.getAddressSize());
          // Note. We do not pass the DWARF format to DWARFExpression, because
          // DW_OP_call_ref, the only operation which depends on the format, is
          // prohibited in call frame instructions, see sec. 6.4.2 in DWARFv5.
          Instructions.back().Expression =
              DWARFExpression(Extractor, Data.getAddressSize());
          *Offset += BlockLength;
          break;
        }
      }
    }
  }

  return Error::success();
}

namespace {


} // end anonymous namespace

ArrayRef<CFIProgram::OperandType[2]> CFIProgram::getOperandTypes() {
  static OperandType OpTypes[DW_CFA_restore+1][2];
  static bool Initialized = false;
  if (Initialized) {
    return ArrayRef<OperandType[2]>(&OpTypes[0], DW_CFA_restore+1);
  }
  Initialized = true;

#define DECLARE_OP2(OP, OPTYPE0, OPTYPE1)       \
  do {                                          \
    OpTypes[OP][0] = OPTYPE0;                   \
    OpTypes[OP][1] = OPTYPE1;                   \
  } while (false)
#define DECLARE_OP1(OP, OPTYPE0) DECLARE_OP2(OP, OPTYPE0, OT_None)
#define DECLARE_OP0(OP) DECLARE_OP1(OP, OT_None)

  DECLARE_OP1(DW_CFA_set_loc, OT_Address);
  DECLARE_OP1(DW_CFA_advance_loc, OT_FactoredCodeOffset);
  DECLARE_OP1(DW_CFA_advance_loc1, OT_FactoredCodeOffset);
  DECLARE_OP1(DW_CFA_advance_loc2, OT_FactoredCodeOffset);
  DECLARE_OP1(DW_CFA_advance_loc4, OT_FactoredCodeOffset);
  DECLARE_OP1(DW_CFA_MIPS_advance_loc8, OT_FactoredCodeOffset);
  DECLARE_OP2(DW_CFA_def_cfa, OT_Register, OT_Offset);
  DECLARE_OP2(DW_CFA_def_cfa_sf, OT_Register, OT_SignedFactDataOffset);
  DECLARE_OP1(DW_CFA_def_cfa_register, OT_Register);
  DECLARE_OP1(DW_CFA_def_cfa_offset, OT_Offset);
  DECLARE_OP1(DW_CFA_def_cfa_offset_sf, OT_SignedFactDataOffset);
  DECLARE_OP1(DW_CFA_def_cfa_expression, OT_Expression);
  DECLARE_OP1(DW_CFA_undefined, OT_Register);
  DECLARE_OP1(DW_CFA_same_value, OT_Register);
  DECLARE_OP2(DW_CFA_offset, OT_Register, OT_UnsignedFactDataOffset);
  DECLARE_OP2(DW_CFA_offset_extended, OT_Register, OT_UnsignedFactDataOffset);
  DECLARE_OP2(DW_CFA_offset_extended_sf, OT_Register, OT_SignedFactDataOffset);
  DECLARE_OP2(DW_CFA_val_offset, OT_Register, OT_UnsignedFactDataOffset);
  DECLARE_OP2(DW_CFA_val_offset_sf, OT_Register, OT_SignedFactDataOffset);
  DECLARE_OP2(DW_CFA_register, OT_Register, OT_Register);
  DECLARE_OP2(DW_CFA_expression, OT_Register, OT_Expression);
  DECLARE_OP2(DW_CFA_val_expression, OT_Register, OT_Expression);
  DECLARE_OP1(DW_CFA_restore, OT_Register);
  DECLARE_OP1(DW_CFA_restore_extended, OT_Register);
  DECLARE_OP0(DW_CFA_remember_state);
  DECLARE_OP0(DW_CFA_restore_state);
  DECLARE_OP0(DW_CFA_GNU_window_save);
  DECLARE_OP1(DW_CFA_GNU_args_size, OT_Offset);
  DECLARE_OP0(DW_CFA_nop);

#undef DECLARE_OP0
#undef DECLARE_OP1
#undef DECLARE_OP2

  return ArrayRef<OperandType[2]>(&OpTypes[0], DW_CFA_restore+1);
}

/// Print \p Opcode's operand number \p OperandIdx which has value \p Operand.
void CFIProgram::printOperand(raw_ostream &OS, const MCRegisterInfo *MRI,
                              bool IsEH, const Instruction &Instr,
                              unsigned OperandIdx, uint64_t Operand) const {
  assert(OperandIdx < 2);
  uint8_t Opcode = Instr.Opcode;
  OperandType Type = getOperandTypes()[Opcode][OperandIdx];

  switch (Type) {
  case OT_Unset: {
    OS << " Unsupported " << (OperandIdx ? "second" : "first") << " operand to";
    auto OpcodeName = CallFrameString(Opcode, Arch);
    if (!OpcodeName.empty())
      OS << " " << OpcodeName;
    else
      OS << format(" Opcode %x",  Opcode);
    break;
  }
  case OT_None:
    break;
  case OT_Address:
    OS << format(" %" PRIx64, Operand);
    break;
  case OT_Offset:
    // The offsets are all encoded in a unsigned form, but in practice
    // consumers use them signed. It's most certainly legacy due to
    // the lack of signed variants in the first Dwarf standards.
    OS << format(" %+" PRId64, int64_t(Operand));
    break;
  case OT_FactoredCodeOffset: // Always Unsigned
    if (CodeAlignmentFactor)
      OS << format(" %" PRId64, Operand * CodeAlignmentFactor);
    else
      OS << format(" %" PRId64 "*code_alignment_factor" , Operand);
    break;
  case OT_SignedFactDataOffset:
    if (DataAlignmentFactor)
      OS << format(" %" PRId64, int64_t(Operand) * DataAlignmentFactor);
    else
      OS << format(" %" PRId64 "*data_alignment_factor" , int64_t(Operand));
    break;
  case OT_UnsignedFactDataOffset:
    if (DataAlignmentFactor)
      OS << format(" %" PRId64, Operand * DataAlignmentFactor);
    else
      OS << format(" %" PRId64 "*data_alignment_factor" , Operand);
    break;
  case OT_Register:
    OS << format(" reg%" PRId64, Operand);
    break;
  case OT_Expression:
    assert(Instr.Expression && "missing DWARFExpression object");
    OS << " ";
    Instr.Expression->print(OS, MRI, nullptr, IsEH);
    break;
  }
}

void CFIProgram::dump(raw_ostream &OS, const MCRegisterInfo *MRI, bool IsEH,
                      unsigned IndentLevel) const {
  for (const auto &Instr : Instructions) {
    uint8_t Opcode = Instr.Opcode;
    if (Opcode & DWARF_CFI_PRIMARY_OPCODE_MASK)
      Opcode &= DWARF_CFI_PRIMARY_OPCODE_MASK;
    OS.indent(2 * IndentLevel);
    OS << CallFrameString(Opcode, Arch) << ":";
    for (unsigned i = 0; i < Instr.Ops.size(); ++i)
      printOperand(OS, MRI, IsEH, Instr, i, Instr.Ops[i]);
    OS << '\n';
  }
}

// Returns the CIE identifier to be used by the requested format.
// CIE ids for .debug_frame sections are defined in Section 7.24 of DWARFv5.
// For CIE ID in .eh_frame sections see
// https://refspecs.linuxfoundation.org/LSB_5.0.0/LSB-Core-generic/LSB-Core-generic/ehframechpt.html
constexpr uint64_t getCIEId(bool IsDWARF64, bool IsEH) {
  if (IsEH)
    return 0;
  if (IsDWARF64)
    return DW64_CIE_ID;
  return DW_CIE_ID;
}

void CIE::dump(raw_ostream &OS, const MCRegisterInfo *MRI, bool IsEH) const {
  // A CIE with a zero length is a terminator entry in the .eh_frame section.
  if (IsEH && Length == 0) {
    OS << format("%08" PRIx64, Offset) << " ZERO terminator\n";
    return;
  }

  OS << format("%08" PRIx64, Offset)
     << format(" %0*" PRIx64, IsDWARF64 ? 16 : 8, Length)
     << format(" %0*" PRIx64, IsDWARF64 && !IsEH ? 16 : 8,
               getCIEId(IsDWARF64, IsEH))
     << " CIE\n";
  OS << format("  Version:               %d\n", Version);
  OS << "  Augmentation:          \"" << Augmentation << "\"\n";
  if (Version >= 4) {
    OS << format("  Address size:          %u\n", (uint32_t)AddressSize);
    OS << format("  Segment desc size:     %u\n",
                 (uint32_t)SegmentDescriptorSize);
  }
  OS << format("  Code alignment factor: %u\n", (uint32_t)CodeAlignmentFactor);
  OS << format("  Data alignment factor: %d\n", (int32_t)DataAlignmentFactor);
  OS << format("  Return address column: %d\n", (int32_t)ReturnAddressRegister);
  if (Personality)
    OS << format("  Personality Address: %016" PRIx64 "\n", *Personality);
  if (!AugmentationData.empty()) {
    OS << "  Augmentation data:    ";
    for (uint8_t Byte : AugmentationData)
      OS << ' ' << hexdigit(Byte >> 4) << hexdigit(Byte & 0xf);
    OS << "\n";
  }
  OS << "\n";
  CFIs.dump(OS, MRI, IsEH);
  OS << "\n";
}

void FDE::dump(raw_ostream &OS, const MCRegisterInfo *MRI, bool IsEH) const {
  OS << format("%08" PRIx64, Offset)
     << format(" %0*" PRIx64, IsDWARF64 ? 16 : 8, Length)
     << format(" %0*" PRIx64, IsDWARF64 && !IsEH ? 16 : 8, CIEPointer)
     << " FDE cie=";
  if (LinkedCIE)
    OS << format("%08" PRIx64, LinkedCIE->getOffset());
  else
    OS << "<invalid offset>";
  OS << format(" pc=%08" PRIx64 "...%08" PRIx64 "\n", InitialLocation,
               InitialLocation + AddressRange);
  if (LSDAAddress)
    OS << format("  LSDA Address: %016" PRIx64 "\n", *LSDAAddress);
  CFIs.dump(OS, MRI, IsEH);
  OS << "\n";
}

DWARFDebugFrame::DWARFDebugFrame(Triple::ArchType Arch,
    bool IsEH, uint64_t EHFrameAddress)
    : Arch(Arch), IsEH(IsEH), EHFrameAddress(EHFrameAddress) {}

DWARFDebugFrame::~DWARFDebugFrame() = default;

static void LLVM_ATTRIBUTE_UNUSED dumpDataAux(DataExtractor Data,
                                              uint64_t Offset, int Length) {
  errs() << "DUMP: ";
  for (int i = 0; i < Length; ++i) {
    uint8_t c = Data.getU8(&Offset);
    errs().write_hex(c); errs() << " ";
  }
  errs() << "\n";
}

Error DWARFDebugFrame::parse(DWARFDataExtractor Data) {
  uint64_t Offset = 0;
  DenseMap<uint64_t, CIE *> CIEs;

  while (Data.isValidOffset(Offset)) {
    uint64_t StartOffset = Offset;

    uint64_t Length;
    DwarfFormat Format;
    std::tie(Length, Format) = Data.getInitialLength(&Offset);
    bool IsDWARF64 = Format == DWARF64;

    // If the Length is 0, then this CIE is a terminator. We add it because some
    // dumper tools might need it to print something special for such entries
    // (e.g. llvm-objdump --dwarf=frames prints "ZERO terminator").
    if (Length == 0) {
      auto Cie = std::make_unique<CIE>(
          IsDWARF64, StartOffset, 0, 0, SmallString<8>(), 0, 0, 0, 0, 0,
          SmallString<8>(), 0, 0, None, None, Arch);
      CIEs[StartOffset] = Cie.get();
      Entries.push_back(std::move(Cie));
      break;
    }

    // At this point, Offset points to the next field after Length.
    // Length is the structure size excluding itself. Compute an offset one
    // past the end of the structure (needed to know how many instructions to
    // read).
    uint64_t StartStructureOffset = Offset;
    uint64_t EndStructureOffset = Offset + Length;

    // The Id field's size depends on the DWARF format
    Error Err = Error::success();
    uint64_t Id = Data.getRelocatedValue((IsDWARF64 && !IsEH) ? 8 : 4, &Offset,
                                         /*SectionIndex=*/nullptr, &Err);
    if (Err)
      return Err;

    if (Id == getCIEId(IsDWARF64, IsEH)) {
      uint8_t Version = Data.getU8(&Offset);
      const char *Augmentation = Data.getCStr(&Offset);
      StringRef AugmentationString(Augmentation ? Augmentation : "");
      uint8_t AddressSize = Version < 4 ? Data.getAddressSize() :
                                          Data.getU8(&Offset);
      Data.setAddressSize(AddressSize);
      uint8_t SegmentDescriptorSize = Version < 4 ? 0 : Data.getU8(&Offset);
      uint64_t CodeAlignmentFactor = Data.getULEB128(&Offset);
      int64_t DataAlignmentFactor = Data.getSLEB128(&Offset);
      uint64_t ReturnAddressRegister =
          Version == 1 ? Data.getU8(&Offset) : Data.getULEB128(&Offset);

      // Parse the augmentation data for EH CIEs
      StringRef AugmentationData("");
      uint32_t FDEPointerEncoding = DW_EH_PE_absptr;
      uint32_t LSDAPointerEncoding = DW_EH_PE_omit;
      Optional<uint64_t> Personality;
      Optional<uint32_t> PersonalityEncoding;
      if (IsEH) {
        Optional<uint64_t> AugmentationLength;
        uint64_t StartAugmentationOffset;
        uint64_t EndAugmentationOffset;

        // Walk the augmentation string to get all the augmentation data.
        for (unsigned i = 0, e = AugmentationString.size(); i != e; ++i) {
          switch (AugmentationString[i]) {
          default:
            return createStringError(
                errc::invalid_argument,
                "unknown augmentation character in entry at 0x%" PRIx64,
                StartOffset);
          case 'L':
            LSDAPointerEncoding = Data.getU8(&Offset);
            break;
          case 'P': {
            if (Personality)
              return createStringError(
                  errc::invalid_argument,
                  "duplicate personality in entry at 0x%" PRIx64, StartOffset);
            PersonalityEncoding = Data.getU8(&Offset);
            Personality = Data.getEncodedPointer(
                &Offset, *PersonalityEncoding,
                EHFrameAddress ? EHFrameAddress + Offset : 0);
            break;
          }
          case 'R':
            FDEPointerEncoding = Data.getU8(&Offset);
            break;
          case 'S':
            // Current frame is a signal trampoline.
            break;
          case 'z':
            if (i)
              return createStringError(
                  errc::invalid_argument,
                  "'z' must be the first character at 0x%" PRIx64, StartOffset);
            // Parse the augmentation length first.  We only parse it if
            // the string contains a 'z'.
            AugmentationLength = Data.getULEB128(&Offset);
            StartAugmentationOffset = Offset;
            EndAugmentationOffset = Offset + *AugmentationLength;
            break;
          case 'B':
            // B-Key is used for signing functions associated with this
            // augmentation string
            break;
          }
        }

        if (AugmentationLength.hasValue()) {
          if (Offset != EndAugmentationOffset)
            return createStringError(errc::invalid_argument,
                                     "parsing augmentation data at 0x%" PRIx64
                                     " failed",
                                     StartOffset);
          AugmentationData = Data.getData().slice(StartAugmentationOffset,
                                                  EndAugmentationOffset);
        }
      }

      auto Cie = std::make_unique<CIE>(
          IsDWARF64, StartOffset, Length, Version, AugmentationString,
          AddressSize, SegmentDescriptorSize, CodeAlignmentFactor,
          DataAlignmentFactor, ReturnAddressRegister, AugmentationData,
          FDEPointerEncoding, LSDAPointerEncoding, Personality,
          PersonalityEncoding, Arch);
      CIEs[StartOffset] = Cie.get();
      Entries.emplace_back(std::move(Cie));
    } else {
      // FDE
      uint64_t CIEPointer = Id;
      uint64_t InitialLocation = 0;
      uint64_t AddressRange = 0;
      Optional<uint64_t> LSDAAddress;
      CIE *Cie = CIEs[IsEH ? (StartStructureOffset - CIEPointer) : CIEPointer];

      if (IsEH) {
        // The address size is encoded in the CIE we reference.
        if (!Cie)
          return createStringError(errc::invalid_argument,
                                   "parsing FDE data at 0x%" PRIx64
                                   " failed due to missing CIE",
                                   StartOffset);
        if (auto Val = Data.getEncodedPointer(
                &Offset, Cie->getFDEPointerEncoding(),
                EHFrameAddress ? EHFrameAddress + Offset : 0)) {
          InitialLocation = *Val;
        }
        if (auto Val = Data.getEncodedPointer(
                &Offset, Cie->getFDEPointerEncoding(), 0)) {
          AddressRange = *Val;
        }

        StringRef AugmentationString = Cie->getAugmentationString();
        if (!AugmentationString.empty()) {
          // Parse the augmentation length and data for this FDE.
          uint64_t AugmentationLength = Data.getULEB128(&Offset);

          uint64_t EndAugmentationOffset = Offset + AugmentationLength;

          // Decode the LSDA if the CIE augmentation string said we should.
          if (Cie->getLSDAPointerEncoding() != DW_EH_PE_omit) {
            LSDAAddress = Data.getEncodedPointer(
                &Offset, Cie->getLSDAPointerEncoding(),
                EHFrameAddress ? Offset + EHFrameAddress : 0);
          }

          if (Offset != EndAugmentationOffset)
            return createStringError(errc::invalid_argument,
                                     "parsing augmentation data at 0x%" PRIx64
                                     " failed",
                                     StartOffset);
        }
      } else {
        InitialLocation = Data.getRelocatedAddress(&Offset);
        AddressRange = Data.getRelocatedAddress(&Offset);
      }

      Entries.emplace_back(new FDE(IsDWARF64, StartOffset, Length, CIEPointer,
                                   InitialLocation, AddressRange, Cie,
                                   LSDAAddress, Arch));
    }

    if (Error E =
            Entries.back()->cfis().parse(Data, &Offset, EndStructureOffset))
      return E;

    if (Offset != EndStructureOffset)
      return createStringError(
          errc::invalid_argument,
          "parsing entry instructions at 0x%" PRIx64 " failed", StartOffset);
  }

  return Error::success();
}

FrameEntry *DWARFDebugFrame::getEntryAtOffset(uint64_t Offset) const {
  auto It = partition_point(Entries, [=](const std::unique_ptr<FrameEntry> &E) {
    return E->getOffset() < Offset;
  });
  if (It != Entries.end() && (*It)->getOffset() == Offset)
    return It->get();
  return nullptr;
}

void DWARFDebugFrame::dump(raw_ostream &OS, const MCRegisterInfo *MRI,
                           Optional<uint64_t> Offset) const {
  if (Offset) {
    if (auto *Entry = getEntryAtOffset(*Offset))
      Entry->dump(OS, MRI, IsEH);
    return;
  }

  OS << "\n";
  for (const auto &Entry : Entries)
    Entry->dump(OS, MRI, IsEH);
}
