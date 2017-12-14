//===- DWARFAcceleratorTable.cpp ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFAcceleratorTable.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFRelocMap.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>
#include <utility>

using namespace llvm;

llvm::Error DWARFAcceleratorTable::extract() {
  uint32_t Offset = 0;

  // Check that we can at least read the header.
  if (!AccelSection.isValidOffset(offsetof(Header, HeaderDataLength)+4))
    return make_error<StringError>("Section too small: cannot read header.",
                                   inconvertibleErrorCode());

  Hdr.Magic = AccelSection.getU32(&Offset);
  Hdr.Version = AccelSection.getU16(&Offset);
  Hdr.HashFunction = AccelSection.getU16(&Offset);
  Hdr.NumBuckets = AccelSection.getU32(&Offset);
  Hdr.NumHashes = AccelSection.getU32(&Offset);
  Hdr.HeaderDataLength = AccelSection.getU32(&Offset);

  // Check that we can read all the hashes and offsets from the
  // section (see SourceLevelDebugging.rst for the structure of the index).
  // We need to substract one because we're checking for an *offset* which is
  // equal to the size for an empty table and hence pointer after the section.
  if (!AccelSection.isValidOffset(sizeof(Hdr) + Hdr.HeaderDataLength +
                                  Hdr.NumBuckets * 4 + Hdr.NumHashes * 8 - 1))
    return make_error<StringError>(
        "Section too small: cannot read buckets and hashes.",
        inconvertibleErrorCode());

  HdrData.DIEOffsetBase = AccelSection.getU32(&Offset);
  uint32_t NumAtoms = AccelSection.getU32(&Offset);

  for (unsigned i = 0; i < NumAtoms; ++i) {
    uint16_t AtomType = AccelSection.getU16(&Offset);
    auto AtomForm = static_cast<dwarf::Form>(AccelSection.getU16(&Offset));
    HdrData.Atoms.push_back(std::make_pair(AtomType, AtomForm));
  }

  IsValid = true;
  return Error::success();
}

uint32_t DWARFAcceleratorTable::getNumBuckets() { return Hdr.NumBuckets; }
uint32_t DWARFAcceleratorTable::getNumHashes() { return Hdr.NumHashes; }
uint32_t DWARFAcceleratorTable::getSizeHdr() { return sizeof(Hdr); }
uint32_t DWARFAcceleratorTable::getHeaderDataLength() {
  return Hdr.HeaderDataLength;
}

ArrayRef<std::pair<DWARFAcceleratorTable::HeaderData::AtomType,
                   DWARFAcceleratorTable::HeaderData::Form>>
DWARFAcceleratorTable::getAtomsDesc() {
  return HdrData.Atoms;
}

bool DWARFAcceleratorTable::validateForms() {
  for (auto Atom : getAtomsDesc()) {
    DWARFFormValue FormValue(Atom.second);
    switch (Atom.first) {
    case dwarf::DW_ATOM_die_offset:
    case dwarf::DW_ATOM_die_tag:
    case dwarf::DW_ATOM_type_flags:
      if ((!FormValue.isFormClass(DWARFFormValue::FC_Constant) &&
           !FormValue.isFormClass(DWARFFormValue::FC_Flag)) ||
          FormValue.getForm() == dwarf::DW_FORM_sdata)
        return false;
    default:
      break;
    }
  }
  return true;
}

std::pair<uint32_t, dwarf::Tag>
DWARFAcceleratorTable::readAtoms(uint32_t &HashDataOffset) {
  uint32_t DieOffset = dwarf::DW_INVALID_OFFSET;
  dwarf::Tag DieTag = dwarf::DW_TAG_null;
  DWARFFormParams FormParams = {Hdr.Version, 0, dwarf::DwarfFormat::DWARF32};

  for (auto Atom : getAtomsDesc()) {
    DWARFFormValue FormValue(Atom.second);
    FormValue.extractValue(AccelSection, &HashDataOffset, FormParams);
    switch (Atom.first) {
    case dwarf::DW_ATOM_die_offset:
      DieOffset = *FormValue.getAsUnsignedConstant();
      break;
    case dwarf::DW_ATOM_die_tag:
      DieTag = (dwarf::Tag)*FormValue.getAsUnsignedConstant();
      break;
    default:
      break;
    }
  }
  return {DieOffset, DieTag};
}

LLVM_DUMP_METHOD void DWARFAcceleratorTable::dump(raw_ostream &OS) const {
  if (!IsValid)
    return;

  // Dump the header.
  OS << "Magic = " << format("0x%08x", Hdr.Magic) << '\n'
     << "Version = " << format("0x%04x", Hdr.Version) << '\n'
     << "Hash function = " << format("0x%08x", Hdr.HashFunction) << '\n'
     << "Bucket count = " << Hdr.NumBuckets << '\n'
     << "Hashes count = " << Hdr.NumHashes << '\n'
     << "HeaderData length = " << Hdr.HeaderDataLength << '\n'
     << "DIE offset base = " << HdrData.DIEOffsetBase << '\n'
     << "Number of atoms = " << HdrData.Atoms.size() << '\n';

  unsigned i = 0;
  SmallVector<DWARFFormValue, 3> AtomForms;
  for (const auto &Atom: HdrData.Atoms) {
    OS << format("Atom[%d] Type: ", i++);
    auto TypeString = dwarf::AtomTypeString(Atom.first);
    if (!TypeString.empty())
      OS << TypeString;
    else
      OS << format("DW_ATOM_Unknown_0x%x", Atom.first);
    OS << " Form: ";
    auto FormString = dwarf::FormEncodingString(Atom.second);
    if (!FormString.empty())
      OS << FormString;
    else
      OS << format("DW_FORM_Unknown_0x%x", Atom.second);
    OS << '\n';
    AtomForms.push_back(DWARFFormValue(Atom.second));
  }

  // Now go through the actual tables and dump them.
  uint32_t Offset = sizeof(Hdr) + Hdr.HeaderDataLength;
  unsigned HashesBase = Offset + Hdr.NumBuckets * 4;
  unsigned OffsetsBase = HashesBase + Hdr.NumHashes * 4;
  DWARFFormParams FormParams = {Hdr.Version, 0, dwarf::DwarfFormat::DWARF32};

  for (unsigned Bucket = 0; Bucket < Hdr.NumBuckets; ++Bucket) {
    unsigned Index = AccelSection.getU32(&Offset);

    OS << format("Bucket[%d]\n", Bucket);
    if (Index == UINT32_MAX) {
      OS << "  EMPTY\n";
      continue;
    }

    for (unsigned HashIdx = Index; HashIdx < Hdr.NumHashes; ++HashIdx) {
      unsigned HashOffset = HashesBase + HashIdx*4;
      unsigned OffsetsOffset = OffsetsBase + HashIdx*4;
      uint32_t Hash = AccelSection.getU32(&HashOffset);

      if (Hash % Hdr.NumBuckets != Bucket)
        break;

      unsigned DataOffset = AccelSection.getU32(&OffsetsOffset);
      OS << format("  Hash = 0x%08x Offset = 0x%08x\n", Hash, DataOffset);
      if (!AccelSection.isValidOffset(DataOffset)) {
        OS << "    Invalid section offset\n";
        continue;
      }
      while (AccelSection.isValidOffsetForDataOfSize(DataOffset, 4)) {
        unsigned StringOffset = AccelSection.getRelocatedValue(4, &DataOffset);
        if (!StringOffset)
          break;
        OS << format("    Name: %08x \"%s\"\n", StringOffset,
                     StringSection.getCStr(&StringOffset));
        unsigned NumData = AccelSection.getU32(&DataOffset);
        for (unsigned Data = 0; Data < NumData; ++Data) {
          OS << format("    Data[%d] => ", Data);
          unsigned i = 0;
          for (auto &Atom : AtomForms) {
            OS << format("{Atom[%d]: ", i++);
            if (Atom.extractValue(AccelSection, &DataOffset, FormParams))
              Atom.dump(OS);
            else
              OS << "Error extracting the value";
            OS << "} ";
          }
          OS << '\n';
        }
      }
    }
  }
}

DWARFAcceleratorTable::ValueIterator::ValueIterator(
    const DWARFAcceleratorTable &AccelTable, unsigned Offset)
    : AccelTable(&AccelTable), DataOffset(Offset) {
  if (!AccelTable.AccelSection.isValidOffsetForDataOfSize(DataOffset, 4))
    return;

  for (const auto &Atom : AccelTable.HdrData.Atoms)
    AtomForms.push_back(DWARFFormValue(Atom.second));

  // Read the first entry.
  NumData = AccelTable.AccelSection.getU32(&DataOffset);
  Next();
}

void DWARFAcceleratorTable::ValueIterator::Next() {
  assert(NumData > 0 && "attempted to increment iterator past the end");
  auto &AccelSection = AccelTable->AccelSection;
  if (Data >= NumData ||
      !AccelSection.isValidOffsetForDataOfSize(DataOffset, 4)) {
    NumData = 0;
    return;
  }
  DWARFFormParams FormParams = {AccelTable->Hdr.Version, 0,
                                dwarf::DwarfFormat::DWARF32};
  for (auto &Atom : AtomForms)
    Atom.extractValue(AccelSection, &DataOffset, FormParams);
  ++Data;
}

iterator_range<DWARFAcceleratorTable::ValueIterator>
DWARFAcceleratorTable::equal_range(StringRef Key) const {
  if (!IsValid)
    return make_range(ValueIterator(), ValueIterator());

  // Find the bucket.
  unsigned HashValue = dwarf::djbHash(Key);
  unsigned Bucket = HashValue % Hdr.NumBuckets;
  unsigned BucketBase = sizeof(Hdr) + Hdr.HeaderDataLength;
  unsigned HashesBase = BucketBase + Hdr.NumBuckets * 4;
  unsigned OffsetsBase = HashesBase + Hdr.NumHashes * 4;

  unsigned BucketOffset = BucketBase + Bucket * 4;
  unsigned Index = AccelSection.getU32(&BucketOffset);

  // Search through all hashes in the bucket.
  for (unsigned HashIdx = Index; HashIdx < Hdr.NumHashes; ++HashIdx) {
    unsigned HashOffset = HashesBase + HashIdx * 4;
    unsigned OffsetsOffset = OffsetsBase + HashIdx * 4;
    uint32_t Hash = AccelSection.getU32(&HashOffset);

    if (Hash % Hdr.NumBuckets != Bucket)
      // We are already in the next bucket.
      break;

    unsigned DataOffset = AccelSection.getU32(&OffsetsOffset);
    unsigned StringOffset = AccelSection.getRelocatedValue(4, &DataOffset);
    if (!StringOffset)
      break;

    // Finally, compare the key.
    if (Key == StringSection.getCStr(&StringOffset))
      return make_range({*this, DataOffset}, ValueIterator());
  }
  return make_range(ValueIterator(), ValueIterator());
}
