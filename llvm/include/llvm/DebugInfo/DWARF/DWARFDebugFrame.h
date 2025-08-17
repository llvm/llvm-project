//===- DWARFDebugFrame.h - Parsing of .debug_frame --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFDEBUGFRAME_H
#define LLVM_DEBUGINFO_DWARF_DWARFDEBUGFRAME_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/iterator.h"
#include "llvm/DebugInfo/DWARF/LowLevel/DWARFCFIProgram.h"
#include "llvm/DebugInfo/DWARF/LowLevel/DWARFExpression.h"
#include "llvm/DebugInfo/DWARF/LowLevel/DWARFUnwindTable.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Triple.h"
#include <memory>
#include <vector>

namespace llvm {

class raw_ostream;
class DWARFDataExtractor;
class MCRegisterInfo;
struct DIDumpOptions;

namespace dwarf {

class CIE;

/// Create an UnwindTable from a Common Information Entry (CIE).
///
/// \param Cie The Common Information Entry to extract the table from. The
/// CFIProgram is retrieved from the \a Cie object and used to create the
/// UnwindTable.
///
/// \returns An error if the DWARF Call Frame Information opcodes have state
/// machine errors, or a valid UnwindTable otherwise.
LLVM_ABI Expected<UnwindTable> createUnwindTable(const CIE *Cie);

class FDE;

/// Create an UnwindTable from a Frame Descriptor Entry (FDE).
///
/// \param Fde The Frame Descriptor Entry to extract the table from. The
/// CFIProgram is retrieved from the \a Fde object and used to create the
/// UnwindTable.
///
/// \returns An error if the DWARF Call Frame Information opcodes have state
/// machine errors, or a valid UnwindTable otherwise.
LLVM_ABI Expected<UnwindTable> createUnwindTable(const FDE *Fde);

/// An entry in either debug_frame or eh_frame. This entry can be a CIE or an
/// FDE.
class FrameEntry {
public:
  enum FrameKind { FK_CIE, FK_FDE };

  FrameEntry(FrameKind K, bool IsDWARF64, uint64_t Offset, uint64_t Length,
             uint64_t CodeAlign, int64_t DataAlign, Triple::ArchType Arch)
      : Kind(K), IsDWARF64(IsDWARF64), Offset(Offset), Length(Length),
        CFIs(CodeAlign, DataAlign, Arch) {}

  virtual ~FrameEntry() = default;

  FrameKind getKind() const { return Kind; }
  uint64_t getOffset() const { return Offset; }
  uint64_t getLength() const { return Length; }
  const CFIProgram &cfis() const { return CFIs; }
  CFIProgram &cfis() { return CFIs; }

  /// Dump the instructions in this CFI fragment
  virtual void dump(raw_ostream &OS, DIDumpOptions DumpOpts) const = 0;

protected:
  const FrameKind Kind;

  const bool IsDWARF64;

  /// Offset of this entry in the section.
  const uint64_t Offset;

  /// Entry length as specified in DWARF.
  const uint64_t Length;

  CFIProgram CFIs;
};

/// DWARF Common Information Entry (CIE)
class LLVM_ABI CIE : public FrameEntry {
public:
  // CIEs (and FDEs) are simply container classes, so the only sensible way to
  // create them is by providing the full parsed contents in the constructor.
  CIE(bool IsDWARF64, uint64_t Offset, uint64_t Length, uint8_t Version,
      SmallString<8> Augmentation, uint8_t AddressSize,
      uint8_t SegmentDescriptorSize, uint64_t CodeAlignmentFactor,
      int64_t DataAlignmentFactor, uint64_t ReturnAddressRegister,
      SmallString<8> AugmentationData, uint32_t FDEPointerEncoding,
      uint32_t LSDAPointerEncoding, std::optional<uint64_t> Personality,
      std::optional<uint32_t> PersonalityEnc, Triple::ArchType Arch)
      : FrameEntry(FK_CIE, IsDWARF64, Offset, Length, CodeAlignmentFactor,
                   DataAlignmentFactor, Arch),
        Version(Version), Augmentation(std::move(Augmentation)),
        AddressSize(AddressSize), SegmentDescriptorSize(SegmentDescriptorSize),
        CodeAlignmentFactor(CodeAlignmentFactor),
        DataAlignmentFactor(DataAlignmentFactor),
        ReturnAddressRegister(ReturnAddressRegister),
        AugmentationData(std::move(AugmentationData)),
        FDEPointerEncoding(FDEPointerEncoding),
        LSDAPointerEncoding(LSDAPointerEncoding), Personality(Personality),
        PersonalityEnc(PersonalityEnc) {}

  static bool classof(const FrameEntry *FE) { return FE->getKind() == FK_CIE; }

  StringRef getAugmentationString() const { return Augmentation; }
  uint64_t getCodeAlignmentFactor() const { return CodeAlignmentFactor; }
  int64_t getDataAlignmentFactor() const { return DataAlignmentFactor; }
  uint8_t getVersion() const { return Version; }
  uint64_t getReturnAddressRegister() const { return ReturnAddressRegister; }
  std::optional<uint64_t> getPersonalityAddress() const { return Personality; }
  std::optional<uint32_t> getPersonalityEncoding() const {
    return PersonalityEnc;
  }

  StringRef getAugmentationData() const { return AugmentationData; }

  uint32_t getFDEPointerEncoding() const { return FDEPointerEncoding; }

  uint32_t getLSDAPointerEncoding() const { return LSDAPointerEncoding; }

  void dump(raw_ostream &OS, DIDumpOptions DumpOpts) const override;

private:
  /// The following fields are defined in section 6.4.1 of the DWARF standard v4
  const uint8_t Version;
  const SmallString<8> Augmentation;
  const uint8_t AddressSize;
  const uint8_t SegmentDescriptorSize;
  const uint64_t CodeAlignmentFactor;
  const int64_t DataAlignmentFactor;
  const uint64_t ReturnAddressRegister;

  // The following are used when the CIE represents an EH frame entry.
  const SmallString<8> AugmentationData;
  const uint32_t FDEPointerEncoding;
  const uint32_t LSDAPointerEncoding;
  const std::optional<uint64_t> Personality;
  const std::optional<uint32_t> PersonalityEnc;
};

/// DWARF Frame Description Entry (FDE)
class LLVM_ABI FDE : public FrameEntry {
public:
  FDE(bool IsDWARF64, uint64_t Offset, uint64_t Length, uint64_t CIEPointer,
      uint64_t InitialLocation, uint64_t AddressRange, CIE *Cie,
      std::optional<uint64_t> LSDAAddress, Triple::ArchType Arch)
      : FrameEntry(FK_FDE, IsDWARF64, Offset, Length,
                   Cie ? Cie->getCodeAlignmentFactor() : 0,
                   Cie ? Cie->getDataAlignmentFactor() : 0, Arch),
        CIEPointer(CIEPointer), InitialLocation(InitialLocation),
        AddressRange(AddressRange), LinkedCIE(Cie), LSDAAddress(LSDAAddress) {}

  ~FDE() override = default;

  const CIE *getLinkedCIE() const { return LinkedCIE; }
  uint64_t getCIEPointer() const { return CIEPointer; }
  uint64_t getInitialLocation() const { return InitialLocation; }
  uint64_t getAddressRange() const { return AddressRange; }
  std::optional<uint64_t> getLSDAAddress() const { return LSDAAddress; }

  void dump(raw_ostream &OS, DIDumpOptions DumpOpts) const override;

  static bool classof(const FrameEntry *FE) { return FE->getKind() == FK_FDE; }

private:
  /// The following fields are defined in section 6.4.1 of the DWARFv3 standard.
  /// Note that CIE pointers in EH FDEs, unlike DWARF FDEs, contain relative
  /// offsets to the linked CIEs. See the following link for more info:
  /// https://refspecs.linuxfoundation.org/LSB_5.0.0/LSB-Core-generic/LSB-Core-generic/ehframechpt.html
  const uint64_t CIEPointer;
  const uint64_t InitialLocation;
  const uint64_t AddressRange;
  const CIE *LinkedCIE;
  const std::optional<uint64_t> LSDAAddress;
};

} // end namespace dwarf

/// A parsed .debug_frame or .eh_frame section
class DWARFDebugFrame {
  const Triple::ArchType Arch;
  // True if this is parsing an eh_frame section.
  const bool IsEH;
  // Not zero for sane pointer values coming out of eh_frame
  const uint64_t EHFrameAddress;

  std::vector<std::unique_ptr<dwarf::FrameEntry>> Entries;
  using iterator = pointee_iterator<decltype(Entries)::const_iterator>;

  /// Return the entry at the given offset or nullptr.
  dwarf::FrameEntry *getEntryAtOffset(uint64_t Offset) const;

public:
  // If IsEH is true, assume it is a .eh_frame section. Otherwise,
  // it is a .debug_frame section. EHFrameAddress should be different
  // than zero for correct parsing of .eh_frame addresses when they
  // use a PC-relative encoding.
  LLVM_ABI DWARFDebugFrame(Triple::ArchType Arch, bool IsEH = false,
                           uint64_t EHFrameAddress = 0);
  LLVM_ABI ~DWARFDebugFrame();

  /// Dump the section data into the given stream.
  LLVM_ABI void dump(raw_ostream &OS, DIDumpOptions DumpOpts,
                     std::optional<uint64_t> Offset) const;

  /// Parse the section from raw data. \p Data is assumed to contain the whole
  /// frame section contents to be parsed.
  LLVM_ABI Error parse(DWARFDataExtractor Data);

  /// Return whether the section has any entries.
  bool empty() const { return Entries.empty(); }

  /// DWARF Frame entries accessors
  iterator begin() const { return Entries.begin(); }
  iterator end() const { return Entries.end(); }
  iterator_range<iterator> entries() const {
    return iterator_range<iterator>(Entries.begin(), Entries.end());
  }

  uint64_t getEHFrameAddress() const { return EHFrameAddress; }
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFDEBUGFRAME_H
