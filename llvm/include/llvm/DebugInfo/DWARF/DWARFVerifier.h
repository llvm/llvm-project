//===- DWARFVerifier.h ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFVERIFIER_H
#define LLVM_DEBUGINFO_DWARF_DWARFVERIFIER_H

#include <cstdint>
#include <map>
#include <set>

namespace llvm {
class raw_ostream;
struct DWARFAttribute;
class DWARFContext;
class DWARFDie;
class DWARFUnit;
class DWARFAcceleratorTable;
class DWARFDataExtractor;

/// A class that verifies DWARF debug information given a DWARF Context.
class DWARFVerifier {
  raw_ostream &OS;
  DWARFContext &DCtx;
  /// A map that tracks all references (converted absolute references) so we
  /// can verify each reference points to a valid DIE and not an offset that
  /// lies between to valid DIEs.
  std::map<uint64_t, std::set<uint32_t>> ReferenceToDIEOffsets;
  uint32_t NumDebugLineErrors = 0;
  uint32_t NumAppleNamesErrors = 0;

  /// Verifies the header of a unit in the .debug_info section.
  ///
  /// This function currently checks for:
  /// - Unit is in 32-bit DWARF format. The function can be modified to
  /// support 64-bit format.
  /// - The DWARF version is valid
  /// - The unit type is valid (if unit is in version >=5)
  /// - The unit doesn't extend beyond .debug_info section
  /// - The address size is valid
  /// - The offset in the .debug_abbrev section is valid
  ///
  /// \param DebugInfoData The .debug_info section data
  /// \param Offset A reference to the offset start of the unit. The offset will
  /// be updated to point to the next unit in .debug_info
  /// \param UnitIndex The index of the unit to be verified
  /// \param UnitType A reference to the type of the unit
  /// \param isUnitDWARF64 A reference to a flag that shows whether the unit is
  /// in 64-bit format.
  ///
  /// \returns true if the header is verified successfully, false otherwise.
  bool verifyUnitHeader(const DWARFDataExtractor DebugInfoData,
                        uint32_t *Offset, unsigned UnitIndex, uint8_t &UnitType,
                        bool &isUnitDWARF64);


  bool verifyUnitContents(DWARFUnit Unit);
  /// Verifies the attribute's DWARF attribute and its value.
  ///
  /// This function currently checks for:
  /// - DW_AT_ranges values is a valid .debug_ranges offset
  /// - DW_AT_stmt_list is a valid .debug_line offset
  ///
  /// \param Die          The DWARF DIE that owns the attribute value
  /// \param AttrValue    The DWARF attribute value to check
  ///
  /// \returns NumErrors The number of errors occured during verification of
  /// attributes' values in a .debug_info section unit
  unsigned verifyDebugInfoAttribute(const DWARFDie &Die,
                                    DWARFAttribute &AttrValue);

  /// Verifies the attribute's DWARF form.
  ///
  /// This function currently checks for:
  /// - All DW_FORM_ref values that are CU relative have valid CU offsets
  /// - All DW_FORM_ref_addr values have valid .debug_info offsets
  /// - All DW_FORM_strp values have valid .debug_str offsets
  ///
  /// \param Die          The DWARF DIE that owns the attribute value
  /// \param AttrValue    The DWARF attribute value to check
  ///
  /// \returns NumErrors The number of errors occured during verification of
  /// attributes' forms in a .debug_info section unit
  unsigned verifyDebugInfoForm(const DWARFDie &Die, DWARFAttribute &AttrValue);

  /// Verifies the all valid references that were found when iterating through
  /// all of the DIE attributes.
  ///
  /// This function will verify that all references point to DIEs whose DIE
  /// offset matches. This helps to ensure if a DWARF link phase moved things
  /// around, that it doesn't create invalid references by failing to relocate
  /// CU relative and absolute references.
  ///
  /// \returns NumErrors The number of errors occured during verification of
  /// references for the .debug_info section
  unsigned verifyDebugInfoReferences();

  /// Verify the the DW_AT_stmt_list encoding and value and ensure that no
  /// compile units that have the same DW_AT_stmt_list value.
  void verifyDebugLineStmtOffsets();

  /// Verify that all of the rows in the line table are valid.
  ///
  /// This function currently checks for:
  /// - addresses within a sequence that decrease in value
  /// - invalid file indexes
  void verifyDebugLineRows();

public:
  DWARFVerifier(raw_ostream &S, DWARFContext &D)
      : OS(S), DCtx(D) {}
  /// Verify the information in the .debug_info section.
  ///
  /// Any errors are reported to the stream that was this object was
  /// constructed with.
  ///
  /// \returns true if the .debug_info verifies successfully, false otherwise.
  bool handleDebugInfo();

  /// Verify the information in the .debug_line section.
  ///
  /// Any errors are reported to the stream that was this object was
  /// constructed with.
  ///
  /// \returns true if the .debug_line verifies successfully, false otherwise.
  bool handleDebugLine();

  /// Verify the information in the .apple_names accelerator table.
  ///
  /// Any errors are reported to the stream that was this object was
  /// constructed with.
  ///
  /// \returns true if the .apple_names verifies successfully, false otherwise.
  bool handleAppleNames();
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFCONTEXT_H
