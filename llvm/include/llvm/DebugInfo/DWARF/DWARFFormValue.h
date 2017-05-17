//===- DWARFFormValue.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFFORMVALUE_H
#define LLVM_DEBUGINFO_DWARFFORMVALUE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Dwarf.h"
#include <cstdint>

namespace llvm {

class DWARFUnit;
class raw_ostream;

class DWARFFormValue {
public:
  enum FormClass {
    FC_Unknown,
    FC_Address,
    FC_Block,
    FC_Constant,
    FC_String,
    FC_Flag,
    FC_Reference,
    FC_Indirect,
    FC_SectionOffset,
    FC_Exprloc
  };

private:
  struct ValueType {
    ValueType() { uval = 0; }

    union {
      uint64_t uval;
      int64_t sval;
      const char *cstr;
    };
    const uint8_t *data = nullptr;
  };

  dwarf::Form Form;             /// Form for this value.
  ValueType Value;              /// Contains all data for the form.
  const DWARFUnit *U = nullptr; /// Remember the DWARFUnit at extract time.

public:
  DWARFFormValue(dwarf::Form F = dwarf::Form(0)) : Form(F) {}

  dwarf::Form getForm() const { return Form; }
  uint64_t getRawUValue() const { return Value.uval; }
  void setForm(dwarf::Form F) { Form = F; }
  void setUValue(uint64_t V) { Value.uval = V; }
  void setSValue(int64_t V) { Value.sval = V; }
  void setPValue(const char *V) { Value.cstr = V; }

  void setBlockValue(const ArrayRef<uint8_t> &Data) {
    Value.data = Data.data();
    setUValue(Data.size());
  }

  bool isFormClass(FormClass FC) const;
  const DWARFUnit *getUnit() const { return U; }
  void dump(raw_ostream &OS) const;

  /// Extracts a value in \p Data at offset \p *OffsetPtr.
  ///
  /// The passed DWARFUnit is allowed to be nullptr, in which
  /// case no relocation processing will be performed and some
  /// kind of forms that depend on Unit information are disallowed.
  /// \param Data The DataExtractor to use.
  /// \param OffsetPtr The offset within DataExtractor where the data starts.
  /// \param U The optional DWARFUnit supplying information for some forms.
  /// \returns whether the extraction succeeded.
  bool extractValue(const DataExtractor &Data, uint32_t *OffsetPtr,
                    const DWARFUnit *U);

  bool isInlinedCStr() const {
    return Value.data != nullptr && Value.data == (const uint8_t *)Value.cstr;
  }

  /// getAsFoo functions below return the extracted value as Foo if only
  /// DWARFFormValue has form class is suitable for representing Foo.
  Optional<uint64_t> getAsReference() const;
  Optional<uint64_t> getAsUnsignedConstant() const;
  Optional<int64_t> getAsSignedConstant() const;
  Optional<const char *> getAsCString() const;
  Optional<uint64_t> getAsAddress() const;
  Optional<uint64_t> getAsSectionOffset() const;
  Optional<ArrayRef<uint8_t>> getAsBlock() const;
  Optional<uint64_t> getAsCStringOffset() const;
  Optional<uint64_t> getAsReferenceUVal() const;

  /// Get the fixed byte size for a given form.
  ///
  /// If the form always has a fixed valid byte size that doesn't depend on a
  /// DWARFUnit, then an Optional with a value will be returned. If the form
  /// can vary in size depending on the DWARFUnit (DWARF version, address byte
  /// size, or DWARF 32/64) and the DWARFUnit is valid, then an Optional with a
  /// valid value is returned. If the form is always encoded using a variable
  /// length storage format (ULEB or SLEB numbers or blocks) or the size
  /// depends on a DWARFUnit and the DWARFUnit is NULL, then None will be
  /// returned.
  /// \param Form The DWARF form to get the fixed byte size for
  /// \param U The DWARFUnit that can be used to help determine the byte size.
  ///
  /// \returns Optional<uint8_t> value with the fixed byte size or None if
  /// \p Form doesn't have a fixed byte size or a DWARFUnit wasn't supplied
  /// and was needed to calculate the byte size.
  static Optional<uint8_t> getFixedByteSize(dwarf::Form Form,
                                            const DWARFUnit *U = nullptr);

  /// Get the fixed byte size for a given form.
  ///
  /// If the form has a fixed byte size given a valid DWARF version and address
  /// byte size, then an Optional with a valid value is returned. If the form
  /// is always encoded using a variable length storage format (ULEB or SLEB
  /// numbers or blocks) then None will be returned.
  ///
  /// \param Form DWARF form to get the fixed byte size for
  /// \param Version DWARF version number.
  /// \param AddrSize size of an address in bytes.
  /// \param Format enum value from llvm::dwarf::DwarfFormat.
  /// \returns Optional<uint8_t> value with the fixed byte size or None if
  /// \p Form doesn't have a fixed byte size.
  static Optional<uint8_t> getFixedByteSize(dwarf::Form Form, uint16_t Version,
                                            uint8_t AddrSize,
                                            llvm::dwarf::DwarfFormat Format);

  /// Skip a form in \p DebugInfoData at offset specified by \p OffsetPtr.
  ///
  /// Skips the bytes for this form in the debug info and updates the offset.
  ///
  /// \param DebugInfoData the .debug_info data to use to skip the value.
  /// \param OffsetPtr a reference to the offset that will be updated.
  /// \param U the DWARFUnit to use when skipping the form in case the form
  /// size differs according to data in the DWARFUnit.
  /// \returns true on success, false if the form was not skipped.
  bool skipValue(DataExtractor DebugInfoData, uint32_t *OffsetPtr,
                 const DWARFUnit *U) const;

  /// Skip a form in \p DebugInfoData at offset specified by \p OffsetPtr.
  ///
  /// Skips the bytes for this form in the debug info and updates the offset.
  ///
  /// \param Form the DW_FORM enumeration that indicates the form to skip.
  /// \param DebugInfoData the .debug_info data to use to skip the value.
  /// \param OffsetPtr a reference to the offset that will be updated.
  /// \param U the DWARFUnit to use when skipping the form in case the form
  /// size differs according to data in the DWARFUnit.
  /// \returns true on success, false if the form was not skipped.
  static bool skipValue(dwarf::Form Form, DataExtractor DebugInfoData,
                        uint32_t *OffsetPtr, const DWARFUnit *U);

  /// Skip a form in \p DebugInfoData at offset specified by \p OffsetPtr.
  ///
  /// Skips the bytes for this form in the debug info and updates the offset.
  ///
  /// \param Form the DW_FORM enumeration that indicates the form to skip.
  /// \param DebugInfoData the .debug_info data to use to skip the value.
  /// \param OffsetPtr a reference to the offset that will be updated.
  /// \param Version DWARF version number.
  /// \param AddrSize size of an address in bytes.
  /// \param Format enum value from llvm::dwarf::DwarfFormat.
  /// \returns true on success, false if the form was not skipped.
  static bool skipValue(dwarf::Form Form, DataExtractor DebugInfoData,
                        uint32_t *OffsetPtr, uint16_t Version, uint8_t AddrSize,
                        llvm::dwarf::DwarfFormat Format);

private:
  void dumpString(raw_ostream &OS) const;
};

namespace dwarf {

/// Take an optional DWARFFormValue and try to extract a string value from it.
///
/// \param V and optional DWARFFormValue to attempt to extract the value from.
/// \returns an optional value that contains a value if the form value
/// was valid and was a string.
inline Optional<const char *> toString(const Optional<DWARFFormValue> &V) {
  if (V)
    return V->getAsCString();
  return None;
}

/// Take an optional DWARFFormValue and extract a string value from it.
///
/// \param V and optional DWARFFormValue to attempt to extract the value from.
/// \param Default the default value to return in case of failure.
/// \returns the string value or Default if the V doesn't have a value or the
/// form value's encoding wasn't a string.
inline const char *toString(const Optional<DWARFFormValue> &V,
                            const char *Default) {
  return toString(V).getValueOr(Default);
}

/// Take an optional DWARFFormValue and try to extract an unsigned constant.
///
/// \param V and optional DWARFFormValue to attempt to extract the value from.
/// \returns an optional value that contains a value if the form value
/// was valid and has a unsigned constant form.
inline Optional<uint64_t> toUnsigned(const Optional<DWARFFormValue> &V) {
  if (V)
    return V->getAsUnsignedConstant();
  return None;
}

/// Take an optional DWARFFormValue and extract a unsigned constant.
///
/// \param V and optional DWARFFormValue to attempt to extract the value from.
/// \param Default the default value to return in case of failure.
/// \returns the extracted unsigned value or Default if the V doesn't have a
/// value or the form value's encoding wasn't an unsigned constant form.
inline uint64_t toUnsigned(const Optional<DWARFFormValue> &V,
                           uint64_t Default) {
  return toUnsigned(V).getValueOr(Default);
}

/// Take an optional DWARFFormValue and try to extract an reference.
///
/// \param V and optional DWARFFormValue to attempt to extract the value from.
/// \returns an optional value that contains a value if the form value
/// was valid and has a reference form.
inline Optional<uint64_t> toReference(const Optional<DWARFFormValue> &V) {
  if (V)
    return V->getAsReference();
  return None;
}

/// Take an optional DWARFFormValue and extract a reference.
///
/// \param V and optional DWARFFormValue to attempt to extract the value from.
/// \param Default the default value to return in case of failure.
/// \returns the extracted reference value or Default if the V doesn't have a
/// value or the form value's encoding wasn't a reference form.
inline uint64_t toReference(const Optional<DWARFFormValue> &V,
                            uint64_t Default) {
  return toReference(V).getValueOr(Default);
}

/// Take an optional DWARFFormValue and try to extract an signed constant.
///
/// \param V and optional DWARFFormValue to attempt to extract the value from.
/// \returns an optional value that contains a value if the form value
/// was valid and has a signed constant form.
inline Optional<int64_t> toSigned(const Optional<DWARFFormValue> &V) {
  if (V)
    return V->getAsSignedConstant();
  return None;
}

/// Take an optional DWARFFormValue and extract a signed integer.
///
/// \param V and optional DWARFFormValue to attempt to extract the value from.
/// \param Default the default value to return in case of failure.
/// \returns the extracted signed integer value or Default if the V doesn't
/// have a value or the form value's encoding wasn't a signed integer form.
inline int64_t toSigned(const Optional<DWARFFormValue> &V, int64_t Default) {
  return toSigned(V).getValueOr(Default);
}

/// Take an optional DWARFFormValue and try to extract an address.
///
/// \param V and optional DWARFFormValue to attempt to extract the value from.
/// \returns an optional value that contains a value if the form value
/// was valid and has a address form.
inline Optional<uint64_t> toAddress(const Optional<DWARFFormValue> &V) {
  if (V)
    return V->getAsAddress();
  return None;
}

/// Take an optional DWARFFormValue and extract a address.
///
/// \param V and optional DWARFFormValue to attempt to extract the value from.
/// \param Default the default value to return in case of failure.
/// \returns the extracted address value or Default if the V doesn't have a
/// value or the form value's encoding wasn't an address form.
inline uint64_t toAddress(const Optional<DWARFFormValue> &V, uint64_t Default) {
  return toAddress(V).getValueOr(Default);
}

/// Take an optional DWARFFormValue and try to extract an section offset.
///
/// \param V and optional DWARFFormValue to attempt to extract the value from.
/// \returns an optional value that contains a value if the form value
/// was valid and has a section offset form.
inline Optional<uint64_t> toSectionOffset(const Optional<DWARFFormValue> &V) {
  if (V)
    return V->getAsSectionOffset();
  return None;
}

/// Take an optional DWARFFormValue and extract a section offset.
///
/// \param V and optional DWARFFormValue to attempt to extract the value from.
/// \param Default the default value to return in case of failure.
/// \returns the extracted section offset value or Default if the V doesn't
/// have a value or the form value's encoding wasn't a section offset form.
inline uint64_t toSectionOffset(const Optional<DWARFFormValue> &V,
                                uint64_t Default) {
  return toSectionOffset(V).getValueOr(Default);
}

/// Take an optional DWARFFormValue and try to extract block data.
///
/// \param V and optional DWARFFormValue to attempt to extract the value from.
/// \returns an optional value that contains a value if the form value
/// was valid and has a block form.
inline Optional<ArrayRef<uint8_t>> toBlock(const Optional<DWARFFormValue> &V) {
  if (V)
    return V->getAsBlock();
  return None;
}

} // end namespace dwarf

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARFFORMVALUE_H
