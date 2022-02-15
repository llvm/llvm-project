//===- DWARFFormValue.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFFORMVALUE_H
#define LLVM_DEBUGINFO_DWARF_DWARFFORMVALUE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/Support/DataExtractor.h"
#include <cstdint>

namespace llvm {

class DWARFContext;
class DWARFObject;
class DWARFDataExtractor;
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
    ValueType(int64_t V) : sval(V) {}
    ValueType(uint64_t V) : uval(V) {}
    ValueType(const char *V) : cstr(V) {}

    union {
      uint64_t uval;
      int64_t sval;
      const char *cstr;
    };
    const uint8_t *data = nullptr;
    uint64_t SectionIndex;      /// Section index for reference forms.
  };

  dwarf::Form Form;             /// Form for this value.
  dwarf::DwarfFormat Format =
      dwarf::DWARF32;           /// Remember the DWARF format at extract time.
  ValueType Value;              /// Contains all data for the form.
  const DWARFUnit *U = nullptr; /// Remember the DWARFUnit at extract time.
  const DWARFContext *C = nullptr; /// Context for extract time.

  DWARFFormValue(dwarf::Form F, ValueType V) : Form(F), Value(V) {}

public:
  DWARFFormValue(dwarf::Form F = dwarf::Form(0)) : Form(F) {}

  static DWARFFormValue createFromSValue(dwarf::Form F, int64_t V);
  static DWARFFormValue createFromUValue(dwarf::Form F, uint64_t V);
  static DWARFFormValue createFromPValue(dwarf::Form F, const char *V);
  static DWARFFormValue createFromBlockValue(dwarf::Form F,
                                             ArrayRef<uint8_t> D);
  static DWARFFormValue createFromUnit(dwarf::Form F, const DWARFUnit *Unit,
                                       uint64_t *OffsetPtr);

  dwarf::Form getForm() const { return Form; }
  uint64_t getRawUValue() const { return Value.uval; }

  bool isFormClass(FormClass FC) const;
  const DWARFUnit *getUnit() const { return U; }
  void dump(raw_ostream &OS, DIDumpOptions DumpOpts = DIDumpOptions()) const;
  void dumpSectionedAddress(raw_ostream &OS, DIDumpOptions DumpOpts,
                            object::SectionedAddress SA) const;
  void dumpAddress(raw_ostream &OS, uint64_t Address) const;
  static void dumpAddress(raw_ostream &OS, uint8_t AddressSize,
                          uint64_t Address);
  static void dumpAddressSection(const DWARFObject &Obj, raw_ostream &OS,
                                 DIDumpOptions DumpOpts, uint64_t SectionIndex);

  /// Extracts a value in \p Data at offset \p *OffsetPtr. The information
  /// in \p FormParams is needed to interpret some forms. The optional
  /// \p Context and \p Unit allows extracting information if the form refers
  /// to other sections (e.g., .debug_str).
  bool extractValue(const DWARFDataExtractor &Data, uint64_t *OffsetPtr,
                    dwarf::FormParams FormParams,
                    const DWARFContext *Context = nullptr,
                    const DWARFUnit *Unit = nullptr);

  bool extractValue(const DWARFDataExtractor &Data, uint64_t *OffsetPtr,
                    dwarf::FormParams FormParams, const DWARFUnit *U) {
    return extractValue(Data, OffsetPtr, FormParams, nullptr, U);
  }

  /// getAsFoo functions below return the extracted value as Foo if only
  /// DWARFFormValue has form class is suitable for representing Foo.
  Optional<uint64_t> getAsReference() const;
  struct UnitOffset {
    DWARFUnit *Unit;
    uint64_t Offset;
  };
  Optional<UnitOffset> getAsRelativeReference() const;
  Optional<uint64_t> getAsUnsignedConstant() const;
  Optional<int64_t> getAsSignedConstant() const;
  Expected<const char *> getAsCString() const;
  Optional<uint64_t> getAsAddress() const;
  Optional<object::SectionedAddress> getAsSectionedAddress() const;
  Optional<uint64_t> getAsSectionOffset() const;
  Optional<ArrayRef<uint8_t>> getAsBlock() const;
  Optional<uint64_t> getAsCStringOffset() const;
  Optional<uint64_t> getAsReferenceUVal() const;
  /// Correctly extract any file paths from a form value.
  ///
  /// These attributes can be in the from DW_AT_decl_file or DW_AT_call_file
  /// attributes. We need to use the file index in the correct DWARFUnit's line
  /// table prologue, and each DWARFFormValue has the DWARFUnit the form value
  /// was extracted from.
  ///
  /// \param Kind The kind of path to extract.
  ///
  /// \returns A valid string value on success, or llvm::None if the form class
  /// is not FC_Constant, or if the file index is not valid.
  Optional<std::string>
  getAsFile(DILineInfoSpecifier::FileLineInfoKind Kind) const;

  /// Skip a form's value in \p DebugInfoData at the offset specified by
  /// \p OffsetPtr.
  ///
  /// Skips the bytes for the current form and updates the offset.
  ///
  /// \param DebugInfoData The data where we want to skip the value.
  /// \param OffsetPtr A reference to the offset that will be updated.
  /// \param Params DWARF parameters to help interpret forms.
  /// \returns true on success, false if the form was not skipped.
  bool skipValue(DataExtractor DebugInfoData, uint64_t *OffsetPtr,
                 const dwarf::FormParams Params) const {
    return DWARFFormValue::skipValue(Form, DebugInfoData, OffsetPtr, Params);
  }

  /// Skip a form's value in \p DebugInfoData at the offset specified by
  /// \p OffsetPtr.
  ///
  /// Skips the bytes for the specified form and updates the offset.
  ///
  /// \param Form The DW_FORM enumeration that indicates the form to skip.
  /// \param DebugInfoData The data where we want to skip the value.
  /// \param OffsetPtr A reference to the offset that will be updated.
  /// \param FormParams DWARF parameters to help interpret forms.
  /// \returns true on success, false if the form was not skipped.
  static bool skipValue(dwarf::Form Form, DataExtractor DebugInfoData,
                        uint64_t *OffsetPtr,
                        const dwarf::FormParams FormParams);

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
  if (!V)
    return None;
  Expected<const char*> E = V->getAsCString();
  if (!E) {
    consumeError(E.takeError());
    return None;
  }
  return *E;
}

/// Take an optional DWARFFormValue and try to extract a string value from it.
///
/// \param V and optional DWARFFormValue to attempt to extract the value from.
/// \returns an optional value that contains a value if the form value
/// was valid and was a string.
inline StringRef toStringRef(const Optional<DWARFFormValue> &V,
                             StringRef Default = {}) {
  if (!V)
    return Default;
  auto S = V->getAsCString();
  if (!S) {
    consumeError(S.takeError());
    return Default;
  }
  if (!*S)
    return Default;
  return *S;
}

/// Take an optional DWARFFormValue and extract a string value from it.
///
/// \param V and optional DWARFFormValue to attempt to extract the value from.
/// \param Default the default value to return in case of failure.
/// \returns the string value or Default if the V doesn't have a value or the
/// form value's encoding wasn't a string.
inline const char *toString(const Optional<DWARFFormValue> &V,
                            const char *Default) {
  if (auto E = toString(V))
    return *E;
  return Default;
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

inline Optional<object::SectionedAddress>
toSectionedAddress(const Optional<DWARFFormValue> &V) {
  if (V)
    return V->getAsSectionedAddress();
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

#endif // LLVM_DEBUGINFO_DWARF_DWARFFORMVALUE_H
