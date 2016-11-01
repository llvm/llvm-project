//===-- DWARFFormValue.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFFORMVALUE_H
#define LLVM_DEBUGINFO_DWARFFORMVALUE_H

#include "llvm/ADT/Optional.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Dwarf.h"

namespace llvm {

template <typename T> class ArrayRef;
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
    ValueType() : data(nullptr) {
      uval = 0;
    }

    union {
      uint64_t uval;
      int64_t sval;
      const char* cstr;
    };
    const uint8_t* data;
  };

  dwarf::Form Form; // Form for this value.
  ValueType Value; // Contains all data for the form.
  const DWARFUnit *U; // Remember the DWARFUnit at extract time.

public:
  DWARFFormValue(dwarf::Form F = dwarf::Form(0)) : Form(F), U(nullptr) {}
  dwarf::Form getForm() const { return Form; }
  bool isFormClass(FormClass FC) const;
  const DWARFUnit *getUnit() const { return U; }
  void dump(raw_ostream &OS) const;

  /// \brief extracts a value in data at offset *offset_ptr.
  ///
  /// The passed DWARFUnit is allowed to be nullptr, in which
  /// case no relocation processing will be performed and some
  /// kind of forms that depend on Unit information are disallowed.
  /// \returns whether the extraction succeeded.
  bool extractValue(const DataExtractor &Data, uint32_t *OffsetPtr,
                    const DWARFUnit *U);
  bool isInlinedCStr() const {
    return Value.data != nullptr && Value.data == (const uint8_t*)Value.cstr;
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

  bool skipValue(DataExtractor debug_info_data, uint32_t *offset_ptr,
                 const DWARFUnit *U) const;
  static bool skipValue(dwarf::Form form, DataExtractor debug_info_data,
                        uint32_t *offset_ptr, const DWARFUnit *u);
  static bool skipValue(dwarf::Form form, DataExtractor debug_info_data,
                        uint32_t *offset_ptr, uint16_t Version,
                        uint8_t AddrSize);

  static ArrayRef<uint8_t> getFixedFormSizes(uint8_t AddrSize,
                                             uint16_t Version);
private:
  void dumpString(raw_ostream &OS) const;
};

}

#endif
