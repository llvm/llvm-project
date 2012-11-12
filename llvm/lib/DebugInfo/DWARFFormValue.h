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

#include "llvm/Support/DataExtractor.h"

namespace llvm {

class DWARFCompileUnit;
class raw_ostream;

class DWARFFormValue {
public:
  struct ValueType {
    ValueType() : data(NULL) {
      uval = 0;
    }

    union {
      uint64_t uval;
      int64_t sval;
      const char* cstr;
    };
    const uint8_t* data;
  };

  enum {
    eValueTypeInvalid = 0,
    eValueTypeUnsigned,
    eValueTypeSigned,
    eValueTypeCStr,
    eValueTypeBlock
  };

private:
  uint16_t Form;   // Form for this value.
  ValueType Value; // Contains all data for the form.

public:
  DWARFFormValue(uint16_t form = 0) : Form(form) {}
  uint16_t getForm() const { return Form; }
  const ValueType& value() const { return Value; }
  void dump(raw_ostream &OS, const DWARFCompileUnit* cu) const;
  bool extractValue(DataExtractor data, uint32_t *offset_ptr,
                    const DWARFCompileUnit *cu);
  bool isInlinedCStr() const {
    return Value.data != NULL && Value.data == (const uint8_t*)Value.cstr;
  }
  const uint8_t *BlockData() const;
  uint64_t getReference(const DWARFCompileUnit* cu) const;

  /// Resolve any compile unit specific references so that we don't need
  /// the compile unit at a later time in order to work with the form
  /// value.
  bool resolveCompileUnitReferences(const DWARFCompileUnit* cu);
  uint64_t getUnsigned() const { return Value.uval; }
  int64_t getSigned() const { return Value.sval; }
  const char *getAsCString(const DataExtractor *debug_str_data_ptr) const;
  bool skipValue(DataExtractor debug_info_data, uint32_t *offset_ptr,
                 const DWARFCompileUnit *cu) const;
  static bool skipValue(uint16_t form, DataExtractor debug_info_data,
                        uint32_t *offset_ptr, const DWARFCompileUnit *cu);
  static bool isBlockForm(uint16_t form);
  static bool isDataForm(uint16_t form);
  static const uint8_t *getFixedFormSizesForAddressSize(uint8_t addr_size);
};

}

#endif
