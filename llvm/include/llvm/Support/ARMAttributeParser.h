//===--- ARMAttributeParser.h - ARM Attribute Information Printer ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ARMATTRIBUTEPARSER_H
#define LLVM_SUPPORT_ARMATTRIBUTEPARSER_H

#include "ARMBuildAttributes.h"
#include "ScopedPrinter.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

#include <map>

namespace llvm {
class StringRef;

class ARMAttributeParser {
  ScopedPrinter *sw;

  std::map<unsigned, unsigned> attributes;
  DataExtractor de{ArrayRef<uint8_t>{}, true, 0};
  DataExtractor::Cursor cursor{0};

  struct DisplayHandler {
    ARMBuildAttrs::AttrType attribute;
    Error (ARMAttributeParser::*routine)(ARMBuildAttrs::AttrType);
  };
  static const DisplayHandler displayRoutines[];

  Error parseAttributeList(uint32_t length);
  void parseIndexList(SmallVectorImpl<uint8_t> &indexList);
  Error parseSubsection(uint32_t length);
  Error parseStringAttribute(const char *name, ARMBuildAttrs::AttrType tag,
                             const ArrayRef<const char *> array);
  void printAttribute(unsigned tag, unsigned value, StringRef valueDesc);

  Error stringAttribute(ARMBuildAttrs::AttrType tag);

  Error CPU_arch(ARMBuildAttrs::AttrType tag);
  Error CPU_arch_profile(ARMBuildAttrs::AttrType tag);
  Error ARM_ISA_use(ARMBuildAttrs::AttrType tag);
  Error THUMB_ISA_use(ARMBuildAttrs::AttrType tag);
  Error FP_arch(ARMBuildAttrs::AttrType tag);
  Error WMMX_arch(ARMBuildAttrs::AttrType tag);
  Error Advanced_SIMD_arch(ARMBuildAttrs::AttrType tag);
  Error MVE_arch(ARMBuildAttrs::AttrType tag);
  Error PCS_config(ARMBuildAttrs::AttrType tag);
  Error ABI_PCS_R9_use(ARMBuildAttrs::AttrType tag);
  Error ABI_PCS_RW_data(ARMBuildAttrs::AttrType tag);
  Error ABI_PCS_RO_data(ARMBuildAttrs::AttrType tag);
  Error ABI_PCS_GOT_use(ARMBuildAttrs::AttrType tag);
  Error ABI_PCS_wchar_t(ARMBuildAttrs::AttrType tag);
  Error ABI_FP_rounding(ARMBuildAttrs::AttrType tag);
  Error ABI_FP_denormal(ARMBuildAttrs::AttrType tag);
  Error ABI_FP_exceptions(ARMBuildAttrs::AttrType tag);
  Error ABI_FP_user_exceptions(ARMBuildAttrs::AttrType tag);
  Error ABI_FP_number_model(ARMBuildAttrs::AttrType tag);
  Error ABI_align_needed(ARMBuildAttrs::AttrType tag);
  Error ABI_align_preserved(ARMBuildAttrs::AttrType tag);
  Error ABI_enum_size(ARMBuildAttrs::AttrType tag);
  Error ABI_HardFP_use(ARMBuildAttrs::AttrType tag);
  Error ABI_VFP_args(ARMBuildAttrs::AttrType tag);
  Error ABI_WMMX_args(ARMBuildAttrs::AttrType tag);
  Error ABI_optimization_goals(ARMBuildAttrs::AttrType tag);
  Error ABI_FP_optimization_goals(ARMBuildAttrs::AttrType tag);
  Error compatibility(ARMBuildAttrs::AttrType tag);
  Error CPU_unaligned_access(ARMBuildAttrs::AttrType tag);
  Error FP_HP_extension(ARMBuildAttrs::AttrType tag);
  Error ABI_FP_16bit_format(ARMBuildAttrs::AttrType tag);
  Error MPextension_use(ARMBuildAttrs::AttrType tag);
  Error DIV_use(ARMBuildAttrs::AttrType tag);
  Error DSP_extension(ARMBuildAttrs::AttrType tag);
  Error T2EE_use(ARMBuildAttrs::AttrType tag);
  Error Virtualization_use(ARMBuildAttrs::AttrType tag);
  Error nodefaults(ARMBuildAttrs::AttrType tag);

public:
  ARMAttributeParser(ScopedPrinter *sw) : sw(sw) {}
  ARMAttributeParser() : sw(nullptr) {}
  ~ARMAttributeParser() { static_cast<void>(!cursor.takeError()); }

  Error parse(ArrayRef<uint8_t> section, support::endianness endian);

  bool hasAttribute(unsigned tag) const { return attributes.count(tag); }

  unsigned getAttributeValue(unsigned tag) const {
    return attributes.find(tag)->second;
  }
};

}

#endif

