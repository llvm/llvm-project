//===- ARMAttributeParser.cpp - ARM Attribute Information Printer ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ARMAttributeParser.h"
#include "llvm/ADT/STLArrayExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace llvm;
using namespace llvm::ARMBuildAttrs;

#define ATTRIBUTE_HANDLER(attr)                                                \
  { ARMBuildAttrs::attr, &ARMAttributeParser::attr }

const ARMAttributeParser::DisplayHandler ARMAttributeParser::displayRoutines[] =
    {
        {ARMBuildAttrs::CPU_raw_name, &ARMAttributeParser::stringAttribute},
        {ARMBuildAttrs::CPU_name, &ARMAttributeParser::stringAttribute},
        ATTRIBUTE_HANDLER(CPU_arch),
        ATTRIBUTE_HANDLER(CPU_arch_profile),
        ATTRIBUTE_HANDLER(ARM_ISA_use),
        ATTRIBUTE_HANDLER(THUMB_ISA_use),
        ATTRIBUTE_HANDLER(FP_arch),
        ATTRIBUTE_HANDLER(WMMX_arch),
        ATTRIBUTE_HANDLER(Advanced_SIMD_arch),
        ATTRIBUTE_HANDLER(MVE_arch),
        ATTRIBUTE_HANDLER(PCS_config),
        ATTRIBUTE_HANDLER(ABI_PCS_R9_use),
        ATTRIBUTE_HANDLER(ABI_PCS_RW_data),
        ATTRIBUTE_HANDLER(ABI_PCS_RO_data),
        ATTRIBUTE_HANDLER(ABI_PCS_GOT_use),
        ATTRIBUTE_HANDLER(ABI_PCS_wchar_t),
        ATTRIBUTE_HANDLER(ABI_FP_rounding),
        ATTRIBUTE_HANDLER(ABI_FP_denormal),
        ATTRIBUTE_HANDLER(ABI_FP_exceptions),
        ATTRIBUTE_HANDLER(ABI_FP_user_exceptions),
        ATTRIBUTE_HANDLER(ABI_FP_number_model),
        ATTRIBUTE_HANDLER(ABI_align_needed),
        ATTRIBUTE_HANDLER(ABI_align_preserved),
        ATTRIBUTE_HANDLER(ABI_enum_size),
        ATTRIBUTE_HANDLER(ABI_HardFP_use),
        ATTRIBUTE_HANDLER(ABI_VFP_args),
        ATTRIBUTE_HANDLER(ABI_WMMX_args),
        ATTRIBUTE_HANDLER(ABI_optimization_goals),
        ATTRIBUTE_HANDLER(ABI_FP_optimization_goals),
        ATTRIBUTE_HANDLER(compatibility),
        ATTRIBUTE_HANDLER(CPU_unaligned_access),
        ATTRIBUTE_HANDLER(FP_HP_extension),
        ATTRIBUTE_HANDLER(ABI_FP_16bit_format),
        ATTRIBUTE_HANDLER(MPextension_use),
        ATTRIBUTE_HANDLER(DIV_use),
        ATTRIBUTE_HANDLER(DSP_extension),
        ATTRIBUTE_HANDLER(T2EE_use),
        ATTRIBUTE_HANDLER(Virtualization_use),
        ATTRIBUTE_HANDLER(PAC_extension),
        ATTRIBUTE_HANDLER(BTI_extension),
        ATTRIBUTE_HANDLER(PACRET_use),
        ATTRIBUTE_HANDLER(BTI_use),
        ATTRIBUTE_HANDLER(nodefaults),
};

#undef ATTRIBUTE_HANDLER

Error ARMAttributeParser::stringAttribute(AttrType tag) {
  StringRef tagName =
      ELFAttrs::attrTypeAsString(tag, tagToStringMap, /*hasTagPrefix=*/false);
  StringRef desc = de.getCStrRef(cursor);

  if (sw) {
    DictScope scope(*sw, "Attribute");
    sw->printNumber("Tag", tag);
    if (!tagName.empty())
      sw->printString("TagName", tagName);
    sw->printString("Value", desc);
  }
  return Error::success();
}

Error ARMAttributeParser::CPU_arch(AttrType tag) {
  static const char *strings[] = {
    "Pre-v4", "ARM v4", "ARM v4T", "ARM v5T", "ARM v5TE", "ARM v5TEJ", "ARM v6",
    "ARM v6KZ", "ARM v6T2", "ARM v6K", "ARM v7", "ARM v6-M", "ARM v6S-M",
    "ARM v7E-M", "ARM v8-A", "ARM v8-R",
    "ARM v8-M Baseline", "ARM v8-M Mainline", nullptr, nullptr, nullptr,
    "ARM v8.1-M Mainline", "ARM v9-A"
  };
  return parseStringAttribute("CPU_arch", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::CPU_arch_profile(AttrType tag) {
  uint64_t value = de.getULEB128(cursor);

  StringRef profile;
  switch (value) {
  default: profile = "Unknown"; break;
  case 'A': profile = "Application"; break;
  case 'R': profile = "Real-time"; break;
  case 'M': profile = "Microcontroller"; break;
  case 'S': profile = "Classic"; break;
  case 0: profile = "None"; break;
  }

  printAttribute(tag, value, profile);
  return Error::success();
}

Error ARMAttributeParser::ARM_ISA_use(AttrType tag) {
  static const char *strings[] = {"Not Permitted", "Permitted"};
  return parseStringAttribute("ARM_ISA_use", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::THUMB_ISA_use(AttrType tag) {
  static const char *strings[] = {"Not Permitted", "Thumb-1", "Thumb-2", "Permitted"};
  return parseStringAttribute("THUMB_ISA_use", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::FP_arch(AttrType tag) {
  static const char *strings[] = {
      "Not Permitted", "VFPv1",     "VFPv2",      "VFPv3",         "VFPv3-D16",
      "VFPv4",         "VFPv4-D16", "ARMv8-a FP", "ARMv8-a FP-D16"};
  return parseStringAttribute("FP_arch", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::WMMX_arch(AttrType tag) {
  static const char *strings[] = {"Not Permitted", "WMMXv1", "WMMXv2"};
  return parseStringAttribute("WMMX_arch", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::Advanced_SIMD_arch(AttrType tag) {
  static const char *strings[] = {"Not Permitted", "NEONv1", "NEONv2+FMA",
                                  "ARMv8-a NEON", "ARMv8.1-a NEON"};
  return parseStringAttribute("Advanced_SIMD_arch", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::MVE_arch(AttrType tag) {
  static const char *strings[] = {"Not Permitted", "MVE integer",
                                  "MVE integer and float"};
  return parseStringAttribute("MVE_arch", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::PCS_config(AttrType tag) {
  static const char *strings[] = {
    "None", "Bare Platform", "Linux Application", "Linux DSO", "Palm OS 2004",
    "Reserved (Palm OS)", "Symbian OS 2004", "Reserved (Symbian OS)"};
  return parseStringAttribute("PCS_config", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::ABI_PCS_R9_use(AttrType tag) {
  static const char *strings[] = {"v6", "Static Base", "TLS", "Unused"};
  return parseStringAttribute("ABI_PCS_R9_use", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::ABI_PCS_RW_data(AttrType tag) {
  static const char *strings[] = {"Absolute", "PC-relative", "SB-relative",
                                  "Not Permitted"};
  return parseStringAttribute("ABI_PCS_RW_data", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::ABI_PCS_RO_data(AttrType tag) {
  static const char *strings[] = {"Absolute", "PC-relative", "Not Permitted"};
  return parseStringAttribute("ABI_PCS_RO_data", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::ABI_PCS_GOT_use(AttrType tag) {
  static const char *strings[] = {"Not Permitted", "Direct", "GOT-Indirect"};
  return parseStringAttribute("ABI_PCS_GOT_use", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::ABI_PCS_wchar_t(AttrType tag) {
  static const char *strings[] = {"Not Permitted", "Unknown", "2-byte",
                                  "Unknown", "4-byte"};
  return parseStringAttribute("ABI_PCS_wchar_t", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::ABI_FP_rounding(AttrType tag) {
  static const char *strings[] = {"IEEE-754", "Runtime"};
  return parseStringAttribute("ABI_FP_rounding", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::ABI_FP_denormal(AttrType tag) {
  static const char *strings[] = {"Unsupported", "IEEE-754", "Sign Only"};
  return parseStringAttribute("ABI_FP_denormal", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::ABI_FP_exceptions(AttrType tag) {
  static const char *strings[] = {"Not Permitted", "IEEE-754"};
  return parseStringAttribute("ABI_FP_exceptions", tag, makeArrayRef(strings));
}
Error ARMAttributeParser::ABI_FP_user_exceptions(AttrType tag) {
  static const char *strings[] = {"Not Permitted", "IEEE-754"};
  return parseStringAttribute("ABI_FP_user_exceptions", tag,
                              makeArrayRef(strings));
}

Error ARMAttributeParser::ABI_FP_number_model(AttrType tag) {
  static const char *strings[] = {"Not Permitted", "Finite Only", "RTABI",
                                  "IEEE-754"};
  return parseStringAttribute("ABI_FP_number_model", tag,
                              makeArrayRef(strings));
}

Error ARMAttributeParser::ABI_align_needed(AttrType tag) {
  static const char *strings[] = {"Not Permitted", "8-byte alignment",
                                  "4-byte alignment", "Reserved"};

  uint64_t value = de.getULEB128(cursor);

  std::string description;
  if (value < array_lengthof(strings))
    description = strings[value];
  else if (value <= 12)
    description = "8-byte alignment, " + utostr(1ULL << value) +
                  "-byte extended alignment";
  else
    description = "Invalid";

  printAttribute(tag, value, description);
  return Error::success();
}

Error ARMAttributeParser::ABI_align_preserved(AttrType tag) {
  static const char *strings[] = {"Not Required", "8-byte data alignment",
                                  "8-byte data and code alignment", "Reserved"};

  uint64_t value = de.getULEB128(cursor);

  std::string description;
  if (value < array_lengthof(strings))
    description = std::string(strings[value]);
  else if (value <= 12)
    description = std::string("8-byte stack alignment, ") +
                  utostr(1ULL << value) + std::string("-byte data alignment");
  else
    description = "Invalid";

  printAttribute(tag, value, description);
  return Error::success();
}

Error ARMAttributeParser::ABI_enum_size(AttrType tag) {
  static const char *strings[] = {"Not Permitted", "Packed", "Int32",
                                  "External Int32"};
  return parseStringAttribute("ABI_enum_size", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::ABI_HardFP_use(AttrType tag) {
  static const char *strings[] = {"Tag_FP_arch", "Single-Precision", "Reserved",
                                  "Tag_FP_arch (deprecated)"};
  return parseStringAttribute("ABI_HardFP_use", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::ABI_VFP_args(AttrType tag) {
  static const char *strings[] = {"AAPCS", "AAPCS VFP", "Custom",
                                  "Not Permitted"};
  return parseStringAttribute("ABI_VFP_args", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::ABI_WMMX_args(AttrType tag) {
  static const char *strings[] = {"AAPCS", "iWMMX", "Custom"};
  return parseStringAttribute("ABI_WMMX_args", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::ABI_optimization_goals(AttrType tag) {
  static const char *strings[] = {
    "None", "Speed", "Aggressive Speed", "Size", "Aggressive Size", "Debugging",
    "Best Debugging"
  };
  return parseStringAttribute("ABI_optimization_goals", tag,
                              makeArrayRef(strings));
}

Error ARMAttributeParser::ABI_FP_optimization_goals(AttrType tag) {
  static const char *strings[] = {
      "None",     "Speed",        "Aggressive Speed", "Size", "Aggressive Size",
      "Accuracy", "Best Accuracy"};
  return parseStringAttribute("ABI_FP_optimization_goals", tag,
                              makeArrayRef(strings));
}

Error ARMAttributeParser::compatibility(AttrType tag) {
  uint64_t integer = de.getULEB128(cursor);
  StringRef string = de.getCStrRef(cursor);

  if (sw) {
    DictScope scope(*sw, "Attribute");
    sw->printNumber("Tag", tag);
    sw->startLine() << "Value: " << integer << ", " << string << '\n';
    sw->printString("TagName",
                    ELFAttrs::attrTypeAsString(tag, tagToStringMap,
                                               /*hasTagPrefix=*/false));
    switch (integer) {
    case 0:
      sw->printString("Description", StringRef("No Specific Requirements"));
      break;
    case 1:
      sw->printString("Description", StringRef("AEABI Conformant"));
      break;
    default:
      sw->printString("Description", StringRef("AEABI Non-Conformant"));
      break;
    }
  }
  return Error::success();
}

Error ARMAttributeParser::CPU_unaligned_access(AttrType tag) {
  static const char *strings[] = {"Not Permitted", "v6-style"};
  return parseStringAttribute("CPU_unaligned_access", tag,
                              makeArrayRef(strings));
}

Error ARMAttributeParser::FP_HP_extension(AttrType tag) {
  static const char *strings[] = {"If Available", "Permitted"};
  return parseStringAttribute("FP_HP_extension", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::ABI_FP_16bit_format(AttrType tag) {
  static const char *strings[] = {"Not Permitted", "IEEE-754", "VFPv3"};
  return parseStringAttribute("ABI_FP_16bit_format", tag,
                              makeArrayRef(strings));
}

Error ARMAttributeParser::MPextension_use(AttrType tag) {
  static const char *strings[] = {"Not Permitted", "Permitted"};
  return parseStringAttribute("MPextension_use", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::DIV_use(AttrType tag) {
  static const char *strings[] = {"If Available", "Not Permitted", "Permitted"};
  return parseStringAttribute("DIV_use", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::DSP_extension(AttrType tag) {
  static const char *strings[] = {"Not Permitted", "Permitted"};
  return parseStringAttribute("DSP_extension", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::T2EE_use(AttrType tag) {
  static const char *strings[] = {"Not Permitted", "Permitted"};
  return parseStringAttribute("T2EE_use", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::Virtualization_use(AttrType tag) {
  static const char *strings[] = {"Not Permitted", "TrustZone",
                                  "Virtualization Extensions",
                                  "TrustZone + Virtualization Extensions"};
  return parseStringAttribute("Virtualization_use", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::PAC_extension(ARMBuildAttrs::AttrType tag) {
  static const char *strings[] = {"Not Permitted", "Permitted in NOP space",
                                  "Permitted"};
  return parseStringAttribute("PAC_extension", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::BTI_extension(ARMBuildAttrs::AttrType tag) {
  static const char *strings[] = {"Not Permitted", "Permitted in NOP space",
                                  "Permitted"};
  return parseStringAttribute("BTI_extension", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::PACRET_use(ARMBuildAttrs::AttrType tag) {
  static const char *strings[] = {"Not Used", "Used"};
  return parseStringAttribute("PACRET_use", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::BTI_use(ARMBuildAttrs::AttrType tag) {
  static const char *strings[] = {"Not Used", "Used"};
  return parseStringAttribute("BTI_use", tag, makeArrayRef(strings));
}

Error ARMAttributeParser::nodefaults(AttrType tag) {
  uint64_t value = de.getULEB128(cursor);
  printAttribute(tag, value, "Unspecified Tags UNDEFINED");
  return Error::success();
}

Error ARMAttributeParser::handler(uint64_t tag, bool &handled) {
  handled = false;
  for (unsigned AHI = 0, AHE = array_lengthof(displayRoutines); AHI != AHE;
       ++AHI) {
    if (uint64_t(displayRoutines[AHI].attribute) == tag) {
      if (Error e =
              (this->*displayRoutines[AHI].routine)(static_cast<AttrType>(tag)))
        return e;
      handled = true;
      break;
    }
  }

  return Error::success();
}
