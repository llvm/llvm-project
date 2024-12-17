//===-- ARMBuildAttrs.cpp - ARM Build Attributes --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ARMBuildAttributes.h"


namespace llvm {
  class StringRef;
  namespace ARMBuildAttrs {
    // AArch64 build attributes
    StringRef getSubsectionTag() { return "aeabi_subsection"; }
    StringRef getAttrTag() { return "aeabi_attribute"; }

    StringRef getVendorName(unsigned Vendor) {
      switch (Vendor) {
      case AEABI_FEATURE_AND_BITS:
        return VendorName[AEABI_FEATURE_AND_BITS];
      case AEABI_PAUTHABI:
        return VendorName[AEABI_PAUTHABI];
      case VENDOR_NOT_FOUND:
      [[fallthrough]];
      default:
        assert(0 && "unknown AArch64 vendor");
        return "";
      }
    }
    VendorID getVendorID(StringRef Vendor) {
      if(Vendor == VendorName[AEABI_FEATURE_AND_BITS]) {
        return AEABI_FEATURE_AND_BITS;
      }
      if(Vendor == VendorName[AEABI_PAUTHABI]) {
        return AEABI_PAUTHABI;
      }
      assert(0 && "unknown AArch64 vendor");
      return VENDOR_NOT_FOUND;
    }
    StringRef getSubsectionUnknownError() { return "unknown AArch64 build attributes subsection"; }

    StringRef getOptionalStr(unsigned Optional) {
      switch (Optional) {
      case REQUIRED:
        return OptionalStr[REQUIRED];
      case OPTIONAL:
        return OptionalStr[OPTIONAL];
      case OPTIONAL_NOT_FOUND:
      [[fallthrough]];
      default:
        assert(0 && "unknown AArch64 Optional option");
        return "";
      }
    }
    SubsectionOptional getOptionalID(StringRef Optional) {
      if(Optional == OptionalStr[REQUIRED])
        return REQUIRED;
      if(Optional == OptionalStr[OPTIONAL])
        return OPTIONAL;
      assert(0 && "unknown AArch64 Optional option");
      return OPTIONAL_NOT_FOUND;
    }
    StringRef getSubsectionOptionalUnknownError() { return "unknown AArch64 build attributes optionality, expecting required|optional"; }

    StringRef getTypeStr(unsigned Type) {
      switch (Type) {
      case ULEB128:
        return TypeStr[ULEB128];
      case NTBS:
        return TypeStr[NTBS];
      case TYPE_NOT_FOUND:
      [[fallthrough]];
      default:
        assert(0 && "unknown AArch64 subsection type");
        return "";
      }
    }
    SubsectionType getTypeID(StringRef Type) {
      if(Type == TypeStr[ULEB128] || Type == (TypeStr[ULEB128].upper()))
        return ULEB128;
      if(Type == TypeStr[NTBS]|| Type == (TypeStr[NTBS].upper()))
        return NTBS;
      assert(0 && "unknown AArch64 subsection type");
      return TYPE_NOT_FOUND;
    }
    StringRef getSubsectionTypeUnknownError() { return "unknown AArch64 build attributes type, expecting uleb128 or ntbs"; }

    StringRef getPauthABITagsStr(unsigned PauthABITag) {
      switch(PauthABITag) {
        case TAG_PAUTH_PLATFORM:
          return PauthABITagsStr[TAG_PAUTH_PLATFORM - 1]; // Tag_PAuth_Platform = 1 in accordance with the spec
        case TAG_PAUTH_SCHEMA:
          return PauthABITagsStr[TAG_PAUTH_SCHEMA - 1]; // Tag_PAuth_Schema = 2 in accordance with the spec
        case PAUTHABI_TAG_NOT_FOUND:
          [[fallthrough]];
        default:
          assert(0 && "unknown pauthabi tag");
          return "";
      }
    }
    PauthABITags getPauthABITagsID(StringRef PauthABITag) {
      if(PauthABITag == PauthABITagsStr[TAG_PAUTH_PLATFORM - 1])
        return TAG_PAUTH_PLATFORM;
      if(PauthABITag == PauthABITagsStr[TAG_PAUTH_SCHEMA - 1])
        return TAG_PAUTH_SCHEMA;
      assert(0 && "unknown FPauthABI tag");
      return PAUTHABI_TAG_NOT_FOUND;
    }
    StringRef getPauthabiTagError() { return "unknown tag for the AArch64 Pauthabi subsection"; }

    StringRef getFeatureAndBitsTagsStr(unsigned FeatureAndBitsTag) {
      switch(FeatureAndBitsTag) {
        case TAG_FEATURE_BTI:
          return FeatureAndBitsTagsStr[TAG_FEATURE_BTI];
        case TAG_FEATURE_PAC:
          return FeatureAndBitsTagsStr[TAG_FEATURE_PAC];
        case TAG_FEATURE_GCS:
          return FeatureAndBitsTagsStr[TAG_FEATURE_GCS];
        case FEATURE_AND_BITS_TAG_NOT_FOUND:
          [[fallthrough]];
        default:
          assert(0 && "unknown feature and bits tag");
          return "";
      }
    }
    FeatureAndBitsTags getFeatureAndBitsTagsID(StringRef FeatureAndBitsTag) {
      if(FeatureAndBitsTag == FeatureAndBitsTagsStr[TAG_FEATURE_BTI])
        return TAG_FEATURE_BTI;
      if(FeatureAndBitsTag == FeatureAndBitsTagsStr[TAG_FEATURE_PAC])
        return TAG_FEATURE_PAC;
      if(FeatureAndBitsTag == FeatureAndBitsTagsStr[TAG_FEATURE_GCS])
        return TAG_FEATURE_GCS;
      assert(0 && "unknown Feature and Bits tag");
      return FEATURE_AND_BITS_TAG_NOT_FOUND;
    }
    StringRef getFeatureAndBitsTagError() { return "unknown tag for the AArch64 Feature And Bits subsection"; }
    ///--- AArch64 build attributes
  } // namespace ARMBuildAttrs
} // namespace llvm


using namespace llvm;

static const TagNameItem tagData[] = {
    {ARMBuildAttrs::File, "Tag_File"},
    {ARMBuildAttrs::Section, "Tag_Section"},
    {ARMBuildAttrs::Symbol, "Tag_Symbol"},
    {ARMBuildAttrs::CPU_raw_name, "Tag_CPU_raw_name"},
    {ARMBuildAttrs::CPU_name, "Tag_CPU_name"},
    {ARMBuildAttrs::CPU_arch, "Tag_CPU_arch"},
    {ARMBuildAttrs::CPU_arch_profile, "Tag_CPU_arch_profile"},
    {ARMBuildAttrs::ARM_ISA_use, "Tag_ARM_ISA_use"},
    {ARMBuildAttrs::THUMB_ISA_use, "Tag_THUMB_ISA_use"},
    {ARMBuildAttrs::FP_arch, "Tag_FP_arch"},
    {ARMBuildAttrs::WMMX_arch, "Tag_WMMX_arch"},
    {ARMBuildAttrs::Advanced_SIMD_arch, "Tag_Advanced_SIMD_arch"},
    {ARMBuildAttrs::MVE_arch, "Tag_MVE_arch"},
    {ARMBuildAttrs::PCS_config, "Tag_PCS_config"},
    {ARMBuildAttrs::ABI_PCS_R9_use, "Tag_ABI_PCS_R9_use"},
    {ARMBuildAttrs::ABI_PCS_RW_data, "Tag_ABI_PCS_RW_data"},
    {ARMBuildAttrs::ABI_PCS_RO_data, "Tag_ABI_PCS_RO_data"},
    {ARMBuildAttrs::ABI_PCS_GOT_use, "Tag_ABI_PCS_GOT_use"},
    {ARMBuildAttrs::ABI_PCS_wchar_t, "Tag_ABI_PCS_wchar_t"},
    {ARMBuildAttrs::ABI_FP_rounding, "Tag_ABI_FP_rounding"},
    {ARMBuildAttrs::ABI_FP_denormal, "Tag_ABI_FP_denormal"},
    {ARMBuildAttrs::ABI_FP_exceptions, "Tag_ABI_FP_exceptions"},
    {ARMBuildAttrs::ABI_FP_user_exceptions, "Tag_ABI_FP_user_exceptions"},
    {ARMBuildAttrs::ABI_FP_number_model, "Tag_ABI_FP_number_model"},
    {ARMBuildAttrs::ABI_align_needed, "Tag_ABI_align_needed"},
    {ARMBuildAttrs::ABI_align_preserved, "Tag_ABI_align_preserved"},
    {ARMBuildAttrs::ABI_enum_size, "Tag_ABI_enum_size"},
    {ARMBuildAttrs::ABI_HardFP_use, "Tag_ABI_HardFP_use"},
    {ARMBuildAttrs::ABI_VFP_args, "Tag_ABI_VFP_args"},
    {ARMBuildAttrs::ABI_WMMX_args, "Tag_ABI_WMMX_args"},
    {ARMBuildAttrs::ABI_optimization_goals, "Tag_ABI_optimization_goals"},
    {ARMBuildAttrs::ABI_FP_optimization_goals, "Tag_ABI_FP_optimization_goals"},
    {ARMBuildAttrs::compatibility, "Tag_compatibility"},
    {ARMBuildAttrs::CPU_unaligned_access, "Tag_CPU_unaligned_access"},
    {ARMBuildAttrs::FP_HP_extension, "Tag_FP_HP_extension"},
    {ARMBuildAttrs::ABI_FP_16bit_format, "Tag_ABI_FP_16bit_format"},
    {ARMBuildAttrs::MPextension_use, "Tag_MPextension_use"},
    {ARMBuildAttrs::DIV_use, "Tag_DIV_use"},
    {ARMBuildAttrs::DSP_extension, "Tag_DSP_extension"},
    {ARMBuildAttrs::PAC_extension, "Tag_PAC_extension"},
    {ARMBuildAttrs::BTI_extension, "Tag_BTI_extension"},
    {ARMBuildAttrs::BTI_use, "Tag_BTI_use"},
    {ARMBuildAttrs::PACRET_use, "Tag_PACRET_use"},
    {ARMBuildAttrs::nodefaults, "Tag_nodefaults"},
    {ARMBuildAttrs::also_compatible_with, "Tag_also_compatible_with"},
    {ARMBuildAttrs::T2EE_use, "Tag_T2EE_use"},
    {ARMBuildAttrs::conformance, "Tag_conformance"},
    {ARMBuildAttrs::Virtualization_use, "Tag_Virtualization_use"},

    // Legacy Names
    {ARMBuildAttrs::FP_arch, "Tag_VFP_arch"},
    {ARMBuildAttrs::FP_HP_extension, "Tag_VFP_HP_extension"},
    {ARMBuildAttrs::ABI_align_needed, "Tag_ABI_align8_needed"},
    {ARMBuildAttrs::ABI_align_preserved, "Tag_ABI_align8_preserved"},
};

constexpr TagNameMap ARMAttributeTags{tagData};
const TagNameMap &llvm::ARMBuildAttrs::getARMAttributeTags() {
  return ARMAttributeTags;
}
