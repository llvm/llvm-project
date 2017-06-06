//===--- AMDGPUCodeObjectMetadata.cpp ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief AMDGPU Code Object Metadata definitions and in-memory
/// representations.
///
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/AMDGPUCodeObjectMetadata.h"
#include "llvm/Support/YAMLTraits.h"

using namespace llvm::AMDGPU;
using namespace llvm::AMDGPU::CodeObject;

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(uint32_t)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(std::string)
LLVM_YAML_IS_SEQUENCE_VECTOR(Kernel::Arg::Metadata)
LLVM_YAML_IS_SEQUENCE_VECTOR(Kernel::Metadata)

namespace llvm {
namespace yaml {

template <>
struct ScalarEnumerationTraits<AccessQualifier> {
  static void enumeration(IO &YIO, AccessQualifier &EN) {
    YIO.enumCase(EN, "Default", AccessQualifier::Default);
    YIO.enumCase(EN, "ReadOnly", AccessQualifier::ReadOnly);
    YIO.enumCase(EN, "WriteOnly", AccessQualifier::WriteOnly);
    YIO.enumCase(EN, "ReadWrite", AccessQualifier::ReadWrite);
  }
};

template <>
struct ScalarEnumerationTraits<AddressSpaceQualifier> {
  static void enumeration(IO &YIO, AddressSpaceQualifier &EN) {
    YIO.enumCase(EN, "Private", AddressSpaceQualifier::Private);
    YIO.enumCase(EN, "Global", AddressSpaceQualifier::Global);
    YIO.enumCase(EN, "Constant", AddressSpaceQualifier::Constant);
    YIO.enumCase(EN, "Local", AddressSpaceQualifier::Local);
    YIO.enumCase(EN, "Generic", AddressSpaceQualifier::Generic);
    YIO.enumCase(EN, "Region", AddressSpaceQualifier::Region);
  }
};

template <>
struct ScalarEnumerationTraits<ValueKind> {
  static void enumeration(IO &YIO, ValueKind &EN) {
    YIO.enumCase(EN, "ByValue", ValueKind::ByValue);
    YIO.enumCase(EN, "GlobalBuffer", ValueKind::GlobalBuffer);
    YIO.enumCase(EN, "DynamicSharedPointer", ValueKind::DynamicSharedPointer);
    YIO.enumCase(EN, "Sampler", ValueKind::Sampler);
    YIO.enumCase(EN, "Image", ValueKind::Image);
    YIO.enumCase(EN, "Pipe", ValueKind::Pipe);
    YIO.enumCase(EN, "Queue", ValueKind::Queue);
    YIO.enumCase(EN, "HiddenGlobalOffsetX", ValueKind::HiddenGlobalOffsetX);
    YIO.enumCase(EN, "HiddenGlobalOffsetY", ValueKind::HiddenGlobalOffsetY);
    YIO.enumCase(EN, "HiddenGlobalOffsetZ", ValueKind::HiddenGlobalOffsetZ);
    YIO.enumCase(EN, "HiddenNone", ValueKind::HiddenNone);
    YIO.enumCase(EN, "HiddenPrintfBuffer", ValueKind::HiddenPrintfBuffer);
    YIO.enumCase(EN, "HiddenDefaultQueue", ValueKind::HiddenDefaultQueue);
    YIO.enumCase(EN, "HiddenCompletionAction",
                 ValueKind::HiddenCompletionAction);
  }
};

template <>
struct ScalarEnumerationTraits<ValueType> {
  static void enumeration(IO &YIO, ValueType &EN) {
    YIO.enumCase(EN, "Struct", ValueType::Struct);
    YIO.enumCase(EN, "I8", ValueType::I8);
    YIO.enumCase(EN, "U8", ValueType::U8);
    YIO.enumCase(EN, "I16", ValueType::I16);
    YIO.enumCase(EN, "U16", ValueType::U16);
    YIO.enumCase(EN, "F16", ValueType::F16);
    YIO.enumCase(EN, "I32", ValueType::I32);
    YIO.enumCase(EN, "U32", ValueType::U32);
    YIO.enumCase(EN, "F32", ValueType::F32);
    YIO.enumCase(EN, "I64", ValueType::I64);
    YIO.enumCase(EN, "U64", ValueType::U64);
    YIO.enumCase(EN, "F64", ValueType::F64);
  }
};

template <>
struct MappingTraits<Kernel::Attrs::Metadata> {
  static void mapping(IO &YIO, Kernel::Attrs::Metadata &MD) {
    YIO.mapOptional(Kernel::Attrs::Key::ReqdWorkGroupSize,
                    MD.mReqdWorkGroupSize, std::vector<uint32_t>());
    YIO.mapOptional(Kernel::Attrs::Key::WorkGroupSizeHint,
                    MD.mWorkGroupSizeHint, std::vector<uint32_t>());
    YIO.mapOptional(Kernel::Attrs::Key::VecTypeHint,
                    MD.mVecTypeHint, std::string());
  }
};

template <>
struct MappingTraits<Kernel::Arg::Metadata> {
  static void mapping(IO &YIO, Kernel::Arg::Metadata &MD) {
    YIO.mapRequired(Kernel::Arg::Key::Size, MD.mSize);
    YIO.mapRequired(Kernel::Arg::Key::Align, MD.mAlign);
    YIO.mapRequired(Kernel::Arg::Key::ValueKind, MD.mValueKind);
    YIO.mapRequired(Kernel::Arg::Key::ValueType, MD.mValueType);
    YIO.mapOptional(Kernel::Arg::Key::PointeeAlign, MD.mPointeeAlign,
                    uint32_t(0));
    YIO.mapOptional(Kernel::Arg::Key::AccQual, MD.mAccQual,
                    AccessQualifier::Unknown);
    YIO.mapOptional(Kernel::Arg::Key::AddrSpaceQual, MD.mAddrSpaceQual,
                    AddressSpaceQualifier::Unknown);
    YIO.mapOptional(Kernel::Arg::Key::IsConst, MD.mIsConst, false);
    YIO.mapOptional(Kernel::Arg::Key::IsPipe, MD.mIsPipe, false);
    YIO.mapOptional(Kernel::Arg::Key::IsRestrict, MD.mIsRestrict, false);
    YIO.mapOptional(Kernel::Arg::Key::IsVolatile, MD.mIsVolatile, false);
    YIO.mapOptional(Kernel::Arg::Key::Name, MD.mName, std::string());
    YIO.mapOptional(Kernel::Arg::Key::TypeName, MD.mTypeName, std::string());
  }
};

template <>
struct MappingTraits<Kernel::CodeProps::Metadata> {
  static void mapping(IO &YIO, Kernel::CodeProps::Metadata &MD) {
    YIO.mapOptional(Kernel::CodeProps::Key::KernargSegmentSize,
                    MD.mKernargSegmentSize, uint64_t(0));
    YIO.mapOptional(Kernel::CodeProps::Key::WorkgroupGroupSegmentSize,
                    MD.mWorkgroupGroupSegmentSize, uint32_t(0));
    YIO.mapOptional(Kernel::CodeProps::Key::WorkitemPrivateSegmentSize,
                    MD.mWorkitemPrivateSegmentSize, uint32_t(0));
    YIO.mapOptional(Kernel::CodeProps::Key::WavefrontNumSGPRs,
                    MD.mWavefrontNumSGPRs, uint16_t(0));
    YIO.mapOptional(Kernel::CodeProps::Key::WorkitemNumVGPRs,
                    MD.mWorkitemNumVGPRs, uint16_t(0));
    YIO.mapOptional(Kernel::CodeProps::Key::KernargSegmentAlign,
                    MD.mKernargSegmentAlign, uint8_t(0));
    YIO.mapOptional(Kernel::CodeProps::Key::GroupSegmentAlign,
                    MD.mGroupSegmentAlign, uint8_t(0));
    YIO.mapOptional(Kernel::CodeProps::Key::PrivateSegmentAlign,
                    MD.mPrivateSegmentAlign, uint8_t(0));
    YIO.mapOptional(Kernel::CodeProps::Key::WavefrontSize,
                    MD.mWavefrontSize, uint8_t(0));
  }
};

template <>
struct MappingTraits<Kernel::DebugProps::Metadata> {
  static void mapping(IO &YIO, Kernel::DebugProps::Metadata &MD) {
    YIO.mapOptional(Kernel::DebugProps::Key::DebuggerABIVersion,
                    MD.mDebuggerABIVersion, std::vector<uint32_t>());
    YIO.mapOptional(Kernel::DebugProps::Key::ReservedNumVGPRs,
                    MD.mReservedNumVGPRs, uint16_t(0));
    YIO.mapOptional(Kernel::DebugProps::Key::ReservedFirstVGPR,
                    MD.mReservedFirstVGPR, uint16_t(-1));
    YIO.mapOptional(Kernel::DebugProps::Key::PrivateSegmentBufferSGPR,
                    MD.mPrivateSegmentBufferSGPR, uint16_t(-1));
    YIO.mapOptional(Kernel::DebugProps::Key::WavefrontPrivateSegmentOffsetSGPR,
                    MD.mWavefrontPrivateSegmentOffsetSGPR, uint16_t(-1));
  }
};

template <>
struct MappingTraits<Kernel::Metadata> {
  static void mapping(IO &YIO, Kernel::Metadata &MD) {
    YIO.mapRequired(Kernel::Key::Name, MD.mName);
    YIO.mapOptional(Kernel::Key::Language, MD.mLanguage, std::string());
    YIO.mapOptional(Kernel::Key::LanguageVersion, MD.mLanguageVersion,
                    std::vector<uint32_t>());
    if (!MD.mAttrs.empty() || !YIO.outputting())
      YIO.mapOptional(Kernel::Key::Attrs, MD.mAttrs);
    if (!MD.mArgs.empty() || !YIO.outputting())
      YIO.mapOptional(Kernel::Key::Args, MD.mArgs);
    if (!MD.mCodeProps.empty() || !YIO.outputting())
      YIO.mapOptional(Kernel::Key::CodeProps, MD.mCodeProps);
    if (!MD.mDebugProps.empty() || !YIO.outputting())
      YIO.mapOptional(Kernel::Key::DebugProps, MD.mDebugProps);
  }
};

template <>
struct MappingTraits<CodeObject::Metadata> {
  static void mapping(IO &YIO, CodeObject::Metadata &MD) {
    YIO.mapRequired(Key::Version, MD.mVersion);
    YIO.mapOptional(Key::Printf, MD.mPrintf, std::vector<std::string>());
    if (!MD.mKernels.empty() || !YIO.outputting())
      YIO.mapOptional(Key::Kernels, MD.mKernels);
  }
};

} // end namespace yaml

namespace AMDGPU {
namespace CodeObject {

/* static */
std::error_code Metadata::fromYamlString(
    std::string YamlString, Metadata &CodeObjectMetadata) {
  yaml::Input YamlInput(YamlString);
  YamlInput >> CodeObjectMetadata;
  return YamlInput.error();
}

/* static */
std::error_code Metadata::toYamlString(
    Metadata CodeObjectMetadata, std::string &YamlString) {
  raw_string_ostream YamlStream(YamlString);
  yaml::Output YamlOutput(YamlStream, nullptr, std::numeric_limits<int>::max());
  YamlOutput << CodeObjectMetadata;
  return std::error_code();
}

} // end namespace CodeObject
} // end namespace AMDGPU
} // end namespace llvm
