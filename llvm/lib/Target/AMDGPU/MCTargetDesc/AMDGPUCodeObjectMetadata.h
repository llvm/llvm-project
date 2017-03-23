//===--- AMDGPUCodeObjectMetadata.h -----------------------------*- C++ -*-===//
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

#ifndef LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUCODEOBJECTMETADATA_H
#define LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUCODEOBJECTMETADATA_H

#include <cstdint>
#include <string>
#include <system_error>
#include <vector>

namespace llvm {
namespace AMDGPU {

//===----------------------------------------------------------------------===//
// Code Object Metadata.
//===----------------------------------------------------------------------===//
namespace CodeObject {

/// \brief Code object metadata major version.
constexpr uint32_t MetadataVersionMajor = 1;
/// \brief Code object metadata minor version.
constexpr uint32_t MetadataVersionMinor = 0;

/// \brief Code object metadata beginning assembler directive.
constexpr char MetadataAssemblerDirectiveBegin[] =
    ".amdgpu_code_object_metadata";
/// \brief Code object metadata ending assembler directive.
constexpr char MetadataAssemblerDirectiveEnd[] =
    ".end_amdgpu_code_object_metadata";

/// \brief Access qualifiers.
enum class AccessQualifier : uint8_t {
  Default   = 0,
  ReadOnly  = 1,
  WriteOnly = 2,
  ReadWrite = 3,
  Unknown   = 0xff
};

/// \brief Address space qualifiers.
enum class AddressSpaceQualifier : uint8_t {
  Private  = 0,
  Global   = 1,
  Constant = 2,
  Local    = 3,
  Generic  = 4,
  Region   = 5,
  Unknown  = 0xff
};

/// \brief Value kinds.
enum class ValueKind : uint8_t {
  ByValue                = 0,
  GlobalBuffer           = 1,
  DynamicSharedPointer   = 2,
  Sampler                = 3,
  Image                  = 4,
  Pipe                   = 5,
  Queue                  = 6,
  HiddenGlobalOffsetX    = 7,
  HiddenGlobalOffsetY    = 8,
  HiddenGlobalOffsetZ    = 9,
  HiddenNone             = 10,
  HiddenPrintfBuffer     = 11,
  HiddenDefaultQueue     = 12,
  HiddenCompletionAction = 13,
  Unknown                = 0xff
};

/// \brief Value types.
enum class ValueType : uint8_t {
  Struct  = 0,
  I8      = 1,
  U8      = 2,
  I16     = 3,
  U16     = 4,
  F16     = 5,
  I32     = 6,
  U32     = 7,
  F32     = 8,
  I64     = 9,
  U64     = 10,
  F64     = 11,
  Unknown = 0xff
};

//===----------------------------------------------------------------------===//
// Instruction Set Architecture Metadata (ISA).
//===----------------------------------------------------------------------===//
namespace Isa {

namespace Key {
/// \brief Key for Isa::Metadata::mWavefrontSize.
constexpr char WavefrontSize[] = "WavefrontSize";
/// \brief Key for Isa::Metadata::mLocalMemorySize.
constexpr char LocalMemorySize[] = "LocalMemorySize";
/// \brief Key for Isa::Metadata::mEUsPerCU.
constexpr char EUsPerCU[] = "EUsPerCU";
/// \brief Key for Isa::Metadata::mMaxWavesPerEU.
constexpr char MaxWavesPerEU[] = "MaxWavesPerEU";
/// \brief Key for Isa::Metadata::mMaxFlatWorkGroupSize.
constexpr char MaxFlatWorkGroupSize[] = "MaxFlatWorkGroupSize";
/// \brief Key for Isa::Metadata::mSGPRAllocGranule.
constexpr char SGPRAllocGranule[] = "SGPRAllocGranule";
/// \brief Key for Isa::Metadata::mTotalNumSGPRs.
constexpr char TotalNumSGPRs[] = "TotalNumSGPRs";
/// \brief Key for Isa::Metadata::mAddressableNumSGPRs.
constexpr char AddressableNumSGPRs[] = "AddressableNumSGPRs";
/// \brief Key for Isa::Metadata::mVGPRAllocGranule.
constexpr char VGPRAllocGranule[] = "VGPRAllocGranule";
/// \brief Key for Isa::Metadata::mTotalNumVGPRs.
constexpr char TotalNumVGPRs[] = "TotalNumVGPRs";
/// \brief Key for Isa::Metadata::mAddressableNumVGPRs.
constexpr char AddressableNumVGPRs[] = "AddressableNumVGPRs";
} // end namespace Key

/// \brief In-memory representation of instruction set architecture metadata.
struct Metadata final {
  /// \brief Wavefront size. Required.
  uint32_t mWavefrontSize = 0;
  /// \brief Local memory size in bytes. Required.
  uint32_t mLocalMemorySize = 0;
  /// \brief Number of execution units per compute unit. Required.
  uint32_t mEUsPerCU = 0;
  /// \brief Maximum number of waves per execution unit. Required.
  uint32_t mMaxWavesPerEU = 0;
  /// \brief Maximum flat work group size. Required.
  uint32_t mMaxFlatWorkGroupSize = 0;
  /// \brief SGPR allocation granularity. Required.
  uint32_t mSGPRAllocGranule = 0;
  /// \brief Total number of SGPRs. Required.
  uint32_t mTotalNumSGPRs = 0;
  /// \brief Addressable number of SGPRs. Required.
  uint32_t mAddressableNumSGPRs = 0;
  /// \brief VGPR allocation granularity. Required.
  uint32_t mVGPRAllocGranule = 0;
  /// \brief Total number of VGPRs. Required.
  uint32_t mTotalNumVGPRs = 0;
  /// \brief Addressable number of VGPRs. Required.
  uint32_t mAddressableNumVGPRs = 0;

  /// \brief Default constructor.
  Metadata() = default;
};

} // end namespace Isa

//===----------------------------------------------------------------------===//
// Kernel Metadata.
//===----------------------------------------------------------------------===//
namespace Kernel {

//===----------------------------------------------------------------------===//
// Kernel Attributes Metadata.
//===----------------------------------------------------------------------===//
namespace Attrs {

namespace Key {
/// \brief Key for Kernel::Attr::Metadata::mReqdWorkGroupSize.
constexpr char ReqdWorkGroupSize[] = "ReqdWorkGroupSize";
/// \brief Key for Kernel::Attr::Metadata::mWorkGroupSizeHint.
constexpr char WorkGroupSizeHint[] = "WorkGroupSizeHint";
/// \brief Key for Kernel::Attr::Metadata::mVecTypeHint.
constexpr char VecTypeHint[] = "VecTypeHint";
} // end namespace Key

/// \brief In-memory representation of kernel attributes metadata.
struct Metadata final {
  /// \brief 'reqd_work_group_size' attribute. Optional.
  std::vector<uint32_t> mReqdWorkGroupSize = std::vector<uint32_t>();
  /// \brief 'work_group_size_hint' attribute. Optional.
  std::vector<uint32_t> mWorkGroupSizeHint = std::vector<uint32_t>();
  /// \brief 'vec_type_hint' attribute. Optional.
  std::string mVecTypeHint = std::string();

  /// \brief Default constructor.
  Metadata() = default;

  /// \returns True if kernel attributes metadata is empty, false otherwise.
  bool empty() const {
    return mReqdWorkGroupSize.empty() &&
           mWorkGroupSizeHint.empty() &&
           mVecTypeHint.empty();
  }

  /// \returns True if kernel attributes metadata is not empty, false otherwise.
  bool notEmpty() const {
    return !empty();
  }
};

} // end namespace Attrs

//===----------------------------------------------------------------------===//
// Kernel Argument Metadata.
//===----------------------------------------------------------------------===//
namespace Arg {

namespace Key {
/// \brief Key for Kernel::Arg::Metadata::mSize.
constexpr char Size[] = "Size";
/// \brief Key for Kernel::Arg::Metadata::mAlign.
constexpr char Align[] = "Align";
/// \brief Key for Kernel::Arg::Metadata::mValueKind.
constexpr char Kind[] = "Kind";
/// \brief Key for Kernel::Arg::Metadata::mValueType.
constexpr char ValueType[] = "ValueType";
/// \brief Key for Kernel::Arg::Metadata::mPointeeAlign.
constexpr char PointeeAlign[] = "PointeeAlign";
/// \brief Key for Kernel::Arg::Metadata::mAccQual.
constexpr char AccQual[] = "AccQual";
/// \brief Key for Kernel::Arg::Metadata::mAddrSpaceQual.
constexpr char AddrSpaceQual[] = "AddrSpaceQual";
/// \brief Key for Kernel::Arg::Metadata::mIsConst.
constexpr char IsConst[] = "IsConst";
/// \brief Key for Kernel::Arg::Metadata::mIsPipe.
constexpr char IsPipe[] = "IsPipe";
/// \brief Key for Kernel::Arg::Metadata::mIsRestrict.
constexpr char IsRestrict[] = "IsRestrict";
/// \brief Key for Kernel::Arg::Metadata::mIsVolatile.
constexpr char IsVolatile[] = "IsVolatile";
/// \brief Key for Kernel::Arg::Metadata::mName.
constexpr char Name[] = "Name";
/// \brief Key for Kernel::Arg::Metadata::mTypeName.
constexpr char TypeName[] = "TypeName";
} // end namespace Key

/// \brief In-memory representation of kernel argument metadata.
struct Metadata final {
  /// \brief Size in bytes. Required.
  uint32_t mSize = 0;
  /// \brief Alignment in bytes. Required.
  uint32_t mAlign = 0;
  /// \brief Value kind. Required.
  ValueKind mValueKind = ValueKind::Unknown;
  /// \brief Value type. Required.
  ValueType mValueType = ValueType::Unknown;
  /// \brief Pointee alignment in bytes. Optional.
  uint32_t mPointeeAlign = 0;
  /// \brief Access qualifier. Optional.
  AccessQualifier mAccQual = AccessQualifier::Unknown;
  /// \brief Address space qualifier. Optional.
  AddressSpaceQualifier mAddrSpaceQual = AddressSpaceQualifier::Unknown;
  /// \brief True if 'const' qualifier is specified. Optional.
  bool mIsConst = false;
  /// \brief True if 'pipe' qualifier is specified. Optional.
  bool mIsPipe = false;
  /// \brief True if 'restrict' qualifier is specified. Optional.
  bool mIsRestrict = false;
  /// \brief True if 'volatile' qualifier is specified. Optional.
  bool mIsVolatile = false;
  /// \brief Name. Optional.
  std::string mName = std::string();
  /// \brief Type name. Optional.
  std::string mTypeName = std::string();

  /// \brief Default constructor.
  Metadata() = default;
};

} // end namespace Arg

//===----------------------------------------------------------------------===//
// Kernel Code Properties Metadata.
//===----------------------------------------------------------------------===//
namespace CodeProps {

namespace Key {
/// \brief Key for Kernel::CodeProps::mKernargSegmentSize.
constexpr char KernargSegmentSize[] = "KernargSegmentSize";
/// \brief Key for Kernel::CodeProps::mWorkgroupGroupSegmentSize.
constexpr char WorkgroupGroupSegmentSize[] = "WorkgroupGroupSegmentSize";
/// \brief Key for Kernel::CodeProps::mWorkitemPrivateSegmentSize.
constexpr char WorkitemPrivateSegmentSize[] = "WorkitemPrivateSegmentSize";
/// \brief Key for Kernel::CodeProps::mWavefrontNumSGPRs.
constexpr char WavefrontNumSGPRs[] = "WavefrontNumSGPRs";
/// \brief Key for Kernel::CodeProps::mWorkitemNumVGPRs.
constexpr char WorkitemNumVGPRs[] = "WorkitemNumVGPRs";
/// \brief Key for Kernel::CodeProps::mKernargSegmentAlign.
constexpr char KernargSegmentAlign[] = "KernargSegmentAlign";
/// \brief Key for Kernel::CodeProps::mGroupSegmentAlign.
constexpr char GroupSegmentAlign[] = "GroupSegmentAlign";
/// \brief Key for Kernel::CodeProps::mPrivateSegmentAlign.
constexpr char PrivateSegmentAlign[] = "PrivateSegmentAlign";
/// \brief Key for Kernel::CodeProps::mWavefrontSize.
constexpr char WavefrontSize[] = "WavefrontSize";
} // end namespace Key

/// \brief In-memory representation of kernel code properties metadata.
struct Metadata final {
  /// \brief Size in bytes of the kernarg segment memory. Kernarg segment memory
  /// holds the values of the arguments to the kernel. Optional.
  uint64_t mKernargSegmentSize = 0;
  /// \brief Size in bytes of the group segment memory required by a workgroup.
  /// This value does not include any dynamically allocated group segment memory
  /// that may be added when the kernel is dispatched. Optional.
  uint32_t mWorkgroupGroupSegmentSize = 0;
  /// \brief Size in bytes of the private segment memory required by a workitem.
  /// Private segment memory includes arg, spill and private segments. Optional.
  uint32_t mWorkitemPrivateSegmentSize = 0;
  /// \brief Total number of SGPRs used by a wavefront. Optional.
  uint16_t mWavefrontNumSGPRs = 0;
  /// \brief Total number of VGPRs used by a workitem. Optional.
  uint16_t mWorkitemNumVGPRs = 0;
  /// \brief Maximum byte alignment of variables used by the kernel in the
  /// kernarg memory segment. Expressed as a power of two. Optional.
  uint8_t mKernargSegmentAlign = 0;
  /// \brief Maximum byte alignment of variables used by the kernel in the
  /// group memory segment. Expressed as a power of two. Optional.
  uint8_t mGroupSegmentAlign = 0;
  /// \brief Maximum byte alignment of variables used by the kernel in the
  /// private memory segment. Expressed as a power of two. Optional.
  uint8_t mPrivateSegmentAlign = 0;
  /// \brief Wavefront size. Expressed as a power of two. Optional.
  uint8_t mWavefrontSize = 0;

  /// \brief Default constructor.
  Metadata() = default;

  /// \returns True if kernel code properties metadata is empty, false
  /// otherwise.
  bool empty() const {
    return !notEmpty();
  }

  /// \returns True if kernel code properties metadata is not empty, false
  /// otherwise.
  bool notEmpty() const {
    return mKernargSegmentSize || mWorkgroupGroupSegmentSize ||
           mWorkitemPrivateSegmentSize || mWavefrontNumSGPRs ||
           mWorkitemNumVGPRs || mKernargSegmentAlign || mGroupSegmentAlign ||
           mPrivateSegmentAlign || mWavefrontSize;
  }
};

} // end namespace CodeProps

namespace Key {
/// \brief Key for Kernel::Metadata::mName.
constexpr char Name[] = "Name";
/// \brief Key for Kernel::Metadata::mLanguage.
constexpr char Language[] = "Language";
/// \brief Key for Kernel::Metadata::mLanguageVersion.
constexpr char LanguageVersion[] = "LanguageVersion";
/// \brief Key for Kernel::Metadata::mAttrs.
constexpr char Attrs[] = "Attrs";
/// \brief Key for Kernel::Metadata::mArgs.
constexpr char Args[] = "Args";
/// \brief Key for Kernel::Metadata::mCodeProps.
constexpr char CodeProps[] = "CodeProps";
} // end namespace Key

/// \brief In-memory representation of kernel metadata.
struct Metadata final {
  /// \brief Name. Required.
  std::string mName = std::string();
  /// \brief Language. Optional.
  std::string mLanguage = std::string();
  /// \brief Language version. Optional.
  std::vector<uint32_t> mLanguageVersion = std::vector<uint32_t>();
  /// \brief Attributes metadata. Optional.
  Attrs::Metadata mAttrs = Attrs::Metadata();
  /// \brief Arguments metadata. Optional.
  std::vector<Arg::Metadata> mArgs = std::vector<Arg::Metadata>();
  /// \brief Code properties metadata. Optional.
  CodeProps::Metadata mCodeProps = CodeProps::Metadata();

  /// \brief Default constructor.
  Metadata() = default;
};

} // end namespace Kernel

namespace Key {
/// \brief Key for CodeObject::Metadata::mVersion.
constexpr char Version[] = "Version";
/// \brief Key for CodeObject::Metadata::mIsa.
constexpr char Isa[] = "Isa";
/// \brief Key for CodeObject::Metadata::mPrintf.
constexpr char Printf[] = "Printf";
/// \brief Key for CodeObject::Metadata::mKernels.
constexpr char Kernels[] = "Kernels";
} // end namespace Key

/// \brief In-memory representation of code object metadata.
struct Metadata final {
  /// \brief Code object metadata version. Required.
  std::vector<uint32_t> mVersion = std::vector<uint32_t>();
  /// \brief Instruction set architecture metadata. Optional.
  Isa::Metadata mIsa = Isa::Metadata();
  /// \brief Printf metadata. Optional.
  std::vector<std::string> mPrintf = std::vector<std::string>();
  /// \brief Kernels metadata. Optional.
  std::vector<Kernel::Metadata> mKernels = std::vector<Kernel::Metadata>();

  /// \brief Default constructor.
  Metadata() = default;

  /// \brief Converts \p YamlString to \p CodeObjectMetadata.
  static std::error_code fromYamlString(std::string YamlString,
                                        Metadata &CodeObjectMetadata);

  /// \brief Converts \p CodeObjectMetadata to \p YamlString.
  static std::error_code toYamlString(Metadata CodeObjectMetadata,
                                      std::string &YamlString);
};

} // end namespace CodeObject
} // end namespace AMDGPU
} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUCODEOBJECTMETADATA_H
