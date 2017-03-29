//===--- AMDGPUCodeObjectMetadataStreamer.cpp -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief AMDGPU Code Object Metadata Streamer.
///
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUCodeObjectMetadataStreamer.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/YAMLTraits.h"

using namespace llvm::AMDGPU;
using namespace llvm::AMDGPU::CodeObject;

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(uint32_t)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(std::string)
LLVM_YAML_IS_SEQUENCE_VECTOR(Kernel::Arg::Metadata)
LLVM_YAML_IS_SEQUENCE_VECTOR(Kernel::Metadata)

namespace llvm {

static cl::opt<bool> DumpCodeObjectMetadata(
    "amdgpu-dump-comd",
    cl::desc("Dump AMDGPU Code Object Metadata"));
static cl::opt<bool> VerifyCodeObjectMetadata(
    "amdgpu-verify-comd",
    cl::desc("Verify AMDGPU Code Object Metadata"));

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

/* static */
std::error_code CodeObject::Metadata::fromYamlString(
    std::string YamlString, CodeObject::Metadata &CodeObjectMetadata) {
  yaml::Input YamlInput(YamlString);
  YamlInput >> CodeObjectMetadata;
  return YamlInput.error();
}

/* static */
std::error_code CodeObject::Metadata::toYamlString(
    CodeObject::Metadata CodeObjectMetadata, std::string &YamlString) {
  raw_string_ostream YamlStream(YamlString);
  yaml::Output YamlOutput(YamlStream, nullptr, std::numeric_limits<int>::max());
  YamlOutput << CodeObjectMetadata;
  return std::error_code();
}

namespace CodeObject {

void MetadataStreamer::dump(StringRef YamlString) const {
  errs() << "AMDGPU Code Object Metadata:\n" << YamlString << '\n';
}

void MetadataStreamer::verify(StringRef YamlString) const {
  errs() << "AMDGPU Code Object Metadata Parser Test: ";

  CodeObject::Metadata FromYamlString;
  if (Metadata::fromYamlString(YamlString, FromYamlString)) {
    errs() << "FAIL\n";
    return;
  }

  std::string ToYamlString;
  if (Metadata::toYamlString(FromYamlString, ToYamlString)) {
    errs() << "FAIL\n";
    return;
  }

  errs() << (YamlString == ToYamlString ? "PASS" : "FAIL") << '\n';
  if (YamlString != ToYamlString) {
    errs() << "Original input: " << YamlString << '\n'
           << "Produced output: " << ToYamlString << '\n';
  }
}

AccessQualifier MetadataStreamer::getAccessQualifier(StringRef AccQual) const {
  if (AccQual.empty())
    return AccessQualifier::Unknown;

  return StringSwitch<AccessQualifier>(AccQual)
             .Case("read_only",  AccessQualifier::ReadOnly)
             .Case("write_only", AccessQualifier::WriteOnly)
             .Case("read_write", AccessQualifier::ReadWrite)
             .Default(AccessQualifier::Default);
}

AddressSpaceQualifier MetadataStreamer::getAddressSpaceQualifer(
    unsigned AddressSpace) const {
  if (AddressSpace == AMDGPUASI.PRIVATE_ADDRESS)
    return AddressSpaceQualifier::Private;
  if (AddressSpace == AMDGPUASI.GLOBAL_ADDRESS)
    return AddressSpaceQualifier::Global;
  if (AddressSpace == AMDGPUASI.CONSTANT_ADDRESS)
    return AddressSpaceQualifier::Constant;
  if (AddressSpace == AMDGPUASI.LOCAL_ADDRESS)
    return AddressSpaceQualifier::Local;
  if (AddressSpace == AMDGPUASI.FLAT_ADDRESS)
    return AddressSpaceQualifier::Generic;
  if (AddressSpace == AMDGPUASI.REGION_ADDRESS)
    return AddressSpaceQualifier::Region;

  llvm_unreachable("Unknown address space qualifier");
}

ValueKind MetadataStreamer::getValueKind(Type *Ty, StringRef TypeQual,
                                         StringRef BaseTypeName) const {
  if (TypeQual.find("pipe") != StringRef::npos)
    return ValueKind::Pipe;

  return StringSwitch<ValueKind>(BaseTypeName)
             .Case("sampler_t", ValueKind::Sampler)
             .Case("queue_t", ValueKind::Queue)
             .Cases("image1d_t",
                    "image1d_array_t",
                    "image1d_buffer_t",
                    "image2d_t" ,
                    "image2d_array_t",
                    "image2d_array_depth_t",
                    "image2d_array_msaa_t"
                    "image2d_array_msaa_depth_t"
                    "image2d_depth_t",
                    "image2d_msaa_t",
                    "image2d_msaa_depth_t",
                    "image3d_t", ValueKind::Image)
             .Default(isa<PointerType>(Ty) ?
                          (Ty->getPointerAddressSpace() ==
                           AMDGPUASI.LOCAL_ADDRESS ?
                           ValueKind::DynamicSharedPointer :
                           ValueKind::GlobalBuffer) :
                      ValueKind::ByValue);
}

ValueType MetadataStreamer::getValueType(Type *Ty, StringRef TypeName) const {
  switch (Ty->getTypeID()) {
  case Type::IntegerTyID: {
    auto Signed = !TypeName.startswith("u");
    switch (Ty->getIntegerBitWidth()) {
    case 8:
      return Signed ? ValueType::I8 : ValueType::U8;
    case 16:
      return Signed ? ValueType::I16 : ValueType::U16;
    case 32:
      return Signed ? ValueType::I32 : ValueType::U32;
    case 64:
      return Signed ? ValueType::I64 : ValueType::U64;
    default:
      return ValueType::Struct;
    }
  }
  case Type::HalfTyID:
    return ValueType::F16;
  case Type::FloatTyID:
    return ValueType::F32;
  case Type::DoubleTyID:
    return ValueType::F64;
  case Type::PointerTyID:
    return getValueType(Ty->getPointerElementType(), TypeName);
  case Type::VectorTyID:
    return getValueType(Ty->getVectorElementType(), TypeName);
  default:
    return ValueType::Struct;
  }
}

std::string MetadataStreamer::getTypeName(Type *Ty, bool Signed) const {
  switch (Ty->getTypeID()) {
  case Type::IntegerTyID: {
    if (!Signed)
      return (Twine('u') + getTypeName(Ty, true)).str();

    auto BitWidth = Ty->getIntegerBitWidth();
    switch (BitWidth) {
    case 8:
      return "char";
    case 16:
      return "short";
    case 32:
      return "int";
    case 64:
      return "long";
    default:
      return (Twine('i') + Twine(BitWidth)).str();
    }
  }
  case Type::HalfTyID:
    return "half";
  case Type::FloatTyID:
    return "float";
  case Type::DoubleTyID:
    return "double";
  case Type::VectorTyID: {
    auto VecTy = cast<VectorType>(Ty);
    auto ElTy = VecTy->getElementType();
    auto NumElements = VecTy->getVectorNumElements();
    return (Twine(getTypeName(ElTy, Signed)) + Twine(NumElements)).str();
  }
  default:
    return "unknown";
  }
}

std::vector<uint32_t> MetadataStreamer::getWorkGroupDimensions(
    MDNode *Node) const {
  std::vector<uint32_t> Dims;
  if (Node->getNumOperands() != 3)
    return Dims;

  for (auto &Op : Node->operands())
    Dims.push_back(mdconst::extract<ConstantInt>(Op)->getZExtValue());
  return Dims;
}

void MetadataStreamer::emitVersion() {
  auto &Version = CodeObjectMetadata.mVersion;

  Version.push_back(MetadataVersionMajor);
  Version.push_back(MetadataVersionMinor);
}

void MetadataStreamer::emitPrintf(const Module &Mod) {
  auto &Printf = CodeObjectMetadata.mPrintf;

  auto Node = Mod.getNamedMetadata("llvm.printf.fmts");
  if (!Node)
    return;

  for (auto Op : Node->operands())
    if (Op->getNumOperands())
      Printf.push_back(cast<MDString>(Op->getOperand(0))->getString());
}

void MetadataStreamer::emitKernelLanguage(const Function &Func) {
  auto &Kernel = CodeObjectMetadata.mKernels.back();

  // TODO: What about other languages?
  auto Node = Func.getParent()->getNamedMetadata("opencl.ocl.version");
  if (!Node || !Node->getNumOperands())
    return;
  auto Op0 = Node->getOperand(0);
  if (Op0->getNumOperands() <= 1)
    return;

  Kernel.mLanguage = "OpenCL C";
  Kernel.mLanguageVersion.push_back(
      mdconst::extract<ConstantInt>(Op0->getOperand(0))->getZExtValue());
  Kernel.mLanguageVersion.push_back(
      mdconst::extract<ConstantInt>(Op0->getOperand(1))->getZExtValue());
}

void MetadataStreamer::emitKernelAttrs(const Function &Func) {
  auto &Attrs = CodeObjectMetadata.mKernels.back().mAttrs;

  if (auto Node = Func.getMetadata("reqd_work_group_size"))
    Attrs.mReqdWorkGroupSize = getWorkGroupDimensions(Node);
  if (auto Node = Func.getMetadata("work_group_size_hint"))
    Attrs.mWorkGroupSizeHint = getWorkGroupDimensions(Node);
  if (auto Node = Func.getMetadata("vec_type_hint")) {
    Attrs.mVecTypeHint = getTypeName(
        cast<ValueAsMetadata>(Node->getOperand(0))->getType(),
        mdconst::extract<ConstantInt>(Node->getOperand(1))->getZExtValue());
  }
}

void MetadataStreamer::emitKernelArgs(const Function &Func) {
  for (auto &Arg : Func.args())
    emitKernelArg(Arg);

  // TODO: What about other languages?
  if (!Func.getParent()->getNamedMetadata("opencl.ocl.version"))
    return;

  auto &DL = Func.getParent()->getDataLayout();
  auto Int64Ty = Type::getInt64Ty(Func.getContext());

  emitKernelArg(DL, Int64Ty, ValueKind::HiddenGlobalOffsetX);
  emitKernelArg(DL, Int64Ty, ValueKind::HiddenGlobalOffsetY);
  emitKernelArg(DL, Int64Ty, ValueKind::HiddenGlobalOffsetZ);

  if (!Func.getParent()->getNamedMetadata("llvm.printf.fmts"))
    return;

  auto Int8PtrTy = Type::getInt8PtrTy(Func.getContext(),
                                      AMDGPUASI.GLOBAL_ADDRESS);
  emitKernelArg(DL, Int8PtrTy, ValueKind::HiddenPrintfBuffer);
}

void MetadataStreamer::emitKernelArg(const Argument &Arg) {
  auto Func = Arg.getParent();
  auto ArgNo = Arg.getArgNo();
  const MDNode *Node;

  StringRef TypeQual;
  Node = Func->getMetadata("kernel_arg_type_qual");
  if (Node && ArgNo < Node->getNumOperands())
    TypeQual = cast<MDString>(Node->getOperand(ArgNo))->getString();

  StringRef BaseTypeName;
  Node = Func->getMetadata("kernel_arg_base_type");
  if (Node && ArgNo < Node->getNumOperands())
    BaseTypeName = cast<MDString>(Node->getOperand(ArgNo))->getString();

  StringRef AccQual;
  Node = Func->getMetadata("kernel_arg_access_qual");
  if (Node && ArgNo < Node->getNumOperands())
    AccQual = cast<MDString>(Node->getOperand(ArgNo))->getString();

  StringRef Name;
  Node = Func->getMetadata("kernel_arg_name");
  if (Node && ArgNo < Node->getNumOperands())
    Name = cast<MDString>(Node->getOperand(ArgNo))->getString();

  StringRef TypeName;
  Node = Func->getMetadata("kernel_arg_type");
  if (Node && ArgNo < Node->getNumOperands())
    TypeName = cast<MDString>(Node->getOperand(ArgNo))->getString();

  emitKernelArg(Func->getParent()->getDataLayout(), Arg.getType(),
                getValueKind(Arg.getType(), TypeQual, BaseTypeName), TypeQual,
                BaseTypeName, AccQual, Name, TypeName);
}

void MetadataStreamer::emitKernelArg(const DataLayout &DL, Type *Ty,
                                     ValueKind ValueKind, StringRef TypeQual,
                                     StringRef BaseTypeName, StringRef AccQual,
                                     StringRef Name, StringRef TypeName) {
  CodeObjectMetadata.mKernels.back().mArgs.push_back(Kernel::Arg::Metadata());
  auto &Arg = CodeObjectMetadata.mKernels.back().mArgs.back();

  Arg.mSize = DL.getTypeAllocSize(Ty);
  Arg.mAlign = DL.getABITypeAlignment(Ty);
  Arg.mValueKind = ValueKind;
  Arg.mValueType = getValueType(Ty, BaseTypeName);

  if (auto PtrTy = dyn_cast<PointerType>(Ty)) {
    auto ElTy = PtrTy->getElementType();
    if (PtrTy->getAddressSpace() == AMDGPUASI.LOCAL_ADDRESS && ElTy->isSized())
      Arg.mPointeeAlign = DL.getABITypeAlignment(ElTy);
  }

  Arg.mAccQual = getAccessQualifier(AccQual);

  if (auto PtrTy = dyn_cast<PointerType>(Ty))
    Arg.mAddrSpaceQual = getAddressSpaceQualifer(PtrTy->getAddressSpace());

  SmallVector<StringRef, 1> SplitTypeQuals;
  TypeQual.split(SplitTypeQuals, " ", -1, false);
  for (StringRef Key : SplitTypeQuals) {
    auto P = StringSwitch<bool*>(Key)
                 .Case("const",    &Arg.mIsConst)
                 .Case("pipe",     &Arg.mIsPipe)
                 .Case("restrict", &Arg.mIsRestrict)
                 .Case("volatile", &Arg.mIsVolatile)
                 .Default(nullptr);
    if (P)
      *P = true;
  }

  Arg.mName = Name;
  Arg.mTypeName = TypeName;
}

void MetadataStreamer::emitKernelCodeProps(
    const amd_kernel_code_t &KernelCode) {
  auto &CodeProps = CodeObjectMetadata.mKernels.back().mCodeProps;

  CodeProps.mKernargSegmentSize = KernelCode.kernarg_segment_byte_size;
  CodeProps.mWorkgroupGroupSegmentSize =
      KernelCode.workgroup_group_segment_byte_size;
  CodeProps.mWorkitemPrivateSegmentSize =
      KernelCode.workitem_private_segment_byte_size;
  CodeProps.mWavefrontNumSGPRs = KernelCode.wavefront_sgpr_count;
  CodeProps.mWorkitemNumVGPRs = KernelCode.workitem_vgpr_count;
  CodeProps.mKernargSegmentAlign = KernelCode.kernarg_segment_alignment;
  CodeProps.mGroupSegmentAlign = KernelCode.group_segment_alignment;
  CodeProps.mPrivateSegmentAlign = KernelCode.private_segment_alignment;
  CodeProps.mWavefrontSize = KernelCode.wavefront_size;
}

void MetadataStreamer::emitKernelDebugProps(
    const amd_kernel_code_t &KernelCode) {
  if (!(KernelCode.code_properties & AMD_CODE_PROPERTY_IS_DEBUG_SUPPORTED))
    return;

  auto &DebugProps = CodeObjectMetadata.mKernels.back().mDebugProps;

  // FIXME: Need to pass down debugger ABI version through features. This is ok
  // for now because we only have one version.
  DebugProps.mDebuggerABIVersion.push_back(1);
  DebugProps.mDebuggerABIVersion.push_back(0);
  DebugProps.mReservedNumVGPRs = KernelCode.reserved_vgpr_count;
  DebugProps.mReservedFirstVGPR = KernelCode.reserved_vgpr_first;
  DebugProps.mPrivateSegmentBufferSGPR =
      KernelCode.debug_private_segment_buffer_sgpr;
  DebugProps.mWavefrontPrivateSegmentOffsetSGPR =
      KernelCode.debug_wavefront_private_segment_offset_sgpr;
}

void MetadataStreamer::begin(const Module &Mod) {
  AMDGPUASI = getAMDGPUAS(Mod);
  emitVersion();
  emitPrintf(Mod);
}

void MetadataStreamer::emitKernel(const Function &Func,
                                  const amd_kernel_code_t &KernelCode) {
  if (Func.getCallingConv() != CallingConv::AMDGPU_KERNEL)
    return;

  CodeObjectMetadata.mKernels.push_back(Kernel::Metadata());
  auto &Kernel = CodeObjectMetadata.mKernels.back();

  Kernel.mName = Func.getName();
  emitKernelLanguage(Func);
  emitKernelAttrs(Func);
  emitKernelArgs(Func);
  emitKernelCodeProps(KernelCode);
  emitKernelDebugProps(KernelCode);
}

ErrorOr<std::string> MetadataStreamer::toYamlString() {
  std::string YamlString;
  if (auto Error = Metadata::toYamlString(CodeObjectMetadata, YamlString))
    return Error;

  if (DumpCodeObjectMetadata)
    dump(YamlString);
  if (VerifyCodeObjectMetadata)
    verify(YamlString);

  return YamlString;
}

ErrorOr<std::string> MetadataStreamer::toYamlString(StringRef YamlString) {
  if (auto Error = Metadata::fromYamlString(YamlString, CodeObjectMetadata))
    return Error;

  return toYamlString();
}

} // end namespace CodeObject
} // end namespace AMDGPU
} // end namespace llvm
