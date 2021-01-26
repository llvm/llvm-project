//===--- AMDGPUHSAMetadataStreamer.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// AMDGPU HSA Metadata Streamer.
///
//
//===----------------------------------------------------------------------===//

#include "AMDGPUHSAMetadataStreamer.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUTargetStreamer.h"
#include "SIMachineFunctionInfo.h"
#include "SIProgramInfo.h"
#include "llvm/IR/Module.h"
using namespace llvm;

static std::pair<Type *, Align> getArgumentTypeAlign(const Argument &Arg,
                                                     const DataLayout &DL) {
  Type *Ty = Arg.getType();
  MaybeAlign ArgAlign;
  if (Arg.hasByRefAttr()) {
    Ty = Arg.getParamByRefType();
    ArgAlign = Arg.getParamAlign();
  }

  if (!ArgAlign)
    ArgAlign = DL.getABITypeAlign(Ty);

  return std::make_pair(Ty, *ArgAlign);
}

namespace llvm {

static cl::opt<bool> DumpHSAMetadata(
    "amdgpu-dump-hsa-metadata",
    cl::desc("Dump AMDGPU HSA Metadata"));
static cl::opt<bool> VerifyHSAMetadata(
    "amdgpu-verify-hsa-metadata",
    cl::desc("Verify AMDGPU HSA Metadata"));

namespace AMDGPU {
namespace HSAMD {

//===----------------------------------------------------------------------===//
// HSAMetadataStreamerV2
//===----------------------------------------------------------------------===//
void MetadataStreamerV2::dump(StringRef HSAMetadataString) const {
  errs() << "AMDGPU HSA Metadata:\n" << HSAMetadataString << '\n';
}

void MetadataStreamerV2::verify(StringRef HSAMetadataString) const {
  errs() << "AMDGPU HSA Metadata Parser Test: ";

  HSAMD::Metadata FromHSAMetadataString;
  if (fromString(HSAMetadataString, FromHSAMetadataString)) {
    errs() << "FAIL\n";
    return;
  }

  std::string ToHSAMetadataString;
  if (toString(FromHSAMetadataString, ToHSAMetadataString)) {
    errs() << "FAIL\n";
    return;
  }

  errs() << (HSAMetadataString == ToHSAMetadataString ? "PASS" : "FAIL")
         << '\n';
  if (HSAMetadataString != ToHSAMetadataString) {
    errs() << "Original input: " << HSAMetadataString << '\n'
           << "Produced output: " << ToHSAMetadataString << '\n';
  }
}

AccessQualifier
MetadataStreamerV2::getAccessQualifier(StringRef AccQual) const {
  if (AccQual.empty())
    return AccessQualifier::Unknown;

  return StringSwitch<AccessQualifier>(AccQual)
             .Case("read_only",  AccessQualifier::ReadOnly)
             .Case("write_only", AccessQualifier::WriteOnly)
             .Case("read_write", AccessQualifier::ReadWrite)
             .Default(AccessQualifier::Default);
}

AddressSpaceQualifier
MetadataStreamerV2::getAddressSpaceQualifier(
    unsigned AddressSpace) const {
  switch (AddressSpace) {
  case AMDGPUAS::PRIVATE_ADDRESS:
    return AddressSpaceQualifier::Private;
  case AMDGPUAS::GLOBAL_ADDRESS:
    return AddressSpaceQualifier::Global;
  case AMDGPUAS::CONSTANT_ADDRESS:
    return AddressSpaceQualifier::Constant;
  case AMDGPUAS::LOCAL_ADDRESS:
    return AddressSpaceQualifier::Local;
  case AMDGPUAS::FLAT_ADDRESS:
    return AddressSpaceQualifier::Generic;
  case AMDGPUAS::REGION_ADDRESS:
    return AddressSpaceQualifier::Region;
  default:
    return AddressSpaceQualifier::Unknown;
  }
}

ValueKind MetadataStreamerV2::getValueKind(Type *Ty, StringRef TypeQual,
                                           StringRef BaseTypeName) const {
  if (TypeQual.find("pipe") != StringRef::npos)
    return ValueKind::Pipe;

  return StringSwitch<ValueKind>(BaseTypeName)
             .Case("image1d_t", ValueKind::Image)
             .Case("image1d_array_t", ValueKind::Image)
             .Case("image1d_buffer_t", ValueKind::Image)
             .Case("image2d_t", ValueKind::Image)
             .Case("image2d_array_t", ValueKind::Image)
             .Case("image2d_array_depth_t", ValueKind::Image)
             .Case("image2d_array_msaa_t", ValueKind::Image)
             .Case("image2d_array_msaa_depth_t", ValueKind::Image)
             .Case("image2d_depth_t", ValueKind::Image)
             .Case("image2d_msaa_t", ValueKind::Image)
             .Case("image2d_msaa_depth_t", ValueKind::Image)
             .Case("image3d_t", ValueKind::Image)
             .Case("sampler_t", ValueKind::Sampler)
             .Case("queue_t", ValueKind::Queue)
             .Default(isa<PointerType>(Ty) ?
                          (Ty->getPointerAddressSpace() ==
                           AMDGPUAS::LOCAL_ADDRESS ?
                           ValueKind::DynamicSharedPointer :
                           ValueKind::GlobalBuffer) :
                      ValueKind::ByValue);
}

std::string MetadataStreamerV2::getTypeName(Type *Ty, bool Signed) const {
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
  case Type::FixedVectorTyID: {
    auto VecTy = cast<FixedVectorType>(Ty);
    auto ElTy = VecTy->getElementType();
    auto NumElements = VecTy->getNumElements();
    return (Twine(getTypeName(ElTy, Signed)) + Twine(NumElements)).str();
  }
  default:
    return "unknown";
  }
}

std::vector<uint32_t>
MetadataStreamerV2::getWorkGroupDimensions(MDNode *Node) const {
  std::vector<uint32_t> Dims;
  if (Node->getNumOperands() != 3)
    return Dims;

  for (auto &Op : Node->operands())
    Dims.push_back(mdconst::extract<ConstantInt>(Op)->getZExtValue());
  return Dims;
}

Kernel::CodeProps::Metadata
MetadataStreamerV2::getHSACodeProps(const MachineFunction &MF,
                                    const SIProgramInfo &ProgramInfo) const {
  const GCNSubtarget &STM = MF.getSubtarget<GCNSubtarget>();
  const SIMachineFunctionInfo &MFI = *MF.getInfo<SIMachineFunctionInfo>();
  HSAMD::Kernel::CodeProps::Metadata HSACodeProps;
  const Function &F = MF.getFunction();

  assert(F.getCallingConv() == CallingConv::AMDGPU_KERNEL ||
         F.getCallingConv() == CallingConv::SPIR_KERNEL);

  Align MaxKernArgAlign;
  HSACodeProps.mKernargSegmentSize = STM.getKernArgSegmentSize(F,
                                                               MaxKernArgAlign);
  HSACodeProps.mGroupSegmentFixedSize = ProgramInfo.LDSSize;
  HSACodeProps.mPrivateSegmentFixedSize = ProgramInfo.ScratchSize;
  HSACodeProps.mKernargSegmentAlign =
      std::max(MaxKernArgAlign, Align(4)).value();
  HSACodeProps.mWavefrontSize = STM.getWavefrontSize();
  HSACodeProps.mNumSGPRs = ProgramInfo.NumSGPR;
  HSACodeProps.mNumVGPRs = ProgramInfo.NumVGPR;
  HSACodeProps.mMaxFlatWorkGroupSize = MFI.getMaxFlatWorkGroupSize();
  HSACodeProps.mIsDynamicCallStack = ProgramInfo.DynamicCallStack;
  HSACodeProps.mIsXNACKEnabled = STM.isXNACKEnabled();
  HSACodeProps.mNumSpilledSGPRs = MFI.getNumSpilledSGPRs();
  HSACodeProps.mNumSpilledVGPRs = MFI.getNumSpilledVGPRs();

  return HSACodeProps;
}

Kernel::DebugProps::Metadata
MetadataStreamerV2::getHSADebugProps(const MachineFunction &MF,
                                     const SIProgramInfo &ProgramInfo) const {
  return HSAMD::Kernel::DebugProps::Metadata();
}

void MetadataStreamerV2::emitVersion() {
  auto &Version = HSAMetadata.mVersion;

  Version.push_back(VersionMajor);
  Version.push_back(VersionMinor);
}

void MetadataStreamerV2::emitPrintf(const Module &Mod) {
  auto &Printf = HSAMetadata.mPrintf;

  auto Node = Mod.getNamedMetadata("llvm.printf.fmts");
  if (!Node)
    return;

  for (auto Op : Node->operands())
    if (Op->getNumOperands())
      Printf.push_back(
          std::string(cast<MDString>(Op->getOperand(0))->getString()));
}

void MetadataStreamerV2::emitKernelLanguage(const Function &Func) {
  auto &Kernel = HSAMetadata.mKernels.back();

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

void MetadataStreamerV2::emitKernelAttrs(const Function &Func) {
  auto &Attrs = HSAMetadata.mKernels.back().mAttrs;

  if (auto Node = Func.getMetadata("reqd_work_group_size"))
    Attrs.mReqdWorkGroupSize = getWorkGroupDimensions(Node);
  if (auto Node = Func.getMetadata("work_group_size_hint"))
    Attrs.mWorkGroupSizeHint = getWorkGroupDimensions(Node);
  if (auto Node = Func.getMetadata("vec_type_hint")) {
    Attrs.mVecTypeHint = getTypeName(
        cast<ValueAsMetadata>(Node->getOperand(0))->getType(),
        mdconst::extract<ConstantInt>(Node->getOperand(1))->getZExtValue());
  }
  if (Func.hasFnAttribute("runtime-handle")) {
    Attrs.mRuntimeHandle =
        Func.getFnAttribute("runtime-handle").getValueAsString().str();
  }
}

void MetadataStreamerV2::emitKernelArgs(const Function &Func) {
  for (auto &Arg : Func.args())
    emitKernelArg(Arg);

  emitHiddenKernelArgs(Func);
}

void MetadataStreamerV2::emitKernelArg(const Argument &Arg) {
  auto Func = Arg.getParent();
  auto ArgNo = Arg.getArgNo();
  const MDNode *Node;

  StringRef Name;
  Node = Func->getMetadata("kernel_arg_name");
  if (Node && ArgNo < Node->getNumOperands())
    Name = cast<MDString>(Node->getOperand(ArgNo))->getString();
  else if (Arg.hasName())
    Name = Arg.getName();

  StringRef TypeName;
  Node = Func->getMetadata("kernel_arg_type");
  if (Node && ArgNo < Node->getNumOperands())
    TypeName = cast<MDString>(Node->getOperand(ArgNo))->getString();

  StringRef BaseTypeName;
  Node = Func->getMetadata("kernel_arg_base_type");
  if (Node && ArgNo < Node->getNumOperands())
    BaseTypeName = cast<MDString>(Node->getOperand(ArgNo))->getString();

  StringRef AccQual;
  if (Arg.getType()->isPointerTy() && Arg.onlyReadsMemory() &&
      Arg.hasNoAliasAttr()) {
    AccQual = "read_only";
  } else {
    Node = Func->getMetadata("kernel_arg_access_qual");
    if (Node && ArgNo < Node->getNumOperands())
      AccQual = cast<MDString>(Node->getOperand(ArgNo))->getString();
  }

  StringRef TypeQual;
  Node = Func->getMetadata("kernel_arg_type_qual");
  if (Node && ArgNo < Node->getNumOperands())
    TypeQual = cast<MDString>(Node->getOperand(ArgNo))->getString();

  const DataLayout &DL = Func->getParent()->getDataLayout();

  MaybeAlign PointeeAlign;
  if (auto PtrTy = dyn_cast<PointerType>(Arg.getType())) {
    if (PtrTy->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS) {
      // FIXME: Should report this for all address spaces
      PointeeAlign = DL.getValueOrABITypeAlignment(Arg.getParamAlign(),
                                                   PtrTy->getElementType());
    }
  }

  Type *ArgTy;
  Align ArgAlign;
  std::tie(ArgTy, ArgAlign) = getArgumentTypeAlign(Arg, DL);

  emitKernelArg(DL, ArgTy, ArgAlign,
                getValueKind(ArgTy, TypeQual, BaseTypeName), PointeeAlign, Name,
                TypeName, BaseTypeName, AccQual, TypeQual);
}

void MetadataStreamerV2::emitKernelArg(const DataLayout &DL, Type *Ty,
                                       Align Alignment, ValueKind ValueKind,
                                       MaybeAlign PointeeAlign, StringRef Name,
                                       StringRef TypeName,
                                       StringRef BaseTypeName,
                                       StringRef AccQual, StringRef TypeQual) {
  HSAMetadata.mKernels.back().mArgs.push_back(Kernel::Arg::Metadata());
  auto &Arg = HSAMetadata.mKernels.back().mArgs.back();

  Arg.mName = std::string(Name);
  Arg.mTypeName = std::string(TypeName);
  Arg.mSize = DL.getTypeAllocSize(Ty);
  Arg.mAlign = Alignment.value();
  Arg.mValueKind = ValueKind;
  Arg.mPointeeAlign = PointeeAlign ? PointeeAlign->value() : 0;

  if (auto PtrTy = dyn_cast<PointerType>(Ty))
    Arg.mAddrSpaceQual = getAddressSpaceQualifier(PtrTy->getAddressSpace());

  Arg.mAccQual = getAccessQualifier(AccQual);

  // TODO: Emit Arg.mActualAccQual.

  SmallVector<StringRef, 1> SplitTypeQuals;
  TypeQual.split(SplitTypeQuals, " ", -1, false);
  for (StringRef Key : SplitTypeQuals) {
    auto P = StringSwitch<bool*>(Key)
                 .Case("const",    &Arg.mIsConst)
                 .Case("restrict", &Arg.mIsRestrict)
                 .Case("volatile", &Arg.mIsVolatile)
                 .Case("pipe",     &Arg.mIsPipe)
                 .Default(nullptr);
    if (P)
      *P = true;
  }
}

void MetadataStreamerV2::emitHiddenKernelArgs(const Function &Func) {
  int HiddenArgNumBytes =
      getIntegerAttribute(Func, "amdgpu-implicitarg-num-bytes", 0);

  if (!HiddenArgNumBytes)
    return;

  auto &DL = Func.getParent()->getDataLayout();
  auto Int64Ty = Type::getInt64Ty(Func.getContext());

  if (HiddenArgNumBytes >= 8)
    emitKernelArg(DL, Int64Ty, Align(8), ValueKind::HiddenGlobalOffsetX);
  if (HiddenArgNumBytes >= 16)
    emitKernelArg(DL, Int64Ty, Align(8), ValueKind::HiddenGlobalOffsetY);
  if (HiddenArgNumBytes >= 24)
    emitKernelArg(DL, Int64Ty, Align(8), ValueKind::HiddenGlobalOffsetZ);

  auto Int8PtrTy = Type::getInt8PtrTy(Func.getContext(),
                                      AMDGPUAS::GLOBAL_ADDRESS);

  // Emit "printf buffer" argument if printf is used, otherwise emit dummy
  // "none" argument.
  if (HiddenArgNumBytes >= 32) {
    if (Func.getParent()->getNamedMetadata("llvm.printf.fmts"))
      emitKernelArg(DL, Int8PtrTy, Align(8), ValueKind::HiddenPrintfBuffer);
    else if (Func.getParent()->getFunction("__ockl_hostcall_internal")) {
      // The printf runtime binding pass should have ensured that hostcall and
      // printf are not used in the same module.
      assert(!Func.getParent()->getNamedMetadata("llvm.printf.fmts"));
      emitKernelArg(DL, Int8PtrTy, Align(8), ValueKind::HiddenHostcallBuffer);
    } else
      emitKernelArg(DL, Int8PtrTy, Align(8), ValueKind::HiddenNone);
  }

  // Emit "default queue" and "completion action" arguments if enqueue kernel is
  // used, otherwise emit dummy "none" arguments.
  if (HiddenArgNumBytes >= 48) {
    if (Func.hasFnAttribute("calls-enqueue-kernel")) {
      emitKernelArg(DL, Int8PtrTy, Align(8), ValueKind::HiddenDefaultQueue);
      emitKernelArg(DL, Int8PtrTy, Align(8), ValueKind::HiddenCompletionAction);
    } else {
      emitKernelArg(DL, Int8PtrTy, Align(8), ValueKind::HiddenNone);
      emitKernelArg(DL, Int8PtrTy, Align(8), ValueKind::HiddenNone);
    }
  }

  // Emit the pointer argument for multi-grid object.
  if (HiddenArgNumBytes >= 56)
    emitKernelArg(DL, Int8PtrTy, Align(8), ValueKind::HiddenMultiGridSyncArg);
}

bool MetadataStreamerV2::emitTo(AMDGPUTargetStreamer &TargetStreamer) {
  return TargetStreamer.EmitHSAMetadata(getHSAMetadata());
}

void MetadataStreamerV2::begin(const Module &Mod) {
  emitVersion();
  emitPrintf(Mod);
}

void MetadataStreamerV2::end() {
  std::string HSAMetadataString;
  if (toString(HSAMetadata, HSAMetadataString))
    return;

  if (DumpHSAMetadata)
    dump(HSAMetadataString);
  if (VerifyHSAMetadata)
    verify(HSAMetadataString);
}

void MetadataStreamerV2::emitKernel(const MachineFunction &MF,
                                    const SIProgramInfo &ProgramInfo) {
  auto &Func = MF.getFunction();
  if (Func.getCallingConv() != CallingConv::AMDGPU_KERNEL)
    return;

  auto CodeProps = getHSACodeProps(MF, ProgramInfo);
  auto DebugProps = getHSADebugProps(MF, ProgramInfo);

  HSAMetadata.mKernels.push_back(Kernel::Metadata());
  auto &Kernel = HSAMetadata.mKernels.back();

  Kernel.mName = std::string(Func.getName());
  Kernel.mSymbolName = (Twine(Func.getName()) + Twine("@kd")).str();
  emitKernelLanguage(Func);
  emitKernelAttrs(Func);
  emitKernelArgs(Func);
  HSAMetadata.mKernels.back().mCodeProps = CodeProps;
  HSAMetadata.mKernels.back().mDebugProps = DebugProps;
}

//===----------------------------------------------------------------------===//
// HSAMetadataStreamerV3
//===----------------------------------------------------------------------===//

void MetadataStreamerV3::dump(StringRef HSAMetadataString) const {
  errs() << "AMDGPU HSA Metadata:\n" << HSAMetadataString << '\n';
}

void MetadataStreamerV3::verify(StringRef HSAMetadataString) const {
  errs() << "AMDGPU HSA Metadata Parser Test: ";

  msgpack::Document FromHSAMetadataString;

  if (!FromHSAMetadataString.fromYAML(HSAMetadataString)) {
    errs() << "FAIL\n";
    return;
  }

  std::string ToHSAMetadataString;
  raw_string_ostream StrOS(ToHSAMetadataString);
  FromHSAMetadataString.toYAML(StrOS);

  errs() << (HSAMetadataString == StrOS.str() ? "PASS" : "FAIL") << '\n';
  if (HSAMetadataString != ToHSAMetadataString) {
    errs() << "Original input: " << HSAMetadataString << '\n'
           << "Produced output: " << StrOS.str() << '\n';
  }
}

Optional<StringRef>
MetadataStreamerV3::getAccessQualifier(StringRef AccQual) const {
  return StringSwitch<Optional<StringRef>>(AccQual)
      .Case("read_only", StringRef("read_only"))
      .Case("write_only", StringRef("write_only"))
      .Case("read_write", StringRef("read_write"))
      .Default(None);
}

Optional<StringRef>
MetadataStreamerV3::getAddressSpaceQualifier(unsigned AddressSpace) const {
  switch (AddressSpace) {
  case AMDGPUAS::PRIVATE_ADDRESS:
    return StringRef("private");
  case AMDGPUAS::GLOBAL_ADDRESS:
    return StringRef("global");
  case AMDGPUAS::CONSTANT_ADDRESS:
    return StringRef("constant");
  case AMDGPUAS::LOCAL_ADDRESS:
    return StringRef("local");
  case AMDGPUAS::FLAT_ADDRESS:
    return StringRef("generic");
  case AMDGPUAS::REGION_ADDRESS:
    return StringRef("region");
  default:
    return None;
  }
}

StringRef MetadataStreamerV3::getValueKind(Type *Ty, StringRef TypeQual,
                                           StringRef BaseTypeName) const {
  if (TypeQual.find("pipe") != StringRef::npos)
    return "pipe";

  return StringSwitch<StringRef>(BaseTypeName)
      .Case("image1d_t", "image")
      .Case("image1d_array_t", "image")
      .Case("image1d_buffer_t", "image")
      .Case("image2d_t", "image")
      .Case("image2d_array_t", "image")
      .Case("image2d_array_depth_t", "image")
      .Case("image2d_array_msaa_t", "image")
      .Case("image2d_array_msaa_depth_t", "image")
      .Case("image2d_depth_t", "image")
      .Case("image2d_msaa_t", "image")
      .Case("image2d_msaa_depth_t", "image")
      .Case("image3d_t", "image")
      .Case("sampler_t", "sampler")
      .Case("queue_t", "queue")
      .Default(isa<PointerType>(Ty)
                   ? (Ty->getPointerAddressSpace() == AMDGPUAS::LOCAL_ADDRESS
                          ? "dynamic_shared_pointer"
                          : "global_buffer")
                   : "by_value");
}

std::string MetadataStreamerV3::getTypeName(Type *Ty, bool Signed) const {
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
  case Type::FixedVectorTyID: {
    auto VecTy = cast<FixedVectorType>(Ty);
    auto ElTy = VecTy->getElementType();
    auto NumElements = VecTy->getNumElements();
    return (Twine(getTypeName(ElTy, Signed)) + Twine(NumElements)).str();
  }
  default:
    return "unknown";
  }
}

msgpack::ArrayDocNode
MetadataStreamerV3::getWorkGroupDimensions(MDNode *Node) const {
  auto Dims = HSAMetadataDoc->getArrayNode();
  if (Node->getNumOperands() != 3)
    return Dims;

  for (auto &Op : Node->operands())
    Dims.push_back(Dims.getDocument()->getNode(
        uint64_t(mdconst::extract<ConstantInt>(Op)->getZExtValue())));
  return Dims;
}

void MetadataStreamerV3::emitVersion() {
  auto Version = HSAMetadataDoc->getArrayNode();
  Version.push_back(Version.getDocument()->getNode(VersionMajor));
  Version.push_back(Version.getDocument()->getNode(VersionMinor));
  getRootMetadata("amdhsa.version") = Version;
}

void MetadataStreamerV3::emitPrintf(const Module &Mod) {
  auto Node = Mod.getNamedMetadata("llvm.printf.fmts");
  if (!Node)
    return;

  auto Printf = HSAMetadataDoc->getArrayNode();
  for (auto Op : Node->operands())
    if (Op->getNumOperands())
      Printf.push_back(Printf.getDocument()->getNode(
          cast<MDString>(Op->getOperand(0))->getString(), /*Copy=*/true));
  getRootMetadata("amdhsa.printf") = Printf;
}

void MetadataStreamerV3::emitKernelLanguage(const Function &Func,
                                            msgpack::MapDocNode Kern) {
  // TODO: What about other languages?
  auto Node = Func.getParent()->getNamedMetadata("opencl.ocl.version");
  if (!Node || !Node->getNumOperands())
    return;
  auto Op0 = Node->getOperand(0);
  if (Op0->getNumOperands() <= 1)
    return;

  Kern[".language"] = Kern.getDocument()->getNode("OpenCL C");
  auto LanguageVersion = Kern.getDocument()->getArrayNode();
  LanguageVersion.push_back(Kern.getDocument()->getNode(
      mdconst::extract<ConstantInt>(Op0->getOperand(0))->getZExtValue()));
  LanguageVersion.push_back(Kern.getDocument()->getNode(
      mdconst::extract<ConstantInt>(Op0->getOperand(1))->getZExtValue()));
  Kern[".language_version"] = LanguageVersion;
}

void MetadataStreamerV3::emitKernelAttrs(const Function &Func,
                                         msgpack::MapDocNode Kern) {

  if (auto Node = Func.getMetadata("reqd_work_group_size"))
    Kern[".reqd_workgroup_size"] = getWorkGroupDimensions(Node);
  if (auto Node = Func.getMetadata("work_group_size_hint"))
    Kern[".workgroup_size_hint"] = getWorkGroupDimensions(Node);
  if (auto Node = Func.getMetadata("vec_type_hint")) {
    Kern[".vec_type_hint"] = Kern.getDocument()->getNode(
        getTypeName(
            cast<ValueAsMetadata>(Node->getOperand(0))->getType(),
            mdconst::extract<ConstantInt>(Node->getOperand(1))->getZExtValue()),
        /*Copy=*/true);
  }
  if (Func.hasFnAttribute("runtime-handle")) {
    Kern[".device_enqueue_symbol"] = Kern.getDocument()->getNode(
        Func.getFnAttribute("runtime-handle").getValueAsString().str(),
        /*Copy=*/true);
  }
}

void MetadataStreamerV3::emitKernelArgs(const Function &Func,
                                        msgpack::MapDocNode Kern) {
  unsigned Offset = 0;
  auto Args = HSAMetadataDoc->getArrayNode();
  for (auto &Arg : Func.args())
    emitKernelArg(Arg, Offset, Args);

  emitHiddenKernelArgs(Func, Offset, Args);

  Kern[".args"] = Args;
}

void MetadataStreamerV3::emitKernelArg(const Argument &Arg, unsigned &Offset,
                                       msgpack::ArrayDocNode Args) {
  auto Func = Arg.getParent();
  auto ArgNo = Arg.getArgNo();
  const MDNode *Node;

  StringRef Name;
  Node = Func->getMetadata("kernel_arg_name");
  if (Node && ArgNo < Node->getNumOperands())
    Name = cast<MDString>(Node->getOperand(ArgNo))->getString();
  else if (Arg.hasName())
    Name = Arg.getName();

  StringRef TypeName;
  Node = Func->getMetadata("kernel_arg_type");
  if (Node && ArgNo < Node->getNumOperands())
    TypeName = cast<MDString>(Node->getOperand(ArgNo))->getString();

  StringRef BaseTypeName;
  Node = Func->getMetadata("kernel_arg_base_type");
  if (Node && ArgNo < Node->getNumOperands())
    BaseTypeName = cast<MDString>(Node->getOperand(ArgNo))->getString();

  StringRef AccQual;
  if (Arg.getType()->isPointerTy() && Arg.onlyReadsMemory() &&
      Arg.hasNoAliasAttr()) {
    AccQual = "read_only";
  } else {
    Node = Func->getMetadata("kernel_arg_access_qual");
    if (Node && ArgNo < Node->getNumOperands())
      AccQual = cast<MDString>(Node->getOperand(ArgNo))->getString();
  }

  StringRef TypeQual;
  Node = Func->getMetadata("kernel_arg_type_qual");
  if (Node && ArgNo < Node->getNumOperands())
    TypeQual = cast<MDString>(Node->getOperand(ArgNo))->getString();

  const DataLayout &DL = Func->getParent()->getDataLayout();

  MaybeAlign PointeeAlign;
  Type *Ty = Arg.hasByRefAttr() ? Arg.getParamByRefType() : Arg.getType();

  // FIXME: Need to distinguish in memory alignment from pointer alignment.
  if (auto PtrTy = dyn_cast<PointerType>(Ty)) {
    if (PtrTy->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS) {
      PointeeAlign = DL.getValueOrABITypeAlignment(Arg.getParamAlign(),
                                                   PtrTy->getElementType());
    }
  }

  // There's no distinction between byval aggregates and raw aggregates.
  Type *ArgTy;
  Align ArgAlign;
  std::tie(ArgTy, ArgAlign) = getArgumentTypeAlign(Arg, DL);

  emitKernelArg(DL, ArgTy, ArgAlign,
                getValueKind(ArgTy, TypeQual, BaseTypeName), Offset, Args,
                PointeeAlign, Name, TypeName, BaseTypeName, AccQual, TypeQual);
}

void MetadataStreamerV3::emitKernelArg(
    const DataLayout &DL, Type *Ty, Align Alignment, StringRef ValueKind,
    unsigned &Offset, msgpack::ArrayDocNode Args, MaybeAlign PointeeAlign,
    StringRef Name, StringRef TypeName, StringRef BaseTypeName,
    StringRef AccQual, StringRef TypeQual) {
  auto Arg = Args.getDocument()->getMapNode();

  if (!Name.empty())
    Arg[".name"] = Arg.getDocument()->getNode(Name, /*Copy=*/true);
  if (!TypeName.empty())
    Arg[".type_name"] = Arg.getDocument()->getNode(TypeName, /*Copy=*/true);
  auto Size = DL.getTypeAllocSize(Ty);
  Arg[".size"] = Arg.getDocument()->getNode(Size);
  Offset = alignTo(Offset, Alignment);
  Arg[".offset"] = Arg.getDocument()->getNode(Offset);
  Offset += Size;
  Arg[".value_kind"] = Arg.getDocument()->getNode(ValueKind, /*Copy=*/true);
  if (PointeeAlign)
    Arg[".pointee_align"] = Arg.getDocument()->getNode(PointeeAlign->value());

  if (auto PtrTy = dyn_cast<PointerType>(Ty))
    if (auto Qualifier = getAddressSpaceQualifier(PtrTy->getAddressSpace()))
      Arg[".address_space"] = Arg.getDocument()->getNode(*Qualifier, /*Copy=*/true);

  if (auto AQ = getAccessQualifier(AccQual))
    Arg[".access"] = Arg.getDocument()->getNode(*AQ, /*Copy=*/true);

  // TODO: Emit Arg[".actual_access"].

  SmallVector<StringRef, 1> SplitTypeQuals;
  TypeQual.split(SplitTypeQuals, " ", -1, false);
  for (StringRef Key : SplitTypeQuals) {
    if (Key == "const")
      Arg[".is_const"] = Arg.getDocument()->getNode(true);
    else if (Key == "restrict")
      Arg[".is_restrict"] = Arg.getDocument()->getNode(true);
    else if (Key == "volatile")
      Arg[".is_volatile"] = Arg.getDocument()->getNode(true);
    else if (Key == "pipe")
      Arg[".is_pipe"] = Arg.getDocument()->getNode(true);
  }

  Args.push_back(Arg);
}

void MetadataStreamerV3::emitHiddenKernelArgs(const Function &Func,
                                              unsigned &Offset,
                                              msgpack::ArrayDocNode Args) {
  int HiddenArgNumBytes =
      getIntegerAttribute(Func, "amdgpu-implicitarg-num-bytes", 0);

  if (!HiddenArgNumBytes)
    return;

  auto &DL = Func.getParent()->getDataLayout();
  auto Int64Ty = Type::getInt64Ty(Func.getContext());

  if (HiddenArgNumBytes >= 8)
    emitKernelArg(DL, Int64Ty, Align(8), "hidden_global_offset_x", Offset,
                  Args);
  if (HiddenArgNumBytes >= 16)
    emitKernelArg(DL, Int64Ty, Align(8), "hidden_global_offset_y", Offset,
                  Args);
  if (HiddenArgNumBytes >= 24)
    emitKernelArg(DL, Int64Ty, Align(8), "hidden_global_offset_z", Offset,
                  Args);

  auto Int8PtrTy =
      Type::getInt8PtrTy(Func.getContext(), AMDGPUAS::GLOBAL_ADDRESS);

  // Emit "printf buffer" argument if printf is used, otherwise emit dummy
  // "none" argument.
  if (HiddenArgNumBytes >= 32) {
    if (Func.getParent()->getNamedMetadata("llvm.printf.fmts"))
      emitKernelArg(DL, Int8PtrTy, Align(8), "hidden_printf_buffer", Offset,
                    Args);
    else if (Func.getParent()->getFunction("__ockl_hostcall_internal")) {
      // The printf runtime binding pass should have ensured that hostcall and
      // printf are not used in the same module.
      assert(!Func.getParent()->getNamedMetadata("llvm.printf.fmts"));
      emitKernelArg(DL, Int8PtrTy, Align(8), "hidden_hostcall_buffer", Offset,
                    Args);
    } else
      emitKernelArg(DL, Int8PtrTy, Align(8), "hidden_none", Offset, Args);
  }

  // Emit "default queue" and "completion action" arguments if enqueue kernel is
  // used, otherwise emit dummy "none" arguments.
  if (HiddenArgNumBytes >= 48) {
    if (Func.hasFnAttribute("calls-enqueue-kernel")) {
      emitKernelArg(DL, Int8PtrTy, Align(8), "hidden_default_queue", Offset,
                    Args);
      emitKernelArg(DL, Int8PtrTy, Align(8), "hidden_completion_action", Offset,
                    Args);
    } else {
      emitKernelArg(DL, Int8PtrTy, Align(8), "hidden_none", Offset, Args);
      emitKernelArg(DL, Int8PtrTy, Align(8), "hidden_none", Offset, Args);
    }
  }

  // Emit the pointer argument for multi-grid object.
  if (HiddenArgNumBytes >= 56)
    emitKernelArg(DL, Int8PtrTy, Align(8), "hidden_multigrid_sync_arg", Offset,
                  Args);
}

msgpack::MapDocNode
MetadataStreamerV3::getHSAKernelProps(const MachineFunction &MF,
                                      const SIProgramInfo &ProgramInfo) const {
  const GCNSubtarget &STM = MF.getSubtarget<GCNSubtarget>();
  const SIMachineFunctionInfo &MFI = *MF.getInfo<SIMachineFunctionInfo>();
  const Function &F = MF.getFunction();

  auto Kern = HSAMetadataDoc->getMapNode();

  Align MaxKernArgAlign;
  Kern[".kernarg_segment_size"] = Kern.getDocument()->getNode(
      STM.getKernArgSegmentSize(F, MaxKernArgAlign));
  Kern[".group_segment_fixed_size"] =
      Kern.getDocument()->getNode(ProgramInfo.LDSSize);
  Kern[".private_segment_fixed_size"] =
      Kern.getDocument()->getNode(ProgramInfo.ScratchSize);
  Kern[".kernarg_segment_align"] =
      Kern.getDocument()->getNode(std::max(Align(4), MaxKernArgAlign).value());
  Kern[".wavefront_size"] =
      Kern.getDocument()->getNode(STM.getWavefrontSize());
  Kern[".sgpr_count"] = Kern.getDocument()->getNode(ProgramInfo.NumSGPR);
  Kern[".vgpr_count"] = Kern.getDocument()->getNode(ProgramInfo.NumVGPR);
  Kern[".max_flat_workgroup_size"] =
      Kern.getDocument()->getNode(MFI.getMaxFlatWorkGroupSize());
  Kern[".sgpr_spill_count"] =
      Kern.getDocument()->getNode(MFI.getNumSpilledSGPRs());
  Kern[".vgpr_spill_count"] =
      Kern.getDocument()->getNode(MFI.getNumSpilledVGPRs());

  return Kern;
}

bool MetadataStreamerV3::emitTo(AMDGPUTargetStreamer &TargetStreamer) {
  return TargetStreamer.EmitHSAMetadata(*HSAMetadataDoc, true);
}

void MetadataStreamerV3::begin(const Module &Mod) {
  emitVersion();
  emitPrintf(Mod);
  getRootMetadata("amdhsa.kernels") = HSAMetadataDoc->getArrayNode();
}

void MetadataStreamerV3::end() {
  std::string HSAMetadataString;
  raw_string_ostream StrOS(HSAMetadataString);
  HSAMetadataDoc->toYAML(StrOS);

  if (DumpHSAMetadata)
    dump(StrOS.str());
  if (VerifyHSAMetadata)
    verify(StrOS.str());
}

void MetadataStreamerV3::emitKernel(const MachineFunction &MF,
                                    const SIProgramInfo &ProgramInfo) {
  auto &Func = MF.getFunction();
  auto Kern = getHSAKernelProps(MF, ProgramInfo);

  assert(Func.getCallingConv() == CallingConv::AMDGPU_KERNEL ||
         Func.getCallingConv() == CallingConv::SPIR_KERNEL);

  auto Kernels =
      getRootMetadata("amdhsa.kernels").getArray(/*Convert=*/true);

  {
    Kern[".name"] = Kern.getDocument()->getNode(Func.getName());
    Kern[".symbol"] = Kern.getDocument()->getNode(
        (Twine(Func.getName()) + Twine(".kd")).str(), /*Copy=*/true);
    emitKernelLanguage(Func, Kern);
    emitKernelAttrs(Func, Kern);
    emitKernelArgs(Func, Kern);
  }

  Kernels.push_back(Kern);
}

} // end namespace HSAMD
} // end namespace AMDGPU
} // end namespace llvm
