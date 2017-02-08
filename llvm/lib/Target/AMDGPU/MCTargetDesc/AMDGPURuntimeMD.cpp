//===-- AMDGPURuntimeMD.cpp - Generates runtime metadata ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// Generates AMDGPU runtime metadata for YAML mapping.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPURuntimeMetadata.h"
#include "MCTargetDesc/AMDGPURuntimeMD.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/YAMLTraits.h"
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

using namespace llvm;
using namespace ::AMDGPU::RuntimeMD;

static cl::opt<bool>
DumpRuntimeMD("amdgpu-dump-rtmd",
              cl::desc("Dump AMDGPU runtime metadata"));

static cl::opt<bool>
CheckRuntimeMDParser("amdgpu-check-rtmd-parser", cl::Hidden,
                     cl::desc("Check AMDGPU runtime metadata YAML parser"));

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(uint8_t)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(uint32_t)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(std::string)
LLVM_YAML_IS_SEQUENCE_VECTOR(Kernel::Metadata)
LLVM_YAML_IS_SEQUENCE_VECTOR(KernelArg::Metadata)

namespace llvm {
namespace yaml {

template <> struct MappingTraits<KernelArg::Metadata> {
  static void mapping(IO &YamlIO, KernelArg::Metadata &A) {
    YamlIO.mapRequired(KeyName::ArgSize, A.Size);
    YamlIO.mapRequired(KeyName::ArgAlign, A.Align);
    YamlIO.mapOptional(KeyName::ArgPointeeAlign, A.PointeeAlign, 0U);
    YamlIO.mapRequired(KeyName::ArgKind, A.Kind);
    YamlIO.mapRequired(KeyName::ArgValueType, A.ValueType);
    YamlIO.mapOptional(KeyName::ArgTypeName, A.TypeName, std::string());
    YamlIO.mapOptional(KeyName::ArgName, A.Name, std::string());
    YamlIO.mapOptional(KeyName::ArgAddrQual, A.AddrQual, INVALID_ADDR_QUAL);
    YamlIO.mapOptional(KeyName::ArgAccQual, A.AccQual, INVALID_ACC_QUAL);
    YamlIO.mapOptional(KeyName::ArgIsVolatile, A.IsVolatile, uint8_t(0));
    YamlIO.mapOptional(KeyName::ArgIsConst, A.IsConst, uint8_t(0));
    YamlIO.mapOptional(KeyName::ArgIsRestrict, A.IsRestrict, uint8_t(0));
    YamlIO.mapOptional(KeyName::ArgIsPipe, A.IsPipe, uint8_t(0));
  }
  static const bool flow = true;
};

template <> struct MappingTraits<Kernel::Metadata> {
  static void mapping(IO &YamlIO, Kernel::Metadata &K) {
    YamlIO.mapRequired(KeyName::KernelName, K.Name);
    YamlIO.mapOptional(KeyName::Language, K.Language, std::string());
    YamlIO.mapOptional(KeyName::LanguageVersion, K.LanguageVersion);
    YamlIO.mapOptional(KeyName::ReqdWorkGroupSize, K.ReqdWorkGroupSize);
    YamlIO.mapOptional(KeyName::WorkGroupSizeHint, K.WorkGroupSizeHint);
    YamlIO.mapOptional(KeyName::VecTypeHint, K.VecTypeHint, std::string());
    YamlIO.mapOptional(KeyName::KernelIndex, K.KernelIndex,
        INVALID_KERNEL_INDEX);
    YamlIO.mapOptional(KeyName::NoPartialWorkGroups, K.NoPartialWorkGroups,
        uint8_t(0));
    YamlIO.mapRequired(KeyName::Args, K.Args);
  }
  static const bool flow = true;
};

template <> struct MappingTraits<IsaInfo::Metadata> {
  static void mapping(IO &YamlIO, IsaInfo::Metadata &I) {
    YamlIO.mapRequired(KeyName::IsaInfoWavefrontSize, I.WavefrontSize);
    YamlIO.mapRequired(KeyName::IsaInfoLocalMemorySize, I.LocalMemorySize);
    YamlIO.mapRequired(KeyName::IsaInfoEUsPerCU, I.EUsPerCU);
    YamlIO.mapRequired(KeyName::IsaInfoMaxWavesPerEU, I.MaxWavesPerEU);
    YamlIO.mapRequired(KeyName::IsaInfoMaxFlatWorkGroupSize,
        I.MaxFlatWorkGroupSize);
    YamlIO.mapRequired(KeyName::IsaInfoSGPRAllocGranule, I.SGPRAllocGranule);
    YamlIO.mapRequired(KeyName::IsaInfoTotalNumSGPRs, I.TotalNumSGPRs);
    YamlIO.mapRequired(KeyName::IsaInfoAddressableNumSGPRs,
        I.AddressableNumSGPRs);
    YamlIO.mapRequired(KeyName::IsaInfoVGPRAllocGranule, I.VGPRAllocGranule);
    YamlIO.mapRequired(KeyName::IsaInfoTotalNumVGPRs, I.TotalNumVGPRs);
    YamlIO.mapRequired(KeyName::IsaInfoAddressableNumVGPRs,
        I.AddressableNumVGPRs);
  }
  static const bool flow = true;
};

template <> struct MappingTraits<Program::Metadata> {
  static void mapping(IO &YamlIO, Program::Metadata &Prog) {
    YamlIO.mapRequired(KeyName::MDVersion, Prog.MDVersionSeq);
    YamlIO.mapRequired(KeyName::IsaInfo, Prog.IsaInfo);
    YamlIO.mapOptional(KeyName::PrintfInfo, Prog.PrintfInfo);
    YamlIO.mapOptional(KeyName::Kernels, Prog.Kernels);
  }
  static const bool flow = true;
};

} // end namespace yaml
} // end namespace llvm

// Get a vector of three integer values from MDNode \p Node;
static std::vector<uint32_t> getThreeInt32(MDNode *Node) {
  assert(Node->getNumOperands() == 3);
  std::vector<uint32_t> V;
  for (const MDOperand &Op : Node->operands()) {
    const ConstantInt *CI = mdconst::extract<ConstantInt>(Op);
    V.push_back(CI->getZExtValue());
  }
  return V;
}

static std::string getOCLTypeName(Type *Ty, bool Signed) {
  switch (Ty->getTypeID()) {
  case Type::HalfTyID:
    return "half";
  case Type::FloatTyID:
    return "float";
  case Type::DoubleTyID:
    return "double";
  case Type::IntegerTyID: {
    if (!Signed)
      return (Twine('u') + getOCLTypeName(Ty, true)).str();
    unsigned BW = Ty->getIntegerBitWidth();
    switch (BW) {
    case 8:
      return "char";
    case 16:
      return "short";
    case 32:
      return "int";
    case 64:
      return "long";
    default:
      return (Twine('i') + Twine(BW)).str();
    }
  }
  case Type::VectorTyID: {
    VectorType *VecTy = cast<VectorType>(Ty);
    Type *EleTy = VecTy->getElementType();
    unsigned Size = VecTy->getVectorNumElements();
    return (Twine(getOCLTypeName(EleTy, Signed)) + Twine(Size)).str();
  }
  default:
    return "unknown";
  }
}

static KernelArg::ValueType getRuntimeMDValueType(
  Type *Ty, StringRef TypeName) {
  switch (Ty->getTypeID()) {
  case Type::HalfTyID:
    return KernelArg::F16;
  case Type::FloatTyID:
    return KernelArg::F32;
  case Type::DoubleTyID:
    return KernelArg::F64;
  case Type::IntegerTyID: {
    bool Signed = !TypeName.startswith("u");
    switch (Ty->getIntegerBitWidth()) {
    case 8:
      return Signed ? KernelArg::I8 : KernelArg::U8;
    case 16:
      return Signed ? KernelArg::I16 : KernelArg::U16;
    case 32:
      return Signed ? KernelArg::I32 : KernelArg::U32;
    case 64:
      return Signed ? KernelArg::I64 : KernelArg::U64;
    default:
      // Runtime does not recognize other integer types. Report as struct type.
      return KernelArg::Struct;
    }
  }
  case Type::VectorTyID:
    return getRuntimeMDValueType(Ty->getVectorElementType(), TypeName);
  case Type::PointerTyID:
    return getRuntimeMDValueType(Ty->getPointerElementType(), TypeName);
  default:
    return KernelArg::Struct;
  }
}

static KernelArg::AddressSpaceQualifer getRuntimeAddrSpace(
    AMDGPUAS::AddressSpaces A) {
  switch (A) {
  case AMDGPUAS::GLOBAL_ADDRESS:
    return KernelArg::Global;
  case AMDGPUAS::CONSTANT_ADDRESS:
    return KernelArg::Constant;
  case AMDGPUAS::LOCAL_ADDRESS:
    return KernelArg::Local;
  case AMDGPUAS::FLAT_ADDRESS:
    return KernelArg::Generic;
  case AMDGPUAS::REGION_ADDRESS:
    return KernelArg::Region;
  default:
    return KernelArg::Private;
  }
}

static KernelArg::Metadata getRuntimeMDForKernelArg(const DataLayout &DL,
    Type *T, KernelArg::Kind Kind, StringRef BaseTypeName = "",
    StringRef TypeName = "", StringRef ArgName = "", StringRef TypeQual = "",
    StringRef AccQual = "") {
  KernelArg::Metadata Arg;

  // Set ArgSize and ArgAlign.
  Arg.Size = DL.getTypeAllocSize(T);
  Arg.Align = DL.getABITypeAlignment(T);
  if (auto PT = dyn_cast<PointerType>(T)) {
    auto ET = PT->getElementType();
    if (PT->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS && ET->isSized())
      Arg.PointeeAlign = DL.getABITypeAlignment(ET);
  }

  // Set ArgTypeName.
  Arg.TypeName = TypeName;

  // Set ArgName.
  Arg.Name = ArgName;

  // Set ArgIsVolatile, ArgIsRestrict, ArgIsConst and ArgIsPipe.
  SmallVector<StringRef, 1> SplitQ;
  TypeQual.split(SplitQ, " ", -1, false /* Drop empty entry */);

  for (StringRef KeyName : SplitQ) {
    auto *P = StringSwitch<uint8_t *>(KeyName)
      .Case("volatile", &Arg.IsVolatile)
      .Case("restrict", &Arg.IsRestrict)
      .Case("const",    &Arg.IsConst)
      .Case("pipe",     &Arg.IsPipe)
      .Default(nullptr);
    if (P)
      *P = 1;
  }

  // Set ArgKind.
  Arg.Kind = Kind;

  // Set ArgValueType.
  Arg.ValueType = getRuntimeMDValueType(T, BaseTypeName);

  // Set ArgAccQual.
  if (!AccQual.empty()) {
    Arg.AccQual = StringSwitch<KernelArg::AccessQualifer>(AccQual)
      .Case("read_only",  KernelArg::ReadOnly)
      .Case("write_only", KernelArg::WriteOnly)
      .Case("read_write", KernelArg::ReadWrite)
      .Default(KernelArg::AccNone);
  }

  // Set ArgAddrQual.
  if (auto *PT = dyn_cast<PointerType>(T)) {
    Arg.AddrQual = getRuntimeAddrSpace(static_cast<AMDGPUAS::AddressSpaces>(
        PT->getAddressSpace()));
  }

  return Arg;
}

static Kernel::Metadata getRuntimeMDForKernel(const Function &F) {
  Kernel::Metadata Kernel;
  Kernel.Name = F.getName();
  auto &M = *F.getParent();

  // Set Language and LanguageVersion.
  if (auto MD = M.getNamedMetadata("opencl.ocl.version")) {
    if (MD->getNumOperands() != 0) {
      auto Node = MD->getOperand(0);
      if (Node->getNumOperands() > 1) {
        Kernel.Language = "OpenCL C";
        uint16_t Major = mdconst::extract<ConstantInt>(Node->getOperand(0))
                         ->getZExtValue();
        uint16_t Minor = mdconst::extract<ConstantInt>(Node->getOperand(1))
                         ->getZExtValue();
        Kernel.LanguageVersion.push_back(Major);
        Kernel.LanguageVersion.push_back(Minor);
      }
    }
  }

  const DataLayout &DL = F.getParent()->getDataLayout();
  for (auto &Arg : F.args()) {
    unsigned I = Arg.getArgNo();
    Type *T = Arg.getType();
    auto TypeName = dyn_cast<MDString>(F.getMetadata(
        "kernel_arg_type")->getOperand(I))->getString();
    auto BaseTypeName = cast<MDString>(F.getMetadata(
        "kernel_arg_base_type")->getOperand(I))->getString();
    StringRef ArgName;
    if (auto ArgNameMD = F.getMetadata("kernel_arg_name"))
      ArgName = cast<MDString>(ArgNameMD->getOperand(I))->getString();
    auto TypeQual = cast<MDString>(F.getMetadata(
        "kernel_arg_type_qual")->getOperand(I))->getString();
    auto AccQual = cast<MDString>(F.getMetadata(
        "kernel_arg_access_qual")->getOperand(I))->getString();
    KernelArg::Kind Kind;
    if (TypeQual.find("pipe") != StringRef::npos)
      Kind = KernelArg::Pipe;
    else Kind = StringSwitch<KernelArg::Kind>(BaseTypeName)
      .Case("sampler_t", KernelArg::Sampler)
      .Case("queue_t",   KernelArg::Queue)
      .Cases("image1d_t", "image1d_array_t", "image1d_buffer_t",
             "image2d_t" , "image2d_array_t",  KernelArg::Image)
      .Cases("image2d_depth_t", "image2d_array_depth_t",
             "image2d_msaa_t", "image2d_array_msaa_t",
             "image2d_msaa_depth_t",  KernelArg::Image)
      .Cases("image2d_array_msaa_depth_t", "image3d_t",
             KernelArg::Image)
      .Default(isa<PointerType>(T) ?
                   (T->getPointerAddressSpace() == AMDGPUAS::LOCAL_ADDRESS ?
                   KernelArg::DynamicSharedPointer :
                   KernelArg::GlobalBuffer) :
                   KernelArg::ByValue);
    Kernel.Args.emplace_back(getRuntimeMDForKernelArg(DL, T, Kind,
        BaseTypeName, TypeName, ArgName, TypeQual, AccQual));
  }

  // Emit hidden kernel arguments for OpenCL kernels.
  if (F.getParent()->getNamedMetadata("opencl.ocl.version")) {
    auto Int64T = Type::getInt64Ty(F.getContext());
    Kernel.Args.emplace_back(getRuntimeMDForKernelArg(DL, Int64T,
        KernelArg::HiddenGlobalOffsetX));
    Kernel.Args.emplace_back(getRuntimeMDForKernelArg(DL, Int64T,
        KernelArg::HiddenGlobalOffsetY));
    Kernel.Args.emplace_back(getRuntimeMDForKernelArg(DL, Int64T,
        KernelArg::HiddenGlobalOffsetZ));
    if (F.getParent()->getNamedMetadata("llvm.printf.fmts")) {
      auto Int8PtrT = Type::getInt8PtrTy(F.getContext(),
          KernelArg::Global);
      Kernel.Args.emplace_back(getRuntimeMDForKernelArg(DL, Int8PtrT,
          KernelArg::HiddenPrintfBuffer));
    }
  }

  // Set ReqdWorkGroupSize, WorkGroupSizeHint, and VecTypeHint.
  if (auto RWGS = F.getMetadata("reqd_work_group_size"))
    Kernel.ReqdWorkGroupSize = getThreeInt32(RWGS);

  if (auto WGSH = F.getMetadata("work_group_size_hint"))
    Kernel.WorkGroupSizeHint = getThreeInt32(WGSH);

  if (auto VTH = F.getMetadata("vec_type_hint"))
    Kernel.VecTypeHint = getOCLTypeName(cast<ValueAsMetadata>(
      VTH->getOperand(0))->getType(), mdconst::extract<ConstantInt>(
      VTH->getOperand(1))->getZExtValue());

  return Kernel;
}

Program::Metadata::Metadata(const std::string &YAML) {
  yaml::Input Input(YAML);
  Input >> *this;
}

std::string Program::Metadata::toYAML() {
  std::string Text;
  raw_string_ostream Stream(Text);
  yaml::Output Output(Stream, nullptr,
                      std::numeric_limits<int>::max() /* do not wrap line */);
  Output << *this;
  return Stream.str();
}

Program::Metadata Program::Metadata::fromYAML(const std::string &S) {
  return Program::Metadata(S);
}

// Check if the YAML string can be parsed.
static void checkRuntimeMDYAMLString(const std::string &YAML) {
  auto P = Program::Metadata::fromYAML(YAML);
  auto S = P.toYAML();
  errs() << "AMDGPU runtime metadata parser test "
         << (YAML == S ? "passes" : "fails") << ".\n";
  if (YAML != S) {
    errs() << "First output: " << YAML << '\n'
           << "Second output: " << S << '\n';
  }
}

std::string llvm::getRuntimeMDYAMLString(const FeatureBitset &Features,
                                         const Module &M) {
  Program::Metadata Prog;
  Prog.MDVersionSeq.push_back(MDVersion);
  Prog.MDVersionSeq.push_back(MDRevision);

  IsaInfo::Metadata &IIM = Prog.IsaInfo;
  IIM.WavefrontSize = AMDGPU::IsaInfo::getWavefrontSize(Features);
  IIM.LocalMemorySize = AMDGPU::IsaInfo::getLocalMemorySize(Features);
  IIM.EUsPerCU = AMDGPU::IsaInfo::getEUsPerCU(Features);
  IIM.MaxWavesPerEU = AMDGPU::IsaInfo::getMaxWavesPerEU(Features);
  IIM.MaxFlatWorkGroupSize = AMDGPU::IsaInfo::getMaxFlatWorkGroupSize(Features);
  IIM.SGPRAllocGranule = AMDGPU::IsaInfo::getSGPRAllocGranule(Features);
  IIM.TotalNumSGPRs = AMDGPU::IsaInfo::getTotalNumSGPRs(Features);
  IIM.AddressableNumSGPRs = AMDGPU::IsaInfo::getAddressableNumSGPRs(Features);
  IIM.VGPRAllocGranule = AMDGPU::IsaInfo::getVGPRAllocGranule(Features);
  IIM.TotalNumVGPRs = AMDGPU::IsaInfo::getTotalNumVGPRs(Features);
  IIM.AddressableNumVGPRs = AMDGPU::IsaInfo::getAddressableNumVGPRs(Features);

  // Set PrintfInfo.
  if (auto MD = M.getNamedMetadata("llvm.printf.fmts")) {
    for (unsigned I = 0; I < MD->getNumOperands(); ++I) {
      auto Node = MD->getOperand(I);
      if (Node->getNumOperands() > 0)
        Prog.PrintfInfo.push_back(cast<MDString>(Node->getOperand(0))
            ->getString());
    }
  }

  // Set Kernels.
  for (auto &F: M.functions()) {
    if (!F.getMetadata("kernel_arg_type"))
      continue;
    Prog.Kernels.emplace_back(getRuntimeMDForKernel(F));
  }

  auto YAML = Prog.toYAML();

  if (DumpRuntimeMD)
    errs() << "AMDGPU runtime metadata:\n" << YAML << '\n';

  if (CheckRuntimeMDParser)
    checkRuntimeMDYAMLString(YAML);

  return YAML;
}
