//===- DXILShaderFlags.cpp - DXIL Shader Flags helper objects -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper objects and APIs for working with DXIL
///       Shader Flags.
///
//===----------------------------------------------------------------------===//

#include "DXILShaderFlags.h"
#include "DirectX.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::dxil;

static bool hasUAVsAtEveryStage(const DXILResourceMap &DRM,
                                const ModuleMetadataInfo &MMDI) {
  if (DRM.uavs().empty())
    return false;

  switch (MMDI.ShaderProfile) {
  default:
    return false;
  case Triple::EnvironmentType::Compute:
  case Triple::EnvironmentType::Pixel:
    return false;
  case Triple::EnvironmentType::Vertex:
  case Triple::EnvironmentType::Geometry:
  case Triple::EnvironmentType::Hull:
  case Triple::EnvironmentType::Domain:
    return true;
  case Triple::EnvironmentType::Library:
  case Triple::EnvironmentType::RayGeneration:
  case Triple::EnvironmentType::Intersection:
  case Triple::EnvironmentType::AnyHit:
  case Triple::EnvironmentType::ClosestHit:
  case Triple::EnvironmentType::Miss:
  case Triple::EnvironmentType::Callable:
  case Triple::EnvironmentType::Mesh:
  case Triple::EnvironmentType::Amplification:
    return MMDI.ValidatorVersion < VersionTuple(1, 8);
  }
}

static bool checkWaveOps(Intrinsic::ID IID) {
  // Currently unsupported intrinsics
  // case Intrinsic::dx_wave_getlanecount:
  // case Intrinsic::dx_wave_allequal:
  // case Intrinsic::dx_wave_ballot:
  // case Intrinsic::dx_wave_readfirst:
  // case Intrinsic::dx_wave_reduce.and:
  // case Intrinsic::dx_wave_reduce.or:
  // case Intrinsic::dx_wave_reduce.xor:
  // case Intrinsic::dx_wave_prefixop:
  // case Intrinsic::dx_quad.readat:
  // case Intrinsic::dx_quad.readacrossx:
  // case Intrinsic::dx_quad.readacrossy:
  // case Intrinsic::dx_quad.readacrossdiagonal:
  // case Intrinsic::dx_wave_prefixballot:
  // case Intrinsic::dx_wave_match:
  // case Intrinsic::dx_wavemulti.*:
  // case Intrinsic::dx_wavemulti.ballot:
  // case Intrinsic::dx_quad.vote:
  switch (IID) {
  default:
    return false;
  case Intrinsic::dx_wave_is_first_lane:
  case Intrinsic::dx_wave_getlaneindex:
  case Intrinsic::dx_wave_any:
  case Intrinsic::dx_wave_all:
  case Intrinsic::dx_wave_readlane:
  case Intrinsic::dx_wave_active_countbits:
  // Wave Active Op Variants
  case Intrinsic::dx_wave_reduce_sum:
  case Intrinsic::dx_wave_reduce_usum:
  case Intrinsic::dx_wave_reduce_max:
  case Intrinsic::dx_wave_reduce_umax:
    return true;
  }
}

/// Update the shader flags mask based on the given instruction.
/// \param CSF Shader flags mask to update.
/// \param I Instruction to check.
void ModuleShaderFlags::updateFunctionFlags(ComputedShaderFlags &CSF,
                                            const Instruction &I,
                                            DXILResourceTypeMap &DRTM,
                                            const ModuleMetadataInfo &MMDI) {
  if (!CSF.Doubles)
    CSF.Doubles = I.getType()->isDoubleTy();

  if (!CSF.Doubles) {
    for (const Value *Op : I.operands()) {
      if (Op->getType()->isDoubleTy()) {
        CSF.Doubles = true;
        break;
      }
    }
  }

  if (CSF.Doubles) {
    switch (I.getOpcode()) {
    case Instruction::FDiv:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
      CSF.DX11_1_DoubleExtensions = true;
      break;
    }
  }

  if (!CSF.LowPrecisionPresent)
    CSF.LowPrecisionPresent =
        I.getType()->isIntegerTy(16) || I.getType()->isHalfTy();

  if (!CSF.LowPrecisionPresent) {
    for (const Value *Op : I.operands()) {
      if (Op->getType()->isIntegerTy(16) || Op->getType()->isHalfTy()) {
        CSF.LowPrecisionPresent = true;
        break;
      }
    }
  }

  if (CSF.LowPrecisionPresent) {
    if (CSF.NativeLowPrecisionMode)
      CSF.NativeLowPrecision = true;
    else
      CSF.MinimumPrecision = true;
  }

  if (!CSF.Int64Ops)
    CSF.Int64Ops = I.getType()->isIntegerTy(64);

  if (!CSF.Int64Ops) {
    for (const Value *Op : I.operands()) {
      if (Op->getType()->isIntegerTy(64)) {
        CSF.Int64Ops = true;
        break;
      }
    }
  }

  if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
    switch (II->getIntrinsicID()) {
    default:
      break;
    case Intrinsic::dx_resource_handlefrombinding: {
      dxil::ResourceTypeInfo &RTI = DRTM[cast<TargetExtType>(II->getType())];

      // Set ResMayNotAlias if DXIL validator version >= 1.8 and the function
      // uses UAVs
      if (!CSF.ResMayNotAlias && CanSetResMayNotAlias &&
          MMDI.ValidatorVersion >= VersionTuple(1, 8) && RTI.isUAV())
        CSF.ResMayNotAlias = true;

      switch (RTI.getResourceKind()) {
      case dxil::ResourceKind::StructuredBuffer:
      case dxil::ResourceKind::RawBuffer:
        CSF.EnableRawAndStructuredBuffers = true;
        break;
      default:
        break;
      }
      break;
    }
    case Intrinsic::dx_resource_load_typedbuffer: {
      dxil::ResourceTypeInfo &RTI =
          DRTM[cast<TargetExtType>(II->getArgOperand(0)->getType())];
      if (RTI.isTyped())
        CSF.TypedUAVLoadAdditionalFormats |= RTI.getTyped().ElementCount > 1;
      break;
    }
    }
  }
  // Handle call instructions
  if (auto *CI = dyn_cast<CallInst>(&I)) {
    const Function *CF = CI->getCalledFunction();
    // Merge-in shader flags mask of the called function in the current module
    if (FunctionFlags.contains(CF))
      CSF.merge(FunctionFlags[CF]);

    // TODO: Set DX11_1_DoubleExtensions if I is a call to DXIL intrinsic
    // DXIL::Opcode::Fma https://github.com/llvm/llvm-project/issues/114554

    CSF.WaveOps |= checkWaveOps(CI->getIntrinsicID());
  }
}

/// Set shader flags that apply to all functions within the module
ComputedShaderFlags
ModuleShaderFlags::gatherGlobalModuleFlags(const Module &M,
                                           const DXILResourceMap &DRM,
                                           const ModuleMetadataInfo &MMDI) {

  ComputedShaderFlags CSF;

  // Set DisableOptimizations flag based on the presence of OptimizeNone
  // attribute of entry functions.
  if (MMDI.EntryPropertyVec.size() > 0) {
    CSF.DisableOptimizations = MMDI.EntryPropertyVec[0].Entry->hasFnAttribute(
        llvm::Attribute::OptimizeNone);
    // Ensure all entry functions have the same optimization attribute
    for (const auto &EntryFunProps : MMDI.EntryPropertyVec)
      if (CSF.DisableOptimizations !=
          EntryFunProps.Entry->hasFnAttribute(llvm::Attribute::OptimizeNone))
        EntryFunProps.Entry->getContext().diagnose(DiagnosticInfoUnsupported(
            *(EntryFunProps.Entry), "Inconsistent optnone attribute "));
  }

  CSF.UAVsAtEveryStage = hasUAVsAtEveryStage(DRM, MMDI);

  // Set the Max64UAVs flag if the number of UAVs is > 8
  uint32_t NumUAVs = 0;
  for (auto &UAV : DRM.uavs())
    if (MMDI.ValidatorVersion < VersionTuple(1, 6))
      NumUAVs++;
    else // MMDI.ValidatorVersion >= VersionTuple(1, 6)
      NumUAVs += UAV.getBinding().Size;
  if (NumUAVs > 8)
    CSF.Max64UAVs = true;

  // Set the module flag that enables native low-precision execution mode.
  // NativeLowPrecisionMode can only be set when the command line option
  // -enable-16bit-types is provided. This is indicated by the dx.nativelowprec
  // module flag being set
  // This flag is needed even if the module does not use 16-bit types because a
  // corresponding debug module may include 16-bit types, and tools that use the
  // debug module may expect it to have the same flags as the original
  if (auto *NativeLowPrec = mdconst::extract_or_null<ConstantInt>(
          M.getModuleFlag("dx.nativelowprec")))
    if (MMDI.ShaderModelVersion >= VersionTuple(6, 2))
      CSF.NativeLowPrecisionMode = NativeLowPrec->getValue().getBoolValue();

  // Set ResMayNotAlias to true if DXIL validator version < 1.8 and there
  // are UAVs present globally.
  if (CanSetResMayNotAlias && MMDI.ValidatorVersion < VersionTuple(1, 8))
    CSF.ResMayNotAlias = !DRM.uavs().empty();

  return CSF;
}

/// Construct ModuleShaderFlags for module Module M
void ModuleShaderFlags::initialize(Module &M, DXILResourceTypeMap &DRTM,
                                   const DXILResourceMap &DRM,
                                   const ModuleMetadataInfo &MMDI) {

  CanSetResMayNotAlias = MMDI.DXILVersion >= VersionTuple(1, 7);
  // The command line option -res-may-alias will set the dx.resmayalias module
  // flag to 1, thereby disabling the ability to set the ResMayNotAlias flag
  if (auto *ResMayAlias = mdconst::extract_or_null<ConstantInt>(
          M.getModuleFlag("dx.resmayalias")))
    if (ResMayAlias->getValue().getBoolValue())
      CanSetResMayNotAlias = false;

  ComputedShaderFlags GlobalSFMask = gatherGlobalModuleFlags(M, DRM, MMDI);

  CallGraph CG(M);

  // Compute Shader Flags Mask for all functions using post-order visit of SCC
  // of the call graph.
  for (scc_iterator<CallGraph *> SCCI = scc_begin(&CG); !SCCI.isAtEnd();
       ++SCCI) {
    const std::vector<CallGraphNode *> &CurSCC = *SCCI;

    // Union of shader masks of all functions in CurSCC
    ComputedShaderFlags SCCSF;
    // List of functions in CurSCC that are neither external nor declarations
    // and hence whose flags are collected
    SmallVector<Function *> CurSCCFuncs;
    for (CallGraphNode *CGN : CurSCC) {
      Function *F = CGN->getFunction();
      if (!F)
        continue;

      if (F->isDeclaration()) {
        assert(!F->getName().starts_with("dx.op.") &&
               "DXIL Shader Flag analysis should not be run post-lowering.");
        continue;
      }

      ComputedShaderFlags CSF = GlobalSFMask;
      for (const auto &BB : *F)
        for (const auto &I : BB)
          updateFunctionFlags(CSF, I, DRTM, MMDI);
      // Update combined shader flags mask for all functions in this SCC
      SCCSF.merge(CSF);

      CurSCCFuncs.push_back(F);
    }

    // Update combined shader flags mask for all functions of the module
    CombinedSFMask.merge(SCCSF);

    // Shader flags mask of each of the functions in an SCC of the call graph is
    // the union of all functions in the SCC. Update shader flags masks of
    // functions in CurSCC accordingly. This is trivially true if SCC contains
    // one function.
    for (Function *F : CurSCCFuncs)
      // Merge SCCSF with that of F
      FunctionFlags[F].merge(SCCSF);
  }
}

void ComputedShaderFlags::print(raw_ostream &OS) const {
  uint64_t FlagVal = (uint64_t) * this;
  OS << formatv("; Shader Flags Value: {0:x8}\n;\n", FlagVal);
  if (FlagVal == 0)
    return;
  OS << "; Note: shader requires additional functionality:\n";
#define SHADER_FEATURE_FLAG(FeatureBit, DxilModuleNum, FlagName, Str)          \
  if (FlagName)                                                                \
    (OS << ";").indent(7) << Str << "\n";
#include "llvm/BinaryFormat/DXContainerConstants.def"
  OS << "; Note: extra DXIL module flags:\n";
#define DXIL_MODULE_FLAG(DxilModuleBit, FlagName, Str)                         \
  if (FlagName)                                                                \
    (OS << ";").indent(7) << Str << "\n";
#include "llvm/BinaryFormat/DXContainerConstants.def"
  OS << ";\n";
}

/// Return the shader flags mask of the specified function Func.
const ComputedShaderFlags &
ModuleShaderFlags::getFunctionFlags(const Function *Func) const {
  auto Iter = FunctionFlags.find(Func);
  assert((Iter != FunctionFlags.end() && Iter->first == Func) &&
         "Get Shader Flags : No Shader Flags Mask exists for function");
  return Iter->second;
}

//===----------------------------------------------------------------------===//
// ShaderFlagsAnalysis and ShaderFlagsAnalysisPrinterPass

// Provide an explicit template instantiation for the static ID.
AnalysisKey ShaderFlagsAnalysis::Key;

ModuleShaderFlags ShaderFlagsAnalysis::run(Module &M,
                                           ModuleAnalysisManager &AM) {
  DXILResourceTypeMap &DRTM = AM.getResult<DXILResourceTypeAnalysis>(M);
  DXILResourceMap &DRM = AM.getResult<DXILResourceAnalysis>(M);
  const ModuleMetadataInfo MMDI = AM.getResult<DXILMetadataAnalysis>(M);

  ModuleShaderFlags MSFI;
  MSFI.initialize(M, DRTM, DRM, MMDI);

  return MSFI;
}

PreservedAnalyses ShaderFlagsAnalysisPrinter::run(Module &M,
                                                  ModuleAnalysisManager &AM) {
  const ModuleShaderFlags &FlagsInfo = AM.getResult<ShaderFlagsAnalysis>(M);
  // Print description of combined shader flags for all module functions
  OS << "; Combined Shader Flags for Module\n";
  FlagsInfo.getCombinedFlags().print(OS);
  // Print shader flags mask for each of the module functions
  OS << "; Shader Flags for Module Functions\n";
  for (const auto &F : M.getFunctionList()) {
    if (F.isDeclaration())
      continue;
    const ComputedShaderFlags &SFMask = FlagsInfo.getFunctionFlags(&F);
    OS << formatv("; Function {0} : {1:x8}\n;\n", F.getName(),
                  (uint64_t)(SFMask));
  }

  return PreservedAnalyses::all();
}

//===----------------------------------------------------------------------===//
// ShaderFlagsAnalysis and ShaderFlagsAnalysisPrinterPass

bool ShaderFlagsAnalysisWrapper::runOnModule(Module &M) {
  DXILResourceTypeMap &DRTM =
      getAnalysis<DXILResourceTypeWrapperPass>().getResourceTypeMap();
  DXILResourceMap &DRM =
      getAnalysis<DXILResourceWrapperPass>().getResourceMap();
  const ModuleMetadataInfo MMDI =
      getAnalysis<DXILMetadataAnalysisWrapperPass>().getModuleMetadata();

  MSFI.initialize(M, DRTM, DRM, MMDI);
  return false;
}

void ShaderFlagsAnalysisWrapper::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<DXILResourceTypeWrapperPass>();
  AU.addRequiredTransitive<DXILResourceWrapperPass>();
  AU.addRequired<DXILMetadataAnalysisWrapperPass>();
}

char ShaderFlagsAnalysisWrapper::ID = 0;

INITIALIZE_PASS_BEGIN(ShaderFlagsAnalysisWrapper, "dx-shader-flag-analysis",
                      "DXIL Shader Flag Analysis", true, true)
INITIALIZE_PASS_DEPENDENCY(DXILResourceTypeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DXILMetadataAnalysisWrapperPass)
INITIALIZE_PASS_END(ShaderFlagsAnalysisWrapper, "dx-shader-flag-analysis",
                    "DXIL Shader Flag Analysis", true, true)
