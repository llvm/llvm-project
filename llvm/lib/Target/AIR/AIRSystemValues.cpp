//===- AIRSystemValues.cpp - AIR system-value lowering --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRSystemValues.h"
#include "AIRWriter/AIRVersion.h"
#include "AIRWriter/PointeeTypeMap.h"
#include "AIR.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

#include <cstdio>

using namespace llvm;

#define DEBUG_TYPE "air-air-system-values"

// AIR system value intrinsic-name prefixes.
static constexpr StringLiteral kCallPid("air.threadgroup_position_in_grid");
static constexpr StringLiteral kCallTid("air.thread_position_in_grid");
static constexpr StringLiteral kCallTidTG("air.thread_position_in_threadgroup");
static constexpr StringLiteral kCallSimdlane("air.thread_index_in_simdgroup");
static constexpr StringLiteral kCallNumProg("air.threadgroups_per_grid");
static constexpr StringLiteral kCallTGSize("air.threads_per_threadgroup");
static constexpr StringLiteral kCallSimdSize("air.threads_per_simdgroup");
static constexpr StringLiteral kCallGridSize("air.threads_per_grid");
static constexpr StringLiteral
    kCallTGIndex("air.thread_index_in_threadgroup");
static constexpr StringLiteral
    kCallNumSimd("air.simdgroups_per_threadgroup");
static constexpr StringLiteral
    kCallSimdId("air.simdgroup_index_in_threadgroup");

// AIR metadata-tag strings.
static constexpr StringLiteral
    kMDThreadPositionInGrid("air.thread_position_in_grid");
static constexpr StringLiteral
    kMDThreadgroupPositionInGrid("air.threadgroup_position_in_grid");
static constexpr StringLiteral
    kMDThreadPositionInTG("air.thread_position_in_threadgroup");
static constexpr StringLiteral
    kMDThreadIndexInSimdgroup("air.thread_index_in_simdgroup");
static constexpr StringLiteral
    kMDThreadgroupsPerGrid("air.threadgroups_per_grid");
static constexpr StringLiteral
    kMDThreadsPerThreadgroup("air.threads_per_threadgroup");
static constexpr StringLiteral
    kMDThreadsPerSimdgroup("air.threads_per_simdgroup");
static constexpr StringLiteral kMDThreadsPerGrid("air.threads_per_grid");
static constexpr StringLiteral
    kMDThreadIndexInThreadgroup("air.thread_index_in_threadgroup");
static constexpr StringLiteral
    kMDSimdgroupsPerThreadgroup("air.simdgroups_per_threadgroup");
static constexpr StringLiteral
    kMDSimdgroupIndexInThreadgroup("air.simdgroup_index_in_threadgroup");

static constexpr StringLiteral kMDBuffer("air.buffer");
static constexpr StringLiteral kMDLocationIndex("air.location_index");
static constexpr StringLiteral kMDRead("air.read");
static constexpr StringLiteral kMDReadWrite("air.read_write");
static constexpr StringLiteral kMDAddressSpace("air.address_space");
static constexpr StringLiteral kMDArgTypeSize("air.arg_type_size");
static constexpr StringLiteral kMDArgTypeAlignSize("air.arg_type_align_size");
static constexpr StringLiteral kMDArgTypeName("air.arg_type_name");
static constexpr StringLiteral kMDArgName("air.arg_name");

static constexpr StringLiteral kNMDKernel("air.kernel");
static constexpr StringLiteral kNMDVersion("air.version");
static constexpr StringLiteral kNMDLanguageVersion("air.language_version");

namespace {
struct SysValParam {
  StringRef IntrinsicName;
  StringRef ParamName;
  StringRef MDTag; // the parameter's role in !air.kernel
  bool IsVector;   // true = <3 x i32>, false = i32
  const char *Dims[3];
};

/// A system-value parameter this pass added to a kernel.
struct AddedSysVal {
  unsigned ArgNo;
  const SysValParam *Info;
};
} // namespace

static const SysValParam kSysVals[] = {
    {kCallPid, "pid", kMDThreadgroupPositionInGrid, true,
     {"pid_x", "pid_y", "pid_z"}},
    {kCallTid, "tid", kMDThreadPositionInGrid, true,
     {"tid_x", "tid_y", "tid_z"}},
    {kCallTidTG, "tidtg", kMDThreadPositionInTG, true,
     {"tidtg_x", "tidtg_y", "tidtg_z"}},
    {kCallSimdlane, "simdlane", kMDThreadIndexInSimdgroup, false,
     {"simdlane", "", ""}},
    {kCallNumProg, "numprog", kMDThreadgroupsPerGrid, true,
     {"numprog_x", "numprog_y", "numprog_z"}},
    {kCallTGSize, "tgsize", kMDThreadsPerThreadgroup, true,
     {"tgsize_x", "tgsize_y", "tgsize_z"}},
    {kCallSimdSize, "simdsize", kMDThreadsPerSimdgroup, false,
     {"simdsize", "", ""}},
    {kCallNumSimd, "numsimd", kMDSimdgroupsPerThreadgroup, false,
     {"numsimd", "", ""}},
    {kCallSimdId, "simdid", kMDSimdgroupIndexInThreadgroup, false,
     {"simdid", "", ""}},
    {kCallGridSize, "gridsize", kMDThreadsPerGrid, true,
     {"gridsize_x", "gridsize_y", "gridsize_z"}},
    {kCallTGIndex, "tgindex", kMDThreadIndexInThreadgroup, false,
     {"tgindex", "", ""}},
};

namespace {
/// How !air.kernel describes the type a buffer argument points to.
struct BufferType {
  StringRef Name = "float";
  unsigned Size = 4;
  unsigned Align = 4;
};
} // namespace

// The MSL name of a scalar or of a 2-, 3- or 4-element vector type, or an
// empty string for any other type.
static std::string mslTypeName(Type *T) {
  unsigned N = 1;
  if (auto *VT = dyn_cast<FixedVectorType>(T)) {
    N = VT->getNumElements();
    if (N < 2 || N > 4)
      return "";
    T = VT->getElementType();
  }
  StringRef Name;
  if (T->isHalfTy())
    Name = "half";
  else if (T->isBFloatTy())
    Name = "bfloat";
  else if (T->isFloatTy())
    Name = "float";
  else if (T->isDoubleTy())
    Name = "double";
  else if (T->isIntegerTy(8))
    Name = "char";
  else if (T->isIntegerTy(16))
    Name = "short";
  else if (T->isIntegerTy(32))
    Name = "int";
  else if (T->isIntegerTy(64))
    Name = "long";
  else
    return "";
  return N == 1 ? Name.str() : (Name + Twine(N)).str();
}

// The type a buffer argument points to, from the loads and stores through it
// (as the AIR writer infers it). A pointee that is not a scalar or a short
// vector, or none at all, keeps the default description (float, 4 bytes).
static BufferType describeBuffer(Argument &Arg, const DataLayout &DL,
                                 StringSaver &Names) {
  BufferType BT;
  Type *Pointee = metal::PointeeTypeMap::inferFromUsage(&Arg);
  if (!Pointee)
    return BT;
  std::string Name = mslTypeName(Pointee);
  if (Name.empty())
    return BT;
  BT.Name = Names.save(Name);
  BT.Size = DL.getTypeAllocSize(Pointee);
  BT.Align = DL.getABITypeAlign(Pointee).value();
  return BT;
}

static Function *findDeclByPrefix(Module &M, StringRef Prefix) {
  for (Function &F : M)
    if (F.isDeclaration() && F.getName().starts_with(Prefix))
      return &F;
  return nullptr;
}

static bool airSystemValues(Module &M) {
  auto &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *Vec3 = FixedVectorType::get(I32, 3);
  const DataLayout &DL = M.getDataLayout();
  BumpPtrAllocator Alloc;
  StringSaver Names(Alloc);

  bool Changed = false;
  // The system-value parameters added to each kernel. They, not argument
  // names, say which parameters are system values: a kernel's own argument
  // may be named tid.
  DenseMap<Function *, SmallVector<AddedSysVal, 4>> SysValArgs;

  SmallVector<Function *, 4> Funcs;
  for (Function &F : M)
    if (!F.isDeclaration())
      Funcs.push_back(&F);

  for (Function *FPtr : Funcs) {
    Function &F = *FPtr;

    struct UsedSysVal {
      const SysValParam *Info;
      SmallVector<CallInst *, 4> Calls;
    };
    SmallVector<UsedSysVal, 5> UsedSysVals;

    for (const SysValParam &SV : kSysVals) {
      Function *Decl = findDeclByPrefix(M, SV.IntrinsicName);
      if (!Decl)
        continue;
      SmallVector<CallInst *, 4> Calls;
      for (BasicBlock &BB : F)
        for (Instruction &I : BB)
          if (auto *CI = dyn_cast<CallInst>(&I))
            if (CI->getCalledFunction() == Decl)
              Calls.push_back(CI);
      if (!Calls.empty())
        UsedSysVals.push_back({&SV, std::move(Calls)});
    }

    if (UsedSysVals.empty())
      continue;

    // Build a new function type with extra params at the end.
    auto *OldFTy = F.getFunctionType();
    SmallVector<Type *, 8> NewParamTypes(OldFTy->params());
    for (const UsedSysVal &USV : UsedSysVals)
      NewParamTypes.push_back(USV.Info->IsVector ? Vec3 : I32);

    auto *NewFTy = FunctionType::get(OldFTy->getReturnType(), NewParamTypes,
                                     OldFTy->isVarArg());

    auto *NewF =
        Function::Create(NewFTy, F.getLinkage(), F.getAddressSpace(), "", &M);
    NewF->copyAttributesFrom(&F);
    NewF->copyMetadata(&F, 0);
    NewF->splice(NewF->begin(), &F);

    auto NewArgIt = NewF->arg_begin();
    for (Argument &OldArg : F.args()) {
      NewArgIt->setName(OldArg.getName());
      OldArg.replaceAllUsesWith(&*NewArgIt);
      ++NewArgIt;
    }

    // Keep metadata that names the kernel (e.g. a pre-baked !air.kernel)
    // pointing at the rewritten function; erasing F would null it out.
    ValueAsMetadata::handleRAUW(&F, NewF);

    std::string FName = F.getName().str();
    F.eraseFromParent();
    NewF->setName(FName);

    BasicBlock &EntryBB = NewF->getEntryBlock();

    for (UsedSysVal &USV : UsedSysVals) {
      Argument *SysArg = &*NewArgIt++;
      SysArg->setName(USV.Info->ParamName);
      SysValArgs[NewF].push_back({SysArg->getArgNo(), USV.Info});

      if (USV.Info->IsVector) {
        Value *Components[3];
        for (int K = 0; K < 3; ++K) {
          IRBuilder<> B(&*EntryBB.begin());
          Components[K] = B.CreateExtractElement(
              SysArg, ConstantInt::get(I32, K), USV.Info->Dims[K]);
        }
        for (CallInst *CI : USV.Calls) {
          // The sysval call may return a <3 x i32> vector (extractelement
          // users) instead of the [3 x i32] array aggregate (extractvalue
          // users).  SysArg is the vector form, so it is the exact replacement;
          // the array reconstruction below would assert on a vector type.
          if (CI->getType()->isVectorTy()) {
            CI->replaceAllUsesWith(SysArg);
            CI->eraseFromParent();
            continue;
          }
          SmallVector<ExtractValueInst *, 4> Extracts;
          for (User *U : CI->users())
            if (auto *EV = dyn_cast<ExtractValueInst>(U))
              Extracts.push_back(EV);
          for (ExtractValueInst *EV : Extracts) {
            unsigned Idx = EV->getIndices()[0];
            EV->replaceAllUsesWith(Components[std::min(Idx, 2u)]);
            EV->eraseFromParent();
          }
          // Any remaining users (e.g. a `freeze [3 x i32]` inserted by the
          // mid-end between the call and its extractvalue, or the call passed
          // around as a whole aggregate) must see the REAL system value, not
          // undef.  Reconstruct the [3 x i32] aggregate from Components and
          // replace the call with it.  Building it right before the call keeps
          // it dominating every use.  (Previously this RAUW'd undef, which
          // poisoned a freeze-separated extractvalue -> garbage div/rem inputs
          // for any kernel whose program-id / thread-position feeds integer
          // math; the freeze appears at -O2/-O3.)
          if (!CI->use_empty()) {
            IRBuilder<> B(CI);
            Value *Agg = UndefValue::get(CI->getType());
            for (unsigned K = 0; K < 3; ++K)
              Agg = B.CreateInsertValue(Agg, Components[K], {K});
            CI->replaceAllUsesWith(Agg);
          }
          CI->eraseFromParent();
        }
      } else {
        for (CallInst *CI : USV.Calls) {
          CI->replaceAllUsesWith(SysArg);
          CI->eraseFromParent();
        }
      }
    }

    Changed = true;
  }

  // Remove unused intrinsic declarations.
  for (const SysValParam &SV : kSysVals) {
    if (auto *F = findDeclByPrefix(M, SV.IntrinsicName))
      if (F->use_empty())
        F->eraseFromParent();
  }

  // Emit !air.kernel metadata (always; skip if pre-baked).
  if (!M.getNamedMetadata(kNMDKernel)) {
    auto *KernelMD = M.getOrInsertNamedMetadata(kNMDKernel);

    for (Function &F : M) {
      if (F.isDeclaration())
        continue;

      bool IsCalled = false;
      for (User *U : F.users()) {
        if (auto *CB = dyn_cast<CallBase>(U)) {
          if (CB->getCalledFunction() == &F) {
            IsCalled = true;
            break;
          }
        }
      }
      if (IsCalled)
        continue;

      SmallVector<Metadata *, 16> ParamNodes;
      unsigned ArgIdx = 0;  // device/constant buffer location-index space
      unsigned TGIdx = 0;   // threadgroup buffers have their own index space
      auto *FTy = F.getFunctionType();

      bool HasArgBuf = false;
      if (auto *MAS = F.getMetadata("air.argbuf.member_as")) {
        HasArgBuf = true;
        unsigned PtrSize = M.getDataLayout().getPointerSize(/*AS=*/1);
        unsigned PtrAlign = PtrSize;
        SmallVector<Metadata *, 32> StructInfo;
        for (unsigned Fld = 0; Fld < MAS->getNumOperands(); ++Fld) {
          unsigned MemAS =
              mdconst::extract<ConstantInt>(MAS->getOperand(Fld))->getZExtValue();
          unsigned Offset = Fld * PtrSize;
          char NameBuf[16];
          std::snprintf(NameBuf, sizeof(NameBuf), "%u", Fld);
          StringRef MemName(NameBuf);
          StringRef AccessMode = (MemAS == 2) ? kMDRead : kMDReadWrite;
          MDNode *Detail = MDNode::get(
              Ctx,
              {ConstantAsMetadata::get(ConstantInt::get(I32, Fld)),
               MDString::get(Ctx, kMDBuffer),
               MDString::get(Ctx, kMDLocationIndex),
               ConstantAsMetadata::get(ConstantInt::get(I32, Fld)),
               ConstantAsMetadata::get(ConstantInt::get(I32, 1)),
               MDString::get(Ctx, AccessMode),
               MDString::get(Ctx, kMDAddressSpace),
               ConstantAsMetadata::get(ConstantInt::get(I32, MemAS)),
               MDString::get(Ctx, kMDArgTypeSize),
               ConstantAsMetadata::get(ConstantInt::get(I32, 4)),
               MDString::get(Ctx, kMDArgTypeAlignSize),
               ConstantAsMetadata::get(ConstantInt::get(I32, 4)),
               MDString::get(Ctx, kMDArgTypeName), MDString::get(Ctx, "float"),
               MDString::get(Ctx, kMDArgName), MDString::get(Ctx, MemName)});
          StructInfo.push_back(
              ConstantAsMetadata::get(ConstantInt::get(I32, Offset)));
          StructInfo.push_back(
              ConstantAsMetadata::get(ConstantInt::get(I32, PtrSize)));
          StructInfo.push_back(ConstantAsMetadata::get(ConstantInt::get(I32, 0)));
          StructInfo.push_back(MDString::get(Ctx, "float"));
          StructInfo.push_back(MDString::get(Ctx, MemName));
          StructInfo.push_back(MDString::get(Ctx, "air.indirect_argument"));
          StructInfo.push_back(Detail);
        }
        MDNode *StructTypeInfo = MDNode::get(Ctx, StructInfo);
        unsigned BufSize = MAS->getNumOperands() * PtrSize;
        ParamNodes.push_back(MDNode::get(
            Ctx,
            {ConstantAsMetadata::get(ConstantInt::get(I32, 0)),
             MDString::get(Ctx, "air.indirect_buffer"),
             MDString::get(Ctx, "air.buffer_size"),
             ConstantAsMetadata::get(ConstantInt::get(I32, BufSize)),
             MDString::get(Ctx, kMDLocationIndex),
             ConstantAsMetadata::get(ConstantInt::get(I32, 0)),
             ConstantAsMetadata::get(ConstantInt::get(I32, 1)),
             MDString::get(Ctx, kMDRead),
             MDString::get(Ctx, kMDAddressSpace),
             ConstantAsMetadata::get(ConstantInt::get(I32, 2)),
             MDString::get(Ctx, "air.struct_type_info"), StructTypeInfo,
             MDString::get(Ctx, kMDArgTypeSize),
             ConstantAsMetadata::get(ConstantInt::get(I32, BufSize)),
             MDString::get(Ctx, kMDArgTypeAlignSize),
             ConstantAsMetadata::get(ConstantInt::get(I32, PtrAlign)),
             MDString::get(Ctx, kMDArgTypeName), MDString::get(Ctx, "argbuf"),
             MDString::get(Ctx, kMDArgName), MDString::get(Ctx, "argbuf")}));
      }

      // Buffer params: device AS=1, constant AS=2, threadgroup AS=3. Apple
      // tags all three as "air.buffer" but threadgroup args carry
      // air.address_space=3 and live in a separate location_index counter
      // (xcrun's reduce gives the 3rd buffer, a threadgroup arg, index 0).
      for (unsigned I = 0; I < FTy->getNumParams(); ++I) {
        if (HasArgBuf && I == 0)
          continue;
        Type *ParamTy = FTy->getParamType(I);
        if (!ParamTy->isPointerTy())
          continue;
        unsigned AS = cast<PointerType>(ParamTy)->getAddressSpace();
        if (AS != 1 && AS != 2 && AS != 3)
          continue;

        BufferType BT = describeBuffer(*F.getArg(I), DL, Names);
        StringRef ArgName = F.getArg(I)->getName();
        char NameBuf[16];
        if (ArgName.empty()) {
          std::snprintf(NameBuf, sizeof(NameBuf), "%u", I);
          ArgName = NameBuf;
        }
        unsigned &LocIdx = (AS == 3) ? TGIdx : ArgIdx;
        StringRef AccessMode = (AS == 2) ? kMDRead : kMDReadWrite;
        ParamNodes.push_back(MDNode::get(
            Ctx,
            {ConstantAsMetadata::get(ConstantInt::get(I32, I)),
             MDString::get(Ctx, kMDBuffer),
             MDString::get(Ctx, kMDLocationIndex),
             ConstantAsMetadata::get(ConstantInt::get(I32, LocIdx)),
             ConstantAsMetadata::get(ConstantInt::get(I32, 1)),
             MDString::get(Ctx, AccessMode),
             MDString::get(Ctx, kMDAddressSpace),
             ConstantAsMetadata::get(ConstantInt::get(I32, AS)),
             MDString::get(Ctx, kMDArgTypeSize),
             ConstantAsMetadata::get(ConstantInt::get(I32, BT.Size)),
             MDString::get(Ctx, kMDArgTypeAlignSize),
             ConstantAsMetadata::get(ConstantInt::get(I32, BT.Align)),
             MDString::get(Ctx, kMDArgTypeName), MDString::get(Ctx, BT.Name),
             MDString::get(Ctx, kMDArgName), MDString::get(Ctx, ArgName)}));
        LocIdx++;
      }

      // System-value params, in parameter order.
      auto It = SysValArgs.find(&F);
      if (It != SysValArgs.end()) {
        for (const AddedSysVal &SV : It->second) {
          StringRef TypeName = SV.Info->IsVector ? "uint3" : "uint";
          ParamNodes.push_back(MDNode::get(
              Ctx, {ConstantAsMetadata::get(ConstantInt::get(I32, SV.ArgNo)),
                    MDString::get(Ctx, SV.Info->MDTag),
                    MDString::get(Ctx, kMDArgTypeName),
                    MDString::get(Ctx, TypeName), MDString::get(Ctx, kMDArgName),
                    MDString::get(Ctx, F.getArg(SV.ArgNo)->getName())}));
        }
      }

      auto *EmptyNode = MDNode::get(Ctx, {});
      auto *ParamsNode = MDNode::get(Ctx, ParamNodes);
      KernelMD->addOperand(
          MDNode::get(Ctx, {ValueAsMetadata::get(&F), EmptyNode, ParamsNode}));
    }

    Changed = true;
  }

  // Version metadata: air.version = (2, AIRMinor, 0). The minor is driven by
  // the target macOS major (OSmajor-8), derived from the module triple and
  // falling back to macOS 16 when absent. Empirically verified against Apple's
  // `xcrun metal -mmacosx-version-min=N`.
  auto AIRVer = metal::AIRVersion::fromTriple(M.getTargetTriple().str());

  // The canonical 8-arg device-atomic form (metal::_atomic arg) is only
  // accepted by the Metal compiler at air.version >= (2,9,0) / MSL 4.1. Bump
  // the version for modules that use it; a stale (2,8,0) stamp PSO-crashes the
  // compiler service. Other modules keep their target-derived version.
  unsigned AIRMinor = AIRVer.AIRMinor;
  unsigned MSLMajor = AIRVer.MSLMajor, MSLMinor = AIRVer.MSLMinor;
  if (auto *F = M.getFunction("air.atomic.global.cmpxchg.weak.i32"))
    if (F->getFunctionType()->getNumParams() == 8 && AIRMinor < 9) {
      AIRMinor = 9;
      MSLMajor = 4;
      MSLMinor = 1;
      // The target's own version cannot express the call, and a system
      // without MSL 4.1 rejects the module ("expecting 2.8").
      Ctx.diagnose(DiagnosticInfoGeneric(
          "air.atomic.global.cmpxchg.weak.i32 with memory flags needs AIR "
          "2.9 (MSL 4.1); raising air.version above the target's, so the "
          "library loads only on systems with MSL 4.1",
          DS_Warning));
    }

  auto *VerMD = M.getOrInsertNamedMetadata(kNMDVersion);
  if (VerMD->getNumOperands() == 0) {
    VerMD->addOperand(MDNode::get(
        Ctx, {ConstantAsMetadata::get(
                  ConstantInt::get(I32, metal::AIRVersion::AIRMajor)),
              ConstantAsMetadata::get(ConstantInt::get(I32, AIRMinor)),
              ConstantAsMetadata::get(ConstantInt::get(I32, 0))}));
    Changed = true;
  }

  // Language version metadata = ("Metal", MSLMajor, MSLMinor, 0). The MSL
  // version the target macOS supports: 13->3.0, 14->3.1, 15->3.2, 16->4.0.
  if (!M.getNamedMetadata(kNMDLanguageVersion)) {
    auto *LangMD = M.getOrInsertNamedMetadata(kNMDLanguageVersion);
    LangMD->addOperand(MDNode::get(
        Ctx, {MDString::get(Ctx, "Metal"),
              ConstantAsMetadata::get(ConstantInt::get(I32, MSLMajor)),
              ConstantAsMetadata::get(ConstantInt::get(I32, MSLMinor)),
              ConstantAsMetadata::get(ConstantInt::get(I32, 0))}));
    Changed = true;
  }

  // air.compile_options — Apple's `xcrun metal` always emits these three
  // top-level options. The stricter macOS 13/14/15 AIR driver rejects AIR
  // that lacks them ("Compiler encountered an internal error"); macOS 26 is
  // lenient. Empirically required for the older-OS path.
  if (!M.getNamedMetadata("air.compile_options")) {
    auto *OptsMD = M.getOrInsertNamedMetadata("air.compile_options");
    OptsMD->addOperand(
        MDNode::get(Ctx, {MDString::get(Ctx, "air.compile.denorms_disable")}));
    OptsMD->addOperand(
        MDNode::get(Ctx, {MDString::get(Ctx, "air.compile.fast_math_enable")}));
    OptsMD->addOperand(MDNode::get(
        Ctx, {MDString::get(Ctx, "air.compile.framebuffer_fetch_enable")}));
    Changed = true;
  }

  // Module flags: air.max_* limits.
  if (!M.getModuleFlag("air.max_device_buffers")) {
    M.addModuleFlag(Module::Max, "air.max_device_buffers",
                    ConstantInt::get(I32, 31));
    M.addModuleFlag(Module::Max, "air.max_constant_buffers",
                    ConstantInt::get(I32, 31));
    M.addModuleFlag(Module::Max, "air.max_threadgroup_buffers",
                    ConstantInt::get(I32, 31));
    M.addModuleFlag(Module::Max, "air.max_textures",
                    ConstantInt::get(I32, 128));
    M.addModuleFlag(Module::Max, "air.max_read_write_textures",
                    ConstantInt::get(I32, 8));
    M.addModuleFlag(Module::Max, "air.max_samplers", ConstantInt::get(I32, 16));
  }

  return Changed;
}

PreservedAnalyses AIRSystemValuesPass::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  return airSystemValues(M) ? PreservedAnalyses::none()
                            : PreservedAnalyses::all();
}

bool AIRSystemValuesLegacy::runOnModule(Module &M) {
  return airSystemValues(M);
}

char AIRSystemValuesLegacy::ID = 0;

INITIALIZE_PASS(AIRSystemValuesLegacy, DEBUG_TYPE,
                "AIR System Values", false, false)

ModulePass *llvm::createAIRSystemValuesLegacyPass() {
  return new AIRSystemValuesLegacy();
}
