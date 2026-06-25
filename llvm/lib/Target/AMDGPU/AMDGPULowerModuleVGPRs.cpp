//===- AMDGPULowerModuleVGPRs.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lays out a module's "VGPR as memory" (addrspace(13)) globals into one shared
// register "file" and records where it lives on every function whose call graph
// uses it.
//
// The file is backed by a fixed block of physical VGPRs, so for an address into
// it to be meaningful across calls every function in the call graph must agree
// on (a) each global's byte offset and (b) the file's base register. The
// backend can derive a base per function (just above its ABI inputs) but those
// differ, so (b) is resolved module-wide:
//
//  * Offsets: all globals are packed into one deterministic layout; each
//    global's byte offset is recorded as "amdgpu.vgpr.memory.offset" metadata.
//  * Base: one index, the max ABI-input VGPR boundary over all participating
//    functions, so it clears every function's inputs yet stays as low as
//    possible to preserve occupancy.
//
// The file size and base are attached as the "amdgpu-vgpr-memory-size" and
// "amdgpu-vgpr-memory-base" attributes to every function whose call graph uses
// the file (like LDS, it is live for a using kernel's whole execution, so all
// reachable functions must reserve it). The backend consumes these:
// SIISelLowering reads the offset metadata; SIMachineFunctionInfo reads the
// attributes; SIRegisterInfo::getVGPRMemoryFile reserves [base, base + size).
//
// TODO: one module-wide layout makes every using function reserve all globals,
// and a function reachable from several kernels reserve the file even for a
// kernel that does not use it. A per-kernel layout (as AMDGPULowerModuleLDS
// does, with a table for shared callees) would tighten this.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-lower-module-vgprs"

namespace {

constexpr char SizeAttr[] = "amdgpu-vgpr-memory-size";
constexpr char BaseAttr[] = "amdgpu-vgpr-memory-base";
constexpr char OffsetMD[] = "amdgpu.vgpr.memory.offset";

// The fixed device-function ABI keeps the work-item ID in this register
// (SITargetLowering::allocateSpecialInputVGPRsFixed). The shared file must not
// overlap it.
constexpr unsigned FixedWorkitemRegIdx = 31;

// True if F may read the work-item ID (and so needs its work-item-ID input
// register), per the amdgpu-no-workitem-id-* attributes.
static bool usesWorkitemID(const Function &F) {
  return !F.hasFnAttribute("amdgpu-no-workitem-id-x") ||
         !F.hasFnAttribute("amdgpu-no-workitem-id-y") ||
         !F.hasFnAttribute("amdgpu-no-workitem-id-z");
}

// Upper bound on the low contiguous VGPRs occupied by F's ABI inputs - the
// registers the shared file must sit above. The fixed device-function ABI also
// keeps the work-item ID in the high register v31 (see usesFixedWorkitemReg);
// that is modelled separately, not counted here.
static unsigned inputVGPRBound(const Function &F) {
  // Compute kernels take args in the kernarg segment, not VGPRs; their only
  // VGPR input is the work-item ID, packed into a single low register.
  if (AMDGPU::isKernel(F.getCallingConv()))
    return usesWorkitemID(F) ? 1 : 0;

  // Graphics entry points and ordinary functions pass their arguments in VGPRs
  // (except inreg arguments, which go in SGPRs).
  const DataLayout &DL = F.getParent()->getDataLayout();
  unsigned N = 0;
  for (const Argument &A : F.args()) {
    if (A.hasAttribute(Attribute::InReg))
      continue;
    unsigned Dwords =
        divideCeil(DL.getTypeAllocSize(A.getType()).getFixedValue(), 4u);
    // A multi-dword argument tuple is even-aligned on targets that require
    // aligned VGPR tuples. Model that gap conservatively so the shared base
    // never lands below such an argument register (the backend's overlap check
    // in getVGPRMemoryFile is the backstop if this is ever too low).
    if (Dwords > 1)
      N = alignTo(N, 2u);
    N += Dwords;
  }
  return N;
}

// True if F is a callable (non-entry) device function on the default ABI, which
// keeps the work-item ID in the fixed high register v31. The shared file must
// not overlap v31 in such a function.
static bool usesFixedWorkitemReg(const Function &F) {
  CallingConv::ID CC = F.getCallingConv();
  return !AMDGPU::isEntryFunctionCC(CC) && !AMDGPU::isGraphics(CC) &&
         usesWorkitemID(F);
}

class AMDGPULowerModuleVGPRs : public ModulePass {
public:
  static char ID;
  AMDGPULowerModuleVGPRs() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

  StringRef getPassName() const override { return "AMDGPU Lower Module VGPRs"; }
};

} // end anonymous namespace

char AMDGPULowerModuleVGPRs::ID = 0;
char &llvm::AMDGPULowerModuleVGPRsID = AMDGPULowerModuleVGPRs::ID;

INITIALIZE_PASS(AMDGPULowerModuleVGPRs, DEBUG_TYPE, "AMDGPU Lower Module VGPRs",
                false, false)

ModulePass *llvm::createAMDGPULowerModuleVGPRsPass() {
  return new AMDGPULowerModuleVGPRs();
}

static bool lowerModuleVGPRs(Module &M) {
  SmallVector<GlobalVariable *, 8> Globals;
  for (GlobalVariable &GV : M.globals())
    if (GV.getAddressSpace() == AMDGPUAS::VGPR)
      Globals.push_back(&GV);
  if (Globals.empty())
    return false;

  // In one walk over each defined function, map it to the addrspace(13) globals
  // it directly references and collect its ordinary calls (non-intrinsic,
  // non-inline-asm) for the later reserved-register-clobber check, so the
  // module is not traversed twice.
  DenseMap<Function *, SmallVector<GlobalVariable *, 2>> Uses;
  DenseMap<Function *, SmallVector<const CallBase *, 2>> Calls;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    SmallPtrSet<GlobalVariable *, 4> Seen;
    for (Instruction &I : instructions(F)) {
      if (const auto *CB = dyn_cast<CallBase>(&I))
        if (!CB->isInlineAsm() &&
            CB->getIntrinsicID() == Intrinsic::not_intrinsic)
          Calls[&F].push_back(CB);
      for (Value *Op : I.operands()) {
        // Only pointer operands can name a global; skipping the rest avoids a
        // getUnderlyingObject call per non-pointer operand on every compile.
        if (!Op->getType()->isPtrOrPtrVectorTy())
          continue;
        // getUnderlyingObject sees through constant-expression GEPs/casts, so
        // a global referenced via e.g. getelementptr(@g, off) is found.
        auto *GV = dyn_cast<GlobalVariable>(getUnderlyingObject(Op));
        if (GV && GV->getAddressSpace() == AMDGPUAS::VGPR &&
            Seen.insert(GV).second)
          Uses[&F].push_back(GV);
      }
    }
  }
  if (Uses.empty())
    return true; // nothing references the file

  CallGraph CG(M);
  auto Reachable = [&](Function *Root, SmallPtrSetImpl<Function *> &Out) {
    SmallVector<Function *, 16> Work{Root};
    while (!Work.empty()) {
      Function *F = Work.pop_back_val();
      if (!Out.insert(F).second)
        continue;
      if (CallGraphNode *N = CG[F])
        for (auto &CR : *N)
          if (Function *Callee = CR.second->getFunction())
            if (!Callee->isDeclaration())
              Work.push_back(Callee);
    }
  };

  // Partition functions and globals into independent layout groups: a group
  // covers everything reachable from a using kernel (the file is live for its
  // whole execution, like LDS) plus every function that uses each global. So
  // disjoint kernels get independent (low, occupancy-friendly) bases while
  // shared functions stay in one group. Functions and globals are both
  // GlobalValues, so one union-find covers both.
  EquivalenceClasses<const GlobalValue *> Groups;
  for (auto &[F, GVs] : Uses)
    for (GlobalVariable *GV : GVs)
      Groups.unionSets(F, GV);

  // Functions reachable from each file-using kernel join that kernel's group
  // (so they reserve the file), and kernels sharing any callee merge.
  for (Function &K : M) {
    if (K.isDeclaration() || !AMDGPU::isEntryFunctionCC(K.getCallingConv()))
      continue;
    SmallPtrSet<Function *, 16> R;
    Reachable(&K, R);
    if (llvm::none_of(R, [&](Function *F) { return Uses.count(F); }))
      continue; // this kernel does not use the file
    for (Function *F : R)
      Groups.unionSets(&K, F);
  }

  const DataLayout &DL = M.getDataLayout();
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);

  // Lay out each group independently.
  for (auto It = Groups.begin(), E = Groups.end(); It != E; ++It) {
    const auto *Leader = *It;
    if (!Leader->isLeader())
      continue;
    SmallVector<GlobalVariable *, 8> GroupGlobals;
    SmallVector<Function *, 16> GroupFns;
    for (auto MI = Groups.member_begin(*Leader); MI != Groups.member_end();
         ++MI) {
      const GlobalValue *GV = *MI;
      if (auto *G = dyn_cast<GlobalVariable>(GV))
        GroupGlobals.push_back(const_cast<GlobalVariable *>(G));
      else
        GroupFns.push_back(const_cast<Function *>(cast<Function>(GV)));
    }
    if (GroupGlobals.empty() || GroupFns.empty())
      continue;

    // Deterministic packed layout (sorted by name).
    llvm::stable_sort(GroupGlobals, [](GlobalVariable *A, GlobalVariable *B) {
      return A->getName() < B->getName();
    });
    unsigned Size = 0;
    for (GlobalVariable *GV : GroupGlobals) {
      Align A = std::max(
          DL.getValueOrABITypeAlignment(GV->getAlign(), GV->getValueType()),
          Align(4));
      unsigned Offset = alignTo(Size, A);
      GV->setMetadata(OffsetMD,
                      MDNode::get(Ctx, {ConstantAsMetadata::get(
                                           ConstantInt::get(I32, Offset))}));
      Size = Offset + DL.getTypeAllocSize(GV->getValueType()).getFixedValue();
    }

    // One base for the group: above every member's low ABI-input VGPRs,
    // even-aligned.
    unsigned Base = 0;
    bool ClearsFixedWorkitem = false;
    for (Function *F : GroupFns) {
      Base = std::max(Base, inputVGPRBound(*F));
      ClearsFixedWorkitem |= usesFixedWorkitemReg(*F);
    }
    Base = alignTo(Base, 2u);

    // The fixed device-function ABI keeps the work-item ID in v31. A small file
    // sits below it; if the file would grow into v31, place it above instead
    // (at an occupancy cost) so it never overlaps that input.
    unsigned Dwords = AMDGPU::getVGPRMemoryFileDwords(Size);
    if (ClearsFixedWorkitem && Base <= FixedWorkitemRegIdx &&
        Base + Dwords > FixedWorkitemRegIdx)
      Base = alignTo(FixedWorkitemRegIdx + 1, 2u);

    // The file lives in low, caller-saved VGPRs that only group members
    // reserve. A call to anything outside the group - indirect, external, or a
    // defined non-member - does not reserve the file and clobbers it, so
    // diagnose rather than silently corrupt it. (Direct calls between members
    // are safe; intrinsics don't clobber.) Calls introduced after this pass
    // (e.g. expanded libcalls) and inline asm clobbering a file register are
    // caught later, in AMDGPUPrivateObjectVGPRs, where the machine-level calls
    // and the final reserved registers are known.
    SmallPtrSet<const Function *, 16> GroupFnSet(GroupFns.begin(),
                                                 GroupFns.end());
    for (Function *F : GroupFns)
      for (const CallBase *CB : Calls.lookup(F)) {
        const Function *Callee = CB->getCalledFunction();
        if (!Callee || !GroupFnSet.contains(Callee))
          Ctx.diagnose(DiagnosticInfoUnsupported(
              *F,
              "'VGPR as memory' is not supported in a function that makes an "
              "indirect call or a call outside its call graph",
              CB->getDebugLoc()));
      }

    for (Function *F : GroupFns) {
      F->addFnAttr(SizeAttr, utostr(Size));
      F->addFnAttr(BaseAttr, utostr(Base));
    }
  }
  return true;
}

bool AMDGPULowerModuleVGPRs::runOnModule(Module &M) {
  return lowerModuleVGPRs(M);
}

PreservedAnalyses AMDGPULowerModuleVGPRsPass::run(Module &M,
                                                  ModuleAnalysisManager &) {
  return lowerModuleVGPRs(M) ? PreservedAnalyses::none()
                             : PreservedAnalyses::all();
}
