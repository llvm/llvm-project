//===- X86OptimizeVectorConstants.cpp - Optimize Vector Constants ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass coalesces redundant vector zero and all-ones materializations into
// subregister copies or register replacements from a single larger
// materialization, reducing redundant instructions before register allocation.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include <array>
#include <cassert>
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "x86-optimize-vector-constants"

STATISTIC(NumZeroConstsCoalesced, "Number of zero vector constants coalesced");
STATISTIC(NumAllOnesConstsCoalesced,
          "Number of all-ones vector constants coalesced");

namespace {

/// The kind of vector constant being materialized.
enum class VectorConstKind : unsigned {
  Zero,
  AllOnes,
  Count,
};

/// Describes a recognized vector constant materialization.
struct VectorConstInfo {
  unsigned Size; // 128, 256, or 512
  VectorConstKind Kind;
};

class X86OptimizeVectorConstantsImpl {
  MachineRegisterInfo *MRI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  const X86Subtarget *Subtarget = nullptr;
  bool Changed = false;

  /// Classify a machine instruction as a vector constant materialization.
  /// Returns the size and kind (zero vs all-ones) if recognized.
  static std::optional<VectorConstInfo>
  classifyConstMaterialization(const MachineInstr &MI) {
    switch (MI.getOpcode()) {
    // Zeros
    case X86::V_SET0:
    case X86::AVX512_128_SET0:
    case X86::FsFLD0SH:
    case X86::FsFLD0SS:
    case X86::FsFLD0SD:
    case X86::FsFLD0F128:
    case X86::AVX512_FsFLD0SH:
    case X86::AVX512_FsFLD0SS:
    case X86::AVX512_FsFLD0SD:
    case X86::AVX512_FsFLD0F128:
      return VectorConstInfo{128, VectorConstKind::Zero};
    case X86::AVX_SET0:
    case X86::AVX512_256_SET0:
      return VectorConstInfo{256, VectorConstKind::Zero};
    case X86::AVX512_512_SET0:
      return VectorConstInfo{512, VectorConstKind::Zero};
    // All-Ones
    case X86::V_SETALLONES:
    case X86::AVX512_128_SETALLONES:
      return VectorConstInfo{128, VectorConstKind::AllOnes};
    case X86::AVX1_SETALLONES:
    case X86::AVX2_SETALLONES:
    case X86::AVX512_256_SETALLONES:
      return VectorConstInfo{256, VectorConstKind::AllOnes};
    case X86::AVX512_512_SETALLONES:
      return VectorConstInfo{512, VectorConstKind::AllOnes};
    default:
      return std::nullopt;
    }
  }

  static unsigned getSizeIndex(unsigned Size) {
    switch (Size) {
    case 128:
      return 0;
    case 256:
      return 1;
    case 512:
      return 2;
    }
    llvm_unreachable("Unexpected vector size");
  }

  static unsigned getSubReg(unsigned ParentSize, unsigned ChildSize) {
    // ActiveSetsT only stores a materialization for equal or larger sizes.
    assert(ParentSize >= ChildSize && "Invalid sizes for subregister");
    if (ParentSize == ChildSize)
      return 0;
    if (ParentSize == 512 && ChildSize == 256)
      return X86::sub_ymm;
    if (ParentSize == 512 && ChildSize == 128)
      return X86::sub_xmm;
    if (ParentSize == 256 && ChildSize == 128)
      return X86::sub_xmm;
    llvm_unreachable("Unexpected size combination");
  }

  /// Active[Kind][SizeIdx] tracks the closest dominating materialization of
  /// the given kind and size-or-larger.
  static constexpr unsigned NumKinds =
      static_cast<unsigned>(VectorConstKind::Count);
  static constexpr unsigned NumSizes = 3;
  using ActiveSetsT =
      std::array<std::array<MachineInstr *, NumSizes>, NumKinds>;

  void processNode(MachineDomTreeNode *Node, ActiveSetsT Active);

public:
  bool run(MachineFunction &MF, MachineDominatorTree &MDT);
};

class X86OptimizeVectorConstantsLegacy : public MachineFunctionPass {
public:
  static char ID;

  X86OptimizeVectorConstantsLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "X86 Optimize Vector Constants";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    X86OptimizeVectorConstantsImpl Impl;
    return Impl.run(
        MF, getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree());
  }
};

char X86OptimizeVectorConstantsLegacy::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(X86OptimizeVectorConstantsLegacy, DEBUG_TYPE,
                      "X86 Optimize Vector Constants", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(X86OptimizeVectorConstantsLegacy, DEBUG_TYPE,
                    "X86 Optimize Vector Constants", false, false)

FunctionPass *llvm::createX86OptimizeVectorConstantsLegacyPass() {
  return new X86OptimizeVectorConstantsLegacy();
}

PreservedAnalyses
X86OptimizeVectorConstantsPass::run(MachineFunction &MF,
                                    MachineFunctionAnalysisManager &MFAM) {
  X86OptimizeVectorConstantsImpl Impl;
  auto &MDT = MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  bool Changed = Impl.run(MF, MDT);
  if (!Changed)
    return PreservedAnalyses::all();
  auto PA = PreservedAnalyses::none();
  PA.preserve<MachineDominatorTreeAnalysis>();
  PA.preserve<MachineLoopAnalysis>();
  return PA;
}

void X86OptimizeVectorConstantsImpl::processNode(MachineDomTreeNode *Node,
                                                 ActiveSetsT Active) {
  MachineBasicBlock *MBB = Node->getBlock();

  for (auto &MI : llvm::make_early_inc_range(*MBB)) {
    auto OptInfo = classifyConstMaterialization(MI);
    if (!OptInfo)
      continue;
    if (!MI.getOperand(0).isReg() || !MI.getOperand(0).getReg().isVirtual())
      continue;

    unsigned Size = OptInfo->Size;
    unsigned KindIdx = static_cast<unsigned>(OptInfo->Kind);
    unsigned SizeIdx = getSizeIndex(Size);

    MachineInstr *DomConst = Active[KindIdx][SizeIdx];
    if (DomConst) {
      Register OldReg = MI.getOperand(0).getReg();

      bool SkipCoalesce = false;
      if (!Subtarget->hasAVX()) {
        bool UsesTwoAddressInstr = false;
        for (const MachineOperand &UseMO : MRI->use_operands(OldReg)) {
          const MachineInstr *UseMI = UseMO.getParent();
          for (unsigned I = 0, E = UseMI->getNumOperands(); I != E; ++I) {
            const MachineOperand &MO = UseMI->getOperand(I);
            if (MO.isReg() && MO.isUse() && MO.isTied()) {
              UsesTwoAddressInstr = true;
              break;
            }
          }
          if (UsesTwoAddressInstr)
            break;
        }

        // Avoid coalescing on non-AVX targets if the constant is consumed by an
        // instruction with any tied operands. Extending the constant's live
        // range prevents TwoAddressInstructionPass from using the untied
        // operand as a killed/sacrificial register to commute the instruction,
        // which forces an expensive extra COPY.
        if (UsesTwoAddressInstr) {
          LLVM_DEBUG(dbgs() << "Skipping coalesce due to two-address user on "
                               "non-AVX target\n");
          SkipCoalesce = true;
        }
      }

      if (!SkipCoalesce) {
        auto DomInfo = classifyConstMaterialization(*DomConst);
        assert(DomInfo && "Active constant must classify");
        unsigned DomSize = DomInfo->Size;

        Register NewReg = DomConst->getOperand(0).getReg();
        assert(NewReg.isVirtual() &&
               "Dominator constant must define a virtual register");
        unsigned SubIdx = getSubReg(DomSize, Size);

        const TargetRegisterClass *OldRC = MRI->getRegClass(OldReg);
        const TargetRegisterClass *NewRC = MRI->getRegClass(NewReg);

        LLVM_DEBUG(dbgs() << "Coalescing: " << MI
                          << "  with dominator: " << *DomConst);

        if (DomSize == Size && OldRC == NewRC) {
          // The constant pseudo has no side effects and DomConst dominates this
          // definition, so all uses can be redirected to the dominating vreg.
          assert(MRI->hasOneDef(OldReg) &&
                 "Expected single-def vreg from isel constant pseudo");
          MRI->replaceRegWith(OldReg, NewReg);
        } else {
          // Emit a COPY with a subregister index. Later register allocation
          // and copy coalescing will reconcile any compatible register-class
          // constraints.
          BuildMI(*MI.getParent(), MI, MI.getDebugLoc(),
                  TII->get(TargetOpcode::COPY), OldReg)
              .addReg(NewReg, RegState(), SubIdx);
        }

        // clearKillFlags must be run unconditionally, as either branch adds a
        // new use of NewReg which may invalidate a previous kill flag.
        MRI->clearKillFlags(NewReg);

        MI.eraseFromParent();
        switch (OptInfo->Kind) {
        case VectorConstKind::Zero:
          ++NumZeroConstsCoalesced;
          break;
        case VectorConstKind::AllOnes:
          ++NumAllOnesConstsCoalesced;
          break;
        case VectorConstKind::Count:
          llvm_unreachable("Invalid vector constant kind");
        }
        Changed = true;
        continue;
      }
    }

    // If we didn't coalesce, MI becomes the closest active constant for its
    // kind at its size and all smaller sizes.
    for (unsigned I = 0; I <= SizeIdx; ++I)
      Active[KindIdx][I] = &MI;
  }

  for (auto *Child : *Node)
    processNode(Child, Active);
}

bool X86OptimizeVectorConstantsImpl::run(MachineFunction &MF,
                                         MachineDominatorTree &MDT) {
  this->MRI = &MF.getRegInfo();
  this->TII = MF.getSubtarget().getInstrInfo();
  this->Subtarget = &MF.getSubtarget<X86Subtarget>();
  this->Changed = false;

  ActiveSetsT Active{};
  processNode(MDT.getRootNode(), Active);

  return Changed;
}
