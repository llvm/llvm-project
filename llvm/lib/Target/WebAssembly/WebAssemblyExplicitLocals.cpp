//===-- WebAssemblyExplicitLocals.cpp - Make Locals Explicit --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file converts any remaining registers into WebAssembly locals.
///
/// After register stackification and register coloring, convert non-stackified
/// registers into locals, inserting explicit get_local and set_local
/// instructions.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssembly.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyUtilities.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-explicit-locals"

namespace {
class WebAssemblyExplicitLocals final : public MachineFunctionPass {
  StringRef getPassName() const override {
    return "WebAssembly Explicit Locals";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addPreserved<MachineBlockFrequencyInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyExplicitLocals() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblyExplicitLocals::ID = 0;
FunctionPass *llvm::createWebAssemblyExplicitLocals() {
  return new WebAssemblyExplicitLocals();
}

/// Return a local id number for the given register, assigning it a new one
/// if it doesn't yet have one.
static unsigned getLocalId(DenseMap<unsigned, unsigned> &Reg2Local,
                           unsigned &CurLocal, unsigned Reg) {
  return Reg2Local.insert(std::make_pair(Reg, CurLocal++)).first->second;
}

/// Get the appropriate get_local opcode for the given register class.
static unsigned getGetLocalOpcode(const TargetRegisterClass *RC) {
  if (RC == &WebAssembly::I32RegClass)
    return WebAssembly::GET_LOCAL_I32;
  if (RC == &WebAssembly::I64RegClass)
    return WebAssembly::GET_LOCAL_I64;
  if (RC == &WebAssembly::F32RegClass)
    return WebAssembly::GET_LOCAL_F32;
  if (RC == &WebAssembly::F64RegClass)
    return WebAssembly::GET_LOCAL_F64;
  if (RC == &WebAssembly::V128RegClass)
    return WebAssembly::GET_LOCAL_V128;
  llvm_unreachable("Unexpected register class");
}

/// Get the appropriate set_local opcode for the given register class.
static unsigned getSetLocalOpcode(const TargetRegisterClass *RC) {
  if (RC == &WebAssembly::I32RegClass)
    return WebAssembly::SET_LOCAL_I32;
  if (RC == &WebAssembly::I64RegClass)
    return WebAssembly::SET_LOCAL_I64;
  if (RC == &WebAssembly::F32RegClass)
    return WebAssembly::SET_LOCAL_F32;
  if (RC == &WebAssembly::F64RegClass)
    return WebAssembly::SET_LOCAL_F64;
  if (RC == &WebAssembly::V128RegClass)
    return WebAssembly::SET_LOCAL_V128;
  llvm_unreachable("Unexpected register class");
}

/// Get the appropriate tee_local opcode for the given register class.
static unsigned getTeeLocalOpcode(const TargetRegisterClass *RC) {
  if (RC == &WebAssembly::I32RegClass)
    return WebAssembly::TEE_LOCAL_I32;
  if (RC == &WebAssembly::I64RegClass)
    return WebAssembly::TEE_LOCAL_I64;
  if (RC == &WebAssembly::F32RegClass)
    return WebAssembly::TEE_LOCAL_F32;
  if (RC == &WebAssembly::F64RegClass)
    return WebAssembly::TEE_LOCAL_F64;
  if (RC == &WebAssembly::V128RegClass)
    return WebAssembly::TEE_LOCAL_V128;
  llvm_unreachable("Unexpected register class");
}

/// Get the type associated with the given register class.
static MVT typeForRegClass(const TargetRegisterClass *RC) {
  if (RC == &WebAssembly::I32RegClass)
    return MVT::i32;
  if (RC == &WebAssembly::I64RegClass)
    return MVT::i64;
  if (RC == &WebAssembly::F32RegClass)
    return MVT::f32;
  if (RC == &WebAssembly::F64RegClass)
    return MVT::f64;
  llvm_unreachable("unrecognized register class");
}

/// Given a MachineOperand of a stackified vreg, return the instruction at the
/// start of the expression tree.
static MachineInstr *FindStartOfTree(MachineOperand &MO,
                                     MachineRegisterInfo &MRI,
                                     WebAssemblyFunctionInfo &MFI) {
  unsigned Reg = MO.getReg();
  assert(MFI.isVRegStackified(Reg));
  MachineInstr *Def = MRI.getVRegDef(Reg);

  // Find the first stackified use and proceed from there.
  for (MachineOperand &DefMO : Def->explicit_uses()) {
    if (!DefMO.isReg())
      continue;
    return FindStartOfTree(DefMO, MRI, MFI);
  }

  // If there were no stackified uses, we've reached the start.
  return Def;
}

bool WebAssemblyExplicitLocals::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(dbgs() << "********** Make Locals Explicit **********\n"
                  "********** Function: "
               << MF.getName() << '\n');

  // Disable this pass if we aren't doing direct wasm object emission.
  if (MF.getSubtarget<WebAssemblySubtarget>()
        .getTargetTriple().isOSBinFormatELF())
    return false;

  bool Changed = false;
  MachineRegisterInfo &MRI = MF.getRegInfo();
  WebAssemblyFunctionInfo &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();
  const auto *TII = MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();

  // Map non-stackified virtual registers to their local ids.
  DenseMap<unsigned, unsigned> Reg2Local;

  // Handle ARGUMENTS first to ensure that they get the designated numbers.
  for (MachineBasicBlock::iterator I = MF.begin()->begin(),
                                   E = MF.begin()->end();
       I != E;) {
    MachineInstr &MI = *I++;
    if (!WebAssembly::isArgument(MI))
      break;
    unsigned Reg = MI.getOperand(0).getReg();
    assert(!MFI.isVRegStackified(Reg));
    Reg2Local[Reg] = MI.getOperand(1).getImm();
    MI.eraseFromParent();
    Changed = true;
  }

  // Start assigning local numbers after the last parameter.
  unsigned CurLocal = MFI.getParams().size();

  // Visit each instruction in the function.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &MI = *I++;
      assert(!WebAssembly::isArgument(MI));

      if (MI.isDebugValue() || MI.isLabel())
        continue;

      // Replace tee instructions with tee_local. The difference is that tee
      // instructins have two defs, while tee_local instructions have one def
      // and an index of a local to write to.
      if (WebAssembly::isTee(MI)) {
        assert(MFI.isVRegStackified(MI.getOperand(0).getReg()));
        assert(!MFI.isVRegStackified(MI.getOperand(1).getReg()));
        unsigned OldReg = MI.getOperand(2).getReg();
        const TargetRegisterClass *RC = MRI.getRegClass(OldReg);

        // Stackify the input if it isn't stackified yet.
        if (!MFI.isVRegStackified(OldReg)) {
          unsigned LocalId = getLocalId(Reg2Local, CurLocal, OldReg);
          unsigned NewReg = MRI.createVirtualRegister(RC);
          unsigned Opc = getGetLocalOpcode(RC);
          BuildMI(MBB, &MI, MI.getDebugLoc(), TII->get(Opc), NewReg)
              .addImm(LocalId);
          MI.getOperand(2).setReg(NewReg);
          MFI.stackifyVReg(NewReg);
        }

        // Replace the TEE with a TEE_LOCAL.
        unsigned LocalId =
            getLocalId(Reg2Local, CurLocal, MI.getOperand(1).getReg());
        unsigned Opc = getTeeLocalOpcode(RC);
        BuildMI(MBB, &MI, MI.getDebugLoc(), TII->get(Opc),
                MI.getOperand(0).getReg())
            .addImm(LocalId)
            .addReg(MI.getOperand(2).getReg());

        MI.eraseFromParent();
        Changed = true;
        continue;
      }

      // Insert set_locals for any defs that aren't stackified yet. Currently
      // we handle at most one def.
      assert(MI.getDesc().getNumDefs() <= 1);
      if (MI.getDesc().getNumDefs() == 1) {
        unsigned OldReg = MI.getOperand(0).getReg();
        if (!MFI.isVRegStackified(OldReg) && !MRI.use_empty(OldReg)) {
          unsigned LocalId = getLocalId(Reg2Local, CurLocal, OldReg);
          const TargetRegisterClass *RC = MRI.getRegClass(OldReg);
          unsigned NewReg = MRI.createVirtualRegister(RC);
          auto InsertPt = std::next(MachineBasicBlock::iterator(&MI));
          unsigned Opc = getSetLocalOpcode(RC);
          BuildMI(MBB, InsertPt, MI.getDebugLoc(), TII->get(Opc))
              .addImm(LocalId)
              .addReg(NewReg);
          MI.getOperand(0).setReg(NewReg);
          MFI.stackifyVReg(NewReg);
          Changed = true;
        }
      }

      // Insert get_locals for any uses that aren't stackified yet.
      MachineInstr *InsertPt = &MI;
      for (MachineOperand &MO : reverse(MI.explicit_uses())) {
        if (!MO.isReg())
          continue;

        unsigned OldReg = MO.getReg();

        // If we see a stackified register, prepare to insert subsequent
        // get_locals before the start of its tree.
        if (MFI.isVRegStackified(OldReg)) {
          InsertPt = FindStartOfTree(MO, MRI, MFI);
          continue;
        }

        // Insert a get_local.
        unsigned LocalId = getLocalId(Reg2Local, CurLocal, OldReg);
        const TargetRegisterClass *RC = MRI.getRegClass(OldReg);
        unsigned NewReg = MRI.createVirtualRegister(RC);
        unsigned Opc = getGetLocalOpcode(RC);
        InsertPt =
            BuildMI(MBB, InsertPt, MI.getDebugLoc(), TII->get(Opc), NewReg)
                .addImm(LocalId);
        MO.setReg(NewReg);
        MFI.stackifyVReg(NewReg);
        Changed = true;
      }

      // Coalesce and eliminate COPY instructions.
      if (WebAssembly::isCopy(MI)) {
        MRI.replaceRegWith(MI.getOperand(1).getReg(),
                           MI.getOperand(0).getReg());
        MI.eraseFromParent();
        Changed = true;
      }
    }
  }

  // Define the locals.
  for (size_t i = 0, e = MRI.getNumVirtRegs(); i < e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    auto I = Reg2Local.find(Reg);
    if (I == Reg2Local.end() || I->second < MFI.getParams().size())
      continue;

    MFI.addLocal(typeForRegClass(MRI.getRegClass(Reg)));
    Changed = true;
  }

#ifndef NDEBUG
  // Assert that all registers have been stackified at this point.
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (MI.isDebugValue() || MI.isLabel())
        continue;
      for (const MachineOperand &MO : MI.explicit_operands()) {
        assert(
            (!MO.isReg() || MRI.use_empty(MO.getReg()) ||
             MFI.isVRegStackified(MO.getReg())) &&
            "WebAssemblyExplicitLocals failed to stackify a register operand");
      }
    }
  }
#endif

  return Changed;
}
