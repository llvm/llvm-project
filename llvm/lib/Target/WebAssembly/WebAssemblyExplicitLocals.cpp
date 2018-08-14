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
/// This file converts any remaining registers into WebAssembly locals.
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

// A command-line option to disable this pass, and keep implicit locals and
// stackified registers for the purpose of testing with lit/llc ONLY.
// This produces output which is not valid WebAssembly, and is not supported
// by assemblers/disassemblers and other MC based tools.
static cl::opt<bool> RegisterCodeGenTestMode(
    "wasm-register-codegen-test-mode", cl::Hidden,
    cl::desc("WebAssembly: output stack registers and implicit locals in"
             " instruction output for test purposes only."),
    cl::init(false));
// This one does explicit locals but keeps stackified registers, as required
// by some current tests.
static cl::opt<bool> ExplicitLocalsCodeGenTestMode(
    "wasm-explicit-locals-codegen-test-mode", cl::Hidden,
    cl::desc("WebAssembly: output stack registers and explicit locals in"
             " instruction output for test purposes only."),
    cl::init(false));

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

unsigned regInstructionToStackInstruction(unsigned OpCode);

char WebAssemblyExplicitLocals::ID = 0;
INITIALIZE_PASS(WebAssemblyExplicitLocals, DEBUG_TYPE,
                "Convert registers to WebAssembly locals", false, false)

FunctionPass *llvm::createWebAssemblyExplicitLocals() {
  return new WebAssemblyExplicitLocals();
}

/// Return a local id number for the given register, assigning it a new one
/// if it doesn't yet have one.
static unsigned getLocalId(DenseMap<unsigned, unsigned> &Reg2Local,
                           unsigned &CurLocal, unsigned Reg) {
  auto P = Reg2Local.insert(std::make_pair(Reg, CurLocal));
  if (P.second)
    ++CurLocal;
  return P.first->second;
}

/// Get the appropriate drop opcode for the given register class.
static unsigned getDropOpcode(const TargetRegisterClass *RC) {
  if (RC == &WebAssembly::I32RegClass)
    return WebAssembly::DROP_I32;
  if (RC == &WebAssembly::I64RegClass)
    return WebAssembly::DROP_I64;
  if (RC == &WebAssembly::F32RegClass)
    return WebAssembly::DROP_F32;
  if (RC == &WebAssembly::F64RegClass)
    return WebAssembly::DROP_F64;
  if (RC == &WebAssembly::V128RegClass)
    return WebAssembly::DROP_V128;
  if (RC == &WebAssembly::EXCEPT_REFRegClass)
    return WebAssembly::DROP_EXCEPT_REF;
  llvm_unreachable("Unexpected register class");
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
  if (RC == &WebAssembly::EXCEPT_REFRegClass)
    return WebAssembly::GET_LOCAL_EXCEPT_REF;
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
  if (RC == &WebAssembly::EXCEPT_REFRegClass)
    return WebAssembly::SET_LOCAL_EXCEPT_REF;
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
  if (RC == &WebAssembly::EXCEPT_REFRegClass)
    return WebAssembly::TEE_LOCAL_EXCEPT_REF;
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
  if (RC == &WebAssembly::EXCEPT_REFRegClass)
    return MVT::ExceptRef;
  llvm_unreachable("unrecognized register class");
}

/// Given a MachineOperand of a stackified vreg, return the instruction at the
/// start of the expression tree.
static MachineInstr *findStartOfTree(MachineOperand &MO,
                                     MachineRegisterInfo &MRI,
                                     WebAssemblyFunctionInfo &MFI) {
  unsigned Reg = MO.getReg();
  assert(MFI.isVRegStackified(Reg));
  MachineInstr *Def = MRI.getVRegDef(Reg);

  // Find the first stackified use and proceed from there.
  for (MachineOperand &DefMO : Def->explicit_uses()) {
    if (!DefMO.isReg())
      continue;
    return findStartOfTree(DefMO, MRI, MFI);
  }

  // If there were no stackified uses, we've reached the start.
  return Def;
}

bool WebAssemblyExplicitLocals::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** Make Locals Explicit **********\n"
                       "********** Function: "
                    << MF.getName() << '\n');

  // Disable this pass if directed to do so.
  if (RegisterCodeGenTestMode)
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
    Reg2Local[Reg] = static_cast<unsigned>(MI.getOperand(1).getImm());
    MI.eraseFromParent();
    Changed = true;
  }

  // Start assigning local numbers after the last parameter.
  unsigned CurLocal = static_cast<unsigned>(MFI.getParams().size());

  // Precompute the set of registers that are unused, so that we can insert
  // drops to their defs.
  BitVector UseEmpty(MRI.getNumVirtRegs());
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I < E; ++I)
    UseEmpty[I] = MRI.use_empty(TargetRegisterInfo::index2VirtReg(I));

  // Visit each instruction in the function.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &MI = *I++;
      assert(!WebAssembly::isArgument(MI));

      if (MI.isDebugInstr() || MI.isLabel())
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
        if (!MFI.isVRegStackified(OldReg)) {
          const TargetRegisterClass *RC = MRI.getRegClass(OldReg);
          unsigned NewReg = MRI.createVirtualRegister(RC);
          auto InsertPt = std::next(MachineBasicBlock::iterator(&MI));
          if (MI.getOpcode() == WebAssembly::IMPLICIT_DEF) {
            MI.eraseFromParent();
            Changed = true;
            continue;
          }
          if (UseEmpty[TargetRegisterInfo::virtReg2Index(OldReg)]) {
            unsigned Opc = getDropOpcode(RC);
            MachineInstr *Drop =
                BuildMI(MBB, InsertPt, MI.getDebugLoc(), TII->get(Opc))
                    .addReg(NewReg);
            // After the drop instruction, this reg operand will not be used
            Drop->getOperand(0).setIsKill();
          } else {
            unsigned LocalId = getLocalId(Reg2Local, CurLocal, OldReg);
            unsigned Opc = getSetLocalOpcode(RC);
            BuildMI(MBB, InsertPt, MI.getDebugLoc(), TII->get(Opc))
                .addImm(LocalId)
                .addReg(NewReg);
          }
          MI.getOperand(0).setReg(NewReg);
          // This register operand is now being used by the inserted drop
          // instruction, so make it undead.
          MI.getOperand(0).setIsDead(false);
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

        // Inline asm may have a def in the middle of the operands. Our contract
        // with inline asm register operands is to provide local indices as
        // immediates.
        if (MO.isDef()) {
          assert(MI.getOpcode() == TargetOpcode::INLINEASM);
          unsigned LocalId = getLocalId(Reg2Local, CurLocal, OldReg);
          MRI.removeRegOperandFromUseList(&MO);
          MO = MachineOperand::CreateImm(LocalId);
          continue;
        }

        // If we see a stackified register, prepare to insert subsequent
        // get_locals before the start of its tree.
        if (MFI.isVRegStackified(OldReg)) {
          InsertPt = findStartOfTree(MO, MRI, MFI);
          continue;
        }

        // Our contract with inline asm register operands is to provide local
        // indices as immediates.
        if (MI.getOpcode() == TargetOpcode::INLINEASM) {
          unsigned LocalId = getLocalId(Reg2Local, CurLocal, OldReg);
          MRI.removeRegOperandFromUseList(&MO);
          MO = MachineOperand::CreateImm(LocalId);
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

    if (!ExplicitLocalsCodeGenTestMode) {
      // Remove all uses of stackified registers to bring the instruction format
      // into its final stack form, and transition opcodes to their _S variant.
      // We do this in a seperate loop, since the previous loop adds/removes
      // instructions.
      // See comments in lib/Target/WebAssembly/WebAssemblyInstrFormats.td for
      // details.
      // TODO: the code above creates new registers which are then removed here.
      // That code could be slightly simplified by not doing that, though maybe
      // it is simpler conceptually to keep the code above in "register mode"
      // until this transition point.
      for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
           I != E;) {
        MachineInstr &MI = *I++;
        // FIXME: we are not processing inline assembly, which contains register
        // operands, because it is used by later target generic code.
        if (MI.isDebugInstr() || MI.isLabel() || MI.isInlineAsm())
          continue;
        auto RegOpcode = MI.getOpcode();
        auto StackOpcode = regInstructionToStackInstruction(RegOpcode);
        MI.setDesc(TII->get(StackOpcode));
        // Now remove all register operands.
        for (auto I = MI.getNumOperands(); I; --I) {
          auto &MO = MI.getOperand(I - 1);
          if (MO.isReg()) {
            MI.RemoveOperand(I - 1);
            // TODO: we should also update the MFI here or below to reflect the
            // removed registers? The MFI is about to be deleted anyway, so
            // maybe that is not worth it?
          }
        }
      }
    }
  }

  // Define the locals.
  // TODO: Sort the locals for better compression.
  MFI.setNumLocals(CurLocal - MFI.getParams().size());
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I < E; ++I) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(I);
    auto RL = Reg2Local.find(Reg);
    if (RL == Reg2Local.end() || RL->second < MFI.getParams().size())
      continue;

    MFI.setLocal(RL->second - MFI.getParams().size(),
                 typeForRegClass(MRI.getRegClass(Reg)));
    Changed = true;
  }

  return Changed;
}

unsigned regInstructionToStackInstruction(unsigned OpCode) {
  switch (OpCode) {
  default:
    // You may hit this if you add new instructions, please add them below.
    // For most of these opcodes, this function could have been implemented
    // as "return OpCode + 1", but since table-gen alphabetically sorts them,
    // this cannot be guaranteed (see e.g. BR and BR_IF).
    // The approach below is the same as what the x87 backend does.
    // TODO(wvo): to make this code cleaner, create a custom tablegen
    // code generator that emits the table below automatically.
    llvm_unreachable(
          "unknown WebAssembly instruction in Explicit Locals pass");
  case WebAssembly::ABS_F32: return WebAssembly::ABS_F32_S;
  case WebAssembly::ABS_F64: return WebAssembly::ABS_F64_S;
  case WebAssembly::ADD_F32: return WebAssembly::ADD_F32_S;
  case WebAssembly::ADD_F32x4: return WebAssembly::ADD_F32x4_S;
  case WebAssembly::ADD_F64: return WebAssembly::ADD_F64_S;
  case WebAssembly::ADD_I16x8: return WebAssembly::ADD_I16x8_S;
  case WebAssembly::ADD_I32: return WebAssembly::ADD_I32_S;
  case WebAssembly::ADD_I32x4: return WebAssembly::ADD_I32x4_S;
  case WebAssembly::ADD_I64: return WebAssembly::ADD_I64_S;
  case WebAssembly::ADD_I8x16: return WebAssembly::ADD_I8x16_S;
  case WebAssembly::ADJCALLSTACKDOWN: return WebAssembly::ADJCALLSTACKDOWN_S;
  case WebAssembly::ADJCALLSTACKUP: return WebAssembly::ADJCALLSTACKUP_S;
  case WebAssembly::AND_I32: return WebAssembly::AND_I32_S;
  case WebAssembly::AND_I64: return WebAssembly::AND_I64_S;
  case WebAssembly::ARGUMENT_EXCEPT_REF: return WebAssembly::ARGUMENT_EXCEPT_REF_S;
  case WebAssembly::ARGUMENT_F32: return WebAssembly::ARGUMENT_F32_S;
  case WebAssembly::ARGUMENT_F64: return WebAssembly::ARGUMENT_F64_S;
  case WebAssembly::ARGUMENT_I32: return WebAssembly::ARGUMENT_I32_S;
  case WebAssembly::ARGUMENT_I64: return WebAssembly::ARGUMENT_I64_S;
  case WebAssembly::ARGUMENT_v16i8: return WebAssembly::ARGUMENT_v16i8_S;
  case WebAssembly::ARGUMENT_v4f32: return WebAssembly::ARGUMENT_v4f32_S;
  case WebAssembly::ARGUMENT_v4i32: return WebAssembly::ARGUMENT_v4i32_S;
  case WebAssembly::ARGUMENT_v8i16: return WebAssembly::ARGUMENT_v8i16_S;
  case WebAssembly::ARGUMENT_v2f64: return WebAssembly::ARGUMENT_v2f64_S;
  case WebAssembly::ARGUMENT_v2i64: return WebAssembly::ARGUMENT_v2i64_S;
  case WebAssembly::ATOMIC_LOAD16_U_I32: return WebAssembly::ATOMIC_LOAD16_U_I32_S;
  case WebAssembly::ATOMIC_LOAD16_U_I64: return WebAssembly::ATOMIC_LOAD16_U_I64_S;
  case WebAssembly::ATOMIC_LOAD32_U_I64: return WebAssembly::ATOMIC_LOAD32_U_I64_S;
  case WebAssembly::ATOMIC_LOAD8_U_I32: return WebAssembly::ATOMIC_LOAD8_U_I32_S;
  case WebAssembly::ATOMIC_LOAD8_U_I64: return WebAssembly::ATOMIC_LOAD8_U_I64_S;
  case WebAssembly::ATOMIC_LOAD_I32: return WebAssembly::ATOMIC_LOAD_I32_S;
  case WebAssembly::ATOMIC_LOAD_I64: return WebAssembly::ATOMIC_LOAD_I64_S;
  case WebAssembly::ATOMIC_STORE16_I32: return WebAssembly::ATOMIC_STORE16_I32_S;
  case WebAssembly::ATOMIC_STORE16_I64: return WebAssembly::ATOMIC_STORE16_I64_S;
  case WebAssembly::ATOMIC_STORE32_I64: return WebAssembly::ATOMIC_STORE32_I64_S;
  case WebAssembly::ATOMIC_STORE8_I32: return WebAssembly::ATOMIC_STORE8_I32_S;
  case WebAssembly::ATOMIC_STORE8_I64: return WebAssembly::ATOMIC_STORE8_I64_S;
  case WebAssembly::ATOMIC_STORE_I32: return WebAssembly::ATOMIC_STORE_I32_S;
  case WebAssembly::ATOMIC_STORE_I64: return WebAssembly::ATOMIC_STORE_I64_S;
  case WebAssembly::BLOCK: return WebAssembly::BLOCK_S;
  case WebAssembly::BR: return WebAssembly::BR_S;
  case WebAssembly::BR_IF: return WebAssembly::BR_IF_S;
  case WebAssembly::BR_TABLE_I32: return WebAssembly::BR_TABLE_I32_S;
  case WebAssembly::BR_TABLE_I64: return WebAssembly::BR_TABLE_I64_S;
  case WebAssembly::BR_UNLESS: return WebAssembly::BR_UNLESS_S;
  case WebAssembly::CALL_EXCEPT_REF: return WebAssembly::CALL_EXCEPT_REF_S;
  case WebAssembly::CALL_F32: return WebAssembly::CALL_F32_S;
  case WebAssembly::CALL_F64: return WebAssembly::CALL_F64_S;
  case WebAssembly::CALL_I32: return WebAssembly::CALL_I32_S;
  case WebAssembly::CALL_I64: return WebAssembly::CALL_I64_S;
  case WebAssembly::CALL_INDIRECT_EXCEPT_REF: return WebAssembly::CALL_INDIRECT_EXCEPT_REF_S;
  case WebAssembly::CALL_INDIRECT_F32: return WebAssembly::CALL_INDIRECT_F32_S;
  case WebAssembly::CALL_INDIRECT_F64: return WebAssembly::CALL_INDIRECT_F64_S;
  case WebAssembly::CALL_INDIRECT_I32: return WebAssembly::CALL_INDIRECT_I32_S;
  case WebAssembly::CALL_INDIRECT_I64: return WebAssembly::CALL_INDIRECT_I64_S;
  case WebAssembly::CALL_INDIRECT_VOID: return WebAssembly::CALL_INDIRECT_VOID_S;
  case WebAssembly::CALL_INDIRECT_v16i8: return WebAssembly::CALL_INDIRECT_v16i8_S;
  case WebAssembly::CALL_INDIRECT_v4f32: return WebAssembly::CALL_INDIRECT_v4f32_S;
  case WebAssembly::CALL_INDIRECT_v4i32: return WebAssembly::CALL_INDIRECT_v4i32_S;
  case WebAssembly::CALL_INDIRECT_v8i16: return WebAssembly::CALL_INDIRECT_v8i16_S;
  case WebAssembly::CALL_VOID: return WebAssembly::CALL_VOID_S;
  case WebAssembly::CALL_v16i8: return WebAssembly::CALL_v16i8_S;
  case WebAssembly::CALL_v4f32: return WebAssembly::CALL_v4f32_S;
  case WebAssembly::CALL_v4i32: return WebAssembly::CALL_v4i32_S;
  case WebAssembly::CALL_v8i16: return WebAssembly::CALL_v8i16_S;
  case WebAssembly::CATCHRET: return WebAssembly::CATCHRET_S;
  case WebAssembly::CATCH_ALL: return WebAssembly::CATCH_ALL_S;
  case WebAssembly::CATCH_I32: return WebAssembly::CATCH_I32_S;
  case WebAssembly::CATCH_I64: return WebAssembly::CATCH_I64_S;
  case WebAssembly::CEIL_F32: return WebAssembly::CEIL_F32_S;
  case WebAssembly::CEIL_F64: return WebAssembly::CEIL_F64_S;
  case WebAssembly::CLEANUPRET: return WebAssembly::CLEANUPRET_S;
  case WebAssembly::CLZ_I32: return WebAssembly::CLZ_I32_S;
  case WebAssembly::CLZ_I64: return WebAssembly::CLZ_I64_S;
  case WebAssembly::CONST_F32: return WebAssembly::CONST_F32_S;
  case WebAssembly::CONST_F64: return WebAssembly::CONST_F64_S;
  case WebAssembly::CONST_I32: return WebAssembly::CONST_I32_S;
  case WebAssembly::CONST_I64: return WebAssembly::CONST_I64_S;
  case WebAssembly::COPYSIGN_F32: return WebAssembly::COPYSIGN_F32_S;
  case WebAssembly::COPYSIGN_F64: return WebAssembly::COPYSIGN_F64_S;
  case WebAssembly::COPY_EXCEPT_REF: return WebAssembly::COPY_EXCEPT_REF_S;
  case WebAssembly::COPY_F32: return WebAssembly::COPY_F32_S;
  case WebAssembly::COPY_F64: return WebAssembly::COPY_F64_S;
  case WebAssembly::COPY_I32: return WebAssembly::COPY_I32_S;
  case WebAssembly::COPY_I64: return WebAssembly::COPY_I64_S;
  case WebAssembly::COPY_V128: return WebAssembly::COPY_V128_S;
  case WebAssembly::CTZ_I32: return WebAssembly::CTZ_I32_S;
  case WebAssembly::CTZ_I64: return WebAssembly::CTZ_I64_S;
  case WebAssembly::CURRENT_MEMORY_I32: return WebAssembly::CURRENT_MEMORY_I32_S;
  case WebAssembly::DIV_F32: return WebAssembly::DIV_F32_S;
  case WebAssembly::DIV_F64: return WebAssembly::DIV_F64_S;
  case WebAssembly::DIV_S_I32: return WebAssembly::DIV_S_I32_S;
  case WebAssembly::DIV_S_I64: return WebAssembly::DIV_S_I64_S;
  case WebAssembly::DIV_U_I32: return WebAssembly::DIV_U_I32_S;
  case WebAssembly::DIV_U_I64: return WebAssembly::DIV_U_I64_S;
  case WebAssembly::DROP_EXCEPT_REF: return WebAssembly::DROP_EXCEPT_REF_S;
  case WebAssembly::DROP_F32: return WebAssembly::DROP_F32_S;
  case WebAssembly::DROP_F64: return WebAssembly::DROP_F64_S;
  case WebAssembly::DROP_I32: return WebAssembly::DROP_I32_S;
  case WebAssembly::DROP_I64: return WebAssembly::DROP_I64_S;
  case WebAssembly::DROP_V128: return WebAssembly::DROP_V128_S;
  case WebAssembly::END_BLOCK: return WebAssembly::END_BLOCK_S;
  case WebAssembly::END_FUNCTION: return WebAssembly::END_FUNCTION_S;
  case WebAssembly::END_LOOP: return WebAssembly::END_LOOP_S;
  case WebAssembly::END_TRY: return WebAssembly::END_TRY_S;
  case WebAssembly::EQZ_I32: return WebAssembly::EQZ_I32_S;
  case WebAssembly::EQZ_I64: return WebAssembly::EQZ_I64_S;
  case WebAssembly::EQ_F32: return WebAssembly::EQ_F32_S;
  case WebAssembly::EQ_F64: return WebAssembly::EQ_F64_S;
  case WebAssembly::EQ_I32: return WebAssembly::EQ_I32_S;
  case WebAssembly::EQ_I64: return WebAssembly::EQ_I64_S;
  case WebAssembly::F32_CONVERT_S_I32: return WebAssembly::F32_CONVERT_S_I32_S;
  case WebAssembly::F32_CONVERT_S_I64: return WebAssembly::F32_CONVERT_S_I64_S;
  case WebAssembly::F32_CONVERT_U_I32: return WebAssembly::F32_CONVERT_U_I32_S;
  case WebAssembly::F32_CONVERT_U_I64: return WebAssembly::F32_CONVERT_U_I64_S;
  case WebAssembly::F32_DEMOTE_F64: return WebAssembly::F32_DEMOTE_F64_S;
  case WebAssembly::F32_REINTERPRET_I32: return WebAssembly::F32_REINTERPRET_I32_S;
  case WebAssembly::F64_CONVERT_S_I32: return WebAssembly::F64_CONVERT_S_I32_S;
  case WebAssembly::F64_CONVERT_S_I64: return WebAssembly::F64_CONVERT_S_I64_S;
  case WebAssembly::F64_CONVERT_U_I32: return WebAssembly::F64_CONVERT_U_I32_S;
  case WebAssembly::F64_CONVERT_U_I64: return WebAssembly::F64_CONVERT_U_I64_S;
  case WebAssembly::F64_PROMOTE_F32: return WebAssembly::F64_PROMOTE_F32_S;
  case WebAssembly::F64_REINTERPRET_I64: return WebAssembly::F64_REINTERPRET_I64_S;
  case WebAssembly::FALLTHROUGH_RETURN_EXCEPT_REF: return WebAssembly::FALLTHROUGH_RETURN_EXCEPT_REF_S;
  case WebAssembly::FALLTHROUGH_RETURN_F32: return WebAssembly::FALLTHROUGH_RETURN_F32_S;
  case WebAssembly::FALLTHROUGH_RETURN_F64: return WebAssembly::FALLTHROUGH_RETURN_F64_S;
  case WebAssembly::FALLTHROUGH_RETURN_I32: return WebAssembly::FALLTHROUGH_RETURN_I32_S;
  case WebAssembly::FALLTHROUGH_RETURN_I64: return WebAssembly::FALLTHROUGH_RETURN_I64_S;
  case WebAssembly::FALLTHROUGH_RETURN_VOID: return WebAssembly::FALLTHROUGH_RETURN_VOID_S;
  case WebAssembly::FALLTHROUGH_RETURN_v16i8: return WebAssembly::FALLTHROUGH_RETURN_v16i8_S;
  case WebAssembly::FALLTHROUGH_RETURN_v4f32: return WebAssembly::FALLTHROUGH_RETURN_v4f32_S;
  case WebAssembly::FALLTHROUGH_RETURN_v4i32: return WebAssembly::FALLTHROUGH_RETURN_v4i32_S;
  case WebAssembly::FALLTHROUGH_RETURN_v8i16: return WebAssembly::FALLTHROUGH_RETURN_v8i16_S;
  case WebAssembly::FALLTHROUGH_RETURN_v2f64: return WebAssembly::FALLTHROUGH_RETURN_v2f64_S;
  case WebAssembly::FALLTHROUGH_RETURN_v2i64: return WebAssembly::FALLTHROUGH_RETURN_v2i64_S;
  case WebAssembly::FLOOR_F32: return WebAssembly::FLOOR_F32_S;
  case WebAssembly::FLOOR_F64: return WebAssembly::FLOOR_F64_S;
  case WebAssembly::FP_TO_SINT_I32_F32: return WebAssembly::FP_TO_SINT_I32_F32_S;
  case WebAssembly::FP_TO_SINT_I32_F64: return WebAssembly::FP_TO_SINT_I32_F64_S;
  case WebAssembly::FP_TO_SINT_I64_F32: return WebAssembly::FP_TO_SINT_I64_F32_S;
  case WebAssembly::FP_TO_SINT_I64_F64: return WebAssembly::FP_TO_SINT_I64_F64_S;
  case WebAssembly::FP_TO_UINT_I32_F32: return WebAssembly::FP_TO_UINT_I32_F32_S;
  case WebAssembly::FP_TO_UINT_I32_F64: return WebAssembly::FP_TO_UINT_I32_F64_S;
  case WebAssembly::FP_TO_UINT_I64_F32: return WebAssembly::FP_TO_UINT_I64_F32_S;
  case WebAssembly::FP_TO_UINT_I64_F64: return WebAssembly::FP_TO_UINT_I64_F64_S;
  case WebAssembly::GET_GLOBAL_EXCEPT_REF: return WebAssembly::GET_GLOBAL_EXCEPT_REF_S;
  case WebAssembly::GET_GLOBAL_F32: return WebAssembly::GET_GLOBAL_F32_S;
  case WebAssembly::GET_GLOBAL_F64: return WebAssembly::GET_GLOBAL_F64_S;
  case WebAssembly::GET_GLOBAL_I32: return WebAssembly::GET_GLOBAL_I32_S;
  case WebAssembly::GET_GLOBAL_I64: return WebAssembly::GET_GLOBAL_I64_S;
  case WebAssembly::GET_GLOBAL_V128: return WebAssembly::GET_GLOBAL_V128_S;
  case WebAssembly::GET_LOCAL_EXCEPT_REF: return WebAssembly::GET_LOCAL_EXCEPT_REF_S;
  case WebAssembly::GET_LOCAL_F32: return WebAssembly::GET_LOCAL_F32_S;
  case WebAssembly::GET_LOCAL_F64: return WebAssembly::GET_LOCAL_F64_S;
  case WebAssembly::GET_LOCAL_I32: return WebAssembly::GET_LOCAL_I32_S;
  case WebAssembly::GET_LOCAL_I64: return WebAssembly::GET_LOCAL_I64_S;
  case WebAssembly::GET_LOCAL_V128: return WebAssembly::GET_LOCAL_V128_S;
  case WebAssembly::GE_F32: return WebAssembly::GE_F32_S;
  case WebAssembly::GE_F64: return WebAssembly::GE_F64_S;
  case WebAssembly::GE_S_I32: return WebAssembly::GE_S_I32_S;
  case WebAssembly::GE_S_I64: return WebAssembly::GE_S_I64_S;
  case WebAssembly::GE_U_I32: return WebAssembly::GE_U_I32_S;
  case WebAssembly::GE_U_I64: return WebAssembly::GE_U_I64_S;
  case WebAssembly::GROW_MEMORY_I32: return WebAssembly::GROW_MEMORY_I32_S;
  case WebAssembly::GT_F32: return WebAssembly::GT_F32_S;
  case WebAssembly::GT_F64: return WebAssembly::GT_F64_S;
  case WebAssembly::GT_S_I32: return WebAssembly::GT_S_I32_S;
  case WebAssembly::GT_S_I64: return WebAssembly::GT_S_I64_S;
  case WebAssembly::GT_U_I32: return WebAssembly::GT_U_I32_S;
  case WebAssembly::GT_U_I64: return WebAssembly::GT_U_I64_S;
  case WebAssembly::I32_EXTEND16_S_I32: return WebAssembly::I32_EXTEND16_S_I32_S;
  case WebAssembly::I32_EXTEND8_S_I32: return WebAssembly::I32_EXTEND8_S_I32_S;
  case WebAssembly::I32_REINTERPRET_F32: return WebAssembly::I32_REINTERPRET_F32_S;
  case WebAssembly::I32_TRUNC_S_F32: return WebAssembly::I32_TRUNC_S_F32_S;
  case WebAssembly::I32_TRUNC_S_F64: return WebAssembly::I32_TRUNC_S_F64_S;
  case WebAssembly::I32_TRUNC_S_SAT_F32: return WebAssembly::I32_TRUNC_S_SAT_F32_S;
  case WebAssembly::I32_TRUNC_S_SAT_F64: return WebAssembly::I32_TRUNC_S_SAT_F64_S;
  case WebAssembly::I32_TRUNC_U_F32: return WebAssembly::I32_TRUNC_U_F32_S;
  case WebAssembly::I32_TRUNC_U_F64: return WebAssembly::I32_TRUNC_U_F64_S;
  case WebAssembly::I32_TRUNC_U_SAT_F32: return WebAssembly::I32_TRUNC_U_SAT_F32_S;
  case WebAssembly::I32_TRUNC_U_SAT_F64: return WebAssembly::I32_TRUNC_U_SAT_F64_S;
  case WebAssembly::I32_WRAP_I64: return WebAssembly::I32_WRAP_I64_S;
  case WebAssembly::I64_EXTEND16_S_I64: return WebAssembly::I64_EXTEND16_S_I64_S;
  case WebAssembly::I64_EXTEND32_S_I64: return WebAssembly::I64_EXTEND32_S_I64_S;
  case WebAssembly::I64_EXTEND8_S_I64: return WebAssembly::I64_EXTEND8_S_I64_S;
  case WebAssembly::I64_EXTEND_S_I32: return WebAssembly::I64_EXTEND_S_I32_S;
  case WebAssembly::I64_EXTEND_U_I32: return WebAssembly::I64_EXTEND_U_I32_S;
  case WebAssembly::I64_REINTERPRET_F64: return WebAssembly::I64_REINTERPRET_F64_S;
  case WebAssembly::I64_TRUNC_S_F32: return WebAssembly::I64_TRUNC_S_F32_S;
  case WebAssembly::I64_TRUNC_S_F64: return WebAssembly::I64_TRUNC_S_F64_S;
  case WebAssembly::I64_TRUNC_S_SAT_F32: return WebAssembly::I64_TRUNC_S_SAT_F32_S;
  case WebAssembly::I64_TRUNC_S_SAT_F64: return WebAssembly::I64_TRUNC_S_SAT_F64_S;
  case WebAssembly::I64_TRUNC_U_F32: return WebAssembly::I64_TRUNC_U_F32_S;
  case WebAssembly::I64_TRUNC_U_F64: return WebAssembly::I64_TRUNC_U_F64_S;
  case WebAssembly::I64_TRUNC_U_SAT_F32: return WebAssembly::I64_TRUNC_U_SAT_F32_S;
  case WebAssembly::I64_TRUNC_U_SAT_F64: return WebAssembly::I64_TRUNC_U_SAT_F64_S;
  case WebAssembly::LE_F32: return WebAssembly::LE_F32_S;
  case WebAssembly::LE_F64: return WebAssembly::LE_F64_S;
  case WebAssembly::LE_S_I32: return WebAssembly::LE_S_I32_S;
  case WebAssembly::LE_S_I64: return WebAssembly::LE_S_I64_S;
  case WebAssembly::LE_U_I32: return WebAssembly::LE_U_I32_S;
  case WebAssembly::LE_U_I64: return WebAssembly::LE_U_I64_S;
  case WebAssembly::LOAD16_S_I32: return WebAssembly::LOAD16_S_I32_S;
  case WebAssembly::LOAD16_S_I64: return WebAssembly::LOAD16_S_I64_S;
  case WebAssembly::LOAD16_U_I32: return WebAssembly::LOAD16_U_I32_S;
  case WebAssembly::LOAD16_U_I64: return WebAssembly::LOAD16_U_I64_S;
  case WebAssembly::LOAD32_S_I64: return WebAssembly::LOAD32_S_I64_S;
  case WebAssembly::LOAD32_U_I64: return WebAssembly::LOAD32_U_I64_S;
  case WebAssembly::LOAD8_S_I32: return WebAssembly::LOAD8_S_I32_S;
  case WebAssembly::LOAD8_S_I64: return WebAssembly::LOAD8_S_I64_S;
  case WebAssembly::LOAD8_U_I32: return WebAssembly::LOAD8_U_I32_S;
  case WebAssembly::LOAD8_U_I64: return WebAssembly::LOAD8_U_I64_S;
  case WebAssembly::LOAD_F32: return WebAssembly::LOAD_F32_S;
  case WebAssembly::LOAD_F64: return WebAssembly::LOAD_F64_S;
  case WebAssembly::LOAD_I32: return WebAssembly::LOAD_I32_S;
  case WebAssembly::LOAD_I64: return WebAssembly::LOAD_I64_S;
  case WebAssembly::LOOP: return WebAssembly::LOOP_S;
  case WebAssembly::LT_F32: return WebAssembly::LT_F32_S;
  case WebAssembly::LT_F64: return WebAssembly::LT_F64_S;
  case WebAssembly::LT_S_I32: return WebAssembly::LT_S_I32_S;
  case WebAssembly::LT_S_I64: return WebAssembly::LT_S_I64_S;
  case WebAssembly::LT_U_I32: return WebAssembly::LT_U_I32_S;
  case WebAssembly::LT_U_I64: return WebAssembly::LT_U_I64_S;
  case WebAssembly::MAX_F32: return WebAssembly::MAX_F32_S;
  case WebAssembly::MAX_F64: return WebAssembly::MAX_F64_S;
  case WebAssembly::MEMORY_GROW_I32: return WebAssembly::MEMORY_GROW_I32_S;
  case WebAssembly::MEMORY_SIZE_I32: return WebAssembly::MEMORY_SIZE_I32_S;
  case WebAssembly::MEM_GROW_I32: return WebAssembly::MEM_GROW_I32_S;
  case WebAssembly::MEM_SIZE_I32: return WebAssembly::MEM_SIZE_I32_S;
  case WebAssembly::MIN_F32: return WebAssembly::MIN_F32_S;
  case WebAssembly::MIN_F64: return WebAssembly::MIN_F64_S;
  case WebAssembly::MUL_F32: return WebAssembly::MUL_F32_S;
  case WebAssembly::MUL_F32x4: return WebAssembly::MUL_F32x4_S;
  case WebAssembly::MUL_F64: return WebAssembly::MUL_F64_S;
  case WebAssembly::MUL_I16x8: return WebAssembly::MUL_I16x8_S;
  case WebAssembly::MUL_I32: return WebAssembly::MUL_I32_S;
  case WebAssembly::MUL_I32x4: return WebAssembly::MUL_I32x4_S;
  case WebAssembly::MUL_I64: return WebAssembly::MUL_I64_S;
  case WebAssembly::MUL_I8x16: return WebAssembly::MUL_I8x16_S;
  case WebAssembly::NEAREST_F32: return WebAssembly::NEAREST_F32_S;
  case WebAssembly::NEAREST_F64: return WebAssembly::NEAREST_F64_S;
  case WebAssembly::NEG_F32: return WebAssembly::NEG_F32_S;
  case WebAssembly::NEG_F64: return WebAssembly::NEG_F64_S;
  case WebAssembly::NE_F32: return WebAssembly::NE_F32_S;
  case WebAssembly::NE_F64: return WebAssembly::NE_F64_S;
  case WebAssembly::NE_I32: return WebAssembly::NE_I32_S;
  case WebAssembly::NE_I64: return WebAssembly::NE_I64_S;
  case WebAssembly::NOP: return WebAssembly::NOP_S;
  case WebAssembly::OR_I32: return WebAssembly::OR_I32_S;
  case WebAssembly::OR_I64: return WebAssembly::OR_I64_S;
  case WebAssembly::PCALL_INDIRECT_EXCEPT_REF: return WebAssembly::PCALL_INDIRECT_EXCEPT_REF_S;
  case WebAssembly::PCALL_INDIRECT_F32: return WebAssembly::PCALL_INDIRECT_F32_S;
  case WebAssembly::PCALL_INDIRECT_F64: return WebAssembly::PCALL_INDIRECT_F64_S;
  case WebAssembly::PCALL_INDIRECT_I32: return WebAssembly::PCALL_INDIRECT_I32_S;
  case WebAssembly::PCALL_INDIRECT_I64: return WebAssembly::PCALL_INDIRECT_I64_S;
  case WebAssembly::PCALL_INDIRECT_VOID: return WebAssembly::PCALL_INDIRECT_VOID_S;
  case WebAssembly::PCALL_INDIRECT_v16i8: return WebAssembly::PCALL_INDIRECT_v16i8_S;
  case WebAssembly::PCALL_INDIRECT_v4f32: return WebAssembly::PCALL_INDIRECT_v4f32_S;
  case WebAssembly::PCALL_INDIRECT_v4i32: return WebAssembly::PCALL_INDIRECT_v4i32_S;
  case WebAssembly::PCALL_INDIRECT_v8i16: return WebAssembly::PCALL_INDIRECT_v8i16_S;
  case WebAssembly::POPCNT_I32: return WebAssembly::POPCNT_I32_S;
  case WebAssembly::POPCNT_I64: return WebAssembly::POPCNT_I64_S;
  case WebAssembly::REM_S_I32: return WebAssembly::REM_S_I32_S;
  case WebAssembly::REM_S_I64: return WebAssembly::REM_S_I64_S;
  case WebAssembly::REM_U_I32: return WebAssembly::REM_U_I32_S;
  case WebAssembly::REM_U_I64: return WebAssembly::REM_U_I64_S;
  case WebAssembly::RETHROW: return WebAssembly::RETHROW_S;
  case WebAssembly::RETHROW_TO_CALLER: return WebAssembly::RETHROW_TO_CALLER_S;
  case WebAssembly::RETURN_EXCEPT_REF: return WebAssembly::RETURN_EXCEPT_REF_S;
  case WebAssembly::RETURN_F32: return WebAssembly::RETURN_F32_S;
  case WebAssembly::RETURN_F64: return WebAssembly::RETURN_F64_S;
  case WebAssembly::RETURN_I32: return WebAssembly::RETURN_I32_S;
  case WebAssembly::RETURN_I64: return WebAssembly::RETURN_I64_S;
  case WebAssembly::RETURN_VOID: return WebAssembly::RETURN_VOID_S;
  case WebAssembly::RETURN_v16i8: return WebAssembly::RETURN_v16i8_S;
  case WebAssembly::RETURN_v4f32: return WebAssembly::RETURN_v4f32_S;
  case WebAssembly::RETURN_v4i32: return WebAssembly::RETURN_v4i32_S;
  case WebAssembly::RETURN_v8i16: return WebAssembly::RETURN_v8i16_S;
  case WebAssembly::ROTL_I32: return WebAssembly::ROTL_I32_S;
  case WebAssembly::ROTL_I64: return WebAssembly::ROTL_I64_S;
  case WebAssembly::ROTR_I32: return WebAssembly::ROTR_I32_S;
  case WebAssembly::ROTR_I64: return WebAssembly::ROTR_I64_S;
  case WebAssembly::SELECT_EXCEPT_REF: return WebAssembly::SELECT_EXCEPT_REF_S;
  case WebAssembly::SELECT_F32: return WebAssembly::SELECT_F32_S;
  case WebAssembly::SELECT_F64: return WebAssembly::SELECT_F64_S;
  case WebAssembly::SELECT_I32: return WebAssembly::SELECT_I32_S;
  case WebAssembly::SELECT_I64: return WebAssembly::SELECT_I64_S;
  case WebAssembly::SET_GLOBAL_EXCEPT_REF: return WebAssembly::SET_GLOBAL_EXCEPT_REF_S;
  case WebAssembly::SET_GLOBAL_F32: return WebAssembly::SET_GLOBAL_F32_S;
  case WebAssembly::SET_GLOBAL_F64: return WebAssembly::SET_GLOBAL_F64_S;
  case WebAssembly::SET_GLOBAL_I32: return WebAssembly::SET_GLOBAL_I32_S;
  case WebAssembly::SET_GLOBAL_I64: return WebAssembly::SET_GLOBAL_I64_S;
  case WebAssembly::SET_GLOBAL_V128: return WebAssembly::SET_GLOBAL_V128_S;
  case WebAssembly::SET_LOCAL_EXCEPT_REF: return WebAssembly::SET_LOCAL_EXCEPT_REF_S;
  case WebAssembly::SET_LOCAL_F32: return WebAssembly::SET_LOCAL_F32_S;
  case WebAssembly::SET_LOCAL_F64: return WebAssembly::SET_LOCAL_F64_S;
  case WebAssembly::SET_LOCAL_I32: return WebAssembly::SET_LOCAL_I32_S;
  case WebAssembly::SET_LOCAL_I64: return WebAssembly::SET_LOCAL_I64_S;
  case WebAssembly::SET_LOCAL_V128: return WebAssembly::SET_LOCAL_V128_S;
  case WebAssembly::SHL_I32: return WebAssembly::SHL_I32_S;
  case WebAssembly::SHL_I64: return WebAssembly::SHL_I64_S;
  case WebAssembly::SHR_S_I32: return WebAssembly::SHR_S_I32_S;
  case WebAssembly::SHR_S_I64: return WebAssembly::SHR_S_I64_S;
  case WebAssembly::SHR_U_I32: return WebAssembly::SHR_U_I32_S;
  case WebAssembly::SHR_U_I64: return WebAssembly::SHR_U_I64_S;
  case WebAssembly::SQRT_F32: return WebAssembly::SQRT_F32_S;
  case WebAssembly::SQRT_F64: return WebAssembly::SQRT_F64_S;
  case WebAssembly::STORE16_I32: return WebAssembly::STORE16_I32_S;
  case WebAssembly::STORE16_I64: return WebAssembly::STORE16_I64_S;
  case WebAssembly::STORE32_I64: return WebAssembly::STORE32_I64_S;
  case WebAssembly::STORE8_I32: return WebAssembly::STORE8_I32_S;
  case WebAssembly::STORE8_I64: return WebAssembly::STORE8_I64_S;
  case WebAssembly::STORE_F32: return WebAssembly::STORE_F32_S;
  case WebAssembly::STORE_F64: return WebAssembly::STORE_F64_S;
  case WebAssembly::STORE_I32: return WebAssembly::STORE_I32_S;
  case WebAssembly::STORE_I64: return WebAssembly::STORE_I64_S;
  case WebAssembly::SUB_F32: return WebAssembly::SUB_F32_S;
  case WebAssembly::SUB_F32x4: return WebAssembly::SUB_F32x4_S;
  case WebAssembly::SUB_F64: return WebAssembly::SUB_F64_S;
  case WebAssembly::SUB_I16x8: return WebAssembly::SUB_I16x8_S;
  case WebAssembly::SUB_I32: return WebAssembly::SUB_I32_S;
  case WebAssembly::SUB_I32x4: return WebAssembly::SUB_I32x4_S;
  case WebAssembly::SUB_I64: return WebAssembly::SUB_I64_S;
  case WebAssembly::SUB_I8x16: return WebAssembly::SUB_I8x16_S;
  case WebAssembly::TEE_EXCEPT_REF: return WebAssembly::TEE_EXCEPT_REF_S;
  case WebAssembly::TEE_F32: return WebAssembly::TEE_F32_S;
  case WebAssembly::TEE_F64: return WebAssembly::TEE_F64_S;
  case WebAssembly::TEE_I32: return WebAssembly::TEE_I32_S;
  case WebAssembly::TEE_I64: return WebAssembly::TEE_I64_S;
  case WebAssembly::TEE_LOCAL_EXCEPT_REF: return WebAssembly::TEE_LOCAL_EXCEPT_REF_S;
  case WebAssembly::TEE_LOCAL_F32: return WebAssembly::TEE_LOCAL_F32_S;
  case WebAssembly::TEE_LOCAL_F64: return WebAssembly::TEE_LOCAL_F64_S;
  case WebAssembly::TEE_LOCAL_I32: return WebAssembly::TEE_LOCAL_I32_S;
  case WebAssembly::TEE_LOCAL_I64: return WebAssembly::TEE_LOCAL_I64_S;
  case WebAssembly::TEE_LOCAL_V128: return WebAssembly::TEE_LOCAL_V128_S;
  case WebAssembly::TEE_V128: return WebAssembly::TEE_V128_S;
  case WebAssembly::THROW_I32: return WebAssembly::THROW_I32_S;
  case WebAssembly::THROW_I64: return WebAssembly::THROW_I64_S;
  case WebAssembly::TRUNC_F32: return WebAssembly::TRUNC_F32_S;
  case WebAssembly::TRUNC_F64: return WebAssembly::TRUNC_F64_S;
  case WebAssembly::TRY: return WebAssembly::TRY_S;
  case WebAssembly::UNREACHABLE: return WebAssembly::UNREACHABLE_S;
  case WebAssembly::XOR_I32: return WebAssembly::XOR_I32_S;
  case WebAssembly::XOR_I64: return WebAssembly::XOR_I64_S;
  }
}
