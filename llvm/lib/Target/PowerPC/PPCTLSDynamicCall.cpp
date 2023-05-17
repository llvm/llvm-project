//===---------- PPCTLSDynamicCall.cpp - TLS Dynamic Call Fixup ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass expands ADDItls{ld,gd}LADDR[32] machine instructions into
// separate ADDItls[gd]L[32] and GETtlsADDR[32] instructions, both of
// which define GPR3.  A copy is added from GPR3 to the target virtual
// register of the original instruction.  The GETtlsADDR[32] is really
// a call instruction, so its target register is constrained to be GPR3.
// This is not true of ADDItls[gd]L[32], but there is a legacy linker
// optimization bug that requires the target register of the addi of
// a local- or general-dynamic TLS access sequence to be GPR3.
//
// This is done in a late pass so that TLS variable accesses can be
// fully commoned by MachineCSE.
//
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "PPCInstrBuilder.h"
#include "PPCInstrInfo.h"
#include "PPCTargetMachine.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "ppc-tls-dynamic-call"

namespace {
  struct PPCTLSDynamicCall : public MachineFunctionPass {
    static char ID;
    PPCTLSDynamicCall() : MachineFunctionPass(ID) {
      initializePPCTLSDynamicCallPass(*PassRegistry::getPassRegistry());
    }

    const PPCInstrInfo *TII;

protected:
    bool processBlock(MachineBasicBlock &MBB) {
      bool Changed = false;
      bool NeedFence = true;
      bool Is64Bit = MBB.getParent()->getSubtarget<PPCSubtarget>().isPPC64();
      bool IsAIX = MBB.getParent()->getSubtarget<PPCSubtarget>().isAIXABI();
      bool IsPCREL = false;

      for (MachineBasicBlock::iterator I = MBB.begin(), IE = MBB.end();
           I != IE;) {
        MachineInstr &MI = *I;
        IsPCREL = isPCREL(MI);

        if (MI.getOpcode() != PPC::ADDItlsgdLADDR &&
            MI.getOpcode() != PPC::ADDItlsldLADDR &&
            MI.getOpcode() != PPC::ADDItlsgdLADDR32 &&
            MI.getOpcode() != PPC::ADDItlsldLADDR32 &&
            MI.getOpcode() != PPC::TLSGDAIX &&
            MI.getOpcode() != PPC::TLSGDAIX8 && !IsPCREL) {
          // Although we create ADJCALLSTACKDOWN and ADJCALLSTACKUP
          // as scheduling fences, we skip creating fences if we already
          // have existing ADJCALLSTACKDOWN/UP to avoid nesting,
          // which causes verification error with -verify-machineinstrs.
          if (MI.getOpcode() == PPC::ADJCALLSTACKDOWN)
            NeedFence = false;
          else if (MI.getOpcode() == PPC::ADJCALLSTACKUP)
            NeedFence = true;

          ++I;
          continue;
        }

        LLVM_DEBUG(dbgs() << "TLS Dynamic Call Fixup:\n    " << MI);

        Register OutReg = MI.getOperand(0).getReg();
        Register InReg = PPC::NoRegister;
        Register GPR3 = Is64Bit ? PPC::X3 : PPC::R3;
        Register GPR4 = Is64Bit ? PPC::X4 : PPC::R4;
        if (!IsPCREL)
          InReg = MI.getOperand(1).getReg();
        DebugLoc DL = MI.getDebugLoc();

        unsigned Opc1, Opc2;
        switch (MI.getOpcode()) {
        default:
          llvm_unreachable("Opcode inconsistency error");
        case PPC::ADDItlsgdLADDR:
          Opc1 = PPC::ADDItlsgdL;
          Opc2 = PPC::GETtlsADDR;
          break;
        case PPC::ADDItlsldLADDR:
          Opc1 = PPC::ADDItlsldL;
          Opc2 = PPC::GETtlsldADDR;
          break;
        case PPC::ADDItlsgdLADDR32:
          Opc1 = PPC::ADDItlsgdL32;
          Opc2 = PPC::GETtlsADDR32;
          break;
        case PPC::ADDItlsldLADDR32:
          Opc1 = PPC::ADDItlsldL32;
          Opc2 = PPC::GETtlsldADDR32;
          break;
        case PPC::TLSGDAIX8:
          // TLSGDAIX8 is expanded to two copies and GET_TLS_ADDR, so we only
          // set Opc2 here.
          Opc2 = PPC::GETtlsADDR64AIX;
          break;
        case PPC::TLSGDAIX:
          // TLSGDAIX is expanded to two copies and GET_TLS_ADDR, so we only
          // set Opc2 here.
          Opc2 = PPC::GETtlsADDR32AIX;
          break;
        case PPC::PADDI8pc:
          assert(IsPCREL && "Expecting General/Local Dynamic PCRel");
          Opc1 = PPC::PADDI8pc;
          Opc2 = MI.getOperand(2).getTargetFlags() ==
                         PPCII::MO_GOT_TLSGD_PCREL_FLAG
                     ? PPC::GETtlsADDRPCREL
                     : PPC::GETtlsldADDRPCREL;
        }

        // We create ADJCALLSTACKUP and ADJCALLSTACKDOWN around _tls_get_addr
        // as scheduling fence to avoid it is scheduled before
        // mflr in the prologue and the address in LR is clobbered (PR25839).
        // We don't really need to save data to the stack - the clobbered
        // registers are already saved when the SDNode (e.g. PPCaddiTlsgdLAddr)
        // gets translated to the pseudo instruction (e.g. ADDItlsgdLADDR).
        if (NeedFence)
          BuildMI(MBB, I, DL, TII->get(PPC::ADJCALLSTACKDOWN)).addImm(0)
                                                              .addImm(0);

        if (IsAIX) {
          // The variable offset and region handle are copied in r4 and r3. The
          // copies are followed by GETtlsADDR32AIX/GETtlsADDR64AIX.
          BuildMI(MBB, I, DL, TII->get(TargetOpcode::COPY), GPR4)
              .addReg(MI.getOperand(1).getReg());
          BuildMI(MBB, I, DL, TII->get(TargetOpcode::COPY), GPR3)
              .addReg(MI.getOperand(2).getReg());
          BuildMI(MBB, I, DL, TII->get(Opc2), GPR3).addReg(GPR3).addReg(GPR4);
        } else {
          MachineInstr *Addi;
          if (IsPCREL) {
            Addi = BuildMI(MBB, I, DL, TII->get(Opc1), GPR3).addImm(0);
          } else {
            // Expand into two ops built prior to the existing instruction.
            assert(InReg != PPC::NoRegister && "Operand must be a register");
            Addi = BuildMI(MBB, I, DL, TII->get(Opc1), GPR3).addReg(InReg);
          }

          Addi->addOperand(MI.getOperand(2));

          MachineInstr *Call =
              (BuildMI(MBB, I, DL, TII->get(Opc2), GPR3).addReg(GPR3));
          if (IsPCREL)
            Call->addOperand(MI.getOperand(2));
          else
            Call->addOperand(MI.getOperand(3));
        }
        if (NeedFence)
          BuildMI(MBB, I, DL, TII->get(PPC::ADJCALLSTACKUP)).addImm(0).addImm(0);

        BuildMI(MBB, I, DL, TII->get(TargetOpcode::COPY), OutReg)
          .addReg(GPR3);

        // Move past the original instruction and remove it.
        ++I;
        MI.removeFromParent();

        Changed = true;
      }

      return Changed;
    }

public:
  bool isPCREL(const MachineInstr &MI) {
    return (MI.getOpcode() == PPC::PADDI8pc) &&
           (MI.getOperand(2).getTargetFlags() ==
                PPCII::MO_GOT_TLSGD_PCREL_FLAG ||
            MI.getOperand(2).getTargetFlags() ==
                PPCII::MO_GOT_TLSLD_PCREL_FLAG);
  }

    bool runOnMachineFunction(MachineFunction &MF) override {
      TII = MF.getSubtarget<PPCSubtarget>().getInstrInfo();

      bool Changed = false;

      for (MachineBasicBlock &B : llvm::make_early_inc_range(MF))
        if (processBlock(B))
          Changed = true;

      return Changed;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<LiveIntervals>();
      AU.addRequired<SlotIndexes>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  };
}

INITIALIZE_PASS_BEGIN(PPCTLSDynamicCall, DEBUG_TYPE,
                      "PowerPC TLS Dynamic Call Fixup", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_END(PPCTLSDynamicCall, DEBUG_TYPE,
                    "PowerPC TLS Dynamic Call Fixup", false, false)

char PPCTLSDynamicCall::ID = 0;
FunctionPass*
llvm::createPPCTLSDynamicCallPass() { return new PPCTLSDynamicCall(); }
