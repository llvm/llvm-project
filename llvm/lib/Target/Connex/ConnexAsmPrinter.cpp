//===-- ConnexAsmPrinter.cpp - Connex LLVM assembly writer ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to the Connex assembly language.
//
//===----------------------------------------------------------------------===//

#include "Connex.h"
#include "ConnexConfig.h"
#include "ConnexInstrInfo.h"
#include "ConnexMCInstLower.h"
#include "ConnexTargetMachine.h"
#include "MCTargetDesc/ConnexInstPrinter.h"
#include "TargetInfo/ConnexTargetInfo.h"
#include "Misc.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
// TODO: #include "BTFDebug.h"

using namespace llvm;

// Inspired from llvm/lib/CodeGen/TargetPassConfig.cpp
static cl::opt<bool> EnableCorrectBBsASMPrint(
    "enable-correct-asm-print", cl::Hidden, cl::init(true),
    cl::desc(
        "Correct the BBs of the 2nd innermost loop in loop nests of kernels "
        "and use normally REPEAT for it and 'host-side OPINCAA C++ for' as "
        "the innermost loop"));

static cl::opt<bool> TreatRepeat2ndInnerLoopGlobalTmp(
    "treat-repeat-2nd-inner-loop", cl::Hidden, cl::init(true),
    cl::desc("Treat well 2nd inner loop in kernel and use normally REPEAT "
             "for it and host-side OPINCAA C++ for() as the inner loop"));

#define DEBUG_TYPE "asm-printer"

namespace {

// Declarations for adapted RPO and DFS traversals of the CFG
typedef bool (*CompareBBs)(MachineBasicBlock &b1, MachineBasicBlock &b2);
//
// We declare these vars static outside the class to avoid some strange C++
//   linker errors (used for adapted RPO or DFS traversal of the CFG).
static std::map<MachineBasicBlock *, bool> visitedMBB;
static std::map<MachineBasicBlock *, int> finishingTimeMBB; // DFS finish time
static std::vector<MachineBasicBlock *> sortedListMBB;

bool isMBBWithInlineAsmString(MachineBasicBlock *MBB, std::string strToSearch) {
  LLVM_DEBUG(
      dbgs() << "Entered isMBBWithOPINCAAKernelEndMarker(MBB->getName() = "
             << MBB->getName() << ")\n");

  for (auto MIItr = MBB->begin(), MBBend = MBB->end(); MIItr != MBBend;
       ++MIItr) {
    MachineInstr *MI = &(*MIItr);

    if (MI->isInlineAsm()) {
      LLVM_DEBUG(
          dbgs()
          << "  isMBBWithOPINCAAKernelEndMarker(): found INLINEASM *MI = "
          << *MI << "\n");

      // See http://llvm.org/docs/doxygen/html/classllvm_1_1MachineInstr.html
      for (unsigned index = 0; index < MI->getNumOperands(); index++) {
        MachineOperand *miOpnd;
        miOpnd = &(MI->getOperand(index));

        if (miOpnd->isSymbol()) {
          std::string symStr = miOpnd->getSymbolName();
          LLVM_DEBUG(dbgs() << "  isMBBWithOPINCAAKernelEndMarker(): symStr = "
                            << symStr << "\n");

          if (symStr.find(strToSearch) != std::string::npos) {
            LLVM_DEBUG(
                dbgs()
                << "  isMBBWithOPINCAAKernelEndMarker(): Found INLINEASM "
                   "with strToSearch in the symbol "
                   "operand\n");
            //"with host-side for loop"
            return true;
          }
        }
      }
    }
  }

  return false;
} // End isMBBWithInlineAsmString()

class ConnexAsmPrinter : public AsmPrinter {
#include "ConnexAsmPrinterLoopNests.h"
  /*
  TODO:
private:
  BTFDebug *BTF;
  explicit BPFAsmPrinter(TargetMachine &TM,
                         std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)), BTF(nullptr) {}
  */
public:
  explicit ConnexAsmPrinter(TargetMachine &TM,
                            std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)) {}

  StringRef getPassName() const override { return "Connex Assembly Printer"; }

  /*
  // Inspired from BPF's BPFAsmPrinter::emitInstruction(const MachineInstr *MI)
  void emitInstruction(const MachineInstr *MI) {
    MCInst TmpInst;

    //if (!BTF || !BTF->InstLower(MI, TmpInst)) {
    ConnexMCInstLower MCInstLowering(OutContext, *this);
    MCInstLowering.Lower(MI, TmpInst);
    //}

    EmitToStreamer(*OutStreamer, TmpInst);
  }
  */

  /*
  (From http://llvm.org/docs/doxygen/html/classllvm_1_1MachineFunctionPass.html
     we see SelectionDAGISel and AsmPrinter were the only passes that inherit
     MachineFunctionPass, from this back end.)
  From http://llvm.org/docs/doxygen/html/AsmPrinter_8h_source.html:
   /// Set up the AsmPrinter when we are working on a new module. If your pass
   /// overrides this, it must make sure to explicitly call this implementation.
  */

  bool isVectorBody(StringRef &&strRef) {
#define STR_VECTOR_BODY "vector.body"
#define STR_VECTOR_BODY_PREHEADER ".preheader"

    LLVM_DEBUG(dbgs() << "isVectorBody(): strRef = " << strRef << "\n");

    // We can have several BBs with name vector.bodyXYZT (but we do NOT
    //   search for STR_VECTOR_BODY_PREHEADER, which can be e.g.,
    //   vector.body40.preheader)
    if (strRef.starts_with(StringRef(STR_VECTOR_BODY)) &&
        strRef.ends_with(StringRef(STR_VECTOR_BODY_PREHEADER)))
      return false;

    if (strRef.starts_with(StringRef(STR_VECTOR_BODY)) == false)
      return false;

    LLVM_DEBUG(dbgs() << "isVectorBody(): returning true\n");

    return true;
  } // End isVectorBody()

  void moveToFrontRepeat(MachineBasicBlock *MBB) {
    LLVM_DEBUG(dbgs() << "Entered moveToFrontRepeat(MBB = " << MBB << ")\n");

    // Moving the REPEAT and it's symbolic operand in INLINEASM at the
    //  front of the MBB.
    for (auto MIItr = MBB->begin(); MIItr != MBB->end(); ++MIItr) {
      MachineInstr *MI = &(*MIItr);

      if (MI->getOpcode() == Connex::REPEAT_SYM_IMM) {
        LLVM_DEBUG(
            dbgs() << "moveToFrontRepeat(): Found Connex::REPEAT_SYM_IMM\n");
        MIItr++;

        MachineInstr *MI2 = &(*MIItr);

        if (MI2->isInlineAsm()) {
          LLVM_DEBUG(dbgs() << "moveToFrontRepeat(): Moving the successor "
                               "INLINEASM together with the "
                               "Connex::REPEAT_SYM_IMM\n");

          MBB->remove(MI2);
          MBB->insert(MBB->front(), MI2);
        } else {
          MIItr++;
          MI2 = &(*MIItr);

          LLVM_DEBUG(dbgs() << "moveToFrontRepeat(): Moving the following "
                               "(not successor) INLINEASM together with the "
                               "Connex::REPEAT_SYM_IMM\n");
          if (MI2->isInlineAsm()) {
            MBB->remove(MI2);
            MBB->insert(MBB->front(), MI2);
          } else {
            assert(0 && "Can't find INLINEASM associated to REPEAT_SYM_IMM");
          }
        }

        LLVM_DEBUG(
            dbgs() << "moveToFrontRepeat(): Moving Connex::REPEAT_SYM_IMM\n");

        MBB->remove(MI);
        MBB->insert(MBB->front(), MI);

        break;
      }
    }
  } // End moveToFrontRepeat()

  void moveToFrontInlineAsm(MachineBasicBlock *MBB, std::string strToSearch) {
    LLVM_DEBUG(dbgs() << "Entered moveToFrontInlineAsm(MBB = " << MBB
                      << ", strToSearch = " << strToSearch << ")\n");

    std::string strMBB = MBB->getName().str();

    // Moving strToSearch and it's associated INLINEASM at the
    //  front of the MBB.
    for (auto MIItr = MBB->begin(), MBBend = MBB->end(); MIItr != MBBend;) {
      MachineInstr *MI = &(*MIItr);

      // We avoid iterator invalidation:
      // See some comments on iterator invalidation (when doing remove) at
      // llvm.1065342.n5.nabble.com/deleting-or-replacing-a-MachineInst-td77723.html
      MachineBasicBlock::iterator MIsucc = MIItr;
      MIsucc++;

      if (MI->isInlineAsm()) {
        LLVM_DEBUG(dbgs() << "  moveToFrontInlineAsm(): found INLINEASM *MI = "
                          << *MI << "\n");

        // See http://llvm.org/docs/doxygen/html/classllvm_1_1MachineInstr.html
        for (unsigned index = 0; index < MI->getNumOperands(); index++) {
          MachineOperand *miOpnd;
          miOpnd = &(MI->getOperand(index));

          LLVM_DEBUG(dbgs() << " MI->getOperand(" << index << ") = " << *miOpnd
                            << "\n");

          if (miOpnd->isSymbol()) {
            std::string symStr = miOpnd->getSymbolName();
            LLVM_DEBUG(dbgs() << "  moveToFrontInlineAsm(): symStr = " << symStr
                              << "\n");

            if (symStr.find(strToSearch) != std::string::npos) {
              LLVM_DEBUG(dbgs() << "  moveToFrontInlineAsm(): Found INLINEASM "
                                   "with strToSearch in the symbol "
                                   "operand\n");

              MBB->remove(MI);

              if (strMBB == "entry") {
                // The "entry" MBB normally contains init Connex
                // instructions, so we add marker MI at the end to prevent these
                // init instructions to be put inside a host-side For loop
                // since they will be executed in the For loop body, which is
                // NOT good
                MBB->insert(MBB->getFirstTerminator(), MI);
              } else {
                MBB->insert(MBB->front(), MI);
              }
            }
          }
        }
      }

      // We avoid iterator invalidation
      MIItr = MIsucc;
    }
  } // End moveToFrontInlineAsm()

  /*
   This moves to the front of the MBB a number of 3 (if justOne == false),
     or 1 (if justOne == true) ASM inline expression(s) IF the 1st inline
     expression has OPINCAA kernel begin.

   We require to run first this function with justOne == false and then
            with justOne == true.

   More exactly, in LoopVectorize.cpp we added, among others, the following
     3 ASM inline expressions (consecutively):
       - 1 BEGIN_KERNEL INLINEASM instruction used as loop prologue
       - 1 END_KERNEL INLINEASM instruction used as
            loop prologue (END_KERNEL part)
       - 1 BEGIN_KERNEL INLINEASM instruction for
            the loop.
   We move these 3 instructions to the front of
       MBB when justOne == false. This ensures that eventual
       less-likely case of having a VLOAD_H_SYM_IMM (and inline ASM associated,
       containing the symbolic operand) manually generated by me
       in ConnexISelDAGToDAG.cpp is not going to be first instruction, before
       the OPINCAA loop header ASM inline expression.
     We also make sure that eventual loads from spills are put inside the loop
       prologue.

   We move 1 instruction to the front since in runOnMachineFunction() we put
      all instructions of the predecessor (has to be only 1 predecessor) of
      vector.body at the front of MBB, so we have to move the BEGIN_KERNEL of
      the loop prologue.
  */
  void moveToFront(MachineBasicBlock *MBB, bool justOne) {
    MachineInstr *tmp1, *tmp2, *tmp3; //, *tmp4;
    int counter = 0;

    LLVM_DEBUG(dbgs() << "Entered moveToFront(justOne = " << justOne << ")\n");

    /* We compute MIItrLastLoadAssociatedToSpill, an iterator (pointer) to
       the first instruction after the loads (fills) from spills at the
       beginning of the BB.
    */
    // See http://llvm.org/docs/doxygen/html/classllvm_1_1MachineBasicBlock.html
    /* Important: Make sure we put this initialization after any other MBB
        mutation in order to use it well to move the 3 INLINEASM instructions.
    */
    MachineBasicBlock::iterator MIItrLastLoadAssociatedToSpill = MBB->front();

    if (justOne == false) {
      for (auto MIItr2 = MBB->begin(); MIItr2 != MBB->end(); ++MIItr2) {
        MachineInstr *MI = &(*MIItr2);

        LLVM_DEBUG(dbgs() << "  moveToFront(): *MI = " << *MI
                          << ", MI->getOpcode() = " << MI->getOpcode() << "\n");

        unsigned imm = -1;
        if (MI->getOpcode() == Connex::LD_H) {
          // Inspired from
          //   http://llvm.org/docs/doxygen/html/MachineInstr_8cpp_source.html,
          //   method MachineInstr::isIdenticalTo()
          for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
            const MachineOperand &MO = MI->getOperand(i);

            if (MO.isImm()) {
              imm = MO.getImm();
              LLVM_DEBUG(dbgs() << "    moveToFront(): imm = " << imm << "\n");
              break;
            }
          }

          // If the imm operand > CONNEX_MEM_NUM_ROWS - 32 it (normally)
          //   means that the operation is generated in
          //   ConnexInstrInfo::storeRegToStackSlot() and
          //   ConnexInstrInfo::loadRegFromStackSlot(),
          //   part of a spill or load from spill operation.
          //   Note that on Connex we do not have a stack per se,
          //     but we emulate it at the end of the LS memory.
          if ((imm >= CONNEX_MEM_NUM_ROWS - 32) &&
              (imm < CONNEX_MEM_NUM_ROWS)) {
            MIItrLastLoadAssociatedToSpill = MIItr2;
            MIItrLastLoadAssociatedToSpill++;
          }
        }
      } // end for
    }   // if (justOne == false)

    // Moving the ISD::INLINEASM instruction containing the opincaa kernel
    //     begin at the very front of this BB.
    for (auto MIItr = MBB->begin(); MIItr != MBB->end(); ++MIItr, ++counter) {
      MachineInstr *MI = &(*MIItr);

      if (MI->isInlineAsm()) {
        LLVM_DEBUG(dbgs() << "  moveToFront() found INLINEASM *MI = " << *MI
                          << "\n");

        bool isOpincaaCodeBegin = false;

        // See http://llvm.org/docs/doxygen/html/classllvm_1_1MachineInstr.html
        for (unsigned index = 0; index < MI->getNumOperands(); index++) {
          MachineOperand *miOpndOpincaaCodeBegin; // = NULL;
          miOpndOpincaaCodeBegin = &(MI->getOperand(index));

          LLVM_DEBUG(dbgs() << " MI->getOperand(" << index
                            << ") = " << *miOpndOpincaaCodeBegin << "\n");

          if (miOpndOpincaaCodeBegin->isSymbol()) {
            std::string symStr = miOpndOpincaaCodeBegin->getSymbolName();
            LLVM_DEBUG(dbgs()
                       << "  moveToFront(): symStr = " << symStr << "\n");
            if (symStr.find(STR_OPINCAA_CODE_BEGIN) != std::string::npos) {
              isOpincaaCodeBegin = true;
              break;
            }
          }
        }

        if (isOpincaaCodeBegin) {
          if (counter != 0) {
            // We move only if not at the beginning of MBB
            tmp1 = MI;
            LLVM_DEBUG(dbgs()
                       << "  moveToFront(): moving INLINEASM to the front "
                          "(counter = "
                       << counter << ", justOne = " << justOne << ")\n");

            if (justOne == true) {
              MBB->remove(tmp1);
              MBB->insert(MBB->front(), tmp1);
            } else {
              // We move the next 3 instructions to the front of
              //   MBB, namely:
              //   - 1 BEGIN_KERNEL INLINEASM instruction used as
              //        loop prologue
              //   - 1 END_KERNEL INLINEASM instruction used as
              //        loop prologue (END_KERNEL part)
              //   - 1 BEGIN_KERNEL INLINEASM instruction for
              //        the loop.
              // TODO: check tmp3 and tmp2 are also INLINEASM.

              MIItr++;
              tmp2 = &(*MIItr);

              MIItr++;
              tmp3 = &(*MIItr);

              LLVM_DEBUG(dbgs() << "  moveToFront(): *tmp1 = " << *tmp1 << "\n");
              LLVM_DEBUG(dbgs() << "  moveToFront(): *tmp2 = " << *tmp2 << "\n");
              LLVM_DEBUG(dbgs() << "  moveToFront(): *tmp3 = " << *tmp3 << "\n");
              /*
              MBB->remove(tmp4);
              //MBB->insert(MBB->front(), tmp3);
              */

              MBB->remove(tmp3);

              MBB->remove(tmp2);

              MBB->remove(tmp1);

              // TODO: check that the iterator
              //   MIItrLastLoadAssociatedToSpill does NOT get
              //   invalidated - it seems it is not invalidated even if we
              //   change MBB, which is so because the instruction
              //   to which the iterator points to is NOT changed.
              MBB->insert(MIItrLastLoadAssociatedToSpill, tmp1);
              MBB->insert(MIItrLastLoadAssociatedToSpill, tmp2);
              MBB->insert(MIItrLastLoadAssociatedToSpill, tmp3);
            }
          } // End if (counter != 0)
          break;
        } // End if (isOpincaaCodeBegin)
      }
      // counter++;
    }
  } // End moveToFront()

  // Moving the last ISD::INLINEASM instruction of MBB at the very back of MBB
  void moveToBackLastInlineAsm(MachineBasicBlock *MBB) {
    MachineInstr *tmp1;
    int counter = 0;

    LLVM_DEBUG(dbgs() << "  moveToBackLastInlineAsm(): *MBB = " << *MBB << "\n");

    for (auto MIItr = MBB->rbegin(); MIItr != MBB->rend(); ++MIItr, ++counter) {
      MachineInstr *MI = &(*MIItr);

      if (MI->isInlineAsm()) {
        LLVM_DEBUG(
            dbgs() << "    moveToBackLastInlineAsm() found INLINEASM MI = "
                   << *MI << "\n");

        bool isOpincaaCodeEnd = false;

        // See http://llvm.org/docs/doxygen/html/classllvm_1_1MachineInstr.html
        for (unsigned index = 0; index < MI->getNumOperands(); index++) {
          MachineOperand *miOpndOpincaaCodeEnd;
          miOpndOpincaaCodeEnd = &(MI->getOperand(index));

          LLVM_DEBUG(dbgs() << " MI->getOperand(" << index
                            << ") = " << *miOpndOpincaaCodeEnd << "\n");

          if (miOpndOpincaaCodeEnd->isSymbol()) {
            std::string symStr = miOpndOpincaaCodeEnd->getSymbolName();
            LLVM_DEBUG(dbgs() << "  moveToBackLastInlineAsm(): symStr = "
                              << symStr << "\n");
            if (symStr.find(STR_OPINCAA_CODE_END) != std::string::npos) {
              isOpincaaCodeEnd = true;
              break;
            }
          }
        }

        if (isOpincaaCodeEnd) {
          tmp1 = MI;
          LLVM_DEBUG(dbgs()
                     << "  moveToBackLastInlineAsm(): moving INLINEASM to the "
                        "front (counter = "
                     << counter << ")\n");

          MBB->remove(tmp1);
          MBB->insert(MBB->end(), tmp1);
          break;
        }
      }
    }
  } // End moveToBackLastInlineAsm()

  // We add at the front of vector.body the instructions
  // for the predecessor of vector.body basic-block DIFFERENT than
  // vector.body (normally vector.ph).
  void copyInstructionsFromPred(MachineFunction &MF, MachineBasicBlock &MBB,
                                MachineBasicBlock *&predMBBGood) {

    // See http://llvm.org/docs/doxygen/html/classllvm_1_1MachineBasicBlock.html
    /* (See fossies.org/linux/llvm/lib/CodeGen/DeadMachineInstructionElim.cpp
     *  also, method DeadMachineInstructionElim::runOnMachineFunction() for
     *  an example of iteration backwards).
     */
    unsigned counterPredMBB = 0;

    // rbegin() is a reverse_iterator
    for (auto predMIItr = predMBBGood->rbegin();
         predMIItr != predMBBGood->rend(); predMIItr++, counterPredMBB++) {
      MachineInstr *predMI = &(*predMIItr);

      LLVM_DEBUG(dbgs() << "  copyInstructionsFromPred(): *predMI = " << *predMI
                        << "\n");

      // Need to insert them in different order
      if (predMI->isBundle()) {
        LLVM_DEBUG(dbgs() << " copyInstructionsFromPred(): handling bundle\n");

        const MachineBasicBlock *MBBBundle = predMI->getParent();
        MachineBasicBlock::const_instr_iterator I = predMI->getIterator();

        // Important: We assume we work with finalized bundles
        I++;

        assert(I != MBBBundle->instr_end());
        const MachineInstr *I1 = &(*I);
        LLVM_DEBUG(dbgs() << "  copyInstructionsFromPredConnexAsmPrinter::"
                             "runOnMachineFunction(): *I1 = "
                          << *I1 << "\n");
        //
        I++;

        // Important: We assume we work with bundles with only 2 instructions

        // From http://llvm.org/docs/doxygen/html/classllvm_1_1MachineInstr.html
        //  bool isInsideBundle () const
        // Return true if MI is in a bundle (but not the first MI in a bundle).
        //  bool  isBundled () const
        //    Return true if this instruction part of a bundle.
        //
        assert(I != MBBBundle->instr_end());
        const MachineInstr *I2 = &(*I);

        MachineInstr *newPredMI2 = MF.CloneMachineInstr(I2);
        LLVM_DEBUG(dbgs() << "  copyInstructionsFromPred(): *newPredMI2 = "
                          << *newPredMI2 << "\n");
        MBB.insert(MBB.front(), newPredMI2);

        MachineInstr *newPredMI1 = MF.CloneMachineInstr(I1);
        LLVM_DEBUG(dbgs() << "  copyInstructionsFromPred(): *newPredMI1 = "
                          << *newPredMI1 << "\n");
        MBB.insert(MBB.front(), newPredMI1);

        LLVM_DEBUG(
            dbgs() << " copyInstructionsFromPred(): End handling bundle\n");

        continue;
      }

      // We avoid the last instruction of predMBBGood, since it is an
      //  unconditional JMP
      if (counterPredMBB == 0 &&
          // See
          // http://llvm.org/docs/doxygen/html/classllvm_1_1MachineInstr.html
          predMI->isUnconditionalBranch()) { // predMBBGood->size())
        /* For llc -O3 it removes the JMP at the end of
           vector.ph, hence it merges it with vector.body,
           even if it leaves the entry label of vector.body.
           So we need to check if predMI is JMP with
           isUnconditionalBranch(). */
        LLVM_DEBUG(dbgs() << "  copyInstructionsFromPred(): found a JMP, "
                             "so not copying it in vector.body\n");
        continue;
      }

      /* Important note: EmitInstruction() fails for ISD::INLINEASM
      EmitInstruction(&predMI);
      */

      /* See http://llvm.org/docs/doxygen/html/classllvm_1_1MachineFunction.html
      MachineInstr *CloneMachineInstr(const MachineInstr *Orig);
        CloneMachineInstr - Create a new MachineInstr which is a
        copy of the 'Orig' instruction, identical in all ways except
        the instruction has no parent, prev, or next.
      */
      MachineInstr *newPredMI = MF.CloneMachineInstr(predMI);

      MBB.insert(MBB.front(), newPredMI);
    }

    // I guess normally we should have 2 predecessors, but since I mess
    // up in LoopVectorize.cpp the vector.body block in some cases
    //  (e.g., with a few iterations, in the order of magnitude of the
    //  vector unit width) it can remain with only 1 predecessor.
    //
    // assert(numPredecessors <= 2 &&
    //  "vector.body should have at most 2 predecessors: itself and one more");
  } // End copyInstructionsFromPred()

  // Important: We copy from successor BB (middle.block) to vector.body BB
  void copyInstructionsFromSucc(MachineFunction &MF, MachineBasicBlock &MBB) {
    LLVM_DEBUG(dbgs() << "  copyInstructionsFromSucc(): Move code from succ "
                         "of block "
                      << MBB.getName().data() << "\n");

    int numSuccessors = 0;

    for (auto succMBB : MBB.successors()) {
      numSuccessors++;

      StringRef strSuccMBB = succMBB->getName();
      LLVM_DEBUG(dbgs() << "  copyInstructionsFromSucc(): strSuccMBB = "
                        << strSuccMBB << "\n");

      // See llvm.org/docs/doxygen/html/classllvm_1_1MachineBasicBlock.html
      /* (See fossies.org/linux/llvm/lib/CodeGen/DeadMachineInstructionElim.cpp
       *  also, method DeadMachineInstructionElim::runOnMachineFunction() for
       *  an example of iteration backwards).
       */
      unsigned counterSuccMBB = 0;

      for (auto succMIItr = succMBB->begin(); succMIItr != succMBB->end();
           succMIItr++, counterSuccMBB++) {
        MachineInstr *succMI = &(*succMIItr);

        LLVM_DEBUG(dbgs() << "  copyInstructionsFromSucc(): succMI = "
                          << *succMI << "\n");

        /* We avoid the last instruction of predMBB, since it is an
           unconditional JMP */
        if (
            // See llvm.org/docs/doxygen/html/classllvm_1_1MachineInstr.html
            (succMI->isUnconditionalBranch() ||
             succMI->isConditionalBranch())) { // predMBB->size())
          /* For llc -O3 it removes the JMP at the end of
             vector.ph, hence it merges it with vector.body,
             even if it leaves the entry label of vector.body.
             So we need to check if predMI is JMP with
             isUnconditionalBranch(). */
          LLVM_DEBUG(dbgs() << "copyInstructionsFromSucc(): found a JMP, "
                               "so not copying it in vector.body\n");
          continue;
        }

        /* Important note: EmitInstruction() fails for ISD::INLINEASM
        EmitInstruction(&predMI);
        */

        /* See llvm.org/docs/doxygen/html/classllvm_1_1MachineFunction.html
        MachineInstr *CloneMachineInstr(const MachineInstr *Orig);
          CloneMachineInstr - Create a new MachineInstr which is a
          copy of the 'Orig' instruction, identical in all ways except
          the instruction has no parent, prev, or next.
        */
        MachineInstr *newSuccMI = MF.CloneMachineInstr(succMI);

        // Gives error: "Assertion `!N->getParent() && "machine instruction
        //               already in a basic block"' failed."
        // MBB.insert(MBB.front(), &predMI);
        MBB.insert(MBB.back(), newSuccMI);
      }

      // Instead of break we should check if predMBB is the BB "just"
      //   above predMBBGood or below
      break;
    }

    assert(numSuccessors == 1);
  } // End copyInstructionsFromSucc()

// If commented we traverse nodes in our standard DFS (pre-order).
//   Otherwise we traverse in Reverse post-order (RPO).
#define ADAPTED_RPO

  /* In DFS() we store in the sortedListMBB vector the traversed nodes:
       - in RPO (Reverse post-order)
         - see e.g.
         eli.thegreenplace.net/2015/directed-graph-traversal-orderings-and-applications-to-data-flow-analysis/
       - OR, we can use, if we want, preorder (standard DFS).
     This is required because the MachineBasicBlock class iterates the BBs
       in an (undocumented/unspecified) order (for MatMul it is actually RPO),
       which is bad for our simple source-to-source transformation basically
       implemented with our simple ReplaceLoopsWithOpincaaKernels tool that
       simply copies a section of the Connex assembly code from the test.s
       file that is inbetween the markers
         "// START_OPINCAA_HOST_DEVICE_CODE" and
         "// END_OPINCAA_HOST_DEVICE_CODE".

    Using the MachineBasicBlock BB iterator order results in e.g.:
      - REPEAT instruction being misplaced (not at beginning of loop, but
      actually close to the end of the loop, close to END_REPEAT) - see the
      MatAdd test. This example shows the difference of the order:
      Printing the MBBs, as they are ordered now:
        BB name: = entry
        BB name: = entry
        BB name: = for.cond2.preheader.us.preheader
        BB name: = for.cond2.preheader.us
        BB name: = min.iters.checked
        BB name: = vector.memcheck
        BB name: = vector.memcheck
        BB name: = vector.memcheck
        BB name: = vector.memcheck
        BB name: = vector.memcheck
        BB name: = vector.memcheck
        BB name: = vector.memcheck
        BB name: = vector.memcheck
        BB name: = vector.memcheck
        BB name: = vector.memcheck
        BB name: = vector.body.preheader
        BB name: = vector.body
        BB name: = middle.block
        BB name: = for.body6.us
        BB name: = for.cond2.for.inc20_crit_edge.us
        BB name: = for.end22.loopexit
        BB name: = for.end22
      My pre-order DFS traversal:
        DFS(): BB name: = entry, n = 0x136c038
        DFS(): BB name: = entry, n = 0x13ad750
        DFS(): BB name: = for.cond2.preheader.us.preheader, n = 0x136c0e8
        DFS(): BB name: = for.cond2.preheader.us, n = 0x136c198
        DFS(): BB name: = min.iters.checked, n = 0x136c4e8
        DFS(): BB name: = for.body6.us, n = 0x1381360
        DFS(): BB name: = for.cond2.for.inc20_crit_edge.us, n = 0x1381520
        DFS(): BB name: = for.end22.loopexit, n = 0x13816f0
        DFS(): BB name: = for.end22, n = 0x13817a0
        DFS(): BB name: = vector.memcheck, n = 0x136c598
        DFS(): BB name: = vector.memcheck, n = 0x13ad980
        DFS(): BB name: = vector.memcheck, n = 0x13ada30
        DFS(): BB name: = vector.memcheck, n = 0x13adb20
        DFS(): BB name: = vector.memcheck, n = 0x13adbd0
        DFS(): BB name: = vector.memcheck, n = 0x13adcc0
        DFS(): BB name: = vector.memcheck, n = 0x13add70
        DFS(): BB name: = vector.memcheck, n = 0x13adf30
        DFS(): BB name: = vector.memcheck, n = 0x13adfe0
        DFS(): BB name: = vector.memcheck, n = 0x13ab5f8
        DFS(): BB name: = vector.body.preheader, n = 0x136c648
        DFS(): BB name: = vector.body, n = 0x136c6f8
        DFS(): BB name: = middle.block, n = 0x136c928

   The issue with misplaced REPEAT can also be seen in the MatMul_sizeMat test.
   */
  void DFS(MachineBasicBlock *n) {
    // See http://www.cplusplus.com/reference/map/map/count/
    if (visitedMBB.count(n) != 0)
      return;

    // See http://www.cplusplus.com/reference/map/map/insert/
    visitedMBB.insert(std::pair<MachineBasicBlock *, bool>(n, true));

#ifndef ADAPTED_RPO
    finishingTimeMBB.insert(
        std::pair<MachineBasicBlock *, int>(n, sortedListMBB.size()));
    sortedListMBB.push_back(n);
#endif

    const char *strN = n->getName().data();
    LLVM_DEBUG(dbgs() << "DFS(): BB name of n: = " << strN << ", n = " << n
                      << "\n");

    int numSuccessorsN = 0;
    MachineBasicBlock *successorsN[2];

    // If in the successors we have vector.ph, vector.body, etc we choose those
    //   first.
    for (auto MBB : n->successors()) {
      std::string strMBB = MBB->getName().data();
      LLVM_DEBUG(dbgs() << "  DFS(): successor: MBB name = " << strMBB
                        << ", MBB = " << MBB << "\n");
      if (strMBB == "min.iters.checked" ||
          // small-TODO: check only for "vector.*" not for all below
          strMBB == "vector.memcheck" || strMBB == "vector.ph" ||
          strMBB == "vector.body.preheader" || strMBB == "vector.body") {
        DFS(MBB); // This will update visitedMBB to avoid further visits
      }

      successorsN[numSuccessorsN & 1] = MBB;
      numSuccessorsN++;
    }
    LLVM_DEBUG(dbgs() << "DFS(): numSuccessorsN = " << numSuccessorsN << "\n");

    if (numSuccessorsN == 2) {
      std::string strSuccName0 =
          successorsN[numSuccessorsN & 1]->getName().str();
      std::string strSuccName1 =
          successorsN[(numSuccessorsN + 1) & 1]->getName().str();

      // If we have 2 successors e.g. %for.cond47.preheader.preheader,
      //   %for.cond6.preheader.preheader
      // we choose the one with smaller ID (i.e., in this case 6) number first.
      // #define FOR_COND_STR "for.cond"
      std::string FOR_COND_STR = "for.cond";
      if (startsWith(strSuccName0, FOR_COND_STR) &&
          startsWith(strSuccName1, FOR_COND_STR)) {
        LLVM_DEBUG(dbgs() << "DFS(): strSuccName0 = " << strSuccName0 << "\n");
        LLVM_DEBUG(dbgs() << "DFS(): strSuccName1 = " << strSuccName1 << "\n");

        std::string strSuccNameId0, strSuccNameId1;
        strSuccNameId0 = strSuccName0.substr(FOR_COND_STR.size());
        strSuccNameId1 = strSuccName1.substr(FOR_COND_STR.size());
        strSuccNameId0 = strSuccNameId0.substr(0, strSuccNameId0.find('.'));
        strSuccNameId1 = strSuccNameId1.substr(0, strSuccNameId1.find('.'));

        LLVM_DEBUG(dbgs() << "DFS(): strSuccNameId0 = " << strSuccNameId0
                          << "\n");
        LLVM_DEBUG(dbgs() << "DFS(): strSuccNameId1 = " << strSuccNameId1
                          << "\n");

        if (atoi(strSuccNameId0.c_str()) < atoi(strSuccNameId1.c_str())) {
#ifdef ADAPTED_RPO
          // The 1st successor has bigger ID --> changing the order of the
          //   2 successors
          LLVM_DEBUG(dbgs() << "DFS(): Changing order of the 2 successors.\n");

          DFS(successorsN[(numSuccessorsN + 1) & 1]);
          // DFS(successorsN[numSuccessorsN & 1]);

          // IMPORTANT: Never give return since at the end of this function
          //   we insert finishingTimeMBB.
#else
          assert(0 && "NOT implemented.");
          // DFS(successorsN[numSuccessorsN & 1]);
#endif
        }
      }

// Addressing case encountered for MatMul-512.i16, TS_182_74
// If we have 2 successors if.then... and if.else...
//   we choose if.then... first.
#define IF_THEN "if.then"
#define IF_ELSE "if.else"
//
#define IF_BODY1 "for.body4.us.preheader"
      // MEGA-TODO: this is a NON-general solution
#define IF_BODY2 "for.body4.preheader"
      if ((startsWith(strSuccName0, IF_THEN) &&
           startsWith(strSuccName1, IF_ELSE)) ||
          (startsWith(strSuccName0, IF_BODY1) &&
           startsWith(strSuccName1, IF_BODY2))) {
        LLVM_DEBUG(dbgs() << "DFS(): strSuccName0 = " << strSuccName0 << "\n");
        LLVM_DEBUG(dbgs() << "DFS(): strSuccName1 = " << strSuccName1 << "\n");

#ifdef ADAPTED_RPO
        // The 1st successor has bigger ID --> changing the order of the
        //   2 successors
        LLVM_DEBUG(dbgs() << "DFS(): Changing order of the 2 successors.\n");

        DFS(successorsN[(numSuccessorsN + 1) & 1]);
        // DFS(successorsN[numSuccessorsN & 1]);

        // IMPORTANT: Never give return since at the end of this function
        //   we insert finishingTimeMBB.
#else
        assert(0 && "NOT implemented.");
        // DFS(successorsN[numSuccessorsN & 1]);
#endif
      }
    }

    for (auto MBB : n->successors()) {
      DFS(MBB);
    }

#ifdef ADAPTED_RPO
    // See http://www.cplusplus.com/reference/map/map/insert/
    finishingTimeMBB.insert(
        std::pair<MachineBasicBlock *, int>(n, sortedListMBB.size()));
    sortedListMBB.push_back(n);
#endif
  }

  static bool compareBasicBlocks(MachineBasicBlock &b1, MachineBasicBlock &b2) {
    LLVM_DEBUG(dbgs() << "compareBasicBlocks(): finishingTimeMBB[&b1] = "
                      << finishingTimeMBB[&b1] << ", finishingTimeMBB[&b2] = "
                      << finishingTimeMBB[&b2] << ".\n");

#ifdef ADAPTED_RPO
    return finishingTimeMBB[&b1] > finishingTimeMBB[&b2];
#endif

    // reverse RPO: return finishingTimeMBB[&b1] < finishingTimeMBB[&b2];
  }

  void sortMBBs(MachineFunction &MF) {
    MachineBasicBlock *entryMBB = NULL;

    LLVM_DEBUG(dbgs() << "Printing the MBBs, as they are ordered now:\n");
    // Looking at http://llvm.org/doxygen/classllvm_1_1MachineFunction.html
    //  it seems it's not possible to obtain the root(s) of the MB otherwise.
    for (auto &MBB : MF) {
      if (entryMBB == NULL)
        entryMBB = &MBB;
      std::string strMBB = MBB.getName().str();
      LLVM_DEBUG(dbgs() << "  BB name = " << strMBB << "\n");
    }

    // We now compute the order of the CFG node (BB) traversal
    visitedMBB.clear();
    finishingTimeMBB.clear();
    sortedListMBB.clear();
    //
    DFS(entryMBB);
    // Small Note: We can get inspired form the ReversePostOrderTraversal
    //    LLVM class and create our adapted RPO order class, but note that
    //    using ReversePostOrderTraversal doesn't change the order of MBBs in
    //    the MF object, which is REQUIRED by the EmitFunctionBody() method,
    //      which iterates over the MBBs of MF. This is why we perform the
    //      somewhat-strange MF.sort() below.
    // See http://llvm.org/doxygen/X86WinAllocaExpander_8cpp_source.html#l00146
    //    ReversePostOrderTraversal<MachineFunction*> RPO(&MF);
    //    for (MachineBasicBlock *MBB : RPO) {...}
    // See also https://llvm.org/doxygen/PostOrderIterator_8h_source.html#l00259

#ifdef ADAPTED_RPO
    LLVM_DEBUG(dbgs() << "ConnexAsmPrinter: ADAPTED_RPO sortedListMBB = \n");
    for (int idxSListMBB = sortedListMBB.size() - 1; idxSListMBB >= 0;
         idxSListMBB--)
#else
    LLVM_DEBUG(dbgs() << "ConnexAsmPrinter: DFS order sortedListMBB =.\n");
    for (int idxSListMBB = 0; idxSListMBB < sortedListMBB.size(); idxSListMBB++)
#endif
    {
      MachineBasicBlock *MBB = sortedListMBB[idxSListMBB];

      std::string strMBB = MBB->getName().str();
      LLVM_DEBUG(dbgs() << "  BB name = " << strMBB << ", MBB = " << MBB
                        << "\n");
    }

    // For calling a templated function
    //    see http://www.cplusplus.com/doc/oldtutorial/templates/
    MF.sort<CompareBBs>(compareBasicBlocks);

    LLVM_DEBUG(dbgs() << "  After sort():\n");
    for (auto &MBB : MF) {
      std::string strMBB = MBB.getName().str();
      LLVM_DEBUG(dbgs() << "  BB name = " << strMBB << "\n");
    }
  } // End sortMBBs()

  /// Emit the specified function out to the OutStreamer.
  bool runOnMachineFunction(MachineFunction &MF) override {
    LLVM_DEBUG(dbgs() << "Entered ConnexAsmPrinter::runOnMachineFunction().\n");
    LLVM_DEBUG(dbgs() << " EnableCorrectBBsASMPrint = "
                      << EnableCorrectBBsASMPrint << "\n");

    // We sort the BBs of the MF in a better order to be able to use our
    //  ReplaceLoopsWithOpincaaKernels tool to extract correctly the vector
    //  kernels, in SIMPLE TEXTUAL order, from the .s file generated here.
    sortMBBs(MF);

    int numVectorizedLoops = 0;
    bool TreatRepeat2ndInnerLoopGlobal = false;

    // We read from FILENAME_LOOPNESTS_LOCATIONS the configuration of the loop
    //  nests in order to fill correctly the std::vector
    //  treatRepeat2ndInnerLoop, which we use below.
    readLoopsLocFile(const_cast<char *>(FILENAME_LOOPNESTS_LOCATIONS), true);
    LLVM_DEBUG(
        dbgs() << "runOnMachineFunction(): treatRepeat2ndInnerLoop.size() = "
               << treatRepeat2ndInnerLoop.size() << "\n");

    if (EnableCorrectBBsASMPrint) {
      this->MF = &MF;

      // Inspired from ConnexRegisterInfo.cpp:
      // const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

      // Inspired from llvm.org/docs/doxygen/html/AsmPrinter_8cpp_source.html:

      // See http://llvm.org/docs/doxygen/html/classllvm_1_1MachineFunction.html
      for (auto &MBB : MF) {
        if (numVectorizedLoops >= (int)treatRepeat2ndInnerLoop.size())
          TreatRepeat2ndInnerLoopGlobal = false;
        else
          TreatRepeat2ndInnerLoopGlobal =
              treatRepeat2ndInnerLoop[numVectorizedLoops];

        LLVM_DEBUG(
            dbgs() << "runOnMachineFunction(): TreatRepeat2ndInnerLoopGlobal = "
                   << TreatRepeat2ndInnerLoopGlobal << "\n");
        LLVM_DEBUG(dbgs() << "runOnMachineFunction(): numVectorizedLoops = "
                          << numVectorizedLoops << "\n");

        if (TreatRepeat2ndInnerLoopGlobal == true) {
          // TODO: think a bit: we should always call moveToFrontRepeat()
          //   - we complicate a bit, BUT it is highly unlikely to have a
          //   REPEAT() after the last vector.body

          // A bit inefficient - we try all MBB
          moveToFrontRepeat(&MBB);
        } else {
          // If we do this we risk to have comments like "Map/Reduction part"
          //    after the REPEAT OPINCAA instruction.
          moveToFrontRepeat(&MBB);
        }

        // NOTE: We need to do this check because if we try to split in the
        //   LoopVectorize pass MBB, it will get merged back into one BB after
        //   LV, in opt.
        if (isMBBWithInlineAsmString(&MBB, STR_OPINCAA_CODE_END) == false) {
          LLVM_DEBUG(dbgs() << "isMBBWithInlineAsmString(STR_OPINCAA_CODE_END) "
                               "returned false\n");
          // We take care to put the beginning marker for OPINCAA kernel at the
          //   very front of its basic block, MBB - we try all MBBs.
          LLVM_DEBUG(dbgs()
                     << "Calling moveToFrontInlineAsm(STR_OPINCAA_CODE_BEGIN) "
                        "for MBB = "
                     << MBB.getName() << "\n");

          moveToFrontInlineAsm(&MBB,
                               const_cast<char *>(STR_OPINCAA_CODE_BEGIN));

          LLVM_DEBUG(dbgs()
                     << "Finished calling "
                        "moveToFrontInlineAsm(STR_OPINCAA_CODE_BEGIN)\n");
        }

        if (isVectorBody(MBB.getName()) == false)
          continue;

        numVectorizedLoops++;

        // moveToFrontRepeat(MBB);
        //
        // replaceWithSymbolicIndex(&MBB);
        /* Important:
         *   We move the Inline ASM expressions to the beginning of the BB,
         *     by using moveToFront(),
         *     such that, immediately after (see code below) we put the
         *     instructions of the predecessor of the vector.body BB
         *     at the top and then call moveToFront(&MBB, true) again
         *     to make the code OK.
         */
        // moveToFront(&MBB, false);

        MachineBasicBlock *predMBBGood;
        int numPredecessors = 0;
        for (auto predMBB : MBB.predecessors()) {
          numPredecessors++;

          if (isVectorBody(predMBB->getName()) == true)
            continue;
          else
            predMBBGood = predMBB;
        }

        // I guess normally we should have 2 predecessors, but since I mess
        // up in LoopVectorize.cpp the vector.body block in some cases
        //  (e.g., with a few iterations, in the order of magnitude of the
        //  vector unit width) it can remain with only 1 predecessor.
        assert(numPredecessors <= 2 && "vector.body should have at most "
                                       "2 predecessors: itself and one more");

        if (TreatRepeat2ndInnerLoopGlobal == false) {
          // copyInstructionsFromPred(MF, MBB, predMBBGood);

          // We move the header of the OPINCAA kernel
          moveToFront(predMBBGood, true);
        }

        // Does NOT help: moveToFront(&MBB, true);
        LLVM_DEBUG(dbgs() << "  runOnMachineFunction(): calling "
                             "moveToFrontInlineAsm(&MBB)\n");
        // moveToFront(&MBB, false);
        moveToFrontInlineAsm(&MBB, const_cast<char *>("for ("));

        if (TreatRepeat2ndInnerLoopGlobal == true) {
          moveToBackLastInlineAsm(&MBB);
        }
      } // End for (auto &MBB : MF)
    }   // End if EnableCorrectBBsASMPrint

    SetupMachineFunction(MF);
    emitFunctionBody();

    return false;
  } // End bool runOnMachineFunction(MachineFunction &MF)

  void printOperand(const MachineInstr *MI, int OpNum, raw_ostream &O,
                    const char *Modifier = nullptr);

  void emitInstruction(const MachineInstr *MI) override;

  // Taken from the MSP430 back end
  void printSrcMemOperand(const MachineInstr *MI, int OpNum, raw_ostream &O);

  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                             unsigned AsmVariant, const char *ExtraCode,
                             raw_ostream &OS) {
    LLVM_DEBUG(dbgs() << "Entered PrintAsmMemoryOperand()\n");
    return false;
  }

  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       unsigned AsmVariant, const char *ExtraCode,
                       raw_ostream &OS) {
    LLVM_DEBUG(dbgs() << "Entered PrintAsmOperand()\n");
    return false;
  }

  void PrintSpecial(const MachineInstr *MI, raw_ostream &OS,
                    const char *Code) const {
    LLVM_DEBUG(dbgs() << "Entered PrintSpecial()\n");
  }

  void printOffset(int64_t Offset, raw_ostream &OS) const {
    LLVM_DEBUG(dbgs() << "Entered printOffset()\n");
  }
}; // End class ConnexAsmPrinter

} // End namespace

// TODO: remove since it seems it's NOT called
void ConnexAsmPrinter::printOperand(const MachineInstr *MI, int OpNum,
                                    raw_ostream &O, const char *Modifier) {
  LLVM_DEBUG(dbgs() << "Entered ConnexAsmPrinter::printOperand()\n");
  const MachineOperand &MO = MI->getOperand(OpNum);

  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    O << ConnexInstPrinter::getRegisterName(MO.getReg());
    break;

  case MachineOperand::MO_Immediate: {
    unsigned imm = MO.getImm();
    LLVM_DEBUG(dbgs() << "printOperand(): imm = " << imm << "\n");

    if (imm == CONNEX_MEM_NUM_ROWS + 10) {
      O << STR_LOOP_SYMBOLIC_INDEX;
    } else {
      O << MO.getImm();
    }
    // O << MO.getImm();
    break;
  }

  case MachineOperand::MO_MachineBasicBlock:
    O << *MO.getMBB()->getSymbol();
    break;

  case MachineOperand::MO_GlobalAddress:
    O << *getSymbol(MO.getGlobal());
    break;

  default:
    llvm_unreachable("<unknown operand type>");
  }
}

void ConnexAsmPrinter::printSrcMemOperand(const MachineInstr *MI, int OpNum,
                                          raw_ostream &O) {
  const MachineOperand &Base = MI->getOperand(OpNum);
  const MachineOperand &Disp = MI->getOperand(OpNum + 1);

  // Print displacement first

  // Imm here is in fact global address - print extra modifier.
  if (Disp.isImm() && !Base.getReg())
    O << '&';

  printOperand(MI, OpNum + 1, O, "nohash");

  // Print register base field
  if (Base.getReg()) {
    O << '(';
    printOperand(MI, OpNum, O);
    O << ')';
  }
}

void ConnexAsmPrinter::emitInstruction(const MachineInstr *MI) {
  // We need to store the correspondence between MachineInstr and the lowered
  //  MCInst, since MCInst does not.
  //  This could be used in ConnexInstPrinter.cpp.
  // static const MachineInstr *crtMI;

  LLVM_DEBUG(dbgs() << "Entered ConnexAsmPrinter::emitInstruction()...\n");

  /* Inspired from lib/Target/AMDGPU/AMDGPUMCInstLower.cpp
     (actually it's class AMDGPUAsmPrinter)
  */
  if (MI->isBundle()) {
    LLVM_DEBUG(dbgs() << " emitInstruction(): handling bundle\n");
    const MachineBasicBlock *MBB = MI->getParent();
    // MachineBasicBlock::const_instr_iterator I = ++MI->getIterator();
    MachineBasicBlock::const_instr_iterator I = MI->getIterator();
    I++;

    /*
    From http://llvm.org/docs/doxygen/html/classllvm_1_1MachineInstr.html
      bool isInsideBundle () const
        Return true if MI is in a bundle (but not the first MI in a bundle).
    */
    while (I != MBB->instr_end() && I->isInsideBundle()) {
      emitInstruction(&(*I));
      ++I;
    }

    return;
  }

  ConnexMCInstLower MCInstLowering(OutContext, *this);

  MCInst TmpInst;
  MCInstLowering.Lower(MI, TmpInst);

  // crtMI = MI;

  EmitToStreamer(*OutStreamer, TmpInst);
} // End ConnexAsmPrinter::emitInstruction()

// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeConnexAsmPrinter() {
  RegisterAsmPrinter<ConnexAsmPrinter> Z(getTheConnexTarget());
}
