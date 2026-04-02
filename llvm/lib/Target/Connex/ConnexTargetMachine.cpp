//===-- TargetMachine.cpp - Define TargetMachine for Connex ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the info about the Connex target spec.
//===----------------------------------------------------------------------===//

#include "ConnexTargetMachine.h"
#include "Connex.h"
#include "TargetInfo/ConnexTargetInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetOptions.h"

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "connex-target-config"

// This must be put after #include "llvm/Support/Debug.h"
#include "ConnexTargetTransformInfo.h"

using namespace llvm;

static cl::opt<bool>
    DontTreatCopyInstructions("dont-treat-copy-instructions", cl::Hidden,
                              cl::init(false),
                              cl::desc("Don't treat copy instructions"));

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeConnexTarget() {
  // Register the target - Force static initialization.
  RegisterTargetMachine<ConnexTargetMachine> Z(getTheConnexTarget());
}

static StringRef computeDataLayout(const Triple &TT) {
  /*
  See http://llvm.org/docs/LangRef.html#data-layout for all details regarding
                layout declaration.
    - e
      Specifies that the target lays out data in little-endian form.
    - S<size>
      Specifies the natural alignment of the stack in bits.
      Alignment promotion of stack variables is limited to the natural stack
                alignment to avoid dynamic stack realignment.
      The stack alignment must be a multiple of 8-bits.
      If omitted, the natural stack alignment defaults to “unspecified”, which
                does not prevent any alignment promotions.
    - p[n]:<size>:<abi>:<pref>
      This specifies the size of a pointer and its <abi> and <pref>erred
                alignments for address space n. All sizes are in bits.
                The address space, n, is optional, and if not specified, denotes
                the default address space 0.
                The value of n must be in the range [1,2^23).
    - i<size>:<abi>:<pref>
      This specifies the alignment for an integer type of a given bit <size>.
                The value of <size> must be in the range [1,2^23).
    - n<size1>:<size2>:<size3>...
      This specifies a set of native integer widths for the target CPU in bits.
    - v<size>:<abi>:<pref>
      This specifies the alignment for a vector type of a given bit <size>.

  See also http://llvm.org/docs/WritingAnLLVMBackend.html
    An upper-case “E” in the string indicates a big-endian target data model.
      A lower-case “e” indicates little-endian.
    “p:” is followed by pointer information: size, ABI alignment, and
            preferred alignment.
      If only two figures follow “p:”, then the first value is pointer size,
            and the second value is both ABI and preferred alignment.
    Then a letter for numeric type alignment: “i”, “f”, “v”, or “a”
      (corresponding to integer, floating point, vector, or aggregate).
      “i”, “v”, or “a” are followed by ABI alignment and preferred alignment.
      “f” is followed by three values: the first indicates the size of a long
      double, then ABI alignment, and then ABI preferred alignment.
  */

  // We specify here the data-layout:
  //   - of the CPU, eBPF - actually ABI properties
  //   - only a few alignment properties for the vector types
  //       - see at the end of the string. Note that we can't
  //       specify any other properties for the Connex vector processor.
  // Very Important: The pointer size 64 (of the eBPF CPU), because the
  //   masked.gather/scatter instructions use such pointer normally in LLVM IR,
  //   even if we translate them to writeDataTo/readDataFromConnex() and
  //   Connex vector assembly instructions with indirect memory accesses.
  //
  //  We really need to specify p:64 (not p:16), otherwise we get an error like:
  //   "Do not know how to promote this operator!"
  //    (GlobalAddress<i64* @CONNEX_VL> 0")
  // Important: the string is the one from the (e)BPF back end,
  //   concatenated with the spec for the vector alignment for Connex.

  // return "e-m:e-p:64:64-i64:64-n32:64-S128-v128:128:128-v2048:2048:2048";
  // return
  // "e-m:e-p:64:64:64:64-p1:32:32:32:32-i64:64-n32:64-S128-v128:128:128-"
  //       "v2048:2048:2048"; // 2019_06_25
  return "e-m:e-p:64:64:64:64-p1:64:64:64:64-i64:64-n32:64-S128-v128:128:128-"
         "v2048:2048:2048"; // 2019_06_25
}

static Reloc::Model getEffectiveRelocModel(std::optional<Reloc::Model> RM) {
  if (!RM.has_value())
    return Reloc::PIC_;
  return *RM;
}

// Inspired from XCore/XCoreTargetMachine.cpp
static CodeModel::Model
getEffectiveXCoreCodeModel(std::optional<CodeModel::Model> CM) {
  if (CM) {
    if (*CM != CodeModel::Small && *CM != CodeModel::Large)
      report_fatal_error("Target only supports CodeModel Small or Large");
    return *CM;
  }
  return CodeModel::Small;
}

ConnexTargetMachine::ConnexTargetMachine(const Target &T, const Triple &TT,
                                         StringRef CPU, StringRef FS,
                                         const TargetOptions &Options,
                                         std::optional<Reloc::Model> RM,
                                         std::optional<CodeModel::Model> CM,
                                         CodeGenOptLevel OL, bool JIT)
    // Inspired from lib/Target/BPF/BPFTargetMachine.cpp (from Oct 2025)
    : CodeGenTargetMachineImpl(T, TT.computeDataLayout(), TT, CPU, FS, Options,
                        getEffectiveRelocModel(RM),
                        getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>()),
      Subtarget(TT, std::string(CPU), std::string(FS), *this) {
  initAsmInfo();
}

namespace {

/* I made sure that the iterators don't become invalid by using
     another iterator, e.g. I2succ, which stores the next pointer in the
     data structures.

small-TODO: it might be safer to do a change by moving (maybe also
 erasing) misplaced instrs one per WHERE block (or even per MBB) and then get
 out of the MBB::iterator loop and restart the loop from the beginning again
 until NO more changes are performed - this in order to avoid any (eventual)
 issue with iterator invalidation.
*/
class PassHandleMisplacedInstr : public MachineFunctionPass {
public:
  PassHandleMisplacedInstr() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "PassHandleMisplacedInstr"; }

  /*
  // Very Important: GMS said in 2018 he doesn't like having arithmetic or logic
  //      instruction between predicate and WHERE* instruction:
  #define ALLOW_COPY_BETWEEN_PREDICATE_AND_WHERE_INSTRUCTIONS
    - this case needs to be implemented carefully - I only sketched it a bit, so
        it is not tested either
  */

  void updateUsesOfRegUntilMisplacedInstr(
      MachineBasicBlock::iterator &Ipredicate,
      // We start replacing uses from Ipredicate + 1
      MachineBasicBlock::iterator &I2, // misplaced instr
      MachineBasicBlock::iterator &IE, unsigned regCrt, unsigned regNew) {
    LLVM_DEBUG(dbgs() << "  I2 = " << *I2);

    /* We update all following occurences of the dest register
       of misplaced instr (which was also the dest register of the
        predicate)
         - for both uses and def, until 1st def. */
    MachineBasicBlock::iterator Iupdate;
    Iupdate = Ipredicate;
    Iupdate++;

    for (; Iupdate != I2 && Iupdate != IE; Iupdate++) {
      LLVM_DEBUG(dbgs() << "  Iupdate = " << *Iupdate);

      /* Important: we go in reverse order to make the def last since we
        break at def. */
      for (int idOpnd = Iupdate->getNumOperands() - 1; idOpnd >= 0; idOpnd--) {
        MachineOperand &IOpnd = Iupdate->getOperand((unsigned)idOpnd);

        if (IOpnd.isReg() && IOpnd.getReg() == regCrt) {
          LLVM_DEBUG(dbgs() << "updateUsesOfRegUntilMisplacedInstr(): Updating "
                               "to regNew the register of Iupdate. "
                               " Iupdate = "
                            << *Iupdate);

          /*
          // This does NOT hold because we can have uses of a misplaced instr
          //   dest register before the misplaced instr - see the big WHERE
          //   block of ADD.f16
          assert( (Iupdate->getOpcode() == Connex::WHEREEQ ||
                  Iupdate->getOpcode() == Connex::WHERELT ||
                  Iupdate->getOpcode() == Connex::WHERECRY) &&
                 "We should NOT be arriving here otherwise.");
          */

          if (IOpnd.isDef()) {
            // We break
            Iupdate = IE;
            Iupdate--; // We make it break out of outermost loop
            break;
          }

          IOpnd.setReg(regNew);
        }
      }
    }
  }

  void putMisplacedInstrBeforeWhereBlock(
      MachineBasicBlock &MBB, const TargetInstrInfo *TII,
      MachineInstr *IMI,               // The WHERE* instruction
      MachineBasicBlock::iterator &I2, // misplaced instr
      MachineBasicBlock::iterator &I2plus1, MachineBasicBlock::iterator &IE,
      bool &changedMF, int &destRegisterPredicateOfSplitWhere) {
    /* NOTE: I2 is the misplaced instr instruction
    if (I2.getOperand(0) == Ipredicate.getOperand(0))
      for each instruction from Ipredicate to I2 - 1 replace defs and uses of
        I2.getOperand(0) with CONNEX_RESERVED_REGISTER_01
    */

    /*
    Moving misplaced instr before the WHERE block.

    Normally we move the Misplaced Instr instructions and put them
      in the same order before the predicate.

    important-Note: If we have 2 Misplaced Instr with the same dest register,
    the WHERE block will be surely split at least for
    the 2nd Misplaced Instr. For example, from MatMul-256.f16:

      R(11) = R(23) == R(1);
      NOP;
    );
    EXECUTE_WHERE_EQ(
      R(19) = ISHL(R(21), 10);
      // Assume it's not here: R(19) = R(10) | R(19);
      // Assume it's not here: R(25) = R(1) & R(10);
      R(10) = R(0) | R(0); // COPY
      R(10) = R(26) - R(1);
      R(11) = R(1) << R(11);
      R(10) = R(0) | R(0); // COPY
      R(10) = R(11) & R(20);
    The 2nd COPY forces the WHERE to be split
      - it's actually a different variable.

    Note: although not important, in principle we could
      have non-SPECIALV_H instrs inside WHERE blocks if
      the register is NOT initialized. */
    LLVM_DEBUG(dbgs() << "  moving I2 immediately before the "
                         "predicate instruction linked to the "
                         "WHERE block (Case 1 from paper)\n");

    MachineBasicBlock::iterator Ipredicate = IMI;
    LLVM_DEBUG(dbgs() << "    IMI = " << *IMI << "\n");
    Ipredicate--;
    LLVM_DEBUG(dbgs() << "    Ipredicate = " << *Ipredicate << "\n");

    /*
    if (Ipredicate->getOpcode() != Connex::NOP_BPF)
      LLVM_DEBUG(dbgs() << "PassHandleMisplacedInstr: Warning: "
             "Ipredicate->getOpcode() != Connex::NOP_BPF\n");
    */
    assert(Ipredicate->getOpcode() == Connex::NOP_BPF
           //|| Ipredicate->getOpcode() == Connex::NOP
    );

    /* Ipredicate is pointing at 2 instructions before the
        WHERE* instruction, normally at the predicate
        instruction.*/
    Ipredicate--;

    LLVM_DEBUG(dbgs() << "    Ipredicate = " << *Ipredicate << "\n");

    // Important-TODO: check better: check for right (w.r.t. WHERE) predicate
    // instruction before NOP
    assert(Ipredicate->getOpcode() == Connex::EQ_H ||
           Ipredicate->getOpcode() == Connex::LT_H ||
           Ipredicate->getOpcode() == Connex::ULT_H //);
           ||
           // This is for the case of using lane gating instructions
           //    (DISABLE_CELL, ENABLE_ALL_CELLS)
           Ipredicate->getOpcode() == Connex::EQ_SPECIAL_H ||
           Ipredicate->getOpcode() == Connex::LT_SPECIAL_H ||
           Ipredicate->getOpcode() == Connex::ULT_SPECIAL_H);

    assert(Ipredicate->getOperand(0).isReg() &&
           Ipredicate->getOperand(0).isDef());
    assert(I2->getOperand(0).isReg() && I2->getOperand(0).isDef());

    /*
    // This case can be handled (ONLY) by splitting WHERE block:
    #ifndef ALLOW_COPY_BETWEEN_PREDICATE_AND_WHERE_INSTRUCTIONS
    assert(I2->getOperand(1).getReg() != Ipredicate->getOperand(0).getReg() &&
          "We reached a case that's not treatable by  to implement this case!");
    #endif
    */

    /* Checking for WAR/anti-dependence between the predicate and Misplaced
        Instr instruction
       - if so, then changing order (moving Misplaced Instr before predicate)
         compromises correctness so we make a copy of the respective predicate
         input. */
    // I2 is the Misplaced Instr instruction
    assert(I2->getOperand(0).isReg() && I2->getOperand(0).isDef());
    //
    // Ipredicate is the predicate instruction
    assert(Ipredicate->getOperand(1).isReg() &&
           Ipredicate->getOperand(1).isUse());
    assert(Ipredicate->getOperand(2).isReg() &&
           Ipredicate->getOperand(2).isUse());
    //
    bool sameOpnd1 =
        Ipredicate->getOperand(1).getReg() == I2->getOperand(0).getReg();
    bool sameOpnd2 =
        Ipredicate->getOperand(2).getReg() == I2->getOperand(0).getReg();
    //
    if (sameOpnd1 || sameOpnd2) {
      LLVM_DEBUG(
          dbgs()
          << "Moving Misplaced Instr before WHERE predicate breaks "
             "WAR/anti-dependence relation between Misplaced Instr and "
             "predicate. "
             "--> fixing the problem by making copy of predicate input.\n");

      /* TODO???: if Ipredicate has a use of the dest register of EQ????????????
      then add: a) an instr before Misplaced Instr with
      CONNEX_RESERVED_REGISTER_01 = Rinput_EQ | Rinput_EQ
      */

      /* We preserve the input register of the predicate instruction since it
         will be overwritten by the moved (before the predicate)
         Misplaced Instr instruction:
          we make a copy:
            CONNEX_RESERVED_REGISTER_01 = Rdst_MisplacedInstr |
                                          Rdst_MisplacedInstr
      */
#ifndef ALLOW_COPY_BETWEEN_PREDICATE_AND_WHERE_INSTRUCTIONS
#ifdef COPY_REGISTER_IMPLEMENTED_WITH_ORV_H
      BuildMI(MBB, Ipredicate,
              /* We insert this MachineInstr before Ipredicate.
                Also the Misplaced Instr I2 we move after this, after
                Ipredicate, so I2 will be moved after this new copy */
              IMI->getDebugLoc(), TII->get(Connex::ORV_H),
              CONNEX_RESERVED_REGISTER_01)
          .addReg(I2->getOperand(0).getReg())
          .
          /* Note: I2 (Misplaced Instr) does NOT necessarily have the
             same dest register as Ipredicate. */
          addReg(I2->getOperand(0).getReg());
#else
#error "This case is NOT implemented. Implement it!"
#endif
#endif
      /* This really helps a lot since the Misplaced Instr moved before
           Ipredicate should be visible inside the WHERE block,
           so then we need to make the Ipredicate destination a reserved reg.
           Chances are big (but it's not necessary to be so I think) that since
            sameOpnd1 || sameOpnd2, then we can have Ipredicate with
             Ipredicate->getOperand(0) == I2->getOperand(0);
             and if we leave it like that then we shadow the Misplaced Instr.
             . */
      if (Ipredicate->getOperand(0).getReg() == I2->getOperand(0).getReg())
        Ipredicate->getOperand(0).setReg(CONNEX_RESERVED_REGISTER_01);

      // Note: Ipredicate is the predicate instruction
      /* These checks handle also the case both input operands of Ipredicate
          are the same.
      */
      if (sameOpnd1)
        Ipredicate->getOperand(1).setReg(CONNEX_RESERVED_REGISTER_01);
      if (sameOpnd2)
        Ipredicate->getOperand(2).setReg(CONNEX_RESERVED_REGISTER_01);

      /* We now normally have to update the uses of modified input of
        Ipredicate for the following instructions between the predicate
        and the place where the Misplaced Instr was.
        However, the instructions using the input after predicate are
        only the ones in the WHERE block basically.
      */
      updateUsesOfRegUntilMisplacedInstr(Ipredicate,
                                         I2, // Misplaced Instr
                                         IE, I2->getOperand(0).getReg(),
                                         CONNEX_RESERVED_REGISTER_01);
    } else // MEGA-TODO: think if OK
        if (Ipredicate->getOperand(0).getReg() == I2->getOperand(0).getReg()) {
      // If we have a WAW (output) dependendce
      // Note: Ipredicate is the predicate, I2 is the Misplaced Instr
      LLVM_DEBUG(
          dbgs()
          << "    Found that the Misplaced Instr to be moved "
             "immediately before the predicate of the "
             "WHERE block has the same destination register as the predicate. "
             "This forces us to handle specially "
             "the predicate instr dest register, "
             "since this dest "
             "register is the same as the one of the "
             "Misplaced Instr (hence, a WAW dependence is broken "
             "and the program would become incorrect "
             "otherwise).\n");

      /* We update dest register of of Ipredicate (predicate)
       due to conflict with I2, which we move before it. */
      /*
      if (destRegisterPredicateOfSplitWhere != -1)
        Ipredicate->getOperand(0).setReg(destRegisterPredicateOfSplitWhere);
      else
        Ipredicate->getOperand(0).setReg(CONNEX_RESERVED_REGISTER_01);
      */
      Ipredicate->getOperand(0).setReg(CONNEX_RESERVED_REGISTER_02);
      //
      updateUsesOfRegUntilMisplacedInstr(Ipredicate,
                                         I2, // Misplaced Instr
                                         IE, I2->getOperand(0).getReg(),
                                         CONNEX_RESERVED_REGISTER_02);
    }

    // We move the Misplaced Instr instruction before the predicate
    MBB.remove((&(*I2)));
    // MBB.insert(IMI, I2); // It inserts before IMI
#ifdef ALLOW_COPY_BETWEEN_PREDICATE_AND_WHERE_INSTRUCTIONS
    MBB.insert(Ipredicate,
               IMI); // It inserts immediately before the WHERE instr
#else
    MBB.insert(Ipredicate, (&(*I2)));         // It inserts before Ipredicate
#endif
    changedMF = true;

    // We handle the case of more than 1 Misplaced Instr instr in WHERE block
    // I2plus1 represents the next instr after the Misplaced Instr (before move)
    I2 = I2plus1;
  } // End putMisplacedInstrBeforeWhereBlock()

  inline void
  splitWhereBlock(MachineBasicBlock &MBB, const TargetInstrInfo *TII,
                  MachineBasicBlock::iterator &I, MachineInstr *&IMI,
                  MachineBasicBlock::iterator &I2, // Misplaced Instr instr
                  MachineBasicBlock::iterator &IE, bool &changedMF,
                  int &destRegisterPredicateOfSplitWhere) {
    /* This case handles only the cases we ran so far.
        See MEGA-TODO for limitation of this case. */
    changedMF = true;

    LLVM_DEBUG(dbgs() << "  splitWhereBlock(): IMI = " << *IMI);
    LLVM_DEBUG(dbgs() << "  splitWhereBlock(): I2 = " << *I2 << "\n");

    /* TODO: handle case
       where we have Misplaced Instr between 2 instr like ADD and
       ADDC, which is incorrect because the Misplaced Instr messes
       up the Connex flags. */
    MachineBasicBlock::iterator I2plus1 = I2;
    I2plus1++;
    // I think this does NOT cover all cases but most of them
    assert(
        I2plus1->getOpcode() != Connex::ADDCV_H &&
        I2plus1->getOpcode() != Connex::SUBCV_H &&
        I2plus1->getOpcode() != Connex::ADDCV_SPECIAL_H &&
        I2plus1->getOpcode() != Connex::SUBCV_SPECIAL_H &&
        "We do NOT handle yet ADDCV/SUBCV instructions immediately after "
        "Misplaced Instr for this case (and the corresponding ADD/SUB before "
        "the Misplaced Instr)");

    LLVM_DEBUG(dbgs() << "  splitting WHERE block in 2 s.t. we put I2 "
                         "immediately after new END_WHERE resulting from "
                         "split.\n");
    // I = beginning of new WHERE block
    // const TargetInstrInfo *TII =
    //                       MF.getSubtarget<ConnexSubtarget>().getInstrInfo();

    MachineBasicBlock::iterator Ipredicate = IMI;
    // We make Ipredicate point to the predicate of this WHERE
    // block
    Ipredicate--;
    LLVM_DEBUG(dbgs() << "  splitWhereBlock(): Ipredicate = " << *Ipredicate
                      << "\n");
    assert(Ipredicate->getOpcode() == Connex::NOP_BPF);
    Ipredicate--;
    LLVM_DEBUG(dbgs() << "  splitWhereBlock(): Ipredicate (2 instr before) = "
                      << *Ipredicate << "\n");

    unsigned regDest = CONNEX_RESERVED_REGISTER_02;
    int changedPredicateOpnd = -1;

    // We check Ipredicate, the predicate, is 3-opcode
    assert(((
                // For the standard case:
                (Ipredicate->getOpcode() == Connex::EQ_H ||
                 Ipredicate->getOpcode() == Connex::LT_H ||
                 Ipredicate->getOpcode() == Connex::ULT_H) &&
                Ipredicate->getNumOperands() == 3) ||
            (
                // For disabled lane gating regions
                (Ipredicate->getOpcode() == Connex::EQ_SPECIAL_H ||
                 Ipredicate->getOpcode() == Connex::LT_SPECIAL_H ||
                 Ipredicate->getOpcode() == Connex::ULT_SPECIAL_H) &&
                Ipredicate->getNumOperands() == 4)) &&
           Ipredicate->getOperand(0).isReg() &&
           Ipredicate->getOperand(0).isDef() &&
           Ipredicate->getOperand(1).isReg() &&
           Ipredicate->getOperand(1).isUse() &&
           Ipredicate->getOperand(2).isReg() &&
           Ipredicate->getOperand(2).isUse());

    unsigned predicateInstrOpnd[2];
    predicateInstrOpnd[0] = Ipredicate->getOperand(1).getReg();
    predicateInstrOpnd[1] = Ipredicate->getOperand(2).getReg();

    destRegisterPredicateOfSplitWhere = Ipredicate->getOperand(0).getReg();
    LLVM_DEBUG(
        dbgs()
        << "PassHandleMisplacedInstr: destRegisterPredicateOfSplitWhere = "
        << destRegisterPredicateOfSplitWhere << "\n");

    /*
    assert( (predicateInstrOpnd[0] != CONNEX_RESERVED_REGISTER_02) &&
    (predicateInstrOpnd[1] != CONNEX_RESERVED_REGISTER_02) &&
    // MEGA-TODO: implement this - it happens for ADD/MUL.f16
    "We currently can't handle these cases because we have only 1 reserved "
            "register.");
    */
    unsigned predicateInstrOpcode = Ipredicate->getOpcode();
    unsigned predicateInstrOpndAux[2];

    /* We look if predicateInstrOpnd[*] is updated/redefined
        either in the predicate instruction or in the
        instructions of the
        associated WHERE block before the Misplaced Instr instr.
          - i.e., if predicateInstrOpnd[1] changes then
            use it as predicateInstrOpnd[0].
       If NO change happens we do NOT need to save the
        value of predicateInstrOpnd[*], i.e., to create
         ORV_H below.

     We check this from Ipredicate(+1) (next instr after predicate) to I2(-1)
            (Misplaced Instr instr, exclusive).
      We check if any of the operands of the predicate change.
        NOTE: assert (if both change - we don't want to waste by reserving 2
          Connex registers - maybe we can change the Connex ASM code by hand
            to avoid this).
    */
    /*
    if (Ipredicate->getOperand(0).getReg() ==
                    Ipredicate->getOperand(1).getReg()) {
      // We changed the 1st input operand of the predicate
      changedPredicateOpnd = 0;
    }
    else
    if (Ipredicate->getOperand(0).getReg() ==
                    Ipredicate->getOperand(2).getReg()) {
      // We changed the 2nd input operand of the predicate
      changedPredicateOpnd = 1;
    }
    */

    MachineBasicBlock::iterator Iaux = Ipredicate;
    // Iaux++;
    MachineBasicBlock::iterator IauxEnd = I2; // I2 is Misplaced Instr

    IauxEnd++; // TREAT_ONLY_ONCE_CHANGE_PREDICATE_OPERANDS
    // IauxEnd--;

    /* Important: for the NEW predicate we don't care what we use for the
        destination register.

       We now check for the NEW predicate we create for the split if its input
        operands are updated between the
           original_predicate..Misplaced Instr */
    for (; Iaux != IauxEnd && Iaux != IE; Iaux++) {
      LLVM_DEBUG(dbgs() << "  splitWhereBlock(): Iaux = " << *Iaux << "\n");
      if (Iaux->getNumOperands() >= 1 && Iaux->getOperand(0).isReg() &&
          Iaux->getOperand(0).isDef()) {
        if (Iaux->getOperand(0).getReg() == predicateInstrOpnd[0]) {
          assert((changedPredicateOpnd == -1 || changedPredicateOpnd == 0) &&
                 // MEGA-TODO: handle this assert violation case
                 "It seems both input operands of the "
                 "predicate get updated so we would need to "
                 "reserve 2 Connex registers to handle well "
                 "this case.");
          // We find that we subsequently change the 1st input operand of
          //   the predicate
          changedPredicateOpnd = 0;
        } else if (Iaux->getOperand(0).getReg() == predicateInstrOpnd[1]) {
          /* We find that we subsequently change
             the 2nd input operand of the predicate */
          assert((changedPredicateOpnd == -1 || changedPredicateOpnd == 1) &&
                 // MEGA-TODO: handle this assert violation case
                 "It seems both input operands of the "
                 "predicate get updated so we would need "
                 "to reserve 2 Connex registers to handle "
                 "well this case.");
          changedPredicateOpnd = 1;
        }
      }
    }

    LLVM_DEBUG(dbgs() << "  changedPredicateOpnd = " << changedPredicateOpnd
                      << " (for the input operands of the predicate)\n");

    if (changedPredicateOpnd == -1) {
      // regDest = predicateInstrOpnd[0];
      predicateInstrOpndAux[0] = predicateInstrOpnd[0];
      predicateInstrOpndAux[1] = predicateInstrOpnd[1];
    } else {
      /* Put a copy of the changed input register of the predicate instruction
         before Ipredicate, the initial predicate of this WHERE block. */
#ifdef COPY_REGISTER_IMPLEMENTED_WITH_ORV_H
      if (regDest != predicateInstrOpnd[changedPredicateOpnd]) {
        BuildMI(MBB, Ipredicate, IMI->getDebugLoc(), TII->get(Connex::ORV_H),
                regDest)
            . // The reserved register, CONNEX_RESERVED_REGISTER_02
            addReg(predicateInstrOpnd[changedPredicateOpnd])
            .addReg(predicateInstrOpnd[changedPredicateOpnd]);
      }
#else
#error "This case is NOT implemented. Implement it!"
#endif

      /*
      predicateInstrOpndAux[0] = regDest; // Reserved register
      predicateInstrOpndAux[1] = predicateInstrOpnd[1 - changedPredicateOpnd];
      */
      predicateInstrOpndAux[changedPredicateOpnd] =
          CONNEX_RESERVED_REGISTER_02; // regDest
      predicateInstrOpndAux[1 - changedPredicateOpnd] =
          predicateInstrOpnd[1 - changedPredicateOpnd];
    }

    LLVM_DEBUG(dbgs() << "  predicateInstrOpndAux[0] = "
                      << predicateInstrOpndAux[0] << "\n");
    LLVM_DEBUG(dbgs() << "  predicateInstrOpndAux[1] = "
                      << predicateInstrOpndAux[1] << "\n");

    MachineBasicBlock::iterator I2succ = I2;
    I2succ++;
    BuildMI(MBB,
            I2, // Immediately before the Misplaced Instr instr
            IMI->getDebugLoc(), TII->get(Connex::END_WHERE)
            //, I2->getOperand(0).getReg()
    );
    LLVM_DEBUG(dbgs() << "  Finished creating the END_WHERE\n");

    /*
    // TREAT_ONLY_ONCE_CHANGE_PREDICATE_OPERANDS

     // Ipredicate is predicate
   #if 0
     // Unnecessary check:
     assert(Ipredicate->getOperand(0).getReg() !=
          I2->getOperand(0).getReg());
   #endif

    // This check is actually VAGUELY different from the one above because
    //   the one above inserts a register save (copy) instruction before the
    //   original WHERE, while this new one after the new END_WHERE resulting
    //   from the split.
    // Very Important Note: the new predicate WHERE can have the result stored
   in
    //                     RESERVED_REGISTER.
    //  We now check for conflicts between:
    //     - destination register operand of Misplaced Instr and
    //     - input registers of predicate instruction.
    //
    // Note: I2 is the Misplaced Instr instruction that triggered the split of
    //        WHERE block.
    //
    // Addressing the case, where after the split of WHERE* block we have
    //     something like this immediately after the 1st new WHERE* block,
    //     before the 2nd WHERE* block, where the repeated predicate instruction
    //     (repeated by us) happens to use the register defined in the Misplaced
    //     Instr instruction, which makes the computation incorrect:
    //   END_WHERE;
    //   R(26) = R(10) | R(10); // This COPY (Misplaced Instr) instruction is
    //                          //    the reason of the split
    //   R(30) = R(26) < R(3);
    //   NOP
    //   WHERE*
    //
    //  Note: CONNEX_RESERVED_REGISTER_01 is a reserved register.
    //
    //  To correct the problem in this example we have to copy the value of
    //       R(26) in R(30):
    //   END_WHERE;
    //   R(30) = R(26) | R(26);
    //   R(26) = R(10) | R(10); // This COPY (Misplaced Instr) instruction is
    //                          //    the reason of the split
    //   R(30) = R(30) < R(3);
    //   NOP
    //   WHERE*
    int changeInputPredicateOperandsDueToMisplacedInstr = 0;
    if (predicateInstrOpnd[0] == I2->getOperand(0).getReg()) {
      changeInputPredicateOperandsDueToMisplacedInstr |= 1;
    }
    if (predicateInstrOpnd[1] == I2->getOperand(0).getReg()) {
      changeInputPredicateOperandsDueToMisplacedInstr |= 2;
    }
    //
    assert(changeInputPredicateOperandsDueToMisplacedInstr != 3 &&
           // important-TODO: handle this assert violation case
           "We shouldn't have such a case - doesn't really make sense for a "
           "conditional to have both operands equal.");

    LLVM_DEBUG(dbgs() << "  changeInputPredicateOperandsDueToMisplacedInstr = "
                      << changeInputPredicateOperandsDueToMisplacedInstrMB
                      << "\n");
    // assert(! (changedPredicateOpnd != -1 &&
    //       changeInputPredicateOperandsDueToMisplacedInstr != 0) &&
    //        // TODO: if not merging the 2 cases together, handle this assert
    //        //   violation case,
    //       "We currently can't handle both cases simultaneously.");
    //
    if (changeInputPredicateOperandsDueToMisplacedInstr != 0) {
      LLVM_DEBUG(dbgs()
            << " PassHandleMisplacedInstr::runOnMachineFunction(): correcting "
               "the conflicting register (due to the Misplaced Instr) in the "
               "predicate instruction\n");
      MachineBasicBlock::iterator Icorrect = I2succ;
      //Icorrect++;
     #ifdef COPY_REGISTER_IMPLEMENTED_WITH_ORV_H
      BuildMI(MBB,
              Icorrect, // We insert this MachineInstr after new END_WHERE,
                        //   before Misplaced Instr instr
              IMI->getDebugLoc(),
              TII->get(Connex::ORV_H),
              CONNEX_RESERVED_REGISTER_02).
        addReg(I2->getOperand(0).getReg()).
        addReg(I2->getOperand(0).getReg());
     #else
       #error "This case is NOT implemented. Implement it!"
     #endif


      // Note: Ipredicate is the predicate for the 1st (part) WHERE* block.
      // // Ipredicate->getOperand(1).setReg(CONNEX_RESERVED_REGISTER_02);

      LLVM_DEBUG(dbgs()
            << "PassHandleMisplacedInstr: after WHERE block processed: MBB = ";
            MBB.dump());
      // We check that we don't mess up the program - TODO we should also check
      //   that the iterators are not messed up
      // for (MachineBasicBlock::iterator Inew = MBB.begin(),
      //       IEnew = MBB.end(); Inew != IEnew; ++Inew) {
      //  //MachineInstr *IMI = I;
      //  LLVM_DEBUG(dbgs() << "  runOnMachineFunction(): Inew = "
      //                    << *Inew << "\n");
      // }
    }
    */ // End comment (TREAT_ONLY_ONCE_CHANGE_PREDICATE_OPERANDS)

    // I2succ++;
    LLVM_DEBUG(dbgs() << "  moving I2 immediately after END_WHERE of "
                         "split WHERE block\n");

    // Very Important: We create another predicate, a NOP and a new WHERE*
    // instructions, identical with the (previous) one associated to the
    // WHERE block, EXCEPT the destination register is
    // CONNEX_RESERVED_REGISTER_01 - this is safe.
    BuildMI(
        MBB,
        I2succ, // We insert new instr immediately before I2succ
        IMI->getDebugLoc(), TII->get(predicateInstrOpcode),
        CONNEX_RESERVED_REGISTER_01 // TODO: (2020_12_09) prove it's correct: it
                                    // looks we can use here also _01 reg (prove
                                    // by exploring all cases) instead of
                                    // CONNEX_RESERVED_REGISTER_03
        /*
        // destRegisterPredicateOfSplitWhere is made -1 only after
        // iterating over END_WHERE, below
        destRegisterPredicateOfSplitWhere != -1 ?
          destRegisterPredicateOfSplitWhere :
          regDest // It is CONNEX_RESERVED_REGISTER_02
        */
        )
        .
        // We now change the conflicting register in the predicate
        //   instruction.
        addReg((changedPredicateOpnd == 0)
                   ?
                   /* // (! TREAT_ONLY_ONCE_CHANGE_PREDICATE_OPERANDS)
                   addReg(((changeInputPredicateOperandsDueToMisplacedInstr & 1)
                   == 1) ?
                   */
                   (unsigned)CONNEX_RESERVED_REGISTER_02
                   : predicateInstrOpndAux[0])
        . // predicateInstrOpnd1).
        addReg((changedPredicateOpnd == 1)
                   ?
                   /* // (! TREAT_ONLY_ONCE_CHANGE_PREDICATE_OPERANDS)
                   addReg(((changeInputPredicateOperandsDueToMisplacedInstr & 2)
                   == 2) ?
                   */
                   (unsigned)CONNEX_RESERVED_REGISTER_02
                   : predicateInstrOpndAux[1]);

    BuildMI(MBB, I2succ, IMI->getDebugLoc(), TII->get(Connex::NOP_BPF));
    // TODO: maybe add an addImm(0)?, although it works without

    // We add the same WHERE instr as the one for this block
    /* This gives the following error:
    <<Assertion `!N->getParent() &&
                "machine instruction already in a basic block"' failed.>>
    MBB.insert(I2succ, IMI); // before I2succ
    */
    LLVM_DEBUG(dbgs() << "  splitWhereBlock(): *IMI (for split) = " << *IMI
                      << "\n");

    LLVM_DEBUG(dbgs() << "  splitWhereBlock(): *I2succ = " << *I2succ << "\n");

    /*
    IMI = I2succ;
    LLVM_DEBUG(dbgs() << "   IMI = I2succ = "
                      << *IMI << "\n");
    // Important: This makes IMI NULL since IMI is a MachineInstr
    //   - see 35l_MatMul_f16/SIZE_128/L_required_manual_move_fills/
    //         STDerr_llc_01_old17
    IMI--;
    */
    // From http://llvm.org/doxygen/MachineInstrBuilder_8h_source.html#l00312:
    //  "inserts the newly-built instruction before the given position".
    // See good comments on iterator invalidation at:
    // http://llvm.1065342.n5.nabble.com/
    //      deleting-or-replacing-a-MachineInst-td77723.html
    I = BuildMI(MBB,
                I2succ, // We insert new instr immediately before I2succ
                IMI->getDebugLoc(), TII->get(IMI->getOpcode())
                //, regDest
    );

    // TODO: understand if it generates (due to iterator invalidation??) another
    //    END_WHERE - see Tests/DawnCC/25k_map_i32/MUL_i32/ (output_old06.cpp?)

    // NOTE: I is the new WHERE* instruction just created
    // We update I2 to check for more Misplaced Instr instrs after the new
    //   created WHERE
    I2 = I;
    I2++;

    // We update IMI since we insert Misplaced Instr before the predicate of
    //   WHERE using IMI
    IMI = (&(*I));

    // MachineBasicBlock::iterator Iaux10 = I2succ; Iaux10--;
    LLVM_DEBUG(dbgs() << "   *I2succ = " << *I2succ << "\n");
    LLVM_DEBUG(dbgs() << "   *IMI = " << *IMI << "\n");
    LLVM_DEBUG(dbgs() << "   *I = " << *I << "\n");
    LLVM_DEBUG(dbgs() << "   *I2 = " << *I2 << "\n");

    // break;
    // assert();
    LLVM_DEBUG(dbgs() << "   To check: *IMI = " << *IMI << "\n");

    LLVM_DEBUG(
        dbgs() << "splitWhereBlock(): after splitting WHERE block in 2: MBB = ";
        MBB.dump());
  } // End splitWhereBlock()

  /*
   * Note: The structure of the loop nest with iterators is:
   * I = main loop iterating over all instr of the MBB
   *   IMI = I;
   * I2
   *   if IMI == WHERE*
   *     I2 = I + 1;
   *     for (;; I2++) // <--- here starts handleMisplacedInstrs()
   *       if I2 == ORV_H (or whatever is used to implement the COPY
   *                                               (Misplaced Instr) primitive)
   *         for (I3 = IMI + 1; ; I3++) // used to compute whatToDo;
               if I3 == END_WHERE
                 break;
               compute whatToDo;
  */
  inline void
  handleMisplacedInstrs(MachineBasicBlock &MBB, const TargetInstrInfo *TII,
                        MachineBasicBlock::iterator &I, MachineInstr *&IMI,
                        MachineBasicBlock::iterator &I2,
                        // Misplaced Instr
                        MachineBasicBlock::iterator &IE, bool &changedMF,
                        int &destRegisterPredicateOfSplitWhere) {
    LLVM_DEBUG(dbgs() << "Entered handleMisplacedInstrs()");

    // Iterating over all remaining instructions of the BB
    for (; I2 != IE; /* I2++ */) {
      LLVM_DEBUG(dbgs() << "  I2 = " << *I2);

      // TO_ADAPT: currently copyPhysReg() is implemented with ORV_H

      // Important: NORMALLY, inside WHERE blocks generated
      // with OPINCAA lib's Kernel::genLLVMISelManualCode(),
      // we are guaranteed to have only ORV_SPECIAL_H Connex
      // instructions, so meeting an ORV_H is only when a Misplaced Instr
      // was generated by the TwoAddressInstructionPass.
      if (
#ifdef COPY_REGISTER_IMPLEMENTED_WITH_ORV_H
          I2->getOpcode() == Connex::ORV_H
#else
#error "This case is NOT implemented. Implement it!"
#endif
          || I2->getOpcode() == Connex::LD_FILL_H) {
        // MEGA-TODO: || I2->getOpcode() == Connex::ST_FILL_H

        // The ORV_H instruction implemented in copyPhysReg()
        //    has both input operands equal.
        //    NOTE: the destination register of any instruction
        //        I is I->getOperand(0).

#ifdef COPY_REGISTER_IMPLEMENTED_WITH_ORV_H
        if (I2->getOpcode() == Connex::ORV_H)
          assert(I2->getOperand(1).getReg() == I2->getOperand(2).getReg() &&
                 "I2 is an ORV_H with different input operands. "
                 "Maybe too paranoid check: We do not "
                 "recommend to have emulation OPINCAA kernels "
                 "generated by Kernel::genLLVMISelManualCode() "
                 "with ORV_H inside WHERE blocks (if these "
                 "instructions come from there). But you "
                 "can comment this assert and issue a simple "
                 "warning.");
          /*
          if (I2->getOperand(1).getReg() !=
                    I2->getOperand(2).getReg())
            LLVM_DEBUG(dbgs() << "PassHandleMisplacedInstr: Warning: "
                   "I2->getOperand(1).getReg() != "
                   "I2->getOperand(2).getReg()\n\n");
          */
#endif // COPY_REGISTER_IMPLEMENTED_WITH_ORV_H

        // From http://llvm.org/doxygen/MachineBasicBlock_8h_source.html:
        //   MBB::insert(iterator, MI)
        // "Insert MI into the instruction list before I, possibly inside a
        //          bundle.
        LLVM_DEBUG(dbgs() << "  found Misplaced Instr (COPY/LD_FILL) at I2 = "
                          << *I2
                          << " --> moving it out of the WHERE block to "
                             "preserve correct program semantics.\n");

        // We should move I2 before or after the WHERE block,
        // or split the WHERE block in 2. */
        /* The algo is (a sketch that MIGHT NOT reflect
             totally the implementation):
            NOTE: this is the case that allows having Misplaced Instr between
                   predicate and WHERE instr.
            If the Misplaced Instr doesn't use (doesn't have as source)
                  a register defined in the WHERE block
                  BEFORE the Misplaced Instr (NO RAW/flow dependence relation
                  to be broken)
                  and also the Misplaced Instr doesn't define a register
                  that is used by an instruction before
                  (NO WAR/anti-dependence relation to be broken):
                We move the Misplaced Instr exactly before the
                  WHERE instruction starting the block
            Else
              If the Misplaced Instr doesn't use (doesn't have as source)
                  a register defined in the WHERE block,
                  after the Misplaced Instr (NO WAR dep broken)
                  and also the Misplaced Instr doesn't define a register
                   used by an instruction after it (NO RAW dep broken):
                We move the Misplaced Instr exactly after the END_WHERE
                  instruction ending the block
            Else
              Moving the Misplaced Instr immediately before/after
                   the WHERE block is UNsafe and
                   would change semantics program
                The solution is to split the WHERE block in
                  two and for the 2nd WHERE block to copy the
                  predicate (together with a NOP) just
                  before it.
        */

#ifdef ALLOW_COPY_BETWEEN_PREDICATE_AND_WHERE_INSTRUCTIONS
        MachineBasicBlock::iterator I3 = IMI; // IMI is WHERE instr
        LLVM_DEBUG(dbgs() << "   *I3 = " << *I3 << "\n");

        I3--;
        LLVM_DEBUG(dbgs() << "   *I3 (after 1 -)= " << *I3 << "\n");

        assert(I3->getOpcode() == Connex::NOP ||
               I3->getOpcode() == Connex::NOP_BPF);

        I3--;
        LLVM_DEBUG(dbgs() << "   *I3 (after 2 -)= " << *I3 << "\n");
        assert(I3->getOpcode() == Connex::EQ_H ||
               I3->getOpcode() == Connex::LT_H ||
               I3->getOpcode() == Connex::ULT_H);
#else
        MachineBasicBlock::iterator I3 = IMI; // IMI is WHERE instr
        I3++;
#endif

#define SAFE_SINCE_NO_CONSTRAINT 0
#define UNSAFE_TO_PUT_COPY_BEFORE_WHERE_BLOCK 1
#define UNSAFE_TO_PUT_COPY_AFTER_WHERE_BLOCK 2
#define SAFE_TO_PUT_COPY_IN_SPLIT_WHERE_BLOCK 3
        int whatToDo = SAFE_SINCE_NO_CONSTRAINT;

        // bool I2afterIsInsideWhereBlock = true;
        bool I3IsBeforeI2 = true;

        // Remember: I2 points to the Misplaced Instr instruction
        for (; I3 != IE; I3++) {
          if (I3->getOpcode() == Connex::END_WHERE) {
            break;
          }

          LLVM_DEBUG(dbgs() << "  *I3 = " << *I3);

          if (I3 == I2) {
            I3IsBeforeI2 = false;
            continue;
          }
          LLVM_DEBUG(dbgs() << "    I3IsBeforeI2  =  " << I3IsBeforeI2 << "\n");

          // We look at all operands of instruction I3
          //    Note: I3->getOperand(0) is result of I3; the rest are inputs.
          for (unsigned idOpnd = 0; idOpnd < I3->getNumOperands(); idOpnd++) {
            MachineOperand &I3Opnd = I3->getOperand(idOpnd);

            LLVM_DEBUG(dbgs() << "  I3Opnd (index = " << idOpnd
                              << ") = " << I3Opnd << "\n");

            if (I3Opnd.isReg() && I3Opnd.isUse()) {
              // Remember: I2 points to the Misplaced Instr instruction
              if (I3Opnd.getReg() == I2->getOperand(0).getReg()) {
                if (I3IsBeforeI2) {
                  // RBW dependence w.r.t. Misplaced Instr (I2), which writes
                  // I3 uses or defines the dst-register of I2 (the Misplaced
                  //   Instr instr)
                  LLVM_DEBUG(dbgs() << "    I3, which is before I2, "
                                       "uses (RAW dependence) the "
                                       "dst-register of I2 "
                                       "--> moving I2 before the "
                                       "WHERE block is NOT safe\n");

                  whatToDo |= UNSAFE_TO_PUT_COPY_BEFORE_WHERE_BLOCK;
                  /*
                  LLVM_DEBUG(dbgs() << "    changing I2afterOpnd's reg to = "
                                    << I2->getOperand(0).getReg() << "\n");
                  I2afterOpnd.setReg(I2->getOperand(1).getReg());
                  */
                } else { // NOT I3IsBeforeI2
                  // RAW dependence w.r.t. Misplaced Instr (I2), which writes
                  // I3 uses the dst-register of I2 (the Misplaced Instr)
                  LLVM_DEBUG(dbgs() << "    I3, which is after I2, "
                                       "uses (RAW dependence) the dst-register "
                                       "of I2 --> moving I2 after the "
                                       "WHERE block is NOT safe\n");

                  whatToDo |= UNSAFE_TO_PUT_COPY_AFTER_WHERE_BLOCK;
                }
              } else
                  // Although we are safe on the else branch,
                  //   we put this code here for "completness".
                  if (
#ifdef COPY_REGISTER_IMPLEMENTED_WITH_ORV_H
                      I2->getOpcode() == Connex::ORV_H &&
#endif
                      I3Opnd.getReg() == I2->getOperand(1).getReg()) {
                // RAR dependence - none actually :)
                if (I3IsBeforeI2) {
                  // I3 uses the dst-register of I2 (the Misplaced Instr)
                  LLVM_DEBUG(dbgs() << "    I3, which is before I2, "
                                       "uses(RAR dependence) the "
                                       "src-register of I2 "
                                       "--> everything is safe\n");

                  // whatToDo |= UNSAFE_TO_PUT_COPY_BEFORE_WHERE_BLOCK;
                } else {
                  // I3 uses the dst-register of I2 (the Misplaced Instr)
                  LLVM_DEBUG(dbgs() << "    I3, which is after I2, "
                                       "uses (RAR dependence) the "
                                       "src-register of I2 "
                                       "--> everything is safe\n");

                  // whatToDo |= UNSAFE_TO_PUT_COPY_AFTER_WHERE_BLOCK;
                }
              }
            } // End I3Opnd.isUse()
            else if (I3Opnd.isReg() && I3Opnd.isDef()) {
              // Remember: I2 points to the Misplaced Instr
              if (I3Opnd.getReg() == I2->getOperand(0).getReg()) {
                if (I3IsBeforeI2) {
                  // WAW  dependence w.r.t. Misplaced Instr (I2), which writes
                  // I3 defs the dst-register of I2 (the Misplaced Instr instr)
                  LLVM_DEBUG(dbgs() << "    I3, which is before I2, "
                                       "defs (WAW dependence) the "
                                       "dst-register of I2 --> "
                                       "moving I2 before the "
                                       "WHERE block is NOT safe\n");

                  whatToDo |= UNSAFE_TO_PUT_COPY_BEFORE_WHERE_BLOCK;
                } else {
                  // WAW  dependence w.r.t. Misplaced Instr (I2), which writes
                  // I3 defs the dst-register of I2 (the Misplaced Instr instr)
                  LLVM_DEBUG(dbgs() << "    I3, which is after I2, "
                                       "defs (WAW dependence) the "
                                       "dst-register of I2 --> "
                                       "moving I2 after the "
                                       "WHERE block is NOT safe\n");

                  whatToDo |= UNSAFE_TO_PUT_COPY_AFTER_WHERE_BLOCK;
                }
              } else if (
#ifdef COPY_REGISTER_IMPLEMENTED_WITH_ORV_H
                  I2->getOpcode() == Connex::ORV_H &&
#endif
                  I3Opnd.getReg() == I2->getOperand(1).getReg()) {
                if (I3IsBeforeI2) {
                  // RAW dependence w.r.t. I3, which writes
                  // I3 defs the dst-register of I2 (the Misplaced Instr)
                  LLVM_DEBUG(dbgs() << "    I3, which is before I2, "
                                       "defs (RAW dependence) the src-register "
                                       "of I2 --> moving I2 before the "
                                       "WHERE block is NOT safe\n");

                  whatToDo |= UNSAFE_TO_PUT_COPY_BEFORE_WHERE_BLOCK;
                } else {
                  // RBW dependence w.r.t. I3, which writes
                  // I3 defs the dst-register of I2 (the Misplaced Instr instr)
                  LLVM_DEBUG(dbgs() << "    I3, which is after I2, "
                                       "defs (RAW dependence) the src-register "
                                       "of I2 --> moving I2 after the "
                                       "WHERE block is NOT safe\n");

                  whatToDo |= UNSAFE_TO_PUT_COPY_AFTER_WHERE_BLOCK;
                }
              }
            } // End I3Opnd.isDef()
          }   // End for loop idOpnd
        }     // End for loop with ind-var I3

        /*
         * Note: The structure of the loop nest with iterators is:
         * I = main loop iterating over all instr of the MBB
         *   IMI = I;
         * I2
         *   if IMI == WHERE*
         *     I2 = I + 1;
         *     for (;; I2++) // <--- here starts handleMisplacedInstrs()
         *       if I2 == ORV_H (or whatever is used to implement the COPY
         *                                    (Misplaced Instr) primitive)
         *         for (I3 = IMI + 1; ; I3++) // used to compute whatToDo;
                     if I3 == END_WHERE
                       break;
                     compute whatToDo;
        */

        MachineBasicBlock::iterator I2plus1 = I2;
        //
        // We need to increment it, otherwise it looks that
        //   I2 and I2plus1 are identical after remove()
        //   and insert()
        I2plus1++;
        LLVM_DEBUG(dbgs() << "  runOnMachineFunction(): I2plus1 = " << *I2plus1
                          << "\n");
        LLVM_DEBUG(
            dbgs() << "  runOnMachineFunction(): I2 (before moving I2) = "
                   << *I2 << "\n");
        LLVM_DEBUG(dbgs() << "  whatToDo = " << whatToDo << "\n");

        if ( // whatToDo == SAFE_SINCE_NO_CONSTRAINT ||
            whatToDo == UNSAFE_TO_PUT_COPY_AFTER_WHERE_BLOCK) {
          // Case 1 from paper
          LLVM_DEBUG(dbgs() << "  Case 1 from paper --> calling "
                               "putMisplacedInstrBeforeWhereBlock()");

          // Moving Misplaced Instr before the WHERE block.
          putMisplacedInstrBeforeWhereBlock(MBB, TII, IMI, I2, I2plus1, IE,
                                            changedMF,
                                            destRegisterPredicateOfSplitWhere);
          // break;

        } // End moving I2 just before logical instr linked to WHERE block
        else if (
            // We treat here SAFE_SINCE_NO_CONSTRAINT because moving after WHERE
            //  block doesn't add any auxiliary instruction
            whatToDo == SAFE_SINCE_NO_CONSTRAINT ||
            whatToDo == UNSAFE_TO_PUT_COPY_BEFORE_WHERE_BLOCK) {
          // Case 2 from paper
          // TODO: we should put multiple Misplaced Instr instructions from
          //   this WHERE block in the SAME order after END_WHERE. See if such
          //   cases happen.
          LLVM_DEBUG(dbgs() << "  moving I2 immediately after WHERE block "
                               "(Case 2 from paper)\n");
          assert(I3 != IE);

          LLVM_DEBUG(dbgs()
                     << "  runOnMachineFunction(): *I2 = " << *I2 << "\n");

          // I3 is pointing to END_WHERE (see code above)
          LLVM_DEBUG(dbgs()
                     << "  runOnMachineFunction(): *I3 = " << *I3 << "\n");

          assert((I3->getOpcode() == Connex::END_WHERE) &&
                 "I3 should point to END_WHERE (see code above).");
          /*
          assert( (I3->getOpcode() == Connex::WHEREEQ ||
            I3->getOpcode() == Connex::WHERELT ||
            I3->getOpcode() == Connex::WHERECRY) &&
           "We should NOT be arriving here otherwise.");
          */

          I3++; // Jump over END_WHERE (normally)
          LLVM_DEBUG(dbgs() << "  runOnMachineFunction(): *I3 (after I3++) = "
                            << *I3 << "\n");

          LLVM_DEBUG(dbgs()
                     << "  runOnMachineFunction(): Preparing to remove *I2 = "
                     << *I2 << "      and moving it before *I3 = " << *I3
                     << "\n");
          MBB.remove((&(*I2)));
          MBB.insert(I3, (&(*I2))); // It inserts before I3

          /*
          // This is NOT good for case where we have 2+ Misplaced Instrs
          //   instrs in the WHERE block: I = I3;
          // I2++;
          // I = I2;
          */
          LLVM_DEBUG(dbgs()
                     << "  runOnMachineFunction(): *I2 (after moving I2) = "
                     << *I2 << "\n");
          // I2plus1++;
          LLVM_DEBUG(dbgs() << "  runOnMachineFunction(): *I2plus1 = "
                            << *I2plus1 << "\n");

          // Here we handle the case of more than 1 Misplaced Instr
          // instr in the WHERE block (I2plus1 represents the next
          //   instr after the Misplaced Instr (before move))
          I2 = I2plus1;

          MachineBasicBlock::iterator I2plus2 = I2plus1;
          I2plus2++;
          LLVM_DEBUG(dbgs() << "  runOnMachineFunction(): *I2plus2 = "
                            << *I2plus2 << "\n");

          changedMF = true;
          // This is NOT good for case where we have 2+ Misplaced Instrs
          //   instrs in the WHERE block:  break;
          // We keep searching with I2 for loop in this WHERE block
          //   for more Misplaced Instrs.
        } // End if (whatToDo == UNSAFE_TO_PUT_COPY_BEFORE_WHERE_BLOCK)
        else if (whatToDo == SAFE_TO_PUT_COPY_IN_SPLIT_WHERE_BLOCK) {
          // Case 3 from paper
          LLVM_DEBUG(dbgs()
                     << "  Case 3 from paper --> calling splitWhereBlock()");
          splitWhereBlock(MBB, TII, I, IMI, I2, IE, changedMF,
                          destRegisterPredicateOfSplitWhere);
          LLVM_DEBUG(dbgs() << "  After calling splitWhereBlock(): *IMI = "
                            << *IMI << "\n");
        } // End if SPLIT WHERE block
        else
          // Important: we increment here the iterator over instruction in
          //     WHERE block
          I2++;
      } // End if (I2->getOpcode() == Connex::ORV_H)
      else {
        // Important: we increment here the iterator over instruction in
        //     WHERE block
        I2++;
        // else
      }

      // Note that the END_WHERE takes input node and has a value output
      if (I2->getOpcode() == Connex::END_WHERE) {
        LLVM_DEBUG(dbgs() << "  found END_WHERE --> breaking I2 loop\n");
        I2++;
        I = I2;

        // MEGA-TODO: think if OK here
        destRegisterPredicateOfSplitWhere = -1;

        LLVM_DEBUG(
            dbgs() << "    Making destRegisterPredicateOfSplitWhere = -1\n");

        break;
      }

      LLVM_DEBUG(
          dbgs() << "PassHandleMisplacedInstr: at end of for loop I2, *I2 = "
                 << *I2 << " and *IMI = " << *IMI);
    } // End for loop with ind-var I2
  }   // End handleMisplacedInstrs()

  /// \brief Loop over all of the basic blocks
  bool runOnMachineFunction(MachineFunction &MF) {
    bool changedMF = false;

    // See http://llvm.org/docs/doxygen/html/classllvm_1_1MachineFunction.html
    LLVM_DEBUG(
        dbgs() << "Entered PassHandleMisplacedInstr::runOnMachineFunction(MF = "
               //; MF.dump();
               << MF.getName()
               // dbgs()
               << ")\n");
    // bool Changed = false;

    // Process all basic blocks.
    for (auto &MBB : MF) {
      // int anotherReservedRegister = -1;
      int destRegisterPredicateOfSplitWhere = -1;

      // For the current MBB:
      // See llvm.org/docs/doxygen/html/classllvm_1_1MachineBasicBlock.html
      LLVM_DEBUG(
          dbgs()
          << "PassHandleMisplacedInstr::runOnMachineFunction(): a new MBB = "
          << MBB << "\n");

      const TargetInstrInfo *TII =
          MF.getSubtarget<ConnexSubtarget>().getInstrInfo();

      // See llvm.org/docs/doxygen/html/classllvm_1_1MachineBasicBlock.html
      LLVM_DEBUG(
          dbgs()
          << "PassHandleMisplacedInstr::runOnMachineFunction(): again MBB = "
          << MBB << "\n");

      for (MachineBasicBlock::iterator I = MBB.begin(), IE = MBB.end(); I != IE;
           ++I) {
        MachineInstr *IMI = (&(*I));
        /*
        if (IMI == &MI)
          I++;
          // predMI contains normally instruction VLOAD_H_SYM_IMM
          break;
        // predMI = I;
        */
        LLVM_DEBUG(dbgs() << "  runOnMachineFunction(): *I = " << *I << "\n");
        LLVM_DEBUG(
            dbgs() << "  runOnMachineFunction(): DontTreatCopyInstructions = "
                   << DontTreatCopyInstructions << "\n");

        if (DontTreatCopyInstructions == false) {
          // Important: we move the Misplaced Instr instructions outside
          //  the WHERE block, just like the ARM/Thumb2ITBlockPass.cpp
          //  does (the ARM pass is also registered in addPreSched2()).
          // Note that moving Misplaced Instrs before WHERE (ARM IT) blocks
          //  (as it seems ARM surprisingly is doing, since
          //  MBB::insert(iterator, MI) does "Insert MI into the
          //     instruction list before I, possibly inside a bundle.")
          //   can change semantics in most cases.

          // Important: First we remove any Misplaced Instrs
          //  generated by the TwoAddressInstructionPass and not erased
          //  by RegisterCoalescer (transformed
          //  into ORV_H) instructions inside WHERE* blocks.
          //  This is to handle cases like sequences of manually
          //  selected instructions in ConnexISelDAGToDAG for MULi32,
          //    DIVi16, etc.
          if (IMI->getOpcode() == Connex::WHEREEQ ||
              IMI->getOpcode() == Connex::WHERELT ||
              IMI->getOpcode() == Connex::WHERECRY) {
            LLVM_DEBUG(dbgs() << "runOnMachineFunction(): found WHERE block\n");

            // Removing useless COPY immediately before WHERE* block
            // (between NOP and WHERE*, where it should normally be put).
            //   It is useless - we eye-balled seriously on a few
            //    programs, most notably SSD.f16 on Jul 29-30 2018
            //     (I guess - MEGA-TODO: check if so) always because it is
            //   generated by the WHERE* instruction and,
            //   therefore, it's NOT required.
            //   important-TODO: we should take care of COPY
            //   instructions being moved by the post-RA scheduler. */
            MachineBasicBlock::iterator ItmpToErase = IMI;
            ItmpToErase--;
            if (ItmpToErase->getOpcode() != Connex::NOP_BPF
                //|| ItmpToErase->getOpcode() == Connex::NOP
            ) {
#ifdef COPY_REGISTER_IMPLEMENTED_WITH_ORV_H
              if (ItmpToErase->getOpcode() == Connex::ORV_H) {
#else
#error "This case is NOT implemented. Implement it!"
#endif
                MachineInstr *Iremove = (&(*ItmpToErase));
                // ItmpToErase--;

                // We assert this COPY is related to the WHERE*
                // instruction - if NOT, then the COPY was moved
                // probably by the post-RA scheduler here.
                assert(Iremove->getOperand(0).isReg() &&
                       Iremove->getOperand(0).isDef() &&
                       Iremove->getOperand(0).getReg() ==
                           IMI->getOperand(0).getReg());

                // Checking that it is really safe to remove this COPY
                //    since it is not used by any instruction after it.
                MachineBasicBlock::iterator Icheck = I;
                //
                // We jump over the WHERE* instruction found
                Icheck++;
                LLVM_DEBUG(dbgs() << "  runOnMachineFunction(): Icheck = "
                                  << *Icheck << "\n");
                // Iterating over all remaining instructions of the BB
                for (; Icheck != IE; Icheck++) {
                  LLVM_DEBUG(dbgs() << "  Icheck = " << *Icheck);
                  if (Icheck->getNumOperands() > 0 &&
                      Icheck->getOperand(0).isReg() &&
                      Icheck->getOperand(0).getReg() ==
                          Iremove->getOperand(0).getReg()) {
                    // It normally has to be a def - if it's a use it's bad
                    assert(
                        Icheck->getOperand(0).isDef() &&
                        "PassHandleMisplacedInstr: Found a 'useless' COPY "
                        "that is not useless since it is used after... - "
                        "this is not good --> change ConnexTargetMachine.cpp");
                    break;
                  }
                }

                LLVM_DEBUG(dbgs() << "    Removing useless COPY immediately "
                                     "before the WHERE block.\n");

                MBB.remove(Iremove);
              }
            }

            MachineBasicBlock::iterator I2 = I; // + 1;
            // We jump over the WHERE* instruction found
            I2++;
            LLVM_DEBUG(dbgs()
                       << "  runOnMachineFunction(): *I2 = " << *I2 << "\n");

            // continue;

            handleMisplacedInstrs(MBB, TII, I, IMI,
                                  I2, // Misplaced Instr
                                  IE, changedMF,
                                  destRegisterPredicateOfSplitWhere);

            LLVM_DEBUG(dbgs() << "PassHandleMisplacedInstr: after WHERE "
                                 "block processed: MBB = ";
                       MBB.dump());
            LLVM_DEBUG(dbgs() << "PassHandleMisplacedInstr: *IMI = " << *IMI);
          } // End if WHERE*
        }   // End if (DontTreatCopyInstructions == false)
      }     // End for (MachineBasicBlock::iterator I

    } // End for (auto &MBB : MF)

    LLVM_DEBUG(dbgs() << "  runOnMachineFunction(): changedMF = " << changedMF
                      << "\n");

    return changedMF; // indicates if we changed MF
  }                   // end runOnMachineFunction(MachineFunction &MF)

private:
  MachineRegisterInfo *MRI;

  static char ID;
}; // namespace
char PassHandleMisplacedInstr::ID = 0;

} // End namespace

// Important: We don't use bundles, since we avoid using the post-RA scheduler

namespace llvm {
FunctionPass *createPassHandleMisplacedInstr() {
  return new PassHandleMisplacedInstr();
}
} // namespace llvm

namespace {

// Connex Code Generator Pass Configuration Options.
class ConnexPassConfig : public TargetPassConfig {
public:
  // Inspired from lib/Target/BPF/BPFTargetMachine.cpp (from Oct 2025)
  ConnexPassConfig(ConnexTargetMachine &TM, PassManagerBase &PM)
      // Inspired from lib/Target/BPF/BPFTargetMachine.cpp (from Oct 2025)
      : TargetPassConfig(TM, PM) {}

  ConnexTargetMachine &getConnexTargetMachine() const {
    return getTM<ConnexTargetMachine>();
  }

  // Important: Not executing these methods following defined in the class
  //   results in error:
  //     <<llc: target does not support generation of this file type!>>

  // bool addInstSelector() override;
  // Install an instruction selector pass using
  // the ISelDag to gen Connex code; also register extra passes.
  bool /* ConnexPassConfig:: */ addInstSelector() {
    // The registered pass is run immediately after the 1st List
    //   scheduling, after the ISel pass registered above.
    //   The reason it is NOT directly after the ISel pass is that it seems
    //     that the 1st scheduling
    //     pass is considered to be linked together with ISel.
    addPass(createConnexISelDag(getConnexTargetMachine()));

    return false;
  }

  // From http://llvm.org/docs/doxygen/html/classllvm_1_1TargetPassConfig.html
  //   This method may be implemented by targets that want to run passes
  //       immediately before register allocation.
  void addPreRegAlloc() {}

  void addPostRegAlloc() {}

  // From http://llvm.org/doxygen/classllvm_1_1TargetPassConfig.html:
  //  <<This pass may be implemented by targets that want to run passes
  //      immediately before machine code is emitted.>>
  void addPreEmitPass() {
    LLVM_DEBUG(dbgs() << "Entered ConnexPassConfig::addPreEmitPass().\n");

    addPass(createPassHandleMisplacedInstr());

    // Here we add a stand-alone hazard recognizer pass
    // Very Important: the post-RA hazard recognizer is called iff
    //   we give:
    //     llc -post-RA-scheduler ...
    addPass(&PostRAHazardRecognizerID);
  }
}; // End class ConnexPassConfig

} // end namespace

TargetPassConfig *ConnexTargetMachine::createPassConfig(PassManagerBase &PM) {
  // Inspired from lib/Target/BPF/BPFTargetMachine.cpp (from Oct 2025)
  return new ConnexPassConfig(*this, PM);
}

// Inspired from BPF/BPFTargetMachine.cpp (from Oct 2025)
TargetTransformInfo
ConnexTargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(std::make_unique<ConnexTTIImpl>(this, F));
}
