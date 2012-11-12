//===-- VectorElementize.cpp - Remove unreachable blocks for codegen --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass converts operations on vector types to operations on their
// element types.
//
// For generic binary and unary vector instructions, the conversion is simple.
// Suppose we have
//        av = bv Vop cv
// where av, bv, and cv are vector virtual registers, and Vop is a vector op.
// This gets converted to the following :
//       a1 = b1 Sop c1
//       a2 = b2 Sop c2
//
// VectorToScalarMap maintains the vector vreg to scalar vreg mapping.
// For the above example, the map will look as follows:
// av => [a1, a2]
// bv => [b1, b2]
//
// In addition, initVectorInfo creates the following opcode->opcode map.
// Vop => Sop
// OtherVop => OtherSop
// ...
//
// For vector specific instructions like vecbuild, vecshuffle etc, the
// conversion is different. Look at comments near the functions with
// prefix createVec<...>.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/Constant.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "NVPTX.h"
#include "NVPTXTargetMachine.h"

using namespace llvm;

namespace {

class LLVM_LIBRARY_VISIBILITY VectorElementize : public MachineFunctionPass {
  virtual bool runOnMachineFunction(MachineFunction &F);

  NVPTXTargetMachine &TM;
  MachineRegisterInfo *MRI;
  const NVPTXRegisterInfo *RegInfo;
  const NVPTXInstrInfo *InstrInfo;

  llvm::DenseMap<const TargetRegisterClass *, const TargetRegisterClass *>
  RegClassMap;
  llvm::DenseMap<unsigned, bool> SimpleMoveMap;

  llvm::DenseMap<unsigned, SmallVector<unsigned, 4> > VectorToScalarMap;

  bool isVectorInstr(MachineInstr *);

  SmallVector<unsigned, 4> getScalarRegisters(unsigned);
  unsigned getScalarVersion(unsigned);
  unsigned getScalarVersion(MachineInstr *);

  bool isVectorRegister(unsigned);
  const TargetRegisterClass *getScalarRegClass(const TargetRegisterClass *RC);
  unsigned numCopiesNeeded(MachineInstr *);

  void createLoadCopy(MachineFunction&, MachineInstr *,
                      std::vector<MachineInstr *>&);
  void createStoreCopy(MachineFunction&, MachineInstr *,
                       std::vector<MachineInstr *>&);

  void createVecDest(MachineFunction&, MachineInstr *,
                     std::vector<MachineInstr *>&);

  void createCopies(MachineFunction&, MachineInstr *,
                    std::vector<MachineInstr *>&);

  unsigned copyProp(MachineFunction&);
  unsigned removeDeadMoves(MachineFunction&);

  void elementize(MachineFunction&);

  bool isSimpleMove(MachineInstr *);

  void createVecShuffle(MachineFunction& F, MachineInstr *Instr,
                        std::vector<MachineInstr *>& copies);

  void createVecExtract(MachineFunction& F, MachineInstr *Instr,
                        std::vector<MachineInstr *>& copies);

  void createVecInsert(MachineFunction& F, MachineInstr *Instr,
                       std::vector<MachineInstr *>& copies);

  void createVecBuild(MachineFunction& F, MachineInstr *Instr,
                      std::vector<MachineInstr *>& copies);

public:

  static char ID; // Pass identification, replacement for typeid
  VectorElementize(NVPTXTargetMachine &tm)
  : MachineFunctionPass(ID), TM(tm) {}

  virtual const char *getPassName() const {
    return "Convert LLVM vector types to their element types";
  }
};

char VectorElementize::ID = 1;
}

static cl::opt<bool>
RemoveRedundantMoves("nvptx-remove-redundant-moves",
       cl::desc("NVPTX: Remove redundant moves introduced by vector lowering"),
                     cl::init(true));

#define VECINST(x) ((((x)->getDesc().TSFlags) & NVPTX::VecInstTypeMask) \
    >> NVPTX::VecInstTypeShift)
#define ISVECINST(x) (VECINST(x) != NVPTX::VecNOP)
#define ISVECLOAD(x)    (VECINST(x) == NVPTX::VecLoad)
#define ISVECSTORE(x)   (VECINST(x) == NVPTX::VecStore)
#define ISVECBUILD(x)   (VECINST(x) == NVPTX::VecBuild)
#define ISVECSHUFFLE(x) (VECINST(x) == NVPTX::VecShuffle)
#define ISVECEXTRACT(x) (VECINST(x) == NVPTX::VecExtract)
#define ISVECINSERT(x)  (VECINST(x) == NVPTX::VecInsert)
#define ISVECDEST(x)     (VECINST(x) == NVPTX::VecDest)

bool VectorElementize::isSimpleMove(MachineInstr *mi) {
  if (mi->isCopy())
    return true;
  unsigned TSFlags = (mi->getDesc().TSFlags & NVPTX::SimpleMoveMask)
        >> NVPTX::SimpleMoveShift;
  return (TSFlags == 1);
}

bool VectorElementize::isVectorInstr(MachineInstr *mi) {
  if ((mi->getOpcode() == NVPTX::PHI) ||
      (mi->getOpcode() == NVPTX::IMPLICIT_DEF) || mi->isCopy()) {
    MachineOperand dest = mi->getOperand(0);
    return isVectorRegister(dest.getReg());
  }
  return ISVECINST(mi);
}

unsigned VectorElementize::getScalarVersion(MachineInstr *mi) {
  return getScalarVersion(mi->getOpcode());
}

///=============================================================================
///Instr is assumed to be a vector instruction. For most vector instructions,
///the size of the destination vector register gives the number of scalar copies
///needed. For VecStore, size of getOperand(1) gives the number of scalar copies
///needed. For VecExtract, the dest is a scalar. So getOperand(1) gives the
///number of scalar copies needed.
///=============================================================================
unsigned VectorElementize::numCopiesNeeded(MachineInstr *Instr) {
  unsigned numDefs=0;
  unsigned def;
  for (unsigned i=0, e=Instr->getNumOperands(); i!=e; ++i) {
    MachineOperand oper = Instr->getOperand(i);

    if (!oper.isReg()) continue;
    if (!oper.isDef()) continue;
    def = i;
    numDefs++;
  }
  assert((numDefs <= 1) && "Only 0 or 1 defs supported");

  if (numDefs == 1) {
    unsigned regnum = Instr->getOperand(def).getReg();
    if (ISVECEXTRACT(Instr))
      regnum = Instr->getOperand(1).getReg();
    return getNVPTXVectorSize(MRI->getRegClass(regnum));
  }
  else if (numDefs == 0) {
    assert(ISVECSTORE(Instr)
           && "Only 0 def instruction supported is vector store");

    unsigned regnum = Instr->getOperand(0).getReg();
    return getNVPTXVectorSize(MRI->getRegClass(regnum));
  }
  return 1;
}

const TargetRegisterClass *VectorElementize::
getScalarRegClass(const TargetRegisterClass *RC) {
  assert(isNVPTXVectorRegClass(RC) &&
         "Not a vector register class");
  return getNVPTXElemClass(RC);
}

bool VectorElementize::isVectorRegister(unsigned reg) {
  const TargetRegisterClass *RC=MRI->getRegClass(reg);
  return isNVPTXVectorRegClass(RC);
}

///=============================================================================
///For every vector register 'v' that is not already in the VectorToScalarMap,
///create n scalar registers of the corresponding element type, where n
///is 2 or 4 (getNVPTXVectorSize) and add it VectorToScalarMap.
///=============================================================================
SmallVector<unsigned, 4> VectorElementize::getScalarRegisters(unsigned regnum) {
  assert(isVectorRegister(regnum) && "Expecting a vector register here");
  // Create the scalar registers and put them in the map, if not already there.
  if (VectorToScalarMap.find(regnum) == VectorToScalarMap.end()) {
    const TargetRegisterClass *vecClass = MRI->getRegClass(regnum);
    const TargetRegisterClass *scalarClass = getScalarRegClass(vecClass);

    SmallVector<unsigned, 4> temp;

    for (unsigned i=0, e=getNVPTXVectorSize(vecClass); i!=e; ++i)
      temp.push_back(MRI->createVirtualRegister(scalarClass));

    VectorToScalarMap[regnum] = temp;
  }
  return VectorToScalarMap[regnum];
}

///=============================================================================
///For a vector load of the form
///va <= ldv2 [addr]
///the following multi output instruction is created :
///[v1, v2] <= LD [addr]
///Look at NVPTXVector.td for the definitions of multi output loads.
///=============================================================================
void VectorElementize::createLoadCopy(MachineFunction& F, MachineInstr *Instr,
                                      std::vector<MachineInstr *>& copies) {
  copies.push_back(F.CloneMachineInstr(Instr));

  MachineInstr *copy=copies[0];
  copy->setDesc(InstrInfo->get(getScalarVersion(copy)));

  // Remove the dest, that should be a vector operand.
  MachineOperand dest = copy->getOperand(0);
  unsigned regnum = dest.getReg();

  SmallVector<unsigned, 4> scalarRegs = getScalarRegisters(regnum);
  copy->RemoveOperand(0);

  std::vector<MachineOperand> otherOperands;
  for (unsigned i=0, e=copy->getNumOperands(); i!=e; ++i)
    otherOperands.push_back(copy->getOperand(i));

  for (unsigned i=0, e=copy->getNumOperands(); i!=e; ++i)
    copy->RemoveOperand(0);

  for (unsigned i=0, e=scalarRegs.size(); i!=e; ++i) {
    copy->addOperand(MachineOperand::CreateReg(scalarRegs[i], true));
  }

  for (unsigned i=0, e=otherOperands.size(); i!=e; ++i)
    copy->addOperand(otherOperands[i]);

}

///=============================================================================
///For a vector store of the form
///stv2 va, [addr]
///the following multi input instruction is created :
///ST v1, v2, [addr]
///Look at NVPTXVector.td for the definitions of multi input stores.
///=============================================================================
void VectorElementize::createStoreCopy(MachineFunction& F, MachineInstr *Instr,
                                       std::vector<MachineInstr *>& copies) {
  copies.push_back(F.CloneMachineInstr(Instr));

  MachineInstr *copy=copies[0];
  copy->setDesc(InstrInfo->get(getScalarVersion(copy)));

  MachineOperand src = copy->getOperand(0);
  unsigned regnum = src.getReg();

  SmallVector<unsigned, 4> scalarRegs = getScalarRegisters(regnum);
  copy->RemoveOperand(0);

  std::vector<MachineOperand> otherOperands;
  for (unsigned i=0, e=copy->getNumOperands(); i!=e; ++i)
    otherOperands.push_back(copy->getOperand(i));

  for (unsigned i=0, e=copy->getNumOperands(); i!=e; ++i)
    copy->RemoveOperand(0);

  for (unsigned i=0, e=scalarRegs.size(); i!=e; ++i)
    copy->addOperand(MachineOperand::CreateReg(scalarRegs[i], false));

  for (unsigned i=0, e=otherOperands.size(); i!=e; ++i)
    copy->addOperand(otherOperands[i]);
}

///=============================================================================
///va <= shufflev2 vb, vc, <i1>, <i2>
///gets converted to 2 moves into a1 and a2. The source of the moves depend on
///i1 and i2. i1, i2 can belong to the set {0, 1, 2, 3} for shufflev2. For
///shufflev4 the set is {0,..7}. For example, if i1=3, i2=0, the move
///instructions will be
///a1 <= c2
///a2 <= b1
///=============================================================================
void VectorElementize::createVecShuffle(MachineFunction& F, MachineInstr *Instr,
                                        std::vector<MachineInstr *>& copies) {
  unsigned numcopies=numCopiesNeeded(Instr);

  unsigned destregnum = Instr->getOperand(0).getReg();
  unsigned src1regnum = Instr->getOperand(1).getReg();
  unsigned src2regnum = Instr->getOperand(2).getReg();

  SmallVector<unsigned, 4> dest = getScalarRegisters(destregnum);
  SmallVector<unsigned, 4> src1 = getScalarRegisters(src1regnum);
  SmallVector<unsigned, 4> src2 = getScalarRegisters(src2regnum);

  DebugLoc DL = Instr->getDebugLoc();

  for (unsigned i=0; i<numcopies; i++) {
    MachineInstr *copy = BuildMI(F, DL,
                              InstrInfo->get(getScalarVersion(Instr)), dest[i]);
    MachineOperand which=Instr->getOperand(3+i);
    assert(which.isImm() && "Shuffle operand not a constant");

    int src=which.getImm();
    int elem=src%numcopies;

    if (which.getImm() < numcopies)
      copy->addOperand(MachineOperand::CreateReg(src1[elem], false));
    else
      copy->addOperand(MachineOperand::CreateReg(src2[elem], false));
    copies.push_back(copy);
  }
}

///=============================================================================
///a <= extractv2 va, <i1>
///gets turned into a simple move to the scalar register a. The source depends
///on i1.
///=============================================================================
void VectorElementize::createVecExtract(MachineFunction& F, MachineInstr *Instr,
                                        std::vector<MachineInstr *>& copies) {
  unsigned srcregnum = Instr->getOperand(1).getReg();

  SmallVector<unsigned, 4> src = getScalarRegisters(srcregnum);

  MachineOperand which = Instr->getOperand(2);
  assert(which.isImm() && "Extract operand not a constant");

  DebugLoc DL = Instr->getDebugLoc();

  MachineInstr *copy = BuildMI(F, DL, InstrInfo->get(getScalarVersion(Instr)),
                               Instr->getOperand(0).getReg());
  copy->addOperand(MachineOperand::CreateReg(src[which.getImm()], false));

  copies.push_back(copy);
}

///=============================================================================
///va <= vecinsertv2 vb, c, <i1>
///This instruction copies all elements of vb to va, except the 'i1'th element.
///The scalar value c becomes the 'i1'th element of va.
///This gets translated to 2 (4 for vecinsertv4) moves.
///=============================================================================
void VectorElementize::createVecInsert(MachineFunction& F, MachineInstr *Instr,
                                       std::vector<MachineInstr *>& copies) {
  unsigned numcopies=numCopiesNeeded(Instr);

  unsigned destregnum = Instr->getOperand(0).getReg();
  unsigned srcregnum = Instr->getOperand(1).getReg();

  SmallVector<unsigned, 4> dest = getScalarRegisters(destregnum);
  SmallVector<unsigned, 4> src = getScalarRegisters(srcregnum);

  MachineOperand which=Instr->getOperand(3);
  assert(which.isImm() && "Insert operand not a constant");
  unsigned int elem=which.getImm();

  DebugLoc DL = Instr->getDebugLoc();

  for (unsigned i=0; i<numcopies; i++) {
    MachineInstr *copy = BuildMI(F, DL,
                              InstrInfo->get(getScalarVersion(Instr)), dest[i]);

    if (i != elem)
      copy->addOperand(MachineOperand::CreateReg(src[i], false));
    else
      copy->addOperand(Instr->getOperand(2));

    copies.push_back(copy);
  }

}

///=============================================================================
///va <= buildv2 b1, b2
///gets translated to
///a1 <= b1
///a2 <= b2
///=============================================================================
void VectorElementize::createVecBuild(MachineFunction& F, MachineInstr *Instr,
                                      std::vector<MachineInstr *>& copies) {
  unsigned numcopies=numCopiesNeeded(Instr);

  unsigned destregnum = Instr->getOperand(0).getReg();

  SmallVector<unsigned, 4> dest = getScalarRegisters(destregnum);

  DebugLoc DL = Instr->getDebugLoc();

  for (unsigned i=0; i<numcopies; i++) {
    MachineInstr *copy = BuildMI(F, DL,
                              InstrInfo->get(getScalarVersion(Instr)), dest[i]);

    copy->addOperand(Instr->getOperand(1+i));

    copies.push_back(copy);
  }

}

///=============================================================================
///For a tex inst of the form
///va <= op [scalar operands]
///the following multi output instruction is created :
///[v1, v2] <= op' [scalar operands]
///=============================================================================
void VectorElementize::createVecDest(MachineFunction& F, MachineInstr *Instr,
                                     std::vector<MachineInstr *>& copies) {
  copies.push_back(F.CloneMachineInstr(Instr));

  MachineInstr *copy=copies[0];
  copy->setDesc(InstrInfo->get(getScalarVersion(copy)));

  // Remove the dest, that should be a vector operand.
  MachineOperand dest = copy->getOperand(0);
  unsigned regnum = dest.getReg();

  SmallVector<unsigned, 4> scalarRegs = getScalarRegisters(regnum);
  copy->RemoveOperand(0);

  std::vector<MachineOperand> otherOperands;
  for (unsigned i=0, e=copy->getNumOperands(); i!=e; ++i)
    otherOperands.push_back(copy->getOperand(i));

  for (unsigned i=0, e=copy->getNumOperands(); i!=e; ++i)
    copy->RemoveOperand(0);

  for (unsigned i=0, e=scalarRegs.size(); i!=e; ++i)
    copy->addOperand(MachineOperand::CreateReg(scalarRegs[i], true));

  for (unsigned i=0, e=otherOperands.size(); i!=e; ++i)
    copy->addOperand(otherOperands[i]);
}

///=============================================================================
///Look at the vector instruction type and dispatch to the createVec<...>
///function that creates the scalar copies.
///=============================================================================
void VectorElementize::createCopies(MachineFunction& F, MachineInstr *Instr,
                                    std::vector<MachineInstr *>& copies) {
  if (ISVECLOAD(Instr)) {
    createLoadCopy(F, Instr, copies);
    return;
  }
  if (ISVECSTORE(Instr)) {
    createStoreCopy(F, Instr, copies);
    return;
  }
  if (ISVECSHUFFLE(Instr)) {
    createVecShuffle(F, Instr, copies);
    return;
  }
  if (ISVECEXTRACT(Instr)) {
    createVecExtract(F, Instr, copies);
    return;
  }
  if (ISVECINSERT(Instr)) {
    createVecInsert(F, Instr, copies);
    return;
  }
  if (ISVECDEST(Instr)) {
    createVecDest(F, Instr, copies);
    return;
  }
  if (ISVECBUILD(Instr)) {
    createVecBuild(F, Instr, copies);
    return;
  }

  unsigned numcopies=numCopiesNeeded(Instr);

  for (unsigned i=0; i<numcopies; ++i)
    copies.push_back(F.CloneMachineInstr(Instr));

  for (unsigned i=0; i<numcopies; ++i) {
    MachineInstr *copy = copies[i];

    std::vector<MachineOperand> allOperands;
    std::vector<bool> isDef;

    for (unsigned j=0, e=copy->getNumOperands(); j!=e; ++j) {
      MachineOperand oper = copy->getOperand(j);
      allOperands.push_back(oper);
      if (oper.isReg())
        isDef.push_back(oper.isDef());
      else
        isDef.push_back(false);
    }

    for (unsigned j=0, e=copy->getNumOperands(); j!=e; ++j)
      copy->RemoveOperand(0);

    copy->setDesc(InstrInfo->get(getScalarVersion(Instr)));

    for (unsigned j=0, e=allOperands.size(); j!=e; ++j) {
      MachineOperand oper=allOperands[j];
      if (oper.isReg()) {
        unsigned regnum = oper.getReg();
        if (isVectorRegister(regnum)) {

          SmallVector<unsigned, 4> scalarRegs = getScalarRegisters(regnum);
          copy->addOperand(MachineOperand::CreateReg(scalarRegs[i], isDef[j]));
        }
        else
          copy->addOperand(oper);
      }
      else
        copy->addOperand(oper);
    }
  }
}

///=============================================================================
///Scan through all basic blocks, looking for vector instructions.
///For each vector instruction I, insert the scalar copies before I, and
///add I into toRemove vector. Finally remove all instructions in toRemove.
///=============================================================================
void VectorElementize::elementize(MachineFunction &F) {
  for (MachineFunction::reverse_iterator BI=F.rbegin(), BE=F.rend();
      BI!=BE; ++BI) {
    MachineBasicBlock *BB = &*BI;

    std::vector<MachineInstr *> copies;
    std::vector<MachineInstr *> toRemove;

    for (MachineBasicBlock::iterator II=BB->begin(), IE=BB->end();
        II!=IE; ++II) {
      MachineInstr *Instr = &*II;

      if (!isVectorInstr(Instr))
        continue;

      copies.clear();
      createCopies(F, Instr, copies);
      for (unsigned i=0, e=copies.size(); i!=e; ++i)
        BB->insert(II, copies[i]);

      assert((copies.size() > 0) && "Problem in createCopies");
      toRemove.push_back(Instr);
    }
    for (unsigned i=0, e=toRemove.size(); i!=e; ++i)
      F.DeleteMachineInstr(toRemove[i]->getParent()->remove(toRemove[i]));
  }
}

///=============================================================================
///a <= b
///...
///...
///x <= op(a, ...)
///gets converted to
///
///x <= op(b, ...)
///The original move is still present. This works on SSA form machine code.
///Note that a <= b should be a simple vreg-to-vreg move instruction.
///TBD : I didn't find a function that can do replaceOperand, so I remove
///all operands and add all of them again, replacing the one while adding.
///=============================================================================
unsigned VectorElementize::copyProp(MachineFunction &F) {
  unsigned numReplacements = 0;

  for (MachineFunction::reverse_iterator BI=F.rbegin(), BE=F.rend(); BI!=BE;
      ++BI) {
    MachineBasicBlock *BB = &*BI;

    for (MachineBasicBlock::iterator II=BB->begin(), IE=BB->end(); II!=IE;
        ++II) {
      MachineInstr *Instr = &*II;

      // Don't do copy propagation on PHI as it will cause unnecessary
      // live range overlap.
      if ((Instr->getOpcode() == TargetOpcode::PHI) ||
          (Instr->getOpcode() == TargetOpcode::DBG_VALUE))
        continue;

      bool needsReplacement = false;

      for (unsigned i=0, e=Instr->getNumOperands(); i!=e; ++i) {
        MachineOperand oper = Instr->getOperand(i);
        if (!oper.isReg()) continue;
        if (oper.isDef()) continue;
        if (!RegInfo->isVirtualRegister(oper.getReg())) continue;

        MachineInstr *defInstr = MRI->getVRegDef(oper.getReg());

        if (!defInstr) continue;

        if (!isSimpleMove(defInstr)) continue;

        MachineOperand defSrc = defInstr->getOperand(1);
        if (!defSrc.isReg()) continue;
        if (!RegInfo->isVirtualRegister(defSrc.getReg())) continue;

        needsReplacement = true;

      }
      if (!needsReplacement) continue;

      numReplacements++;

      std::vector<MachineOperand> operands;

      for (unsigned i=0, e=Instr->getNumOperands(); i!=e; ++i) {
        MachineOperand oper = Instr->getOperand(i);
        bool flag = false;
        do {
          if (!(oper.isReg()))
            break;
          if (oper.isDef())
            break;
          if (!(RegInfo->isVirtualRegister(oper.getReg())))
            break;
          MachineInstr *defInstr = MRI->getVRegDef(oper.getReg());
          if (!(isSimpleMove(defInstr)))
            break;
          MachineOperand defSrc = defInstr->getOperand(1);
          if (!(defSrc.isReg()))
            break;
          if (!(RegInfo->isVirtualRegister(defSrc.getReg())))
            break;
          operands.push_back(defSrc);
          flag = true;
        } while (0);
        if (flag == false)
          operands.push_back(oper);
      }

      for (unsigned i=0, e=Instr->getNumOperands(); i!=e; ++i)
        Instr->RemoveOperand(0);
      for (unsigned i=0, e=operands.size(); i!=e; ++i)
        Instr->addOperand(operands[i]);

    }
  }
  return numReplacements;
}

///=============================================================================
///Look for simple vreg-to-vreg instructions whose use_empty() is true, add
///them to deadMoves vector. Then remove all instructions in deadMoves.
///=============================================================================
unsigned VectorElementize::removeDeadMoves(MachineFunction &F) {
  std::vector<MachineInstr *> deadMoves;
  for (MachineFunction::reverse_iterator BI=F.rbegin(), BE=F.rend(); BI!=BE;
      ++BI) {
    MachineBasicBlock *BB = &*BI;

    for (MachineBasicBlock::iterator II=BB->begin(), IE=BB->end(); II!=IE;
        ++II) {
      MachineInstr *Instr = &*II;

      if (!isSimpleMove(Instr)) continue;

      MachineOperand dest = Instr->getOperand(0);
      assert(dest.isReg() && "dest of move not a register");
      assert(RegInfo->isVirtualRegister(dest.getReg()) &&
             "dest of move not a virtual register");

      if (MRI->use_empty(dest.getReg())) {
        deadMoves.push_back(Instr);
      }
    }
  }

  for (unsigned i=0, e=deadMoves.size(); i!=e; ++i)
    F.DeleteMachineInstr(deadMoves[i]->getParent()->remove(deadMoves[i]));

  return deadMoves.size();
}

///=============================================================================
///Main function for this pass.
///=============================================================================
bool VectorElementize::runOnMachineFunction(MachineFunction &F) {
  MRI = &F.getRegInfo();

  RegInfo = TM.getRegisterInfo();
  InstrInfo = TM.getInstrInfo();

  VectorToScalarMap.clear();

  elementize(F);

  if (RemoveRedundantMoves)
    while (1) {
      if (copyProp(F) == 0) break;
      removeDeadMoves(F);
    }

  return true;
}

FunctionPass *llvm::createVectorElementizePass(NVPTXTargetMachine &tm) {
  return new VectorElementize(tm);
}

unsigned VectorElementize::getScalarVersion(unsigned opcode) {
  if (opcode == NVPTX::PHI)
    return opcode;
  if (opcode == NVPTX::IMPLICIT_DEF)
    return opcode;
  switch(opcode) {
  default: llvm_unreachable("Scalar version not set, fix NVPTXVector.td");
  case TargetOpcode::COPY: return TargetOpcode::COPY;
  case NVPTX::AddCCCV2I32: return NVPTX::ADDCCCi32rr;
  case NVPTX::AddCCCV4I32: return NVPTX::ADDCCCi32rr;
  case NVPTX::AddCCV2I32: return NVPTX::ADDCCi32rr;
  case NVPTX::AddCCV4I32: return NVPTX::ADDCCi32rr;
  case NVPTX::Build_Vector2_f32: return NVPTX::FMOV32rr;
  case NVPTX::Build_Vector2_f64: return NVPTX::FMOV64rr;
  case NVPTX::Build_Vector2_i16: return NVPTX::IMOV16rr;
  case NVPTX::Build_Vector2_i32: return NVPTX::IMOV32rr;
  case NVPTX::Build_Vector2_i64: return NVPTX::IMOV64rr;
  case NVPTX::Build_Vector2_i8: return NVPTX::IMOV8rr;
  case NVPTX::Build_Vector4_f32: return NVPTX::FMOV32rr;
  case NVPTX::Build_Vector4_i16: return NVPTX::IMOV16rr;
  case NVPTX::Build_Vector4_i32: return NVPTX::IMOV32rr;
  case NVPTX::Build_Vector4_i8: return NVPTX::IMOV8rr;
  case NVPTX::CVTv2i16tov2i32: return NVPTX::Zint_extendext16to32;
  case NVPTX::CVTv2i64tov2i32: return NVPTX::TRUNC_64to32;
  case NVPTX::CVTv2i8tov2i32: return NVPTX::Zint_extendext8to32;
  case NVPTX::CVTv4i16tov4i32: return NVPTX::Zint_extendext16to32;
  case NVPTX::CVTv4i8tov4i32: return NVPTX::Zint_extendext8to32;
  case NVPTX::F32MAD_ftzV2: return NVPTX::FMAD32_ftzrrr;
  case NVPTX::F32MADV2: return NVPTX::FMAD32rrr;
  case NVPTX::F32MAD_ftzV4: return NVPTX::FMAD32_ftzrrr;
  case NVPTX::F32MADV4: return NVPTX::FMAD32rrr;
  case NVPTX::F32FMA_ftzV2: return NVPTX::FMA32_ftzrrr;
  case NVPTX::F32FMAV2: return NVPTX::FMA32rrr;
  case NVPTX::F32FMA_ftzV4: return NVPTX::FMA32_ftzrrr;
  case NVPTX::F32FMAV4: return NVPTX::FMA32rrr;
  case NVPTX::F64FMAV2: return NVPTX::FMA64rrr;
  case NVPTX::FVecEQV2F32: return NVPTX::FSetEQf32rr_toi32;
  case NVPTX::FVecEQV2F64: return NVPTX::FSetEQf64rr_toi64;
  case NVPTX::FVecEQV4F32: return NVPTX::FSetEQf32rr_toi32;
  case NVPTX::FVecGEV2F32: return NVPTX::FSetGEf32rr_toi32;
  case NVPTX::FVecGEV2F64: return NVPTX::FSetGEf64rr_toi64;
  case NVPTX::FVecGEV4F32: return NVPTX::FSetGEf32rr_toi32;
  case NVPTX::FVecGTV2F32: return NVPTX::FSetGTf32rr_toi32;
  case NVPTX::FVecGTV2F64: return NVPTX::FSetGTf64rr_toi64;
  case NVPTX::FVecGTV4F32: return NVPTX::FSetGTf32rr_toi32;
  case NVPTX::FVecLEV2F32: return NVPTX::FSetLEf32rr_toi32;
  case NVPTX::FVecLEV2F64: return NVPTX::FSetLEf64rr_toi64;
  case NVPTX::FVecLEV4F32: return NVPTX::FSetLEf32rr_toi32;
  case NVPTX::FVecLTV2F32: return NVPTX::FSetLTf32rr_toi32;
  case NVPTX::FVecLTV2F64: return NVPTX::FSetLTf64rr_toi64;
  case NVPTX::FVecLTV4F32: return NVPTX::FSetLTf32rr_toi32;
  case NVPTX::FVecNANV2F32: return NVPTX::FSetNANf32rr_toi32;
  case NVPTX::FVecNANV2F64: return NVPTX::FSetNANf64rr_toi64;
  case NVPTX::FVecNANV4F32: return NVPTX::FSetNANf32rr_toi32;
  case NVPTX::FVecNEV2F32: return NVPTX::FSetNEf32rr_toi32;
  case NVPTX::FVecNEV2F64: return NVPTX::FSetNEf64rr_toi64;
  case NVPTX::FVecNEV4F32: return NVPTX::FSetNEf32rr_toi32;
  case NVPTX::FVecNUMV2F32: return NVPTX::FSetNUMf32rr_toi32;
  case NVPTX::FVecNUMV2F64: return NVPTX::FSetNUMf64rr_toi64;
  case NVPTX::FVecNUMV4F32: return NVPTX::FSetNUMf32rr_toi32;
  case NVPTX::FVecUEQV2F32: return NVPTX::FSetUEQf32rr_toi32;
  case NVPTX::FVecUEQV2F64: return NVPTX::FSetUEQf64rr_toi64;
  case NVPTX::FVecUEQV4F32: return NVPTX::FSetUEQf32rr_toi32;
  case NVPTX::FVecUGEV2F32: return NVPTX::FSetUGEf32rr_toi32;
  case NVPTX::FVecUGEV2F64: return NVPTX::FSetUGEf64rr_toi64;
  case NVPTX::FVecUGEV4F32: return NVPTX::FSetUGEf32rr_toi32;
  case NVPTX::FVecUGTV2F32: return NVPTX::FSetUGTf32rr_toi32;
  case NVPTX::FVecUGTV2F64: return NVPTX::FSetUGTf64rr_toi64;
  case NVPTX::FVecUGTV4F32: return NVPTX::FSetUGTf32rr_toi32;
  case NVPTX::FVecULEV2F32: return NVPTX::FSetULEf32rr_toi32;
  case NVPTX::FVecULEV2F64: return NVPTX::FSetULEf64rr_toi64;
  case NVPTX::FVecULEV4F32: return NVPTX::FSetULEf32rr_toi32;
  case NVPTX::FVecULTV2F32: return NVPTX::FSetULTf32rr_toi32;
  case NVPTX::FVecULTV2F64: return NVPTX::FSetULTf64rr_toi64;
  case NVPTX::FVecULTV4F32: return NVPTX::FSetULTf32rr_toi32;
  case NVPTX::FVecUNEV2F32: return NVPTX::FSetUNEf32rr_toi32;
  case NVPTX::FVecUNEV2F64: return NVPTX::FSetUNEf64rr_toi64;
  case NVPTX::FVecUNEV4F32: return NVPTX::FSetUNEf32rr_toi32;
  case NVPTX::I16MADV2: return NVPTX::MAD16rrr;
  case NVPTX::I16MADV4: return NVPTX::MAD16rrr;
  case NVPTX::I32MADV2: return NVPTX::MAD32rrr;
  case NVPTX::I32MADV4: return NVPTX::MAD32rrr;
  case NVPTX::I64MADV2: return NVPTX::MAD64rrr;
  case NVPTX::I8MADV2: return NVPTX::MAD8rrr;
  case NVPTX::I8MADV4: return NVPTX::MAD8rrr;
  case NVPTX::ShiftLV2I16: return NVPTX::SHLi16rr;
  case NVPTX::ShiftLV2I32: return NVPTX::SHLi32rr;
  case NVPTX::ShiftLV2I64: return NVPTX::SHLi64rr;
  case NVPTX::ShiftLV2I8: return NVPTX::SHLi8rr;
  case NVPTX::ShiftLV4I16: return NVPTX::SHLi16rr;
  case NVPTX::ShiftLV4I32: return NVPTX::SHLi32rr;
  case NVPTX::ShiftLV4I8: return NVPTX::SHLi8rr;
  case NVPTX::ShiftRAV2I16: return NVPTX::SRAi16rr;
  case NVPTX::ShiftRAV2I32: return NVPTX::SRAi32rr;
  case NVPTX::ShiftRAV2I64: return NVPTX::SRAi64rr;
  case NVPTX::ShiftRAV2I8: return NVPTX::SRAi8rr;
  case NVPTX::ShiftRAV4I16: return NVPTX::SRAi16rr;
  case NVPTX::ShiftRAV4I32: return NVPTX::SRAi32rr;
  case NVPTX::ShiftRAV4I8: return NVPTX::SRAi8rr;
  case NVPTX::ShiftRLV2I16: return NVPTX::SRLi16rr;
  case NVPTX::ShiftRLV2I32: return NVPTX::SRLi32rr;
  case NVPTX::ShiftRLV2I64: return NVPTX::SRLi64rr;
  case NVPTX::ShiftRLV2I8: return NVPTX::SRLi8rr;
  case NVPTX::ShiftRLV4I16: return NVPTX::SRLi16rr;
  case NVPTX::ShiftRLV4I32: return NVPTX::SRLi32rr;
  case NVPTX::ShiftRLV4I8: return NVPTX::SRLi8rr;
  case NVPTX::SubCCCV2I32: return NVPTX::SUBCCCi32rr;
  case NVPTX::SubCCCV4I32: return NVPTX::SUBCCCi32rr;
  case NVPTX::SubCCV2I32: return NVPTX::SUBCCi32rr;
  case NVPTX::SubCCV4I32: return NVPTX::SUBCCi32rr;
  case NVPTX::V2F32Div_prec_ftz: return NVPTX::FDIV32rr_prec_ftz;
  case NVPTX::V2F32Div_prec: return NVPTX::FDIV32rr_prec;
  case NVPTX::V2F32Div_ftz: return NVPTX::FDIV32rr_ftz;
  case NVPTX::V2F32Div: return NVPTX::FDIV32rr;
  case NVPTX::V2F32_Select: return NVPTX::SELECTf32rr;
  case NVPTX::V2F64Div: return NVPTX::FDIV64rr;
  case NVPTX::V2F64_Select: return NVPTX::SELECTf64rr;
  case NVPTX::V2I16_Select: return NVPTX::SELECTi16rr;
  case NVPTX::V2I32_Select: return NVPTX::SELECTi32rr;
  case NVPTX::V2I64_Select: return NVPTX::SELECTi64rr;
  case NVPTX::V2I8_Select: return NVPTX::SELECTi8rr;
  case NVPTX::V2f32Extract: return NVPTX::FMOV32rr;
  case NVPTX::V2f32Insert: return NVPTX::FMOV32rr;
  case NVPTX::V2f32Mov: return NVPTX::FMOV32rr;
  case NVPTX::V2f64Extract: return NVPTX::FMOV64rr;
  case NVPTX::V2f64Insert: return NVPTX::FMOV64rr;
  case NVPTX::V2f64Mov: return NVPTX::FMOV64rr;
  case NVPTX::V2i16Extract: return NVPTX::IMOV16rr;
  case NVPTX::V2i16Insert: return NVPTX::IMOV16rr;
  case NVPTX::V2i16Mov: return NVPTX::IMOV16rr;
  case NVPTX::V2i32Extract: return NVPTX::IMOV32rr;
  case NVPTX::V2i32Insert: return NVPTX::IMOV32rr;
  case NVPTX::V2i32Mov: return NVPTX::IMOV32rr;
  case NVPTX::V2i64Extract: return NVPTX::IMOV64rr;
  case NVPTX::V2i64Insert: return NVPTX::IMOV64rr;
  case NVPTX::V2i64Mov: return NVPTX::IMOV64rr;
  case NVPTX::V2i8Extract: return NVPTX::IMOV8rr;
  case NVPTX::V2i8Insert: return NVPTX::IMOV8rr;
  case NVPTX::V2i8Mov: return NVPTX::IMOV8rr;
  case NVPTX::V4F32Div_prec_ftz: return NVPTX::FDIV32rr_prec_ftz;
  case NVPTX::V4F32Div_prec: return NVPTX::FDIV32rr_prec;
  case NVPTX::V4F32Div_ftz: return NVPTX::FDIV32rr_ftz;
  case NVPTX::V4F32Div: return NVPTX::FDIV32rr;
  case NVPTX::V4F32_Select: return NVPTX::SELECTf32rr;
  case NVPTX::V4I16_Select: return NVPTX::SELECTi16rr;
  case NVPTX::V4I32_Select: return NVPTX::SELECTi32rr;
  case NVPTX::V4I8_Select: return NVPTX::SELECTi8rr;
  case NVPTX::V4f32Extract: return NVPTX::FMOV32rr;
  case NVPTX::V4f32Insert: return NVPTX::FMOV32rr;
  case NVPTX::V4f32Mov: return NVPTX::FMOV32rr;
  case NVPTX::V4i16Extract: return NVPTX::IMOV16rr;
  case NVPTX::V4i16Insert: return NVPTX::IMOV16rr;
  case NVPTX::V4i16Mov: return NVPTX::IMOV16rr;
  case NVPTX::V4i32Extract: return NVPTX::IMOV32rr;
  case NVPTX::V4i32Insert: return NVPTX::IMOV32rr;
  case NVPTX::V4i32Mov: return NVPTX::IMOV32rr;
  case NVPTX::V4i8Extract: return NVPTX::IMOV8rr;
  case NVPTX::V4i8Insert: return NVPTX::IMOV8rr;
  case NVPTX::V4i8Mov: return NVPTX::IMOV8rr;
  case NVPTX::VAddV2I16: return NVPTX::ADDi16rr;
  case NVPTX::VAddV2I32: return NVPTX::ADDi32rr;
  case NVPTX::VAddV2I64: return NVPTX::ADDi64rr;
  case NVPTX::VAddV2I8: return NVPTX::ADDi8rr;
  case NVPTX::VAddV4I16: return NVPTX::ADDi16rr;
  case NVPTX::VAddV4I32: return NVPTX::ADDi32rr;
  case NVPTX::VAddV4I8: return NVPTX::ADDi8rr;
  case NVPTX::VAddfV2F32: return NVPTX::FADDf32rr;
  case NVPTX::VAddfV2F32_ftz: return NVPTX::FADDf32rr_ftz;
  case NVPTX::VAddfV2F64: return NVPTX::FADDf64rr;
  case NVPTX::VAddfV4F32: return NVPTX::FADDf32rr;
  case NVPTX::VAddfV4F32_ftz: return NVPTX::FADDf32rr_ftz;
  case NVPTX::VAndV2I16: return NVPTX::ANDb16rr;
  case NVPTX::VAndV2I32: return NVPTX::ANDb32rr;
  case NVPTX::VAndV2I64: return NVPTX::ANDb64rr;
  case NVPTX::VAndV2I8: return NVPTX::ANDb8rr;
  case NVPTX::VAndV4I16: return NVPTX::ANDb16rr;
  case NVPTX::VAndV4I32: return NVPTX::ANDb32rr;
  case NVPTX::VAndV4I8: return NVPTX::ANDb8rr;
  case NVPTX::VMulfV2F32_ftz: return NVPTX::FMULf32rr_ftz;
  case NVPTX::VMulfV2F32: return NVPTX::FMULf32rr;
  case NVPTX::VMulfV2F64: return NVPTX::FMULf64rr;
  case NVPTX::VMulfV4F32_ftz: return NVPTX::FMULf32rr_ftz;
  case NVPTX::VMulfV4F32: return NVPTX::FMULf32rr;
  case NVPTX::VMultHSV2I16: return NVPTX::MULTHSi16rr;
  case NVPTX::VMultHSV2I32: return NVPTX::MULTHSi32rr;
  case NVPTX::VMultHSV2I64: return NVPTX::MULTHSi64rr;
  case NVPTX::VMultHSV2I8: return NVPTX::MULTHSi8rr;
  case NVPTX::VMultHSV4I16: return NVPTX::MULTHSi16rr;
  case NVPTX::VMultHSV4I32: return NVPTX::MULTHSi32rr;
  case NVPTX::VMultHSV4I8: return NVPTX::MULTHSi8rr;
  case NVPTX::VMultHUV2I16: return NVPTX::MULTHUi16rr;
  case NVPTX::VMultHUV2I32: return NVPTX::MULTHUi32rr;
  case NVPTX::VMultHUV2I64: return NVPTX::MULTHUi64rr;
  case NVPTX::VMultHUV2I8: return NVPTX::MULTHUi8rr;
  case NVPTX::VMultHUV4I16: return NVPTX::MULTHUi16rr;
  case NVPTX::VMultHUV4I32: return NVPTX::MULTHUi32rr;
  case NVPTX::VMultHUV4I8: return NVPTX::MULTHUi8rr;
  case NVPTX::VMultV2I16: return NVPTX::MULTi16rr;
  case NVPTX::VMultV2I32: return NVPTX::MULTi32rr;
  case NVPTX::VMultV2I64: return NVPTX::MULTi64rr;
  case NVPTX::VMultV2I8: return NVPTX::MULTi8rr;
  case NVPTX::VMultV4I16: return NVPTX::MULTi16rr;
  case NVPTX::VMultV4I32: return NVPTX::MULTi32rr;
  case NVPTX::VMultV4I8: return NVPTX::MULTi8rr;
  case NVPTX::VNegV2I16: return NVPTX::INEG16;
  case NVPTX::VNegV2I32: return NVPTX::INEG32;
  case NVPTX::VNegV2I64: return NVPTX::INEG64;
  case NVPTX::VNegV2I8: return NVPTX::INEG8;
  case NVPTX::VNegV4I16: return NVPTX::INEG16;
  case NVPTX::VNegV4I32: return NVPTX::INEG32;
  case NVPTX::VNegV4I8: return NVPTX::INEG8;
  case NVPTX::VNegv2f32: return NVPTX::FNEGf32;
  case NVPTX::VNegv2f32_ftz: return NVPTX::FNEGf32_ftz;
  case NVPTX::VNegv2f64: return NVPTX::FNEGf64;
  case NVPTX::VNegv4f32: return NVPTX::FNEGf32;
  case NVPTX::VNegv4f32_ftz: return NVPTX::FNEGf32_ftz;
  case NVPTX::VNotV2I16: return NVPTX::NOT16;
  case NVPTX::VNotV2I32: return NVPTX::NOT32;
  case NVPTX::VNotV2I64: return NVPTX::NOT64;
  case NVPTX::VNotV2I8: return NVPTX::NOT8;
  case NVPTX::VNotV4I16: return NVPTX::NOT16;
  case NVPTX::VNotV4I32: return NVPTX::NOT32;
  case NVPTX::VNotV4I8: return NVPTX::NOT8;
  case NVPTX::VOrV2I16: return NVPTX::ORb16rr;
  case NVPTX::VOrV2I32: return NVPTX::ORb32rr;
  case NVPTX::VOrV2I64: return NVPTX::ORb64rr;
  case NVPTX::VOrV2I8: return NVPTX::ORb8rr;
  case NVPTX::VOrV4I16: return NVPTX::ORb16rr;
  case NVPTX::VOrV4I32: return NVPTX::ORb32rr;
  case NVPTX::VOrV4I8: return NVPTX::ORb8rr;
  case NVPTX::VSDivV2I16: return NVPTX::SDIVi16rr;
  case NVPTX::VSDivV2I32: return NVPTX::SDIVi32rr;
  case NVPTX::VSDivV2I64: return NVPTX::SDIVi64rr;
  case NVPTX::VSDivV2I8: return NVPTX::SDIVi8rr;
  case NVPTX::VSDivV4I16: return NVPTX::SDIVi16rr;
  case NVPTX::VSDivV4I32: return NVPTX::SDIVi32rr;
  case NVPTX::VSDivV4I8: return NVPTX::SDIVi8rr;
  case NVPTX::VSRemV2I16: return NVPTX::SREMi16rr;
  case NVPTX::VSRemV2I32: return NVPTX::SREMi32rr;
  case NVPTX::VSRemV2I64: return NVPTX::SREMi64rr;
  case NVPTX::VSRemV2I8: return NVPTX::SREMi8rr;
  case NVPTX::VSRemV4I16: return NVPTX::SREMi16rr;
  case NVPTX::VSRemV4I32: return NVPTX::SREMi32rr;
  case NVPTX::VSRemV4I8: return NVPTX::SREMi8rr;
  case NVPTX::VSubV2I16: return NVPTX::SUBi16rr;
  case NVPTX::VSubV2I32: return NVPTX::SUBi32rr;
  case NVPTX::VSubV2I64: return NVPTX::SUBi64rr;
  case NVPTX::VSubV2I8: return NVPTX::SUBi8rr;
  case NVPTX::VSubV4I16: return NVPTX::SUBi16rr;
  case NVPTX::VSubV4I32: return NVPTX::SUBi32rr;
  case NVPTX::VSubV4I8: return NVPTX::SUBi8rr;
  case NVPTX::VSubfV2F32_ftz: return NVPTX::FSUBf32rr_ftz;
  case NVPTX::VSubfV2F32: return NVPTX::FSUBf32rr;
  case NVPTX::VSubfV2F64: return NVPTX::FSUBf64rr;
  case NVPTX::VSubfV4F32_ftz: return NVPTX::FSUBf32rr_ftz;
  case NVPTX::VSubfV4F32: return NVPTX::FSUBf32rr;
  case NVPTX::VUDivV2I16: return NVPTX::UDIVi16rr;
  case NVPTX::VUDivV2I32: return NVPTX::UDIVi32rr;
  case NVPTX::VUDivV2I64: return NVPTX::UDIVi64rr;
  case NVPTX::VUDivV2I8: return NVPTX::UDIVi8rr;
  case NVPTX::VUDivV4I16: return NVPTX::UDIVi16rr;
  case NVPTX::VUDivV4I32: return NVPTX::UDIVi32rr;
  case NVPTX::VUDivV4I8: return NVPTX::UDIVi8rr;
  case NVPTX::VURemV2I16: return NVPTX::UREMi16rr;
  case NVPTX::VURemV2I32: return NVPTX::UREMi32rr;
  case NVPTX::VURemV2I64: return NVPTX::UREMi64rr;
  case NVPTX::VURemV2I8: return NVPTX::UREMi8rr;
  case NVPTX::VURemV4I16: return NVPTX::UREMi16rr;
  case NVPTX::VURemV4I32: return NVPTX::UREMi32rr;
  case NVPTX::VURemV4I8: return NVPTX::UREMi8rr;
  case NVPTX::VXorV2I16: return NVPTX::XORb16rr;
  case NVPTX::VXorV2I32: return NVPTX::XORb32rr;
  case NVPTX::VXorV2I64: return NVPTX::XORb64rr;
  case NVPTX::VXorV2I8: return NVPTX::XORb8rr;
  case NVPTX::VXorV4I16: return NVPTX::XORb16rr;
  case NVPTX::VXorV4I32: return NVPTX::XORb32rr;
  case NVPTX::VXorV4I8: return NVPTX::XORb8rr;
  case NVPTX::VecSEQV2I16: return NVPTX::ISetSEQi16rr_toi16;
  case NVPTX::VecSEQV2I32: return NVPTX::ISetSEQi32rr_toi32;
  case NVPTX::VecSEQV2I64: return NVPTX::ISetSEQi64rr_toi64;
  case NVPTX::VecSEQV2I8: return NVPTX::ISetSEQi8rr_toi8;
  case NVPTX::VecSEQV4I16: return NVPTX::ISetSEQi16rr_toi16;
  case NVPTX::VecSEQV4I32: return NVPTX::ISetSEQi32rr_toi32;
  case NVPTX::VecSEQV4I8: return NVPTX::ISetSEQi8rr_toi8;
  case NVPTX::VecSGEV2I16: return NVPTX::ISetSGEi16rr_toi16;
  case NVPTX::VecSGEV2I32: return NVPTX::ISetSGEi32rr_toi32;
  case NVPTX::VecSGEV2I64: return NVPTX::ISetSGEi64rr_toi64;
  case NVPTX::VecSGEV2I8: return NVPTX::ISetSGEi8rr_toi8;
  case NVPTX::VecSGEV4I16: return NVPTX::ISetSGEi16rr_toi16;
  case NVPTX::VecSGEV4I32: return NVPTX::ISetSGEi32rr_toi32;
  case NVPTX::VecSGEV4I8: return NVPTX::ISetSGEi8rr_toi8;
  case NVPTX::VecSGTV2I16: return NVPTX::ISetSGTi16rr_toi16;
  case NVPTX::VecSGTV2I32: return NVPTX::ISetSGTi32rr_toi32;
  case NVPTX::VecSGTV2I64: return NVPTX::ISetSGTi64rr_toi64;
  case NVPTX::VecSGTV2I8: return NVPTX::ISetSGTi8rr_toi8;
  case NVPTX::VecSGTV4I16: return NVPTX::ISetSGTi16rr_toi16;
  case NVPTX::VecSGTV4I32: return NVPTX::ISetSGTi32rr_toi32;
  case NVPTX::VecSGTV4I8: return NVPTX::ISetSGTi8rr_toi8;
  case NVPTX::VecSLEV2I16: return NVPTX::ISetSLEi16rr_toi16;
  case NVPTX::VecSLEV2I32: return NVPTX::ISetSLEi32rr_toi32;
  case NVPTX::VecSLEV2I64: return NVPTX::ISetSLEi64rr_toi64;
  case NVPTX::VecSLEV2I8: return NVPTX::ISetSLEi8rr_toi8;
  case NVPTX::VecSLEV4I16: return NVPTX::ISetSLEi16rr_toi16;
  case NVPTX::VecSLEV4I32: return NVPTX::ISetSLEi32rr_toi32;
  case NVPTX::VecSLEV4I8: return NVPTX::ISetSLEi8rr_toi8;
  case NVPTX::VecSLTV2I16: return NVPTX::ISetSLTi16rr_toi16;
  case NVPTX::VecSLTV2I32: return NVPTX::ISetSLTi32rr_toi32;
  case NVPTX::VecSLTV2I64: return NVPTX::ISetSLTi64rr_toi64;
  case NVPTX::VecSLTV2I8: return NVPTX::ISetSLTi8rr_toi8;
  case NVPTX::VecSLTV4I16: return NVPTX::ISetSLTi16rr_toi16;
  case NVPTX::VecSLTV4I32: return NVPTX::ISetSLTi32rr_toi32;
  case NVPTX::VecSLTV4I8: return NVPTX::ISetSLTi8rr_toi8;
  case NVPTX::VecSNEV2I16: return NVPTX::ISetSNEi16rr_toi16;
  case NVPTX::VecSNEV2I32: return NVPTX::ISetSNEi32rr_toi32;
  case NVPTX::VecSNEV2I64: return NVPTX::ISetSNEi64rr_toi64;
  case NVPTX::VecSNEV2I8: return NVPTX::ISetSNEi8rr_toi8;
  case NVPTX::VecSNEV4I16: return NVPTX::ISetSNEi16rr_toi16;
  case NVPTX::VecSNEV4I32: return NVPTX::ISetSNEi32rr_toi32;
  case NVPTX::VecSNEV4I8: return NVPTX::ISetSNEi8rr_toi8;
  case NVPTX::VecShuffle_v2f32: return NVPTX::FMOV32rr;
  case NVPTX::VecShuffle_v2f64: return NVPTX::FMOV64rr;
  case NVPTX::VecShuffle_v2i16: return NVPTX::IMOV16rr;
  case NVPTX::VecShuffle_v2i32: return NVPTX::IMOV32rr;
  case NVPTX::VecShuffle_v2i64: return NVPTX::IMOV64rr;
  case NVPTX::VecShuffle_v2i8: return NVPTX::IMOV8rr;
  case NVPTX::VecShuffle_v4f32: return NVPTX::FMOV32rr;
  case NVPTX::VecShuffle_v4i16: return NVPTX::IMOV16rr;
  case NVPTX::VecShuffle_v4i32: return NVPTX::IMOV32rr;
  case NVPTX::VecShuffle_v4i8: return NVPTX::IMOV8rr;
  case NVPTX::VecUEQV2I16: return NVPTX::ISetUEQi16rr_toi16;
  case NVPTX::VecUEQV2I32: return NVPTX::ISetUEQi32rr_toi32;
  case NVPTX::VecUEQV2I64: return NVPTX::ISetUEQi64rr_toi64;
  case NVPTX::VecUEQV2I8: return NVPTX::ISetUEQi8rr_toi8;
  case NVPTX::VecUEQV4I16: return NVPTX::ISetUEQi16rr_toi16;
  case NVPTX::VecUEQV4I32: return NVPTX::ISetUEQi32rr_toi32;
  case NVPTX::VecUEQV4I8: return NVPTX::ISetUEQi8rr_toi8;
  case NVPTX::VecUGEV2I16: return NVPTX::ISetUGEi16rr_toi16;
  case NVPTX::VecUGEV2I32: return NVPTX::ISetUGEi32rr_toi32;
  case NVPTX::VecUGEV2I64: return NVPTX::ISetUGEi64rr_toi64;
  case NVPTX::VecUGEV2I8: return NVPTX::ISetUGEi8rr_toi8;
  case NVPTX::VecUGEV4I16: return NVPTX::ISetUGEi16rr_toi16;
  case NVPTX::VecUGEV4I32: return NVPTX::ISetUGEi32rr_toi32;
  case NVPTX::VecUGEV4I8: return NVPTX::ISetUGEi8rr_toi8;
  case NVPTX::VecUGTV2I16: return NVPTX::ISetUGTi16rr_toi16;
  case NVPTX::VecUGTV2I32: return NVPTX::ISetUGTi32rr_toi32;
  case NVPTX::VecUGTV2I64: return NVPTX::ISetUGTi64rr_toi64;
  case NVPTX::VecUGTV2I8: return NVPTX::ISetUGTi8rr_toi8;
  case NVPTX::VecUGTV4I16: return NVPTX::ISetUGTi16rr_toi16;
  case NVPTX::VecUGTV4I32: return NVPTX::ISetUGTi32rr_toi32;
  case NVPTX::VecUGTV4I8: return NVPTX::ISetUGTi8rr_toi8;
  case NVPTX::VecULEV2I16: return NVPTX::ISetULEi16rr_toi16;
  case NVPTX::VecULEV2I32: return NVPTX::ISetULEi32rr_toi32;
  case NVPTX::VecULEV2I64: return NVPTX::ISetULEi64rr_toi64;
  case NVPTX::VecULEV2I8: return NVPTX::ISetULEi8rr_toi8;
  case NVPTX::VecULEV4I16: return NVPTX::ISetULEi16rr_toi16;
  case NVPTX::VecULEV4I32: return NVPTX::ISetULEi32rr_toi32;
  case NVPTX::VecULEV4I8: return NVPTX::ISetULEi8rr_toi8;
  case NVPTX::VecULTV2I16: return NVPTX::ISetULTi16rr_toi16;
  case NVPTX::VecULTV2I32: return NVPTX::ISetULTi32rr_toi32;
  case NVPTX::VecULTV2I64: return NVPTX::ISetULTi64rr_toi64;
  case NVPTX::VecULTV2I8: return NVPTX::ISetULTi8rr_toi8;
  case NVPTX::VecULTV4I16: return NVPTX::ISetULTi16rr_toi16;
  case NVPTX::VecULTV4I32: return NVPTX::ISetULTi32rr_toi32;
  case NVPTX::VecULTV4I8: return NVPTX::ISetULTi8rr_toi8;
  case NVPTX::VecUNEV2I16: return NVPTX::ISetUNEi16rr_toi16;
  case NVPTX::VecUNEV2I32: return NVPTX::ISetUNEi32rr_toi32;
  case NVPTX::VecUNEV2I64: return NVPTX::ISetUNEi64rr_toi64;
  case NVPTX::VecUNEV2I8: return NVPTX::ISetUNEi8rr_toi8;
  case NVPTX::VecUNEV4I16: return NVPTX::ISetUNEi16rr_toi16;
  case NVPTX::VecUNEV4I32: return NVPTX::ISetUNEi32rr_toi32;
  case NVPTX::VecUNEV4I8: return NVPTX::ISetUNEi8rr_toi8;
  case NVPTX::INT_PTX_LDU_G_v2i8_32: return NVPTX::INT_PTX_LDU_G_v2i8_ELE_32;
  case NVPTX::INT_PTX_LDU_G_v4i8_32: return NVPTX::INT_PTX_LDU_G_v4i8_ELE_32;
  case NVPTX::INT_PTX_LDU_G_v2i16_32: return NVPTX::INT_PTX_LDU_G_v2i16_ELE_32;
  case NVPTX::INT_PTX_LDU_G_v4i16_32: return NVPTX::INT_PTX_LDU_G_v4i16_ELE_32;
  case NVPTX::INT_PTX_LDU_G_v2i32_32: return NVPTX::INT_PTX_LDU_G_v2i32_ELE_32;
  case NVPTX::INT_PTX_LDU_G_v4i32_32: return NVPTX::INT_PTX_LDU_G_v4i32_ELE_32;
  case NVPTX::INT_PTX_LDU_G_v2f32_32: return NVPTX::INT_PTX_LDU_G_v2f32_ELE_32;
  case NVPTX::INT_PTX_LDU_G_v4f32_32: return NVPTX::INT_PTX_LDU_G_v4f32_ELE_32;
  case NVPTX::INT_PTX_LDU_G_v2i64_32: return NVPTX::INT_PTX_LDU_G_v2i64_ELE_32;
  case NVPTX::INT_PTX_LDU_G_v2f64_32: return NVPTX::INT_PTX_LDU_G_v2f64_ELE_32;
  case NVPTX::INT_PTX_LDU_G_v2i8_64: return NVPTX::INT_PTX_LDU_G_v2i8_ELE_64;
  case NVPTX::INT_PTX_LDU_G_v4i8_64: return NVPTX::INT_PTX_LDU_G_v4i8_ELE_64;
  case NVPTX::INT_PTX_LDU_G_v2i16_64: return NVPTX::INT_PTX_LDU_G_v2i16_ELE_64;
  case NVPTX::INT_PTX_LDU_G_v4i16_64: return NVPTX::INT_PTX_LDU_G_v4i16_ELE_64;
  case NVPTX::INT_PTX_LDU_G_v2i32_64: return NVPTX::INT_PTX_LDU_G_v2i32_ELE_64;
  case NVPTX::INT_PTX_LDU_G_v4i32_64: return NVPTX::INT_PTX_LDU_G_v4i32_ELE_64;
  case NVPTX::INT_PTX_LDU_G_v2f32_64: return NVPTX::INT_PTX_LDU_G_v2f32_ELE_64;
  case NVPTX::INT_PTX_LDU_G_v4f32_64: return NVPTX::INT_PTX_LDU_G_v4f32_ELE_64;
  case NVPTX::INT_PTX_LDU_G_v2i64_64: return NVPTX::INT_PTX_LDU_G_v2i64_ELE_64;
  case NVPTX::INT_PTX_LDU_G_v2f64_64: return NVPTX::INT_PTX_LDU_G_v2f64_ELE_64;

  case NVPTX::LoadParamV4I32: return NVPTX::LoadParamScalar4I32;
  case NVPTX::LoadParamV4I16: return NVPTX::LoadParamScalar4I16;
  case NVPTX::LoadParamV4I8: return NVPTX::LoadParamScalar4I8;
  case NVPTX::LoadParamV2I64: return NVPTX::LoadParamScalar2I64;
  case NVPTX::LoadParamV2I32: return NVPTX::LoadParamScalar2I32;
  case NVPTX::LoadParamV2I16: return NVPTX::LoadParamScalar2I16;
  case NVPTX::LoadParamV2I8: return NVPTX::LoadParamScalar2I8;
  case NVPTX::LoadParamV4F32: return NVPTX::LoadParamScalar4F32;
  case NVPTX::LoadParamV2F32: return NVPTX::LoadParamScalar2F32;
  case NVPTX::LoadParamV2F64: return NVPTX::LoadParamScalar2F64;
  case NVPTX::StoreParamV4I32: return NVPTX::StoreParamScalar4I32;
  case NVPTX::StoreParamV4I16: return NVPTX::StoreParamScalar4I16;
  case NVPTX::StoreParamV4I8: return NVPTX::StoreParamScalar4I8;
  case NVPTX::StoreParamV2I64: return NVPTX::StoreParamScalar2I64;
  case NVPTX::StoreParamV2I32: return NVPTX::StoreParamScalar2I32;
  case NVPTX::StoreParamV2I16: return NVPTX::StoreParamScalar2I16;
  case NVPTX::StoreParamV2I8: return NVPTX::StoreParamScalar2I8;
  case NVPTX::StoreParamV4F32: return NVPTX::StoreParamScalar4F32;
  case NVPTX::StoreParamV2F32: return NVPTX::StoreParamScalar2F32;
  case NVPTX::StoreParamV2F64: return NVPTX::StoreParamScalar2F64;
  case NVPTX::StoreRetvalV4I32: return NVPTX::StoreRetvalScalar4I32;
  case NVPTX::StoreRetvalV4I16: return NVPTX::StoreRetvalScalar4I16;
  case NVPTX::StoreRetvalV4I8: return NVPTX::StoreRetvalScalar4I8;
  case NVPTX::StoreRetvalV2I64: return NVPTX::StoreRetvalScalar2I64;
  case NVPTX::StoreRetvalV2I32: return NVPTX::StoreRetvalScalar2I32;
  case NVPTX::StoreRetvalV2I16: return NVPTX::StoreRetvalScalar2I16;
  case NVPTX::StoreRetvalV2I8: return NVPTX::StoreRetvalScalar2I8;
  case NVPTX::StoreRetvalV4F32: return NVPTX::StoreRetvalScalar4F32;
  case NVPTX::StoreRetvalV2F32: return NVPTX::StoreRetvalScalar2F32;
  case NVPTX::StoreRetvalV2F64: return NVPTX::StoreRetvalScalar2F64;
  case NVPTX::VecI32toV4I8: return NVPTX::I32toV4I8;
  case NVPTX::VecI64toV4I16: return NVPTX::I64toV4I16;
  case NVPTX::VecI16toV2I8: return NVPTX::I16toV2I8;
  case NVPTX::VecI32toV2I16: return NVPTX::I32toV2I16;
  case NVPTX::VecI64toV2I32: return NVPTX::I64toV2I32;
  case NVPTX::VecF64toV2F32: return NVPTX::F64toV2F32;

  case NVPTX::LD_v2i8_avar: return NVPTX::LDV_i8_v2_avar;
  case NVPTX::LD_v2i8_areg: return NVPTX::LDV_i8_v2_areg;
  case NVPTX::LD_v2i8_ari:  return NVPTX::LDV_i8_v2_ari;
  case NVPTX::LD_v2i8_asi:  return NVPTX::LDV_i8_v2_asi;
  case NVPTX::LD_v4i8_avar: return NVPTX::LDV_i8_v4_avar;
  case NVPTX::LD_v4i8_areg: return NVPTX::LDV_i8_v4_areg;
  case NVPTX::LD_v4i8_ari:  return NVPTX::LDV_i8_v4_ari;
  case NVPTX::LD_v4i8_asi:  return NVPTX::LDV_i8_v4_asi;

  case NVPTX::LD_v2i16_avar: return NVPTX::LDV_i16_v2_avar;
  case NVPTX::LD_v2i16_areg: return NVPTX::LDV_i16_v2_areg;
  case NVPTX::LD_v2i16_ari:  return NVPTX::LDV_i16_v2_ari;
  case NVPTX::LD_v2i16_asi:  return NVPTX::LDV_i16_v2_asi;
  case NVPTX::LD_v4i16_avar: return NVPTX::LDV_i16_v4_avar;
  case NVPTX::LD_v4i16_areg: return NVPTX::LDV_i16_v4_areg;
  case NVPTX::LD_v4i16_ari:  return NVPTX::LDV_i16_v4_ari;
  case NVPTX::LD_v4i16_asi:  return NVPTX::LDV_i16_v4_asi;

  case NVPTX::LD_v2i32_avar: return NVPTX::LDV_i32_v2_avar;
  case NVPTX::LD_v2i32_areg: return NVPTX::LDV_i32_v2_areg;
  case NVPTX::LD_v2i32_ari:  return NVPTX::LDV_i32_v2_ari;
  case NVPTX::LD_v2i32_asi:  return NVPTX::LDV_i32_v2_asi;
  case NVPTX::LD_v4i32_avar: return NVPTX::LDV_i32_v4_avar;
  case NVPTX::LD_v4i32_areg: return NVPTX::LDV_i32_v4_areg;
  case NVPTX::LD_v4i32_ari:  return NVPTX::LDV_i32_v4_ari;
  case NVPTX::LD_v4i32_asi:  return NVPTX::LDV_i32_v4_asi;

  case NVPTX::LD_v2f32_avar: return NVPTX::LDV_f32_v2_avar;
  case NVPTX::LD_v2f32_areg: return NVPTX::LDV_f32_v2_areg;
  case NVPTX::LD_v2f32_ari:  return NVPTX::LDV_f32_v2_ari;
  case NVPTX::LD_v2f32_asi:  return NVPTX::LDV_f32_v2_asi;
  case NVPTX::LD_v4f32_avar: return NVPTX::LDV_f32_v4_avar;
  case NVPTX::LD_v4f32_areg: return NVPTX::LDV_f32_v4_areg;
  case NVPTX::LD_v4f32_ari:  return NVPTX::LDV_f32_v4_ari;
  case NVPTX::LD_v4f32_asi:  return NVPTX::LDV_f32_v4_asi;

  case NVPTX::LD_v2i64_avar: return NVPTX::LDV_i64_v2_avar;
  case NVPTX::LD_v2i64_areg: return NVPTX::LDV_i64_v2_areg;
  case NVPTX::LD_v2i64_ari:  return NVPTX::LDV_i64_v2_ari;
  case NVPTX::LD_v2i64_asi:  return NVPTX::LDV_i64_v2_asi;
  case NVPTX::LD_v2f64_avar: return NVPTX::LDV_f64_v2_avar;
  case NVPTX::LD_v2f64_areg: return NVPTX::LDV_f64_v2_areg;
  case NVPTX::LD_v2f64_ari:  return NVPTX::LDV_f64_v2_ari;
  case NVPTX::LD_v2f64_asi:  return NVPTX::LDV_f64_v2_asi;

  case NVPTX::ST_v2i8_avar: return NVPTX::STV_i8_v2_avar;
  case NVPTX::ST_v2i8_areg: return NVPTX::STV_i8_v2_areg;
  case NVPTX::ST_v2i8_ari:  return NVPTX::STV_i8_v2_ari;
  case NVPTX::ST_v2i8_asi:  return NVPTX::STV_i8_v2_asi;
  case NVPTX::ST_v4i8_avar: return NVPTX::STV_i8_v4_avar;
  case NVPTX::ST_v4i8_areg: return NVPTX::STV_i8_v4_areg;
  case NVPTX::ST_v4i8_ari:  return NVPTX::STV_i8_v4_ari;
  case NVPTX::ST_v4i8_asi:  return NVPTX::STV_i8_v4_asi;

  case NVPTX::ST_v2i16_avar: return NVPTX::STV_i16_v2_avar;
  case NVPTX::ST_v2i16_areg: return NVPTX::STV_i16_v2_areg;
  case NVPTX::ST_v2i16_ari:  return NVPTX::STV_i16_v2_ari;
  case NVPTX::ST_v2i16_asi:  return NVPTX::STV_i16_v2_asi;
  case NVPTX::ST_v4i16_avar: return NVPTX::STV_i16_v4_avar;
  case NVPTX::ST_v4i16_areg: return NVPTX::STV_i16_v4_areg;
  case NVPTX::ST_v4i16_ari:  return NVPTX::STV_i16_v4_ari;
  case NVPTX::ST_v4i16_asi:  return NVPTX::STV_i16_v4_asi;

  case NVPTX::ST_v2i32_avar: return NVPTX::STV_i32_v2_avar;
  case NVPTX::ST_v2i32_areg: return NVPTX::STV_i32_v2_areg;
  case NVPTX::ST_v2i32_ari:  return NVPTX::STV_i32_v2_ari;
  case NVPTX::ST_v2i32_asi:  return NVPTX::STV_i32_v2_asi;
  case NVPTX::ST_v4i32_avar: return NVPTX::STV_i32_v4_avar;
  case NVPTX::ST_v4i32_areg: return NVPTX::STV_i32_v4_areg;
  case NVPTX::ST_v4i32_ari:  return NVPTX::STV_i32_v4_ari;
  case NVPTX::ST_v4i32_asi:  return NVPTX::STV_i32_v4_asi;

  case NVPTX::ST_v2f32_avar: return NVPTX::STV_f32_v2_avar;
  case NVPTX::ST_v2f32_areg: return NVPTX::STV_f32_v2_areg;
  case NVPTX::ST_v2f32_ari:  return NVPTX::STV_f32_v2_ari;
  case NVPTX::ST_v2f32_asi:  return NVPTX::STV_f32_v2_asi;
  case NVPTX::ST_v4f32_avar: return NVPTX::STV_f32_v4_avar;
  case NVPTX::ST_v4f32_areg: return NVPTX::STV_f32_v4_areg;
  case NVPTX::ST_v4f32_ari:  return NVPTX::STV_f32_v4_ari;
  case NVPTX::ST_v4f32_asi:  return NVPTX::STV_f32_v4_asi;

  case NVPTX::ST_v2i64_avar: return NVPTX::STV_i64_v2_avar;
  case NVPTX::ST_v2i64_areg: return NVPTX::STV_i64_v2_areg;
  case NVPTX::ST_v2i64_ari:  return NVPTX::STV_i64_v2_ari;
  case NVPTX::ST_v2i64_asi:  return NVPTX::STV_i64_v2_asi;
  case NVPTX::ST_v2f64_avar: return NVPTX::STV_f64_v2_avar;
  case NVPTX::ST_v2f64_areg: return NVPTX::STV_f64_v2_areg;
  case NVPTX::ST_v2f64_ari:  return NVPTX::STV_f64_v2_ari;
  case NVPTX::ST_v2f64_asi:  return NVPTX::STV_f64_v2_asi;
  }
  return 0;
}
