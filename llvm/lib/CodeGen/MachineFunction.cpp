//===-- MachineFunction.cpp -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Collect native machine code information for a function.  This allows
// target-specific information about the generated code to be stored with each
// function.
//
//===----------------------------------------------------------------------===//

#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Config/config.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
  struct Printer : public MachineFunctionPass {
    static char ID;

    raw_ostream &OS;
    const std::string Banner;

    Printer(raw_ostream &os, const std::string &banner) 
      : MachineFunctionPass(&ID), OS(os), Banner(banner) {}

    const char *getPassName() const { return "MachineFunction Printer"; }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    bool runOnMachineFunction(MachineFunction &MF) {
      OS << "# " << Banner << ":\n";
      MF.print(OS);
      return false;
    }
  };
  char Printer::ID = 0;
}

/// Returns a newly-created MachineFunction Printer pass. The default banner is
/// empty.
///
FunctionPass *llvm::createMachineFunctionPrinterPass(raw_ostream &OS,
                                                     const std::string &Banner){
  return new Printer(OS, Banner);
}

//===----------------------------------------------------------------------===//
// MachineFunction implementation
//===----------------------------------------------------------------------===//

// Out of line virtual method.
MachineFunctionInfo::~MachineFunctionInfo() {}

void ilist_traits<MachineBasicBlock>::deleteNode(MachineBasicBlock *MBB) {
  MBB->getParent()->DeleteMachineBasicBlock(MBB);
}

MachineFunction::MachineFunction(Function *F, const TargetMachine &TM,
                                 unsigned FunctionNum)
  : Fn(F), Target(TM) {
  if (TM.getRegisterInfo())
    RegInfo = new (Allocator.Allocate<MachineRegisterInfo>())
                  MachineRegisterInfo(*TM.getRegisterInfo());
  else
    RegInfo = 0;
  MFInfo = 0;
  FrameInfo = new (Allocator.Allocate<MachineFrameInfo>())
                  MachineFrameInfo(*TM.getFrameInfo());
  if (Fn->hasFnAttr(Attribute::StackAlignment))
    FrameInfo->setMaxAlignment(Attribute::getStackAlignmentFromAttrs(
        Fn->getAttributes().getFnAttributes()));
  ConstantPool = new (Allocator.Allocate<MachineConstantPool>())
                     MachineConstantPool(TM.getTargetData());
  Alignment = TM.getTargetLowering()->getFunctionAlignment(F);
  FunctionNumber = FunctionNum;
  JumpTableInfo = 0;
}

MachineFunction::~MachineFunction() {
  BasicBlocks.clear();
  InstructionRecycler.clear(Allocator);
  BasicBlockRecycler.clear(Allocator);
  if (RegInfo) {
    RegInfo->~MachineRegisterInfo();
    Allocator.Deallocate(RegInfo);
  }
  if (MFInfo) {
    MFInfo->~MachineFunctionInfo();
    Allocator.Deallocate(MFInfo);
  }
  FrameInfo->~MachineFrameInfo();         Allocator.Deallocate(FrameInfo);
  ConstantPool->~MachineConstantPool();   Allocator.Deallocate(ConstantPool);
  
  if (JumpTableInfo) {
    JumpTableInfo->~MachineJumpTableInfo();
    Allocator.Deallocate(JumpTableInfo);
  }
}

/// getOrCreateJumpTableInfo - Get the JumpTableInfo for this function, if it
/// does already exist, allocate one.
MachineJumpTableInfo *MachineFunction::
getOrCreateJumpTableInfo(unsigned EntryKind) {
  if (JumpTableInfo) return JumpTableInfo;
  
  JumpTableInfo = new (Allocator.Allocate<MachineJumpTableInfo>())
    MachineJumpTableInfo((MachineJumpTableInfo::JTEntryKind)EntryKind);
  return JumpTableInfo;
}

/// RenumberBlocks - This discards all of the MachineBasicBlock numbers and
/// recomputes them.  This guarantees that the MBB numbers are sequential,
/// dense, and match the ordering of the blocks within the function.  If a
/// specific MachineBasicBlock is specified, only that block and those after
/// it are renumbered.
void MachineFunction::RenumberBlocks(MachineBasicBlock *MBB) {
  if (empty()) { MBBNumbering.clear(); return; }
  MachineFunction::iterator MBBI, E = end();
  if (MBB == 0)
    MBBI = begin();
  else
    MBBI = MBB;
  
  // Figure out the block number this should have.
  unsigned BlockNo = 0;
  if (MBBI != begin())
    BlockNo = prior(MBBI)->getNumber()+1;
  
  for (; MBBI != E; ++MBBI, ++BlockNo) {
    if (MBBI->getNumber() != (int)BlockNo) {
      // Remove use of the old number.
      if (MBBI->getNumber() != -1) {
        assert(MBBNumbering[MBBI->getNumber()] == &*MBBI &&
               "MBB number mismatch!");
        MBBNumbering[MBBI->getNumber()] = 0;
      }
      
      // If BlockNo is already taken, set that block's number to -1.
      if (MBBNumbering[BlockNo])
        MBBNumbering[BlockNo]->setNumber(-1);

      MBBNumbering[BlockNo] = MBBI;
      MBBI->setNumber(BlockNo);
    }
  }    

  // Okay, all the blocks are renumbered.  If we have compactified the block
  // numbering, shrink MBBNumbering now.
  assert(BlockNo <= MBBNumbering.size() && "Mismatch!");
  MBBNumbering.resize(BlockNo);
}

/// CreateMachineInstr - Allocate a new MachineInstr. Use this instead
/// of `new MachineInstr'.
///
MachineInstr *
MachineFunction::CreateMachineInstr(const TargetInstrDesc &TID,
                                    DebugLoc DL, bool NoImp) {
  return new (InstructionRecycler.Allocate<MachineInstr>(Allocator))
    MachineInstr(TID, DL, NoImp);
}

/// CloneMachineInstr - Create a new MachineInstr which is a copy of the
/// 'Orig' instruction, identical in all ways except the instruction
/// has no parent, prev, or next.
///
MachineInstr *
MachineFunction::CloneMachineInstr(const MachineInstr *Orig) {
  return new (InstructionRecycler.Allocate<MachineInstr>(Allocator))
             MachineInstr(*this, *Orig);
}

/// DeleteMachineInstr - Delete the given MachineInstr.
///
void
MachineFunction::DeleteMachineInstr(MachineInstr *MI) {
  MI->~MachineInstr();
  InstructionRecycler.Deallocate(Allocator, MI);
}

/// CreateMachineBasicBlock - Allocate a new MachineBasicBlock. Use this
/// instead of `new MachineBasicBlock'.
///
MachineBasicBlock *
MachineFunction::CreateMachineBasicBlock(const BasicBlock *bb) {
  return new (BasicBlockRecycler.Allocate<MachineBasicBlock>(Allocator))
             MachineBasicBlock(*this, bb);
}

/// DeleteMachineBasicBlock - Delete the given MachineBasicBlock.
///
void
MachineFunction::DeleteMachineBasicBlock(MachineBasicBlock *MBB) {
  assert(MBB->getParent() == this && "MBB parent mismatch!");
  MBB->~MachineBasicBlock();
  BasicBlockRecycler.Deallocate(Allocator, MBB);
}

MachineMemOperand *
MachineFunction::getMachineMemOperand(const Value *v, unsigned f,
                                      int64_t o, uint64_t s,
                                      unsigned base_alignment) {
  return new (Allocator.Allocate<MachineMemOperand>())
             MachineMemOperand(v, f, o, s, base_alignment);
}

MachineMemOperand *
MachineFunction::getMachineMemOperand(const MachineMemOperand *MMO,
                                      int64_t Offset, uint64_t Size) {
  return new (Allocator.Allocate<MachineMemOperand>())
             MachineMemOperand(MMO->getValue(), MMO->getFlags(),
                               int64_t(uint64_t(MMO->getOffset()) +
                                       uint64_t(Offset)),
                               Size, MMO->getBaseAlignment());
}

MachineInstr::mmo_iterator
MachineFunction::allocateMemRefsArray(unsigned long Num) {
  return Allocator.Allocate<MachineMemOperand *>(Num);
}

std::pair<MachineInstr::mmo_iterator, MachineInstr::mmo_iterator>
MachineFunction::extractLoadMemRefs(MachineInstr::mmo_iterator Begin,
                                    MachineInstr::mmo_iterator End) {
  // Count the number of load mem refs.
  unsigned Num = 0;
  for (MachineInstr::mmo_iterator I = Begin; I != End; ++I)
    if ((*I)->isLoad())
      ++Num;

  // Allocate a new array and populate it with the load information.
  MachineInstr::mmo_iterator Result = allocateMemRefsArray(Num);
  unsigned Index = 0;
  for (MachineInstr::mmo_iterator I = Begin; I != End; ++I) {
    if ((*I)->isLoad()) {
      if (!(*I)->isStore())
        // Reuse the MMO.
        Result[Index] = *I;
      else {
        // Clone the MMO and unset the store flag.
        MachineMemOperand *JustLoad =
          getMachineMemOperand((*I)->getValue(),
                               (*I)->getFlags() & ~MachineMemOperand::MOStore,
                               (*I)->getOffset(), (*I)->getSize(),
                               (*I)->getBaseAlignment());
        Result[Index] = JustLoad;
      }
      ++Index;
    }
  }
  return std::make_pair(Result, Result + Num);
}

std::pair<MachineInstr::mmo_iterator, MachineInstr::mmo_iterator>
MachineFunction::extractStoreMemRefs(MachineInstr::mmo_iterator Begin,
                                     MachineInstr::mmo_iterator End) {
  // Count the number of load mem refs.
  unsigned Num = 0;
  for (MachineInstr::mmo_iterator I = Begin; I != End; ++I)
    if ((*I)->isStore())
      ++Num;

  // Allocate a new array and populate it with the store information.
  MachineInstr::mmo_iterator Result = allocateMemRefsArray(Num);
  unsigned Index = 0;
  for (MachineInstr::mmo_iterator I = Begin; I != End; ++I) {
    if ((*I)->isStore()) {
      if (!(*I)->isLoad())
        // Reuse the MMO.
        Result[Index] = *I;
      else {
        // Clone the MMO and unset the load flag.
        MachineMemOperand *JustStore =
          getMachineMemOperand((*I)->getValue(),
                               (*I)->getFlags() & ~MachineMemOperand::MOLoad,
                               (*I)->getOffset(), (*I)->getSize(),
                               (*I)->getBaseAlignment());
        Result[Index] = JustStore;
      }
      ++Index;
    }
  }
  return std::make_pair(Result, Result + Num);
}

void MachineFunction::dump() const {
  print(dbgs());
}

void MachineFunction::print(raw_ostream &OS) const {
  OS << "# Machine code for function " << Fn->getName() << ":\n";

  // Print Frame Information
  FrameInfo->print(*this, OS);
  
  // Print JumpTable Information
  if (JumpTableInfo)
    JumpTableInfo->print(OS);

  // Print Constant Pool
  ConstantPool->print(OS);
  
  const TargetRegisterInfo *TRI = getTarget().getRegisterInfo();
  
  if (RegInfo && !RegInfo->livein_empty()) {
    OS << "Function Live Ins: ";
    for (MachineRegisterInfo::livein_iterator
         I = RegInfo->livein_begin(), E = RegInfo->livein_end(); I != E; ++I) {
      if (TRI)
        OS << "%" << TRI->getName(I->first);
      else
        OS << " %physreg" << I->first;
      
      if (I->second)
        OS << " in reg%" << I->second;

      if (llvm::next(I) != E)
        OS << ", ";
    }
    OS << '\n';
  }
  if (RegInfo && !RegInfo->liveout_empty()) {
    OS << "Function Live Outs: ";
    for (MachineRegisterInfo::liveout_iterator
         I = RegInfo->liveout_begin(), E = RegInfo->liveout_end(); I != E; ++I){
      if (TRI)
        OS << '%' << TRI->getName(*I);
      else
        OS << "%physreg" << *I;

      if (llvm::next(I) != E)
        OS << " ";
    }
    OS << '\n';
  }
  
  for (const_iterator BB = begin(), E = end(); BB != E; ++BB) {
    OS << '\n';
    BB->print(OS);
  }

  OS << "\n# End machine code for function " << Fn->getName() << ".\n\n";
}

namespace llvm {
  template<>
  struct DOTGraphTraits<const MachineFunction*> : public DefaultDOTGraphTraits {

  DOTGraphTraits (bool isSimple=false) : DefaultDOTGraphTraits(isSimple) {}

    static std::string getGraphName(const MachineFunction *F) {
      return "CFG for '" + F->getFunction()->getNameStr() + "' function";
    }

    std::string getNodeLabel(const MachineBasicBlock *Node,
                             const MachineFunction *Graph) {
      if (isSimple () && Node->getBasicBlock() &&
          !Node->getBasicBlock()->getName().empty())
        return Node->getBasicBlock()->getNameStr() + ":";

      std::string OutStr;
      {
        raw_string_ostream OSS(OutStr);
        
        if (isSimple())
          OSS << Node->getNumber() << ':';
        else
          Node->print(OSS);
      }

      if (OutStr[0] == '\n') OutStr.erase(OutStr.begin());

      // Process string output to make it nicer...
      for (unsigned i = 0; i != OutStr.length(); ++i)
        if (OutStr[i] == '\n') {                            // Left justify
          OutStr[i] = '\\';
          OutStr.insert(OutStr.begin()+i+1, 'l');
        }
      return OutStr;
    }
  };
}

void MachineFunction::viewCFG() const
{
#ifndef NDEBUG
  ViewGraph(this, "mf" + getFunction()->getNameStr());
#else
  errs() << "SelectionDAG::viewGraph is only available in debug builds on "
         << "systems with Graphviz or gv!\n";
#endif // NDEBUG
}

void MachineFunction::viewCFGOnly() const
{
#ifndef NDEBUG
  ViewGraph(this, "mf" + getFunction()->getNameStr(), true);
#else
  errs() << "SelectionDAG::viewGraph is only available in debug builds on "
         << "systems with Graphviz or gv!\n";
#endif // NDEBUG
}

/// addLiveIn - Add the specified physical register as a live-in value and
/// create a corresponding virtual register for it.
unsigned MachineFunction::addLiveIn(unsigned PReg,
                                    const TargetRegisterClass *RC) {
  assert(RC->contains(PReg) && "Not the correct regclass!");
  unsigned VReg = getRegInfo().createVirtualRegister(RC);
  getRegInfo().addLiveIn(PReg, VReg);
  return VReg;
}

/// getDILocation - Get the DILocation for a given DebugLoc object.
DILocation MachineFunction::getDILocation(DebugLoc DL) const {
  unsigned Idx = DL.getIndex();
  assert(Idx < DebugLocInfo.DebugLocations.size() &&
         "Invalid index into debug locations!");
  return DILocation(DebugLocInfo.DebugLocations[Idx]);
}


/// getJTISymbol - Return the MCSymbol for the specified non-empty jump table.
/// If isLinkerPrivate is specified, an 'l' label is returned, otherwise a
/// normal 'L' label is returned.
MCSymbol *MachineFunction::getJTISymbol(unsigned JTI, MCContext &Ctx, 
                                        bool isLinkerPrivate) const {
  assert(JumpTableInfo && "No jump tables");
  
  assert(JTI < JumpTableInfo->getJumpTables().size() && "Invalid JTI!");
  const MCAsmInfo &MAI = *getTarget().getMCAsmInfo();
  
  const char *Prefix = isLinkerPrivate ? MAI.getLinkerPrivateGlobalPrefix() :
                                         MAI.getPrivateGlobalPrefix();
  SmallString<60> Name;
  raw_svector_ostream(Name)
    << Prefix << "JTI" << getFunctionNumber() << '_' << JTI;
  return Ctx.GetOrCreateSymbol(Name.str());
}


//===----------------------------------------------------------------------===//
//  MachineFrameInfo implementation
//===----------------------------------------------------------------------===//

/// CreateFixedObject - Create a new object at a fixed location on the stack.
/// All fixed objects should be created before other objects are created for
/// efficiency. By default, fixed objects are immutable. This returns an
/// index with a negative value.
///
int MachineFrameInfo::CreateFixedObject(uint64_t Size, int64_t SPOffset,
                                        bool Immutable, bool isSS) {
  assert(Size != 0 && "Cannot allocate zero size fixed stack objects!");
  Objects.insert(Objects.begin(), StackObject(Size, 1, SPOffset, Immutable,
                                              isSS));
  return -++NumFixedObjects;
}


BitVector
MachineFrameInfo::getPristineRegs(const MachineBasicBlock *MBB) const {
  assert(MBB && "MBB must be valid");
  const MachineFunction *MF = MBB->getParent();
  assert(MF && "MBB must be part of a MachineFunction");
  const TargetMachine &TM = MF->getTarget();
  const TargetRegisterInfo *TRI = TM.getRegisterInfo();
  BitVector BV(TRI->getNumRegs());

  // Before CSI is calculated, no registers are considered pristine. They can be
  // freely used and PEI will make sure they are saved.
  if (!isCalleeSavedInfoValid())
    return BV;

  for (const unsigned *CSR = TRI->getCalleeSavedRegs(MF); CSR && *CSR; ++CSR)
    BV.set(*CSR);

  // The entry MBB always has all CSRs pristine.
  if (MBB == &MF->front())
    return BV;

  // On other MBBs the saved CSRs are not pristine.
  const std::vector<CalleeSavedInfo> &CSI = getCalleeSavedInfo();
  for (std::vector<CalleeSavedInfo>::const_iterator I = CSI.begin(),
         E = CSI.end(); I != E; ++I)
    BV.reset(I->getReg());

  return BV;
}


void MachineFrameInfo::print(const MachineFunction &MF, raw_ostream &OS) const{
  if (Objects.empty()) return;

  const TargetFrameInfo *FI = MF.getTarget().getFrameInfo();
  int ValOffset = (FI ? FI->getOffsetOfLocalArea() : 0);

  OS << "Frame Objects:\n";

  for (unsigned i = 0, e = Objects.size(); i != e; ++i) {
    const StackObject &SO = Objects[i];
    OS << "  fi#" << (int)(i-NumFixedObjects) << ": ";
    if (SO.Size == ~0ULL) {
      OS << "dead\n";
      continue;
    }
    if (SO.Size == 0)
      OS << "variable sized";
    else
      OS << "size=" << SO.Size;
    OS << ", align=" << SO.Alignment;

    if (i < NumFixedObjects)
      OS << ", fixed";
    if (i < NumFixedObjects || SO.SPOffset != -1) {
      int64_t Off = SO.SPOffset - ValOffset;
      OS << ", at location [SP";
      if (Off > 0)
        OS << "+" << Off;
      else if (Off < 0)
        OS << Off;
      OS << "]";
    }
    OS << "\n";
  }
}

void MachineFrameInfo::dump(const MachineFunction &MF) const {
  print(MF, dbgs());
}

//===----------------------------------------------------------------------===//
//  MachineJumpTableInfo implementation
//===----------------------------------------------------------------------===//

/// getEntrySize - Return the size of each entry in the jump table.
unsigned MachineJumpTableInfo::getEntrySize(const TargetData &TD) const {
  // The size of a jump table entry is 4 bytes unless the entry is just the
  // address of a block, in which case it is the pointer size.
  switch (getEntryKind()) {
  case MachineJumpTableInfo::EK_BlockAddress:
    return TD.getPointerSize();
  case MachineJumpTableInfo::EK_GPRel32BlockAddress:
  case MachineJumpTableInfo::EK_LabelDifference32:
  case MachineJumpTableInfo::EK_Custom32:
    return 4;
  }
  assert(0 && "Unknown jump table encoding!");
  return ~0;
}

/// getEntryAlignment - Return the alignment of each entry in the jump table.
unsigned MachineJumpTableInfo::getEntryAlignment(const TargetData &TD) const {
  // The alignment of a jump table entry is the alignment of int32 unless the
  // entry is just the address of a block, in which case it is the pointer
  // alignment.
  switch (getEntryKind()) {
  case MachineJumpTableInfo::EK_BlockAddress:
    return TD.getPointerABIAlignment();
  case MachineJumpTableInfo::EK_GPRel32BlockAddress:
  case MachineJumpTableInfo::EK_LabelDifference32:
  case MachineJumpTableInfo::EK_Custom32:
    return TD.getABIIntegerTypeAlignment(32);
  }
  assert(0 && "Unknown jump table encoding!");
  return ~0;
}

/// createJumpTableIndex - Create a new jump table entry in the jump table info.
///
unsigned MachineJumpTableInfo::createJumpTableIndex(
                               const std::vector<MachineBasicBlock*> &DestBBs) {
  assert(!DestBBs.empty() && "Cannot create an empty jump table!");
  JumpTables.push_back(MachineJumpTableEntry(DestBBs));
  return JumpTables.size()-1;
}

/// ReplaceMBBInJumpTables - If Old is the target of any jump tables, update
/// the jump tables to branch to New instead.
bool MachineJumpTableInfo::ReplaceMBBInJumpTables(MachineBasicBlock *Old,
                                                  MachineBasicBlock *New) {
  assert(Old != New && "Not making a change?");
  bool MadeChange = false;
  for (size_t i = 0, e = JumpTables.size(); i != e; ++i)
    ReplaceMBBInJumpTable(i, Old, New);
  return MadeChange;
}

/// ReplaceMBBInJumpTable - If Old is a target of the jump tables, update
/// the jump table to branch to New instead.
bool MachineJumpTableInfo::ReplaceMBBInJumpTable(unsigned Idx,
                                                 MachineBasicBlock *Old,
                                                 MachineBasicBlock *New) {
  assert(Old != New && "Not making a change?");
  bool MadeChange = false;
  MachineJumpTableEntry &JTE = JumpTables[Idx];
  for (size_t j = 0, e = JTE.MBBs.size(); j != e; ++j)
    if (JTE.MBBs[j] == Old) {
      JTE.MBBs[j] = New;
      MadeChange = true;
    }
  return MadeChange;
}

void MachineJumpTableInfo::print(raw_ostream &OS) const {
  if (JumpTables.empty()) return;

  OS << "Jump Tables:\n";

  for (unsigned i = 0, e = JumpTables.size(); i != e; ++i) {
    OS << "  jt#" << i << ": ";
    for (unsigned j = 0, f = JumpTables[i].MBBs.size(); j != f; ++j)
      OS << " BB#" << JumpTables[i].MBBs[j]->getNumber();
  }

  OS << '\n';
}

void MachineJumpTableInfo::dump() const { print(dbgs()); }


//===----------------------------------------------------------------------===//
//  MachineConstantPool implementation
//===----------------------------------------------------------------------===//

const Type *MachineConstantPoolEntry::getType() const {
  if (isMachineConstantPoolEntry())
    return Val.MachineCPVal->getType();
  return Val.ConstVal->getType();
}


unsigned MachineConstantPoolEntry::getRelocationInfo() const {
  if (isMachineConstantPoolEntry())
    return Val.MachineCPVal->getRelocationInfo();
  return Val.ConstVal->getRelocationInfo();
}

MachineConstantPool::~MachineConstantPool() {
  for (unsigned i = 0, e = Constants.size(); i != e; ++i)
    if (Constants[i].isMachineConstantPoolEntry())
      delete Constants[i].Val.MachineCPVal;
}

/// CanShareConstantPoolEntry - Test whether the given two constants
/// can be allocated the same constant pool entry.
static bool CanShareConstantPoolEntry(Constant *A, Constant *B,
                                      const TargetData *TD) {
  // Handle the trivial case quickly.
  if (A == B) return true;

  // If they have the same type but weren't the same constant, quickly
  // reject them.
  if (A->getType() == B->getType()) return false;

  // For now, only support constants with the same size.
  if (TD->getTypeStoreSize(A->getType()) != TD->getTypeStoreSize(B->getType()))
    return false;

  // If a floating-point value and an integer value have the same encoding,
  // they can share a constant-pool entry.
  if (ConstantFP *AFP = dyn_cast<ConstantFP>(A))
    if (ConstantInt *BI = dyn_cast<ConstantInt>(B))
      return AFP->getValueAPF().bitcastToAPInt() == BI->getValue();
  if (ConstantFP *BFP = dyn_cast<ConstantFP>(B))
    if (ConstantInt *AI = dyn_cast<ConstantInt>(A))
      return BFP->getValueAPF().bitcastToAPInt() == AI->getValue();

  // Two vectors can share an entry if each pair of corresponding
  // elements could.
  if (ConstantVector *AV = dyn_cast<ConstantVector>(A))
    if (ConstantVector *BV = dyn_cast<ConstantVector>(B)) {
      if (AV->getType()->getNumElements() != BV->getType()->getNumElements())
        return false;
      for (unsigned i = 0, e = AV->getType()->getNumElements(); i != e; ++i)
        if (!CanShareConstantPoolEntry(AV->getOperand(i),
                                       BV->getOperand(i), TD))
          return false;
      return true;
    }

  // TODO: Handle other cases.

  return false;
}

/// getConstantPoolIndex - Create a new entry in the constant pool or return
/// an existing one.  User must specify the log2 of the minimum required
/// alignment for the object.
///
unsigned MachineConstantPool::getConstantPoolIndex(Constant *C, 
                                                   unsigned Alignment) {
  assert(Alignment && "Alignment must be specified!");
  if (Alignment > PoolAlignment) PoolAlignment = Alignment;

  // Check to see if we already have this constant.
  //
  // FIXME, this could be made much more efficient for large constant pools.
  for (unsigned i = 0, e = Constants.size(); i != e; ++i)
    if (!Constants[i].isMachineConstantPoolEntry() &&
        CanShareConstantPoolEntry(Constants[i].Val.ConstVal, C, TD)) {
      if ((unsigned)Constants[i].getAlignment() < Alignment)
        Constants[i].Alignment = Alignment;
      return i;
    }
  
  Constants.push_back(MachineConstantPoolEntry(C, Alignment));
  return Constants.size()-1;
}

unsigned MachineConstantPool::getConstantPoolIndex(MachineConstantPoolValue *V,
                                                   unsigned Alignment) {
  assert(Alignment && "Alignment must be specified!");
  if (Alignment > PoolAlignment) PoolAlignment = Alignment;
  
  // Check to see if we already have this constant.
  //
  // FIXME, this could be made much more efficient for large constant pools.
  int Idx = V->getExistingMachineCPValue(this, Alignment);
  if (Idx != -1)
    return (unsigned)Idx;

  Constants.push_back(MachineConstantPoolEntry(V, Alignment));
  return Constants.size()-1;
}

void MachineConstantPool::print(raw_ostream &OS) const {
  if (Constants.empty()) return;

  OS << "Constant Pool:\n";
  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    OS << "  cp#" << i << ": ";
    if (Constants[i].isMachineConstantPoolEntry())
      Constants[i].Val.MachineCPVal->print(OS);
    else
      OS << *(Value*)Constants[i].Val.ConstVal;
    OS << ", align=" << Constants[i].getAlignment();
    OS << "\n";
  }
}

void MachineConstantPool::dump() const { print(dbgs()); }
