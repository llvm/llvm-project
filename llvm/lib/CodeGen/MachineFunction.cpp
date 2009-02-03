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
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/config.h"
#include <fstream>
#include <sstream>
using namespace llvm;

static AnnotationID MF_AID(
  AnnotationManager::getID("CodeGen::MachineCodeForFunction"));

// Out of line virtual function to home classes.
void MachineFunctionPass::virtfn() {}

namespace {
  struct VISIBILITY_HIDDEN Printer : public MachineFunctionPass {
    static char ID;

    std::ostream *OS;
    const std::string Banner;

    Printer (std::ostream *os, const std::string &banner) 
      : MachineFunctionPass(&ID), OS(os), Banner(banner) {}

    const char *getPassName() const { return "MachineFunction Printer"; }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }

    bool runOnMachineFunction(MachineFunction &MF) {
      (*OS) << Banner;
      MF.print (*OS);
      return false;
    }
  };
  char Printer::ID = 0;
}

/// Returns a newly-created MachineFunction Printer pass. The default output
/// stream is std::cerr; the default banner is empty.
///
FunctionPass *llvm::createMachineFunctionPrinterPass(std::ostream *OS,
                                                     const std::string &Banner){
  return new Printer(OS, Banner);
}

namespace {
  struct VISIBILITY_HIDDEN Deleter : public MachineFunctionPass {
    static char ID;
    Deleter() : MachineFunctionPass(&ID) {}

    const char *getPassName() const { return "Machine Code Deleter"; }

    bool runOnMachineFunction(MachineFunction &MF) {
      // Delete the annotation from the function now.
      MachineFunction::destruct(MF.getFunction());
      return true;
    }
  };
  char Deleter::ID = 0;
}

/// MachineCodeDeletion Pass - This pass deletes all of the machine code for
/// the current function, which should happen after the function has been
/// emitted to a .s file or to memory.
FunctionPass *llvm::createMachineCodeDeleter() {
  return new Deleter();
}



//===---------------------------------------------------------------------===//
// MachineFunction implementation
//===---------------------------------------------------------------------===//

void ilist_traits<MachineBasicBlock>::deleteNode(MachineBasicBlock *MBB) {
  MBB->getParent()->DeleteMachineBasicBlock(MBB);
}

MachineFunction::MachineFunction(const Function *F,
                                 const TargetMachine &TM)
  : Annotation(MF_AID), Fn(F), Target(TM) {
  if (TM.getRegisterInfo())
    RegInfo = new (Allocator.Allocate<MachineRegisterInfo>())
                  MachineRegisterInfo(*TM.getRegisterInfo());
  else
    RegInfo = 0;
  MFInfo = 0;
  FrameInfo = new (Allocator.Allocate<MachineFrameInfo>())
                  MachineFrameInfo(*TM.getFrameInfo());
  ConstantPool = new (Allocator.Allocate<MachineConstantPool>())
                     MachineConstantPool(TM.getTargetData());
  
  // Set up jump table.
  const TargetData &TD = *TM.getTargetData();
  bool IsPic = TM.getRelocationModel() == Reloc::PIC_;
  unsigned EntrySize = IsPic ? 4 : TD.getPointerSize();
  unsigned Alignment = IsPic ? TD.getABITypeAlignment(Type::Int32Ty)
                             : TD.getPointerABIAlignment();
  JumpTableInfo = new (Allocator.Allocate<MachineJumpTableInfo>())
                      MachineJumpTableInfo(EntrySize, Alignment);
}

MachineFunction::~MachineFunction() {
  BasicBlocks.clear();
  InstructionRecycler.clear(Allocator);
  BasicBlockRecycler.clear(Allocator);
  if (RegInfo)
    RegInfo->~MachineRegisterInfo();        Allocator.Deallocate(RegInfo);
  if (MFInfo) {
    MFInfo->~MachineFunctionInfo();       Allocator.Deallocate(MFInfo);
  }
  FrameInfo->~MachineFrameInfo();         Allocator.Deallocate(FrameInfo);
  ConstantPool->~MachineConstantPool();   Allocator.Deallocate(ConstantPool);
  JumpTableInfo->~MachineJumpTableInfo(); Allocator.Deallocate(JumpTableInfo);
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
/// 'Orig' instruction, identical in all ways except the the instruction
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
  // Clear the instructions memoperands. This must be done manually because
  // the instruction's parent pointer is now null, so it can't properly
  // deallocate them on its own.
  MI->clearMemOperands(*this);

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

void MachineFunction::dump() const {
  print(*cerr.stream());
}

void MachineFunction::print(std::ostream &OS) const {
  OS << "# Machine code for " << Fn->getName () << "():\n";

  // Print Frame Information
  FrameInfo->print(*this, OS);
  
  // Print JumpTable Information
  JumpTableInfo->print(OS);

  // Print Constant Pool
  {
    raw_os_ostream OSS(OS);
    ConstantPool->print(OSS);
  }
  
  const TargetRegisterInfo *TRI = getTarget().getRegisterInfo();
  
  if (RegInfo && !RegInfo->livein_empty()) {
    OS << "Live Ins:";
    for (MachineRegisterInfo::livein_iterator
         I = RegInfo->livein_begin(), E = RegInfo->livein_end(); I != E; ++I) {
      if (TRI)
        OS << " " << TRI->getName(I->first);
      else
        OS << " Reg #" << I->first;
      
      if (I->second)
        OS << " in VR#" << I->second << " ";
    }
    OS << "\n";
  }
  if (RegInfo && !RegInfo->liveout_empty()) {
    OS << "Live Outs:";
    for (MachineRegisterInfo::liveout_iterator
         I = RegInfo->liveout_begin(), E = RegInfo->liveout_end(); I != E; ++I)
      if (TRI)
        OS << " " << TRI->getName(*I);
      else
        OS << " Reg #" << *I;
    OS << "\n";
  }
  
  for (const_iterator BB = begin(); BB != end(); ++BB)
    BB->print(OS);

  OS << "\n# End machine code for " << Fn->getName () << "().\n\n";
}

/// CFGOnly flag - This is used to control whether or not the CFG graph printer
/// prints out the contents of basic blocks or not.  This is acceptable because
/// this code is only really used for debugging purposes.
///
static bool CFGOnly = false;

namespace llvm {
  template<>
  struct DOTGraphTraits<const MachineFunction*> : public DefaultDOTGraphTraits {
    static std::string getGraphName(const MachineFunction *F) {
      return "CFG for '" + F->getFunction()->getName() + "' function";
    }

    static std::string getNodeLabel(const MachineBasicBlock *Node,
                                    const MachineFunction *Graph) {
      if (CFGOnly && Node->getBasicBlock() &&
          !Node->getBasicBlock()->getName().empty())
        return Node->getBasicBlock()->getName() + ":";

      std::ostringstream Out;
      if (CFGOnly) {
        Out << Node->getNumber() << ':';
        return Out.str();
      }

      Node->print(Out);

      std::string OutStr = Out.str();
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
  ViewGraph(this, "mf" + getFunction()->getName());
#else
  cerr << "SelectionDAG::viewGraph is only available in debug builds on "
       << "systems with Graphviz or gv!\n";
#endif // NDEBUG
}

void MachineFunction::viewCFGOnly() const
{
  CFGOnly = true;
  viewCFG();
  CFGOnly = false;
}

// The next two methods are used to construct and to retrieve
// the MachineCodeForFunction object for the given function.
// construct() -- Allocates and initializes for a given function and target
// get()       -- Returns a handle to the object.
//                This should not be called before "construct()"
//                for a given Function.
//
MachineFunction&
MachineFunction::construct(const Function *Fn, const TargetMachine &Tar)
{
  assert(Fn->getAnnotation(MF_AID) == 0 &&
         "Object already exists for this function!");
  MachineFunction* mcInfo = new MachineFunction(Fn, Tar);
  Fn->addAnnotation(mcInfo);
  return *mcInfo;
}

void MachineFunction::destruct(const Function *Fn) {
  bool Deleted = Fn->deleteAnnotation(MF_AID);
  assert(Deleted && "Machine code did not exist for function!"); 
  Deleted = Deleted; // silence warning when no assertions.
}

MachineFunction& MachineFunction::get(const Function *F)
{
  MachineFunction *mc = (MachineFunction*)F->getAnnotation(MF_AID);
  assert(mc && "Call construct() method first to allocate the object");
  return *mc;
}

/// getOrCreateDebugLocID - Look up the DebugLocTuple index with the given
/// source file, line, and column. If none currently exists, create add a new
/// new DebugLocTuple and insert it into the DebugIdMap.
unsigned MachineFunction::getOrCreateDebugLocID(unsigned Src, unsigned Line,
                                                unsigned Col) {
  struct DebugLocTuple Tuple(Src, Line, Col);
  DenseMap<DebugLocTuple, unsigned>::iterator II
    = DebugLocInfo.DebugIdMap.find(Tuple);
  if (II != DebugLocInfo.DebugIdMap.end())
    return II->second;
  // Add a new tuple.
  unsigned Id = DebugLocInfo.DebugLocations.size();
  DebugLocInfo.DebugLocations.push_back(Tuple);
  DebugLocInfo.DebugIdMap[Tuple] = Id;
  return Id;
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
                                        bool Immutable) {
  assert(Size != 0 && "Cannot allocate zero size fixed stack objects!");
  Objects.insert(Objects.begin(), StackObject(Size, 1, SPOffset, Immutable));
  return -++NumFixedObjects;
}


void MachineFrameInfo::print(const MachineFunction &MF, std::ostream &OS) const{
  const TargetFrameInfo *FI = MF.getTarget().getFrameInfo();
  int ValOffset = (FI ? FI->getOffsetOfLocalArea() : 0);

  for (unsigned i = 0, e = Objects.size(); i != e; ++i) {
    const StackObject &SO = Objects[i];
    OS << "  <fi#" << (int)(i-NumFixedObjects) << ">: ";
    if (SO.Size == ~0ULL) {
      OS << "dead\n";
      continue;
    }
    if (SO.Size == 0)
      OS << "variable sized";
    else
      OS << "size is " << SO.Size << " byte" << (SO.Size != 1 ? "s," : ",");
    OS << " alignment is " << SO.Alignment << " byte"
       << (SO.Alignment != 1 ? "s," : ",");

    if (i < NumFixedObjects)
      OS << " fixed";
    if (i < NumFixedObjects || SO.SPOffset != -1) {
      int64_t Off = SO.SPOffset - ValOffset;
      OS << " at location [SP";
      if (Off > 0)
        OS << "+" << Off;
      else if (Off < 0)
        OS << Off;
      OS << "]";
    }
    OS << "\n";
  }

  if (HasVarSizedObjects)
    OS << "  Stack frame contains variable sized objects\n";
}

void MachineFrameInfo::dump(const MachineFunction &MF) const {
  print(MF, *cerr.stream());
}


//===----------------------------------------------------------------------===//
//  MachineJumpTableInfo implementation
//===----------------------------------------------------------------------===//

/// getJumpTableIndex - Create a new jump table entry in the jump table info
/// or return an existing one.
///
unsigned MachineJumpTableInfo::getJumpTableIndex(
                               const std::vector<MachineBasicBlock*> &DestBBs) {
  assert(!DestBBs.empty() && "Cannot create an empty jump table!");
  for (unsigned i = 0, e = JumpTables.size(); i != e; ++i)
    if (JumpTables[i].MBBs == DestBBs)
      return i;
  
  JumpTables.push_back(MachineJumpTableEntry(DestBBs));
  return JumpTables.size()-1;
}


void MachineJumpTableInfo::print(std::ostream &OS) const {
  // FIXME: this is lame, maybe we could print out the MBB numbers or something
  // like {1, 2, 4, 5, 3, 0}
  for (unsigned i = 0, e = JumpTables.size(); i != e; ++i) {
    OS << "  <jt#" << i << "> has " << JumpTables[i].MBBs.size() 
       << " entries\n";
  }
}

void MachineJumpTableInfo::dump() const { print(*cerr.stream()); }


//===----------------------------------------------------------------------===//
//  MachineConstantPool implementation
//===----------------------------------------------------------------------===//

const Type *MachineConstantPoolEntry::getType() const {
  if (isMachineConstantPoolEntry())
      return Val.MachineCPVal->getType();
  return Val.ConstVal->getType();
}

MachineConstantPool::~MachineConstantPool() {
  for (unsigned i = 0, e = Constants.size(); i != e; ++i)
    if (Constants[i].isMachineConstantPoolEntry())
      delete Constants[i].Val.MachineCPVal;
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
  unsigned AlignMask = (1 << Alignment)-1;
  for (unsigned i = 0, e = Constants.size(); i != e; ++i)
    if (Constants[i].Val.ConstVal == C && (Constants[i].Offset & AlignMask)== 0)
      return i;
  
  unsigned Offset = 0;
  if (!Constants.empty()) {
    Offset = Constants.back().getOffset();
    Offset += TD->getTypePaddedSize(Constants.back().getType());
    Offset = (Offset+AlignMask)&~AlignMask;
  }
  
  Constants.push_back(MachineConstantPoolEntry(C, Offset));
  return Constants.size()-1;
}

unsigned MachineConstantPool::getConstantPoolIndex(MachineConstantPoolValue *V,
                                                   unsigned Alignment) {
  assert(Alignment && "Alignment must be specified!");
  if (Alignment > PoolAlignment) PoolAlignment = Alignment;
  
  // Check to see if we already have this constant.
  //
  // FIXME, this could be made much more efficient for large constant pools.
  unsigned AlignMask = (1 << Alignment)-1;
  int Idx = V->getExistingMachineCPValue(this, Alignment);
  if (Idx != -1)
    return (unsigned)Idx;
  
  unsigned Offset = 0;
  if (!Constants.empty()) {
    Offset = Constants.back().getOffset();
    Offset += TD->getTypePaddedSize(Constants.back().getType());
    Offset = (Offset+AlignMask)&~AlignMask;
  }
  
  Constants.push_back(MachineConstantPoolEntry(V, Offset));
  return Constants.size()-1;
}

void MachineConstantPool::print(raw_ostream &OS) const {
  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    OS << "  <cp#" << i << "> is";
    if (Constants[i].isMachineConstantPoolEntry())
      Constants[i].Val.MachineCPVal->print(OS);
    else
      OS << *(Value*)Constants[i].Val.ConstVal;
    OS << " , offset=" << Constants[i].getOffset();
    OS << "\n";
  }
}

void MachineConstantPool::dump() const { print(errs()); errs().flush(); }
