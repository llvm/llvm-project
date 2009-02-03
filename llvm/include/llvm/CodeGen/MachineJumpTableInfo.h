//===-- CodeGen/MachineJumpTableInfo.h - Abstract Jump Tables  --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The MachineJumpTableInfo class keeps track of jump tables referenced by
// lowered switch instructions in the MachineFunction.
//
// Instructions reference the address of these jump tables through the use of 
// MO_JumpTableIndex values.  When emitting assembly or machine code, these 
// virtual address references are converted to refer to the address of the 
// function jump tables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEJUMPTABLEINFO_H
#define LLVM_CODEGEN_MACHINEJUMPTABLEINFO_H

#include <vector>
#include <iosfwd>
#include <cassert>

namespace llvm {

class MachineBasicBlock;
class TargetData;

/// MachineJumpTableEntry - One jump table in the jump table info.
///
struct MachineJumpTableEntry {
  /// MBBs - The vector of basic blocks from which to create the jump table.
  std::vector<MachineBasicBlock*> MBBs;
  
  explicit MachineJumpTableEntry(const std::vector<MachineBasicBlock*> &M)
  : MBBs(M) {}
};
  
class MachineJumpTableInfo {
  unsigned EntrySize;
  unsigned Alignment;
  std::vector<MachineJumpTableEntry> JumpTables;
public:
  MachineJumpTableInfo(unsigned Size, unsigned Align)
  : EntrySize(Size), Alignment(Align) {}
    
  /// getJumpTableIndex - Create a new jump table or return an existing one.
  ///
  unsigned getJumpTableIndex(const std::vector<MachineBasicBlock*> &DestBBs);
  
  /// isEmpty - Return true if there are no jump tables.
  ///
  bool isEmpty() const { return JumpTables.empty(); }

  const std::vector<MachineJumpTableEntry> &getJumpTables() const {
    return JumpTables;
  }
  
  /// RemoveJumpTable - Mark the specific index as being dead.  This will cause
  /// it to not be emitted.
  void RemoveJumpTable(unsigned Idx) {
    JumpTables[Idx].MBBs.clear();
  }
  
  /// ReplaceMBBInJumpTables - If Old is the target of any jump tables, update
  /// the jump tables to branch to New instead.
  bool ReplaceMBBInJumpTables(MachineBasicBlock *Old, MachineBasicBlock *New) {
    assert(Old != New && "Not making a change?");
    bool MadeChange = false;
    for (size_t i = 0, e = JumpTables.size(); i != e; ++i) {
      MachineJumpTableEntry &JTE = JumpTables[i];
      for (size_t j = 0, e = JTE.MBBs.size(); j != e; ++j)
        if (JTE.MBBs[j] == Old) {
          JTE.MBBs[j] = New;
          MadeChange = true;
        }
    }
    return MadeChange;
  }
  
  /// getEntrySize - Returns the size of an individual field in a jump table. 
  ///
  unsigned getEntrySize() const { return EntrySize; }
  
  /// getAlignment - returns the target's preferred alignment for jump tables
  unsigned getAlignment() const { return Alignment; }
  
  /// print - Used by the MachineFunction printer to print information about
  /// jump tables.  Implemented in MachineFunction.cpp
  ///
  void print(std::ostream &OS) const;
  void print(std::ostream *OS) const { if (OS) print(*OS); }

  /// dump - Call print(std::cerr) to be called from the debugger.
  ///
  void dump() const;
};

} // End llvm namespace

#endif
