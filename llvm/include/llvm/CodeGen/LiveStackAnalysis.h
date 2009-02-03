//===-- LiveStackAnalysis.h - Live Stack Slot Analysis ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the live stack slot analysis pass. It is analogous to
// live interval analysis except it's analyzing liveness of stack slots rather
// than registers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVESTACK_ANALYSIS_H
#define LLVM_CODEGEN_LIVESTACK_ANALYSIS_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/Support/Allocator.h"
#include <map>

namespace llvm {

  class LiveStacks : public MachineFunctionPass {
    /// Special pool allocator for VNInfo's (LiveInterval val#).
    ///
    BumpPtrAllocator VNInfoAllocator;

    /// s2iMap - Stack slot indices to live interval mapping.
    ///
    typedef std::map<int, LiveInterval> SS2IntervalMap;
    SS2IntervalMap s2iMap;

  public:
    static char ID; // Pass identification, replacement for typeid
    LiveStacks() : MachineFunctionPass(&ID) {}

    typedef SS2IntervalMap::iterator iterator;
    typedef SS2IntervalMap::const_iterator const_iterator;
    const_iterator begin() const { return s2iMap.begin(); }
    const_iterator end() const { return s2iMap.end(); }
    iterator begin() { return s2iMap.begin(); }
    iterator end() { return s2iMap.end(); }
    unsigned getNumIntervals() const { return (unsigned)s2iMap.size(); }

    LiveInterval &getOrCreateInterval(int Slot) {
      SS2IntervalMap::iterator I = s2iMap.find(Slot);
      if (I == s2iMap.end())
        I = s2iMap.insert(I,std::make_pair(Slot,LiveInterval(Slot,0.0F,true)));
      return I->second;
    }

    LiveInterval &getInterval(int Slot) {
      SS2IntervalMap::iterator I = s2iMap.find(Slot);
      assert(I != s2iMap.end() && "Interval does not exist for stack slot");
      return I->second;
    }

    const LiveInterval &getInterval(int Slot) const {
      SS2IntervalMap::const_iterator I = s2iMap.find(Slot);
      assert(I != s2iMap.end() && "Interval does not exist for stack slot");
      return I->second;
    }

    bool hasInterval(unsigned Slot) const {
      return s2iMap.count(Slot);
    }

    BumpPtrAllocator& getVNInfoAllocator() { return VNInfoAllocator; }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual void releaseMemory();

    /// runOnMachineFunction - pass entry point
    virtual bool runOnMachineFunction(MachineFunction&);

    /// print - Implement the dump method.
    virtual void print(std::ostream &O, const Module* = 0) const;
    void print(std::ostream *O, const Module* M = 0) const {
      if (O) print(*O, M);
    }
  };
}

#endif /* LLVM_CODEGEN_LIVESTACK_ANALYSIS_H */
