//===- Loads.h - Local load analysis --------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares simple local analyses for load instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOADS_H
#define LLVM_ANALYSIS_LOADS_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {

class DataLayout;
class MDNode;

/// isDereferenceablePointer - Return true if this is always a dereferenceable
/// pointer. If the context instruction is specified perform context-sensitive
/// analysis and return true if the pointer is dereferenceable at the
/// specified instruction.
bool isDereferenceablePointer(const Value *V, const DataLayout &DL,
                              const Instruction *CtxI = nullptr,
                              const DominatorTree *DT = nullptr,
                              const TargetLibraryInfo *TLI = nullptr);

/// Returns true if V is always a dereferenceable pointer with alignment
/// greater or equal than requested. If the context instruction is specified
/// performs context-sensitive analysis and returns true if the pointer is
/// dereferenceable at the specified instruction.
bool isDereferenceableAndAlignedPointer(const Value *V, unsigned Align,
                                        const DataLayout &DL,
                                        const Instruction *CtxI = nullptr,
                                        const DominatorTree *DT = nullptr,
                                        const TargetLibraryInfo *TLI = nullptr);

/// isSafeToLoadUnconditionally - Return true if we know that executing a load
/// from this value cannot trap.
///
/// If DT and ScanFrom are specified this method performs context-sensitive
/// analysis and returns true if it is safe to load immediately before ScanFrom.
///
/// If it is not obviously safe to load from the specified pointer, we do a
/// quick local scan of the basic block containing ScanFrom, to determine if
/// the address is already accessed.
bool isSafeToLoadUnconditionally(Value *V, unsigned Align,
                                 const DataLayout &DL,
                                 Instruction *ScanFrom = nullptr,
                                 const DominatorTree *DT = nullptr,
                                 const TargetLibraryInfo *TLI = nullptr);

/// DefMaxInstsToScan - the default number of maximum instructions
/// to scan in the block, used by FindAvailableLoadedValue().
extern cl::opt<unsigned> DefMaxInstsToScan;

/// Scan the ScanBB block backwards checking to see if we have the value at
/// the memory address \p Ptr of type \p AccessTy locally available within a
/// small number of instructions. If the value is available, return it.
///
/// If not, return the iterator for the last validated instruction that the
/// value would be live through.  If we scanned the entire block and didn't
/// find something that invalidates *Ptr or provides it, ScanFrom would be
/// left at begin() and this returns null.  ScanFrom could also be left
///
/// MaxInstsToScan specifies the maximum instructions to scan in the block.
/// If it is set to 0, it will scan the whole block. You can also optionally
/// specify an alias analysis implementation, which makes this more precise.
///
/// If AATags is non-null and a load or store is found, the AA tags from the
/// load or store are recorded there.  If there are no AA tags or if no access
/// is found, it is left unmodified.
///
/// IsAtomicMemOp specifies the atomicity of the memory operation that accesses
/// \p *Ptr. We verify atomicity constraints are satisfied when value forwarding
/// from another memory operation that has value \p *Ptr available.
///
/// Note that we assume the \p *Ptr is accessed through a non-volatile but
/// potentially atomic load. Any other constraints should be verified at the
/// caller.
Value *FindAvailableLoadedValue(Value *Ptr, Type *AccessTy, bool IsAtomicMemOp,
                                BasicBlock *ScanBB,
                                BasicBlock::iterator &ScanFrom,
                                unsigned MaxInstsToScan,
                                AliasAnalysis *AA = nullptr,
                                AAMDNodes *AATags = nullptr,
                                bool *IsLoadCSE = nullptr);

/// FindAvailableLoadedValue - Scan the ScanBB block backwards (starting at
/// the instruction before ScanFrom) checking to see if we have the value at
/// the memory address *Ptr locally available within a small number of
///  instructions. If the value is available, return it.
///
/// If not, return the iterator for the last validated instruction that the
/// value would be live through.  If we scanned the entire block and didn't
/// find something that invalidates *Ptr or provides it, ScanFrom would be
/// left at begin() and this returns null.  ScanFrom could also be left
///
/// MaxInstsToScan specifies the maximum instructions to scan in the block.
/// If it is set to 0, it will scan the whole block. You can also optionally
/// specify an alias analysis implementation, which makes this more precise.
///
/// If AATags is non-null and a load or store is found, the AA tags from the
/// load or store are recorded there.  If there are no AA tags or if no access
/// is found, it is left unmodified.
Value *FindAvailableLoadedValue(LoadInst *Load, BasicBlock *ScanBB,
                                BasicBlock::iterator &ScanFrom,
                                unsigned MaxInstsToScan = DefMaxInstsToScan,
                                AliasAnalysis *AA = nullptr,
                                AAMDNodes *AATags = nullptr,
                                bool *IsLoadCSE = nullptr);

}

#endif
