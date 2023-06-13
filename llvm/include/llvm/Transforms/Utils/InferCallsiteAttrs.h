//===- InferCallsiteAttrs.h - Propagate attributes to callsites -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the InferCallsiteAttrs class.
// This class is used to propagate attributes present in the caller function of
// the callsite to the arguments/return/callsite itself.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_INFERCALLSITEATTRS_H
#define LLVM_TRANSFORMS_UTILS_INFERCALLSITEATTRS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

namespace llvm {
class InferCallsiteAttrs {
  enum : uint8_t { kMaybe = 0, kYes = 1, kNo = 2 };

  // Limit maximum amount of instructions we will check. Everything is O(1) so
  // relatively high value is okay.
  static constexpr unsigned kMaxChecks = UINT_MAX;

  struct FunctionInfos {
    uint8_t LandingOrEHPad : 2;
  };

  struct BasicBlockInfos {
    uint8_t Alloca : 2;
    uint8_t UnkMalloc : 2;

    bool isSet() const { return Alloca != kMaybe && UnkMalloc != kMaybe; }
  };

  struct CallsiteInfos {
    uint16_t StoresBetweenReturn : 2;
    uint16_t LoadsBetweenReturn : 2;
    uint16_t NonDirectTransferBetweenReturn : 2;
    uint16_t CallerReturnBasedOnCallsite : 2;
    uint16_t IsLastInsBeforeReturn : 2;
    uint16_t PrecedingAlloca : 2;
    uint16_t PrecedingLocalMalloc : 2;
  };

  DenseMap<const BasicBlock *, BasicBlockInfos> BBInfos;
  DenseMap<const Function *, FunctionInfos> FunctionInfos;
  const Function *Caller;
  const CallBase *CxtCB;

  CallsiteInfos CurCBInfo;
  struct FunctionInfos CurFnInfo;
  bool PreserveCache;

  // Wrapper for attribute checks that check both the context callsite and
  // actual calling function.
  bool checkCallerHasFnAttr(Attribute::AttrKind Attr) const {
    return (CxtCB && CxtCB->hasFnAttr(Attr)) || Caller->hasFnAttribute(Attr);
  };
  bool checkCallerHasParamAttr(unsigned ArgIdx,
                               Attribute::AttrKind Attr) const {
    return (CxtCB && CxtCB->paramHasAttr(ArgIdx, Attr)) ||
           Caller->getArg(ArgIdx)->hasAttribute(Attr);
  };
  bool checkCallerHasReturnAttr(Attribute::AttrKind Attr) const {
    return (CxtCB && CxtCB->hasRetAttr(Attr)) || Caller->hasRetAttribute(Attr);
  };

  bool checkCallerDoesNotThrow() const {
    return (CxtCB && CxtCB->doesNotThrow()) || Caller->doesNotThrow();
  }
  bool checkCallerDoesNotAccessMemory() const {
    return (CxtCB && CxtCB->doesNotAccessMemory()) ||
           Caller->doesNotAccessMemory();
  };
  bool checkCallerOnlyReadsMemory() const {
    return (CxtCB && CxtCB->onlyReadsMemory()) || Caller->onlyReadsMemory();
  };
  bool checkCallerOnlyWritesMemory() const {
    return (CxtCB && CxtCB->onlyWritesMemory()) || Caller->onlyWritesMemory();
  };
  bool checkCallerOnlyAccessesArgMemory() const {
    return (CxtCB && CxtCB->onlyAccessesArgMemory()) ||
           Caller->onlyAccessesArgMemory();
  };
  bool checkCallerOnlyAccessesInaccessibleMemory() const {
    return (CxtCB && CxtCB->onlyAccessesInaccessibleMemory()) ||
           Caller->onlyAccessesInaccessibleMemory();
  };
  bool checkCallerOnlyAccessesInaccessibleMemOrArgMem() const {
    return (CxtCB && CxtCB->onlyAccessesInaccessibleMemOrArgMem()) ||
           Caller->onlyAccessesInaccessibleMemOrArgMem();
  };

  // Check all instructions between callbase and end of basicblock (if that
  // basic block ends in a return). This will cache the analysis information.
  // Will break early if condition is met based on arguments.
  bool checkBetweenCallsiteAndReturn(const CallBase *CB, bool BailOnStore,
                                     bool BailOnLoad,
                                     bool BailOnNonDirectTransfer,
                                     bool BailOnNotReturned);

  // Check all instruction instructions preceding basic blocked (any instruction
  // that may reach the callsite CB). If conditions are met, can set early
  // return using BailOn* arguments.
  bool checkPrecedingBBIns(const CallBase *CB, bool BailOnAlloca,
                           bool BailOnLocalMalloc);

  // Check all basic blocks for conditions. At the moment only condition is if
  // landing/EH pad so will store result and break immediately if one is found.
  // In the future may be extended to check other conditions.
  bool checkAllBBs(bool BailOnPad);

  // Try to propagate nocapture attribute from caller argument to callsite
  // arguments.
  bool tryPropagateNoCapture(CallBase *CB);

  // Try trivial propagations (one where if true for the caller, are
  // automatically true for the callsite without further analysis).
  bool tryTrivialPropagations(CallBase *CB);

  // Try propagations of return attributes (nonnull, noundef, etc...)
  bool tryReturnPropagations(CallBase *CB);

  // Try propagations of memory access attribute (readnone, readonly, etc...).
  bool tryMemoryPropagations(CallBase *CB);

  // Add attributes to callsite, assumes Caller and CxtCB are setup already
  bool inferCallsiteAttributesImpl(CallBase *CB);

public:
  // Set PreserveCacheBetweenFunctions to keep cached information on
  // functions/basicblocks between calls processFunction.
  InferCallsiteAttrs(bool PreserveCacheBetweenFunctions = false)
      : PreserveCache(PreserveCacheBetweenFunctions) {}

  // Call if either 1) BB instructions have changed which may invalidate some of
  // the prior analysis or 2) all previous work no longer applies (in which case
  // clearing the cache improves performance).
  void resetCache();

  // Add attributes to callsites based on the function is called in (or by
  // setting CxtCallsite the exact callsite of the callsite).
  bool inferCallsiteAttributes(CallBase *CB,
                               const CallBase *CxtCallsite = nullptr);

  // Process all callsites in Function ParentFunc. This is more efficient that
  // calling inferCallsiteAttributes in a loop as it 1) avoids some unnecessary
  // cache lookups and 2) does some analysis while searching for callsites.
  bool processFunction(Function *ParentFunc,
                       const CallBase *ParentCallsite = nullptr);
};
} // namespace llvm
#endif
