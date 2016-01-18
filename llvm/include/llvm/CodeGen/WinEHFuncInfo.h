//===-- llvm/CodeGen/WinEHFuncInfo.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Data structures and associated state for Windows exception handling schemes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_WINEHFUNCINFO_H
#define LLVM_CODEGEN_WINEHFUNCINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TinyPtrVector.h"

namespace llvm {
class AllocaInst;
class BasicBlock;
class Constant;
class Function;
class GlobalVariable;
class InvokeInst;
class IntrinsicInst;
class LandingPadInst;
class MCExpr;
class MCSymbol;
class MachineBasicBlock;
class Value;

// The following structs respresent the .xdata tables for various
// Windows-related EH personalities.

typedef PointerUnion<const BasicBlock *, MachineBasicBlock *> MBBOrBasicBlock;

struct CxxUnwindMapEntry {
  int ToState;
  MBBOrBasicBlock Cleanup;
};

/// Similar to CxxUnwindMapEntry, but supports SEH filters.
struct SEHUnwindMapEntry {
  /// If unwinding continues through this handler, transition to the handler at
  /// this state. This indexes into SEHUnwindMap.
  int ToState = -1;

  bool IsFinally = false;

  /// Holds the filter expression function.
  const Function *Filter = nullptr;

  /// Holds the __except or __finally basic block.
  MBBOrBasicBlock Handler;
};

struct WinEHHandlerType {
  int Adjectives;
  /// The CatchObj starts out life as an LLVM alloca, is turned into a frame
  /// index, and after PEI, becomes a raw offset.
  union {
    const AllocaInst *Alloca;
    int FrameIndex;
  } CatchObj = {};
  GlobalVariable *TypeDescriptor;
  MBBOrBasicBlock Handler;
};

struct WinEHTryBlockMapEntry {
  int TryLow = -1;
  int TryHigh = -1;
  int CatchHigh = -1;
  SmallVector<WinEHHandlerType, 1> HandlerArray;
};

enum class ClrHandlerType { Catch, Finally, Fault, Filter };

struct ClrEHUnwindMapEntry {
  MBBOrBasicBlock Handler;
  uint32_t TypeToken;
  int Parent;
  ClrHandlerType HandlerType;
};

struct WinEHFuncInfo {
  DenseMap<const Instruction *, int> EHPadStateMap;
  DenseMap<const CatchReturnInst *, const BasicBlock *>
      CatchRetSuccessorColorMap;
  DenseMap<MCSymbol *, std::pair<int, MCSymbol *>> InvokeToStateMap;
  SmallVector<CxxUnwindMapEntry, 4> CxxUnwindMap;
  SmallVector<WinEHTryBlockMapEntry, 4> TryBlockMap;
  SmallVector<SEHUnwindMapEntry, 4> SEHUnwindMap;
  SmallVector<ClrEHUnwindMapEntry, 4> ClrEHUnwindMap;
  int UnwindHelpFrameIdx = INT_MAX;

  int getLastStateNumber() const { return CxxUnwindMap.size() - 1; }

  void addIPToStateRange(const BasicBlock *PadBB, MCSymbol *InvokeBegin,
                         MCSymbol *InvokeEnd);

  /// localescape index of the 32-bit EH registration node. Set by
  /// WinEHStatePass and used indirectly by SEH filter functions of the parent.
  int EHRegNodeEscapeIndex = INT_MAX;
  const AllocaInst *EHRegNode = nullptr;
  int EHRegNodeFrameIndex = INT_MAX;
  int EHRegNodeEndOffset = INT_MAX;

  WinEHFuncInfo() {}
};

/// Analyze the IR in ParentFn and it's handlers to build WinEHFuncInfo, which
/// describes the state numbers and tables used by __CxxFrameHandler3. This
/// analysis assumes that WinEHPrepare has already been run.
void calculateWinCXXEHStateNumbers(const Function *ParentFn,
                                   WinEHFuncInfo &FuncInfo);

void calculateSEHStateNumbers(const Function *ParentFn,
                              WinEHFuncInfo &FuncInfo);

void calculateClrEHStateNumbers(const Function *Fn, WinEHFuncInfo &FuncInfo);

void calculateCatchReturnSuccessorColors(const Function *Fn,
                                         WinEHFuncInfo &FuncInfo);
}
#endif // LLVM_CODEGEN_WINEHFUNCINFO_H
