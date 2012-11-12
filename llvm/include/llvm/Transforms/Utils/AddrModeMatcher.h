//===- AddrModeMatcher.h - Addressing mode matching facility ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// AddressingModeMatcher - This class exposes a single public method, which is
// used to construct a "maximal munch" of the addressing mode for the target
// specified by TLI for an access to "V" with an access type of AccessTy.  This
// returns the addressing mode that is actually matched by value, but also
// returns the list of instructions involved in that addressing computation in
// AddrModeInsts.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_ADDRMODEMATCHER_H
#define LLVM_TRANSFORMS_UTILS_ADDRMODEMATCHER_H

#include "llvm/AddressingMode.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {

class GlobalValue;
class Instruction;
class Value;
class Type;
class User;
class raw_ostream;

/// ExtAddrMode - This is an extended version of TargetLowering::AddrMode
/// which holds actual Value*'s for register values.
struct ExtAddrMode : public AddrMode {
  Value *BaseReg;
  Value *ScaledReg;
  ExtAddrMode() : BaseReg(0), ScaledReg(0) {}
  void print(raw_ostream &OS) const;
  void dump() const;
  
  bool operator==(const ExtAddrMode& O) const {
    return (BaseReg == O.BaseReg) && (ScaledReg == O.ScaledReg) &&
           (BaseGV == O.BaseGV) && (BaseOffs == O.BaseOffs) &&
           (HasBaseReg == O.HasBaseReg) && (Scale == O.Scale);
  }
};

static inline raw_ostream &operator<<(raw_ostream &OS, const ExtAddrMode &AM) {
  AM.print(OS);
  return OS;
}

class AddressingModeMatcher {
  SmallVectorImpl<Instruction*> &AddrModeInsts;
  const TargetLowering &TLI;

  /// AccessTy/MemoryInst - This is the type for the access (e.g. double) and
  /// the memory instruction that we're computing this address for.
  Type *AccessTy;
  Instruction *MemoryInst;
  
  /// AddrMode - This is the addressing mode that we're building up.  This is
  /// part of the return value of this addressing mode matching stuff.
  ExtAddrMode &AddrMode;
  
  /// IgnoreProfitability - This is set to true when we should not do
  /// profitability checks.  When true, IsProfitableToFoldIntoAddressingMode
  /// always returns true.
  bool IgnoreProfitability;
  
  AddressingModeMatcher(SmallVectorImpl<Instruction*> &AMI,
                        const TargetLowering &T, Type *AT,
                        Instruction *MI, ExtAddrMode &AM)
    : AddrModeInsts(AMI), TLI(T), AccessTy(AT), MemoryInst(MI), AddrMode(AM) {
    IgnoreProfitability = false;
  }
public:
  
  /// Match - Find the maximal addressing mode that a load/store of V can fold,
  /// give an access type of AccessTy.  This returns a list of involved
  /// instructions in AddrModeInsts.
  static ExtAddrMode Match(Value *V, Type *AccessTy,
                           Instruction *MemoryInst,
                           SmallVectorImpl<Instruction*> &AddrModeInsts,
                           const TargetLowering &TLI) {
    ExtAddrMode Result;

    bool Success = 
      AddressingModeMatcher(AddrModeInsts, TLI, AccessTy,
                            MemoryInst, Result).MatchAddr(V, 0);
    (void)Success; assert(Success && "Couldn't select *anything*?");
    return Result;
  }
private:
  bool MatchScaledValue(Value *ScaleReg, int64_t Scale, unsigned Depth);
  bool MatchAddr(Value *V, unsigned Depth);
  bool MatchOperationAddr(User *Operation, unsigned Opcode, unsigned Depth);
  bool IsProfitableToFoldIntoAddressingMode(Instruction *I,
                                            ExtAddrMode &AMBefore,
                                            ExtAddrMode &AMAfter);
  bool ValueAlreadyLiveAtInst(Value *Val, Value *KnownLive1, Value *KnownLive2);
};

} // End llvm namespace

#endif
