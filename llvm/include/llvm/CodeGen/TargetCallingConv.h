//===-- llvm/CodeGen/TargetCallingConv.h - Calling Convention ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines types for working with calling-convention information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_TARGETCALLINGCONV_H
#define LLVM_CODEGEN_TARGETCALLINGCONV_H

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGenTypes/MachineValueType.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <climits>
#include <cstdint>

namespace llvm {
namespace ISD {

  struct ArgFlagsTy {
  public:
    /// Flag bits describing an argument.
    enum Flags : uint32_t {
      NoFlags = 0,
      ZExt = 1U << 0,     ///< Zero extended
      SExt = 1U << 1,     ///< Sign extended
      NoExt = 1U << 2,    ///< No extension
      InReg = 1U << 3,    ///< Passed in register
      SRet = 1U << 4,     ///< Hidden struct-ret ptr
      ByVal = 1U << 5,    ///< Struct passed by value
      ByRef = 1U << 6,    ///< Passed in memory
      Nest = 1U << 7,     ///< Nested fn static chain
      Returned = 1U << 8, ///< Always returned
      Split = 1U << 9,
      InAlloca = 1U << 10,      ///< Passed with inalloca
      Preallocated = 1U << 11,  ///< ByVal without the copy
      SplitEnd = 1U << 12,      ///< Last part of a split
      SwiftSelf = 1U << 13,     ///< Swift self parameter
      SwiftAsync = 1U << 14,    ///< Swift async context parameter
      SwiftError = 1U << 15,    ///< Swift error parameter
      CFGuardTarget = 1U << 16, ///< Control Flow Guard target
      Hva = 1U << 17,           ///< HVA field
      HvaStart = 1U << 18,      ///< HVA structure start
      SecArgPass = 1U << 19,    ///< Second argument
      InConsecutiveRegsLast = 1U << 20,
      InConsecutiveRegs = 1U << 21,
      CopyElisionCandidate = 1U << 22, ///< Argument copy elision candidate
      Pointer = 1U << 23,
      /// Whether this is part of a variable argument list (non-fixed).
      VarArg = 1U << 24,

      LLVM_MARK_AS_BITMASK_ENUM(/* LargestFlag = */ VarArg)
    };

  private:
    Flags FlagVals = NoFlags;
    unsigned MemAlign : 6;  ///< Log 2 of alignment when arg is passed in memory
                            ///< (including byval/byref). The max alignment is
                            ///< verified in IR verification.
    unsigned OrigAlign : 5; ///< Log 2 of original alignment

    unsigned ByValOrByRefSize = 0; ///< Byval or byref struct size

    unsigned PointerAddrSpace = 0; ///< Address space of pointer argument

    void setFlag(Flags Flag, bool Value = true) {
      FlagVals = (FlagVals & ~Flag) | (Value ? Flag : NoFlags);
    }

  public:
    ArgFlagsTy() : MemAlign(0), OrigAlign(0) {
      static_assert(sizeof(*this) == 4 * sizeof(unsigned), "flags are too big");
    }

    /// Return the argument's boolean flags.
    Flags getFlags() const { return FlagVals; }

    bool isZExt() const { return FlagVals & ZExt; }
    void setZExt() { setFlag(ZExt); }

    bool isSExt() const { return FlagVals & SExt; }
    void setSExt() { setFlag(SExt); }

    bool isNoExt() const { return FlagVals & NoExt; }
    void setNoExt() { setFlag(NoExt); }

    bool isInReg() const { return FlagVals & InReg; }
    void setInReg() { setFlag(InReg); }

    bool isSRet() const { return FlagVals & SRet; }
    void setSRet() { setFlag(SRet); }

    bool isByVal() const { return FlagVals & ByVal; }
    void setByVal() { setFlag(ByVal); }

    bool isByRef() const { return FlagVals & ByRef; }
    void setByRef() { setFlag(ByRef); }

    bool isInAlloca() const { return FlagVals & InAlloca; }
    void setInAlloca() { setFlag(InAlloca); }

    bool isPreallocated() const { return FlagVals & Preallocated; }
    void setPreallocated() { setFlag(Preallocated); }

    bool isSwiftSelf() const { return FlagVals & SwiftSelf; }
    void setSwiftSelf() { setFlag(SwiftSelf); }

    bool isSwiftAsync() const { return FlagVals & SwiftAsync; }
    void setSwiftAsync() { setFlag(SwiftAsync); }

    bool isSwiftError() const { return FlagVals & SwiftError; }
    void setSwiftError() { setFlag(SwiftError); }

    bool isCFGuardTarget() const { return FlagVals & CFGuardTarget; }
    void setCFGuardTarget() { setFlag(CFGuardTarget); }

    bool isHva() const { return FlagVals & Hva; }
    void setHva() { setFlag(Hva); }

    bool isHvaStart() const { return FlagVals & HvaStart; }
    void setHvaStart() { setFlag(HvaStart); }

    bool isSecArgPass() const { return FlagVals & SecArgPass; }
    void setSecArgPass() { setFlag(SecArgPass); }

    bool isNest() const { return FlagVals & Nest; }
    void setNest() { setFlag(Nest); }

    bool isReturned() const { return FlagVals & Returned; }
    void setReturned(bool V = true) { setFlag(Returned, V); }

    bool isInConsecutiveRegs() const { return FlagVals & InConsecutiveRegs; }
    void setInConsecutiveRegs(bool Flag = true) {
      setFlag(InConsecutiveRegs, Flag);
    }

    bool isInConsecutiveRegsLast() const {
      return FlagVals & InConsecutiveRegsLast;
    }
    void setInConsecutiveRegsLast(bool Flag = true) {
      setFlag(InConsecutiveRegsLast, Flag);
    }

    bool isSplit() const { return FlagVals & Split; }
    void setSplit() { setFlag(Split); }

    bool isSplitEnd() const { return FlagVals & SplitEnd; }
    void setSplitEnd() { setFlag(SplitEnd); }

    bool isCopyElisionCandidate() const {
      return FlagVals & CopyElisionCandidate;
    }
    void setCopyElisionCandidate() { setFlag(CopyElisionCandidate); }

    bool isPointer() const { return FlagVals & Pointer; }
    void setPointer() { setFlag(Pointer); }

    bool isVarArg() const { return FlagVals & VarArg; }
    void setVarArg() { setFlag(VarArg); }

    Align getNonZeroMemAlign() const {
      return decodeMaybeAlign(MemAlign).valueOrOne();
    }

    void setMemAlign(Align A) {
      MemAlign = encode(A);
      assert(getNonZeroMemAlign() == A && "bitfield overflow");
    }

    Align getNonZeroByValAlign() const {
      assert(isByVal());
      MaybeAlign A = decodeMaybeAlign(MemAlign);
      assert(A && "ByValAlign must be defined");
      return *A;
    }

    Align getNonZeroOrigAlign() const {
      return decodeMaybeAlign(OrigAlign).valueOrOne();
    }

    void setOrigAlign(Align A) {
      OrigAlign = encode(A);
      assert(getNonZeroOrigAlign() == A && "bitfield overflow");
    }

    unsigned getByValSize() const {
      assert(isByVal() && !isByRef());
      return ByValOrByRefSize;
    }
    void setByValSize(unsigned S) {
      assert(isByVal() && !isByRef());
      ByValOrByRefSize = S;
    }

    unsigned getByRefSize() const {
      assert(!isByVal() && isByRef());
      return ByValOrByRefSize;
    }
    void setByRefSize(unsigned S) {
      assert(!isByVal() && isByRef());
      ByValOrByRefSize = S;
    }

    unsigned getPointerAddrSpace() const { return PointerAddrSpace; }
    void setPointerAddrSpace(unsigned AS) { PointerAddrSpace = AS; }
};

  /// InputArg - This struct carries flags and type information about a
  /// single incoming (formal) argument or incoming (from the perspective
  /// of the caller) return value virtual register.
  ///
  struct InputArg {
    ArgFlagsTy Flags;
    /// Legalized type of this argument part.
    MVT VT = MVT::Other;
    /// Usually the non-legalized type of the argument, which is the EVT
    /// corresponding to the OrigTy IR type. However, for post-legalization
    /// libcalls, this will be a legalized type.
    EVT ArgVT;
    /// Original IR type of the argument. For aggregates, this is the type of
    /// an individual aggregate element, not the whole aggregate.
    Type *OrigTy;
    bool Used;

    /// Index original Function's argument.
    unsigned OrigArgIndex;
    /// Sentinel value for implicit machine-level input arguments.
    static const unsigned NoArgIndex = UINT_MAX;

    /// Offset in bytes of current input value relative to the beginning of
    /// original argument. E.g. if argument was splitted into four 32 bit
    /// registers, we got 4 InputArgs with PartOffsets 0, 4, 8 and 12.
    unsigned PartOffset;

    InputArg(ArgFlagsTy Flags, MVT VT, EVT ArgVT, Type *OrigTy, bool Used,
             unsigned OrigArgIndex, unsigned PartOffset)
        : Flags(Flags), VT(VT), ArgVT(ArgVT), OrigTy(OrigTy), Used(Used),
          OrigArgIndex(OrigArgIndex), PartOffset(PartOffset) {}

    bool isOrigArg() const {
      return OrigArgIndex != NoArgIndex;
    }

    unsigned getOrigArgIndex() const {
      assert(OrigArgIndex != NoArgIndex && "Implicit machine-level argument");
      return OrigArgIndex;
    }
  };

  /// OutputArg - This struct carries flags and a value for a
  /// single outgoing (actual) argument or outgoing (from the perspective
  /// of the caller) return value virtual register.
  ///
  struct OutputArg {
    ArgFlagsTy Flags;
    // Legalized type of this argument part.
    MVT VT;
    /// Non-legalized type of the argument. This is the EVT corresponding to
    /// the OrigTy IR type.
    EVT ArgVT;
    /// Original IR type of the argument. For aggregates, this is the type of
    /// an individual aggregate element, not the whole aggregate.
    Type *OrigTy;

    /// Index original Function's argument.
    unsigned OrigArgIndex;

    /// Offset in bytes of current output value relative to the beginning of
    /// original argument. E.g. if argument was splitted into four 32 bit
    /// registers, we got 4 OutputArgs with PartOffsets 0, 4, 8 and 12.
    unsigned PartOffset;

    OutputArg(ArgFlagsTy Flags, MVT VT, EVT ArgVT, Type *OrigTy,
              unsigned OrigArgIndex, unsigned PartOffset)
        : Flags(Flags), VT(VT), ArgVT(ArgVT), OrigTy(OrigTy),
          OrigArgIndex(OrigArgIndex), PartOffset(PartOffset) {}
  };

} // end namespace ISD
} // end namespace llvm

#endif // LLVM_CODEGEN_TARGETCALLINGCONV_H
