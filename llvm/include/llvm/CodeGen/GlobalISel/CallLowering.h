//===- llvm/CodeGen/GlobalISel/CallLowering.h - Call lowering ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes how to lower LLVM calls to machine code calls.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_CALLLOWERING_H
#define LLVM_CODEGEN_GLOBALISEL_CALLLOWERING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/TargetCallingConv.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MachineValueType.h"
#include <cstdint>
#include <functional>

namespace llvm {

class CCState;
class CallBase;
class DataLayout;
class Function;
class MachineIRBuilder;
class MachineOperand;
struct MachinePointerInfo;
class MachineRegisterInfo;
class TargetLowering;
class Type;
class Value;

class CallLowering {
  const TargetLowering *TLI;

  virtual void anchor();
public:
  struct ArgInfo {
    SmallVector<Register, 4> Regs;
    // If the argument had to be split into multiple parts according to the
    // target calling convention, then this contains the original vregs
    // if the argument was an incoming arg.
    SmallVector<Register, 2> OrigRegs;
    Type *Ty;
    SmallVector<ISD::ArgFlagsTy, 4> Flags;
    bool IsFixed;

    ArgInfo(ArrayRef<Register> Regs, Type *Ty,
            ArrayRef<ISD::ArgFlagsTy> Flags = ArrayRef<ISD::ArgFlagsTy>(),
            bool IsFixed = true)
        : Regs(Regs.begin(), Regs.end()), Ty(Ty),
          Flags(Flags.begin(), Flags.end()), IsFixed(IsFixed) {
      if (!Regs.empty() && Flags.empty())
        this->Flags.push_back(ISD::ArgFlagsTy());
      // FIXME: We should have just one way of saying "no register".
      assert(((Ty->isVoidTy() || Ty->isEmptyTy()) ==
              (Regs.empty() || Regs[0] == 0)) &&
             "only void types should have no register");
    }

    ArgInfo() : Ty(nullptr), IsFixed(false) {}
  };

  struct CallLoweringInfo {
    /// Calling convention to be used for the call.
    CallingConv::ID CallConv = CallingConv::C;

    /// Destination of the call. It should be either a register, globaladdress,
    /// or externalsymbol.
    MachineOperand Callee = MachineOperand::CreateImm(0);

    /// Descriptor for the return type of the function.
    ArgInfo OrigRet;

    /// List of descriptors of the arguments passed to the function.
    SmallVector<ArgInfo, 8> OrigArgs;

    /// Valid if the call has a swifterror inout parameter, and contains the
    /// vreg that the swifterror should be copied into after the call.
    Register SwiftErrorVReg;

    MDNode *KnownCallees = nullptr;

    /// True if the call must be tail call optimized.
    bool IsMustTailCall = false;

    /// True if the call passes all target-independent checks for tail call
    /// optimization.
    bool IsTailCall = false;

    /// True if the call was lowered as a tail call. This is consumed by the
    /// legalizer. This allows the legalizer to lower libcalls as tail calls.
    bool LoweredTailCall = false;

    /// True if the call is to a vararg function.
    bool IsVarArg = false;
  };

  /// Argument handling is mostly uniform between the four places that
  /// make these decisions: function formal arguments, call
  /// instruction args, call instruction returns and function
  /// returns. However, once a decision has been made on where an
  /// argument should go, exactly what happens can vary slightly. This
  /// class abstracts the differences.
  struct ValueHandler {
    ValueHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                 CCAssignFn *AssignFn)
      : MIRBuilder(MIRBuilder), MRI(MRI), AssignFn(AssignFn) {}

    virtual ~ValueHandler() = default;

    /// Returns true if the handler is dealing with incoming arguments,
    /// i.e. those that move values from some physical location to vregs.
    virtual bool isIncomingArgumentHandler() const = 0;

    /// Materialize a VReg containing the address of the specified
    /// stack-based object. This is either based on a FrameIndex or
    /// direct SP manipulation, depending on the context. \p MPO
    /// should be initialized to an appropriate description of the
    /// address created.
    virtual Register getStackAddress(uint64_t Size, int64_t Offset,
                                     MachinePointerInfo &MPO) = 0;

    /// The specified value has been assigned to a physical register,
    /// handle the appropriate COPY (either to or from) and mark any
    /// relevant uses/defines as needed.
    virtual void assignValueToReg(Register ValVReg, Register PhysReg,
                                  CCValAssign &VA) = 0;

    /// The specified value has been assigned to a stack
    /// location. Load or store it there, with appropriate extension
    /// if necessary.
    virtual void assignValueToAddress(Register ValVReg, Register Addr,
                                      uint64_t Size, MachinePointerInfo &MPO,
                                      CCValAssign &VA) = 0;

    /// An overload which takes an ArgInfo if additional information about
    /// the arg is needed.
    virtual void assignValueToAddress(const ArgInfo &Arg, Register Addr,
                                      uint64_t Size, MachinePointerInfo &MPO,
                                      CCValAssign &VA) {
      assignValueToAddress(Arg.Regs[0], Addr, Size, MPO, VA);
    }

    /// Handle custom values, which may be passed into one or more of \p VAs.
    /// \return The number of \p VAs that have been assigned after the first
    ///         one, and which should therefore be skipped from further
    ///         processing.
    virtual unsigned assignCustomValue(const ArgInfo &Arg,
                                       ArrayRef<CCValAssign> VAs) {
      // This is not a pure virtual method because not all targets need to worry
      // about custom values.
      llvm_unreachable("Custom values not supported");
    }

    /// Extend a register to the location type given in VA, capped at extending
    /// to at most MaxSize bits. If MaxSizeBits is 0 then no maximum is set.
    Register extendRegister(Register ValReg, CCValAssign &VA,
                            unsigned MaxSizeBits = 0);

    virtual bool assignArg(unsigned ValNo, MVT ValVT, MVT LocVT,
                           CCValAssign::LocInfo LocInfo, const ArgInfo &Info,
                           ISD::ArgFlagsTy Flags, CCState &State) {
      return AssignFn(ValNo, ValVT, LocVT, LocInfo, Flags, State);
    }

    MachineIRBuilder &MIRBuilder;
    MachineRegisterInfo &MRI;
    CCAssignFn *AssignFn;

  private:
    virtual void anchor();
  };

protected:
  /// Getter for generic TargetLowering class.
  const TargetLowering *getTLI() const {
    return TLI;
  }

  /// Getter for target specific TargetLowering class.
  template <class XXXTargetLowering>
    const XXXTargetLowering *getTLI() const {
    return static_cast<const XXXTargetLowering *>(TLI);
  }

  template <typename FuncInfoTy>
  void setArgFlags(ArgInfo &Arg, unsigned OpIdx, const DataLayout &DL,
                   const FuncInfoTy &FuncInfo) const;

  /// Generate instructions for packing \p SrcRegs into one big register
  /// corresponding to the aggregate type \p PackedTy.
  ///
  /// \param SrcRegs should contain one virtual register for each base type in
  ///                \p PackedTy, as returned by computeValueLLTs.
  ///
  /// \return The packed register.
  Register packRegs(ArrayRef<Register> SrcRegs, Type *PackedTy,
                    MachineIRBuilder &MIRBuilder) const;

  /// Generate instructions for unpacking \p SrcReg into the \p DstRegs
  /// corresponding to the aggregate type \p PackedTy.
  ///
  /// \param DstRegs should contain one virtual register for each base type in
  ///        \p PackedTy, as returned by computeValueLLTs.
  void unpackRegs(ArrayRef<Register> DstRegs, Register SrcReg, Type *PackedTy,
                  MachineIRBuilder &MIRBuilder) const;

  /// Invoke Handler::assignArg on each of the given \p Args and then use
  /// \p Callback to move them to the assigned locations.
  ///
  /// \return True if everything has succeeded, false otherwise.
  bool handleAssignments(MachineIRBuilder &MIRBuilder,
                         SmallVectorImpl<ArgInfo> &Args,
                         ValueHandler &Handler) const;
  bool handleAssignments(CCState &CCState,
                         SmallVectorImpl<CCValAssign> &ArgLocs,
                         MachineIRBuilder &MIRBuilder,
                         SmallVectorImpl<ArgInfo> &Args,
                         ValueHandler &Handler) const;

  /// Analyze passed or returned values from a call, supplied in \p ArgInfo,
  /// incorporating info about the passed values into \p CCState.
  ///
  /// Used to check if arguments are suitable for tail call lowering.
  bool analyzeArgInfo(CCState &CCState, SmallVectorImpl<ArgInfo> &Args,
                      CCAssignFn &AssignFnFixed,
                      CCAssignFn &AssignFnVarArg) const;

  /// \returns True if the calling convention for a callee and its caller pass
  /// results in the same way. Typically used for tail call eligibility checks.
  ///
  /// \p Info is the CallLoweringInfo for the call.
  /// \p MF is the MachineFunction for the caller.
  /// \p InArgs contains the results of the call.
  /// \p CalleeAssignFnFixed is the CCAssignFn to be used for the callee for
  /// fixed arguments.
  /// \p CalleeAssignFnVarArg is similar, but for varargs.
  /// \p CallerAssignFnFixed is the CCAssignFn to be used for the caller for
  /// fixed arguments.
  /// \p CallerAssignFnVarArg is similar, but for varargs.
  bool resultsCompatible(CallLoweringInfo &Info, MachineFunction &MF,
                         SmallVectorImpl<ArgInfo> &InArgs,
                         CCAssignFn &CalleeAssignFnFixed,
                         CCAssignFn &CalleeAssignFnVarArg,
                         CCAssignFn &CallerAssignFnFixed,
                         CCAssignFn &CallerAssignFnVarArg) const;

public:
  CallLowering(const TargetLowering *TLI) : TLI(TLI) {}
  virtual ~CallLowering() = default;

  /// \return true if the target is capable of handling swifterror values that
  /// have been promoted to a specified register. The extended versions of
  /// lowerReturn and lowerCall should be implemented.
  virtual bool supportSwiftError() const {
    return false;
  }

  /// This hook must be implemented to lower outgoing return values, described
  /// by \p Val, into the specified virtual registers \p VRegs.
  /// This hook is used by GlobalISel.
  ///
  /// \p SwiftErrorVReg is non-zero if the function has a swifterror parameter
  /// that needs to be implicitly returned.
  ///
  /// \return True if the lowering succeeds, false otherwise.
  virtual bool lowerReturn(MachineIRBuilder &MIRBuilder, const Value *Val,
                           ArrayRef<Register> VRegs,
                           Register SwiftErrorVReg) const {
    if (!supportSwiftError()) {
      assert(SwiftErrorVReg == 0 && "attempt to use unsupported swifterror");
      return lowerReturn(MIRBuilder, Val, VRegs);
    }
    return false;
  }

  /// This hook behaves as the extended lowerReturn function, but for targets
  /// that do not support swifterror value promotion.
  virtual bool lowerReturn(MachineIRBuilder &MIRBuilder, const Value *Val,
                           ArrayRef<Register> VRegs) const {
    return false;
  }

  /// This hook must be implemented to lower the incoming (formal)
  /// arguments, described by \p VRegs, for GlobalISel. Each argument
  /// must end up in the related virtual registers described by \p VRegs.
  /// In other words, the first argument should end up in \c VRegs[0],
  /// the second in \c VRegs[1], and so on. For each argument, there will be one
  /// register for each non-aggregate type, as returned by \c computeValueLLTs.
  /// \p MIRBuilder is set to the proper insertion for the argument
  /// lowering.
  ///
  /// \return True if the lowering succeeded, false otherwise.
  virtual bool lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                    const Function &F,
                                    ArrayRef<ArrayRef<Register>> VRegs) const {
    return false;
  }

  /// This hook must be implemented to lower the given call instruction,
  /// including argument and return value marshalling.
  ///
  ///
  /// \return true if the lowering succeeded, false otherwise.
  virtual bool lowerCall(MachineIRBuilder &MIRBuilder,
                         CallLoweringInfo &Info) const {
    return false;
  }

  /// Lower the given call instruction, including argument and return value
  /// marshalling.
  ///
  /// \p CI is the call/invoke instruction.
  ///
  /// \p ResRegs are the registers where the call's return value should be
  /// stored (or 0 if there is no return value). There will be one register for
  /// each non-aggregate type, as returned by \c computeValueLLTs.
  ///
  /// \p ArgRegs is a list of lists of virtual registers containing each
  /// argument that needs to be passed (argument \c i should be placed in \c
  /// ArgRegs[i]). For each argument, there will be one register for each
  /// non-aggregate type, as returned by \c computeValueLLTs.
  ///
  /// \p SwiftErrorVReg is non-zero if the call has a swifterror inout
  /// parameter, and contains the vreg that the swifterror should be copied into
  /// after the call.
  ///
  /// \p GetCalleeReg is a callback to materialize a register for the callee if
  /// the target determines it cannot jump to the destination based purely on \p
  /// CI. This might be because \p CI is indirect, or because of the limited
  /// range of an immediate jump.
  ///
  /// \return true if the lowering succeeded, false otherwise.
  bool lowerCall(MachineIRBuilder &MIRBuilder, const CallBase &Call,
                 ArrayRef<Register> ResRegs,
                 ArrayRef<ArrayRef<Register>> ArgRegs, Register SwiftErrorVReg,
                 std::function<unsigned()> GetCalleeReg) const;
};

} // end namespace llvm

#endif // LLVM_CODEGEN_GLOBALISEL_CALLLOWERING_H
