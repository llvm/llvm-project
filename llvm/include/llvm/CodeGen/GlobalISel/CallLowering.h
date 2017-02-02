//===-- llvm/CodeGen/GlobalISel/CallLowering.h - Call lowering --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes how to lower LLVM calls to machine code calls.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_CALLLOWERING_H
#define LLVM_CODEGEN_GLOBALISEL_CALLLOWERING_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/Target/TargetCallingConv.h"

namespace llvm {
// Forward declarations.
class MachineIRBuilder;
class MachineOperand;
class TargetLowering;
class Value;

class CallLowering {
  const TargetLowering *TLI;
public:
  struct ArgInfo {
    unsigned Reg;
    Type *Ty;
    ISD::ArgFlagsTy Flags;
    bool IsFixed;

    ArgInfo(unsigned Reg, Type *Ty, ISD::ArgFlagsTy Flags = ISD::ArgFlagsTy{},
            bool IsFixed = true)
        : Reg(Reg), Ty(Ty), Flags(Flags), IsFixed(IsFixed) {}
  };

  /// Argument handling is mostly uniform between the four places that
  /// make these decisions: function formal arguments, call
  /// instruction args, call instruction returns and function
  /// returns. However, once a decision has been made on where an
  /// arugment should go, exactly what happens can vary slightly. This
  /// class abstracts the differences.
  struct ValueHandler {
    /// Materialize a VReg containing the address of the specified
    /// stack-based object. This is either based on a FrameIndex or
    /// direct SP manipulation, depending on the context. \p MPO
    /// should be initialized to an appropriate description of the
    /// address created.
    virtual unsigned getStackAddress(uint64_t Size, int64_t Offset,
                                     MachinePointerInfo &MPO) = 0;

    /// The specified value has been assigned to a physical register,
    /// handle the appropriate COPY (either to or from) and mark any
    /// relevant uses/defines as needed.
    virtual void assignValueToReg(unsigned ValVReg, unsigned PhysReg,
                                  CCValAssign &VA) = 0;

    /// The specified value has been assigned to a stack
    /// location. Load or store it there, with appropriate extension
    /// if necessary.
    virtual void assignValueToAddress(unsigned ValVReg, unsigned Addr,
                                      uint64_t Size, MachinePointerInfo &MPO,
                                      CCValAssign &VA) = 0;

    unsigned extendRegister(unsigned ValReg, CCValAssign &VA);

    virtual bool assignArg(unsigned ValNo, MVT ValVT, MVT LocVT,
                           CCValAssign::LocInfo LocInfo, const ArgInfo &Info,
                           CCState &State) {
      return AssignFn(ValNo, ValVT, LocVT, LocInfo, Info.Flags, State);
    }

    ValueHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                 CCAssignFn *AssignFn)
      : MIRBuilder(MIRBuilder), MRI(MRI), AssignFn(AssignFn) {}

    virtual ~ValueHandler() {}

    MachineIRBuilder &MIRBuilder;
    MachineRegisterInfo &MRI;
    CCAssignFn *AssignFn;
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
  void setArgFlags(ArgInfo &Arg, unsigned OpNum, const DataLayout &DL,
                   const FuncInfoTy &FuncInfo) const;

  /// Invoke Handler::assignArg on each of the given \p Args and then use
  /// \p Callback to move them to the assigned locations.
  ///
  /// \return True if everything has succeeded, false otherwise.
  bool handleAssignments(MachineIRBuilder &MIRBuilder, ArrayRef<ArgInfo> Args,
                         ValueHandler &Callback) const;

public:
  CallLowering(const TargetLowering *TLI) : TLI(TLI) {}
  virtual ~CallLowering() {}

  /// This hook must be implemented to lower outgoing return values, described
  /// by \p Val, into the specified virtual register \p VReg.
  /// This hook is used by GlobalISel.
  ///
  /// \return True if the lowering succeeds, false otherwise.
  virtual bool lowerReturn(MachineIRBuilder &MIRBuilder,
                           const Value *Val, unsigned VReg) const {
    return false;
  }

  /// This hook must be implemented to lower the incoming (formal)
  /// arguments, described by \p Args, for GlobalISel. Each argument
  /// must end up in the related virtual register described by VRegs.
  /// In other words, the first argument should end up in VRegs[0],
  /// the second in VRegs[1], and so on.
  /// \p MIRBuilder is set to the proper insertion for the argument
  /// lowering.
  ///
  /// \return True if the lowering succeeded, false otherwise.
  virtual bool lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                    const Function &F,
                                    ArrayRef<unsigned> VRegs) const {
    return false;
  }

  /// This hook must be implemented to lower the given call instruction,
  /// including argument and return value marshalling.
  ///
  /// \p Callee is the destination of the call. It should be either a register,
  /// globaladdress, or externalsymbol.
  ///
  /// \p ResTy is the type returned by the function
  ///
  /// \p ResReg is the generic virtual register that the returned
  /// value should be lowered into.
  ///
  /// \p ArgTys is a list of the types each member of \p ArgRegs has; used by
  /// the target to decide which register/stack slot should be allocated.
  ///
  /// \p ArgRegs is a list of virtual registers containing each argument that
  /// needs to be passed.
  ///
  /// \return true if the lowering succeeded, false otherwise.
  virtual bool lowerCall(MachineIRBuilder &MIRBuilder,
                         const MachineOperand &Callee, const ArgInfo &OrigRet,
                         ArrayRef<ArgInfo> OrigArgs) const {
    return false;
  }

  /// This hook must be implemented to lower the given call instruction,
  /// including argument and return value marshalling.
  ///
  /// \p CI is either a CallInst or InvokeInst reference (other instantiations
  /// will fail at link time).
  ///
  /// \p ResReg is a register where the call's return value should be stored (or
  /// 0 if there is no return value).
  ///
  /// \p ArgRegs is a list of virtual registers containing each argument that
  /// needs to be passed.
  ///
  /// \p GetCalleeReg is a callback to materialize a register for the callee if
  /// the target determines it cannot jump to the destination based purely on \p
  /// CI. This might be because \p CI is indirect, or because of the limited
  /// range of an immediate jump.
  ///
  /// \return true if the lowering succeeded, false otherwise.
  template <typename CallInstTy>
  bool lowerCall(MachineIRBuilder &MIRBuilder, const CallInstTy &CI,
                 unsigned ResReg, ArrayRef<unsigned> ArgRegs,
                 std::function<unsigned()> GetCalleeReg) const;
};
} // End namespace llvm.

#endif
