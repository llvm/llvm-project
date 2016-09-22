//===-- llvm/lib/Target/AArch64/AArch64CallLowering.h - Call lowering -----===//
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

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64CALLLOWERING
#define LLVM_LIB_TARGET_AARCH64_AARCH64CALLLOWERING

#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/ValueTypes.h"

namespace llvm {

class AArch64TargetLowering;

class AArch64CallLowering: public CallLowering {
 public:

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

    ValueHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI)
        : MIRBuilder(MIRBuilder), MRI(MRI) {}

    virtual ~ValueHandler() {}

    MachineIRBuilder &MIRBuilder;
    MachineRegisterInfo &MRI;
  };

  AArch64CallLowering(const AArch64TargetLowering &TLI);

  bool lowerReturn(MachineIRBuilder &MIRBuiler, const Value *Val,
                   unsigned VReg) const override;

  bool lowerFormalArguments(MachineIRBuilder &MIRBuilder, const Function &F,
                            ArrayRef<unsigned> VRegs) const override;

  bool lowerCall(MachineIRBuilder &MIRBuilder, const MachineOperand &Callee,
                 const ArgInfo &OrigRet,
                 ArrayRef<ArgInfo> OrigArgs) const override;

private:
  typedef std::function<void(MachineIRBuilder &, Type *, unsigned,
                             CCValAssign &)>
      RegHandler;

  typedef std::function<void(MachineIRBuilder &, int, CCValAssign &)>
      MemHandler;

  typedef std::function<void(ArrayRef<unsigned>, ArrayRef<uint64_t>)>
      SplitArgTy;

  void splitToValueTypes(const ArgInfo &OrigArgInfo,
                         SmallVectorImpl<ArgInfo> &SplitArgs,
                         const DataLayout &DL, MachineRegisterInfo &MRI,
                         SplitArgTy SplitArg) const;

  bool handleAssignments(MachineIRBuilder &MIRBuilder, CCAssignFn *AssignFn,
                         ArrayRef<ArgInfo> Args,
                         ValueHandler &Callback) const;
};
} // End of namespace llvm;
#endif
