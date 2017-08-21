//===-- llvm/lib/Target/X86/X86CallLowering.cpp - Call lowering -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the lowering of LLVM calls to machine code calls for
/// GlobalISel.
///
//===----------------------------------------------------------------------===//

#include "X86CallLowering.h"
#include "X86CallingConv.h"
#include "X86ISelLowering.h"
#include "X86InstrInfo.h"
#include "X86TargetMachine.h"

#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineValueType.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

#include "X86GenCallingConv.inc"

X86CallLowering::X86CallLowering(const X86TargetLowering &TLI)
    : CallLowering(&TLI) {}

bool X86CallLowering::splitToValueTypes(const ArgInfo &OrigArg,
                                        SmallVectorImpl<ArgInfo> &SplitArgs,
                                        const DataLayout &DL,
                                        MachineRegisterInfo &MRI,
                                        SplitArgTy PerformArgSplit) const {

  const X86TargetLowering &TLI = *getTLI<X86TargetLowering>();
  LLVMContext &Context = OrigArg.Ty->getContext();

  SmallVector<EVT, 4> SplitVTs;
  SmallVector<uint64_t, 4> Offsets;
  ComputeValueVTs(TLI, DL, OrigArg.Ty, SplitVTs, &Offsets, 0);

  if (SplitVTs.size() != 1) {
    // TODO: support struct/array split
    return false;
  }

  EVT VT = SplitVTs[0];
  unsigned NumParts = TLI.getNumRegisters(Context, VT);

  if (NumParts == 1) {
    // replace the original type ( pointer -> GPR ).
    SplitArgs.emplace_back(OrigArg.Reg, VT.getTypeForEVT(Context),
                           OrigArg.Flags, OrigArg.IsFixed);
    return true;
  }

  SmallVector<unsigned, 8> SplitRegs;

  EVT PartVT = TLI.getRegisterType(Context, VT);
  Type *PartTy = PartVT.getTypeForEVT(Context);

  for (unsigned i = 0; i < NumParts; ++i) {
    ArgInfo Info =
        ArgInfo{MRI.createGenericVirtualRegister(getLLTForType(*PartTy, DL)),
                PartTy, OrigArg.Flags};
    SplitArgs.push_back(Info);
    SplitRegs.push_back(Info.Reg);
  }

  PerformArgSplit(SplitRegs);
  return true;
}

namespace {
struct OutgoingValueHandler : public CallLowering::ValueHandler {
  OutgoingValueHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                       MachineInstrBuilder &MIB, CCAssignFn *AssignFn)
      : ValueHandler(MIRBuilder, MRI, AssignFn), MIB(MIB), StackSize(0),
        DL(MIRBuilder.getMF().getDataLayout()),
        STI(MIRBuilder.getMF().getSubtarget<X86Subtarget>()) {}

  unsigned getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override {

    LLT p0 = LLT::pointer(0, DL.getPointerSizeInBits(0));
    LLT SType = LLT::scalar(DL.getPointerSizeInBits(0));
    unsigned SPReg = MRI.createGenericVirtualRegister(p0);
    MIRBuilder.buildCopy(SPReg, STI.getRegisterInfo()->getStackRegister());

    unsigned OffsetReg = MRI.createGenericVirtualRegister(SType);
    MIRBuilder.buildConstant(OffsetReg, Offset);

    unsigned AddrReg = MRI.createGenericVirtualRegister(p0);
    MIRBuilder.buildGEP(AddrReg, SPReg, OffsetReg);

    MPO = MachinePointerInfo::getStack(MIRBuilder.getMF(), Offset);
    return AddrReg;
  }

  void assignValueToReg(unsigned ValVReg, unsigned PhysReg,
                        CCValAssign &VA) override {
    MIB.addUse(PhysReg, RegState::Implicit);
    unsigned ExtReg = extendRegister(ValVReg, VA);
    MIRBuilder.buildCopy(PhysReg, ExtReg);
  }

  void assignValueToAddress(unsigned ValVReg, unsigned Addr, uint64_t Size,
                            MachinePointerInfo &MPO, CCValAssign &VA) override {

    unsigned ExtReg = extendRegister(ValVReg, VA);
    auto MMO = MIRBuilder.getMF().getMachineMemOperand(
        MPO, MachineMemOperand::MOStore, VA.getLocVT().getStoreSize(),
        /* Alignment */ 0);
    MIRBuilder.buildStore(ExtReg, Addr, *MMO);
  }

  bool assignArg(unsigned ValNo, MVT ValVT, MVT LocVT,
                 CCValAssign::LocInfo LocInfo,
                 const CallLowering::ArgInfo &Info, CCState &State) override {

    if (!Info.IsFixed)
      return true; // TODO: handle variadic function

    bool Res = AssignFn(ValNo, ValVT, LocVT, LocInfo, Info.Flags, State);
    StackSize = State.getNextStackOffset();
    return Res;
  }

  uint64_t getStackSize() { return StackSize; }

protected:
  MachineInstrBuilder &MIB;
  uint64_t StackSize;
  const DataLayout &DL;
  const X86Subtarget &STI;
};
} // End anonymous namespace.

bool X86CallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                  const Value *Val, unsigned VReg) const {

  assert(((Val && VReg) || (!Val && !VReg)) && "Return value without a vreg");

  auto MIB = MIRBuilder.buildInstrNoInsert(X86::RET).addImm(0);

  if (VReg) {
    MachineFunction &MF = MIRBuilder.getMF();
    MachineRegisterInfo &MRI = MF.getRegInfo();
    auto &DL = MF.getDataLayout();
    const Function &F = *MF.getFunction();

    ArgInfo OrigArg{VReg, Val->getType()};
    setArgFlags(OrigArg, AttributeList::ReturnIndex, DL, F);

    SmallVector<ArgInfo, 8> SplitArgs;
    if (!splitToValueTypes(OrigArg, SplitArgs, DL, MRI,
                           [&](ArrayRef<unsigned> Regs) {
                             MIRBuilder.buildUnmerge(Regs, VReg);
                           }))
      return false;

    OutgoingValueHandler Handler(MIRBuilder, MRI, MIB, RetCC_X86);
    if (!handleAssignments(MIRBuilder, SplitArgs, Handler))
      return false;
  }

  MIRBuilder.insertInstr(MIB);
  return true;
}

namespace {
struct IncomingValueHandler : public CallLowering::ValueHandler {
  IncomingValueHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                       CCAssignFn *AssignFn)
      : ValueHandler(MIRBuilder, MRI, AssignFn),
        DL(MIRBuilder.getMF().getDataLayout()) {}

  unsigned getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override {

    auto &MFI = MIRBuilder.getMF().getFrameInfo();
    int FI = MFI.CreateFixedObject(Size, Offset, true);
    MPO = MachinePointerInfo::getFixedStack(MIRBuilder.getMF(), FI);

    unsigned AddrReg = MRI.createGenericVirtualRegister(
        LLT::pointer(0, DL.getPointerSizeInBits(0)));
    MIRBuilder.buildFrameIndex(AddrReg, FI);
    return AddrReg;
  }

  void assignValueToAddress(unsigned ValVReg, unsigned Addr, uint64_t Size,
                            MachinePointerInfo &MPO, CCValAssign &VA) override {

    auto MMO = MIRBuilder.getMF().getMachineMemOperand(
        MPO, MachineMemOperand::MOLoad | MachineMemOperand::MOInvariant, Size,
        0);
    MIRBuilder.buildLoad(ValVReg, Addr, *MMO);
  }

protected:
  const DataLayout &DL;
};

struct FormalArgHandler : public IncomingValueHandler {
  FormalArgHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                   CCAssignFn *AssignFn)
      : IncomingValueHandler(MIRBuilder, MRI, AssignFn) {}

  void assignValueToReg(unsigned ValVReg, unsigned PhysReg,
                        CCValAssign &VA) override {
    MIRBuilder.getMBB().addLiveIn(PhysReg);
    MIRBuilder.buildCopy(ValVReg, PhysReg);
  }
};

struct CallReturnHandler : public IncomingValueHandler {
  CallReturnHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                    CCAssignFn *AssignFn, MachineInstrBuilder &MIB)
      : IncomingValueHandler(MIRBuilder, MRI, AssignFn), MIB(MIB) {}

  void assignValueToReg(unsigned ValVReg, unsigned PhysReg,
                        CCValAssign &VA) override {
    MIB.addDef(PhysReg, RegState::Implicit);
    MIRBuilder.buildCopy(ValVReg, PhysReg);
  }

protected:
  MachineInstrBuilder &MIB;
};

} // namespace

bool X86CallLowering::lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                           const Function &F,
                                           ArrayRef<unsigned> VRegs) const {
  if (F.arg_empty())
    return true;

  // TODO: handle variadic function
  if (F.isVarArg())
    return false;

  MachineFunction &MF = MIRBuilder.getMF();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  auto DL = MF.getDataLayout();

  SmallVector<ArgInfo, 8> SplitArgs;
  unsigned Idx = 0;
  for (auto &Arg : F.args()) {

    // TODO: handle not simple cases.
    if (Arg.hasAttribute(Attribute::ByVal) ||
        Arg.hasAttribute(Attribute::InReg) ||
        Arg.hasAttribute(Attribute::StructRet) ||
        Arg.hasAttribute(Attribute::SwiftSelf) ||
        Arg.hasAttribute(Attribute::SwiftError) ||
        Arg.hasAttribute(Attribute::Nest))
      return false;

    ArgInfo OrigArg(VRegs[Idx], Arg.getType());
    setArgFlags(OrigArg, Idx + AttributeList::FirstArgIndex, DL, F);
    if (!splitToValueTypes(OrigArg, SplitArgs, DL, MRI,
                           [&](ArrayRef<unsigned> Regs) {
                             MIRBuilder.buildMerge(VRegs[Idx], Regs);
                           }))
      return false;
    Idx++;
  }

  MachineBasicBlock &MBB = MIRBuilder.getMBB();
  if (!MBB.empty())
    MIRBuilder.setInstr(*MBB.begin());

  FormalArgHandler Handler(MIRBuilder, MRI, CC_X86);
  if (!handleAssignments(MIRBuilder, SplitArgs, Handler))
    return false;

  // Move back to the end of the basic block.
  MIRBuilder.setMBB(MBB);

  return true;
}

bool X86CallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                CallingConv::ID CallConv,
                                const MachineOperand &Callee,
                                const ArgInfo &OrigRet,
                                ArrayRef<ArgInfo> OrigArgs) const {

  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = *MF.getFunction();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  auto &DL = F.getParent()->getDataLayout();
  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const TargetInstrInfo &TII = *STI.getInstrInfo();
  auto TRI = STI.getRegisterInfo();

  // Handle only Linux C, X86_64_SysV calling conventions for now.
  if (!STI.isTargetLinux() ||
      !(CallConv == CallingConv::C || CallConv == CallingConv::X86_64_SysV))
    return false;

  unsigned AdjStackDown = TII.getCallFrameSetupOpcode();
  auto CallSeqStart = MIRBuilder.buildInstr(AdjStackDown);

  // Create a temporarily-floating call instruction so we can add the implicit
  // uses of arg registers.
  bool Is64Bit = STI.is64Bit();
  unsigned CallOpc = Callee.isReg()
                         ? (Is64Bit ? X86::CALL64r : X86::CALL32r)
                         : (Is64Bit ? X86::CALL64pcrel32 : X86::CALLpcrel32);

  auto MIB = MIRBuilder.buildInstrNoInsert(CallOpc).add(Callee).addRegMask(
      TRI->getCallPreservedMask(MF, CallConv));

  SmallVector<ArgInfo, 8> SplitArgs;
  for (const auto &OrigArg : OrigArgs) {

    // TODO: handle not simple cases.
    if (OrigArg.Flags.isByVal())
      return false;

    if (!splitToValueTypes(OrigArg, SplitArgs, DL, MRI,
                           [&](ArrayRef<unsigned> Regs) {
                             MIRBuilder.buildUnmerge(Regs, OrigArg.Reg);
                           }))
      return false;
  }
  // Do the actual argument marshalling.
  OutgoingValueHandler Handler(MIRBuilder, MRI, MIB, CC_X86);
  if (!handleAssignments(MIRBuilder, SplitArgs, Handler))
    return false;

  // Now we can add the actual call instruction to the correct basic block.
  MIRBuilder.insertInstr(MIB);

  // If Callee is a reg, since it is used by a target specific
  // instruction, it must have a register class matching the
  // constraint of that instruction.
  if (Callee.isReg())
    MIB->getOperand(0).setReg(constrainOperandRegClass(
        MF, *TRI, MRI, *MF.getSubtarget().getInstrInfo(),
        *MF.getSubtarget().getRegBankInfo(), *MIB, MIB->getDesc(),
        Callee.getReg(), 0));

  // Finally we can copy the returned value back into its virtual-register. In
  // symmetry with the arguments, the physical register must be an
  // implicit-define of the call instruction.

  if (OrigRet.Reg) {
    SplitArgs.clear();
    SmallVector<unsigned, 8> NewRegs;

    if (!splitToValueTypes(OrigRet, SplitArgs, DL, MRI,
                           [&](ArrayRef<unsigned> Regs) {
                             NewRegs.assign(Regs.begin(), Regs.end());
                           }))
      return false;

    CallReturnHandler Handler(MIRBuilder, MRI, RetCC_X86, MIB);
    if (!handleAssignments(MIRBuilder, SplitArgs, Handler))
      return false;

    if (!NewRegs.empty())
      MIRBuilder.buildMerge(OrigRet.Reg, NewRegs);
  }

  CallSeqStart.addImm(Handler.getStackSize())
      .addImm(0 /* see getFrameTotalSize */)
      .addImm(0 /* see getFrameAdjustment */);

  unsigned AdjStackUp = TII.getCallFrameDestroyOpcode();
  MIRBuilder.buildInstr(AdjStackUp)
      .addImm(Handler.getStackSize())
      .addImm(0 /* NumBytesForCalleeToPop */);

  return true;
}
