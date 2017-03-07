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
#include "X86ISelLowering.h"
#include "X86InstrInfo.h"
#include "X86TargetMachine.h"
#include "X86CallingConv.h"

#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineValueType.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

#include "X86GenCallingConv.inc"

#ifndef LLVM_BUILD_GLOBAL_ISEL
#error "This shouldn't be built without GISel"
#endif

X86CallLowering::X86CallLowering(const X86TargetLowering &TLI)
    : CallLowering(&TLI) {}

void X86CallLowering::splitToValueTypes(const ArgInfo &OrigArg,
                                        SmallVectorImpl<ArgInfo> &SplitArgs,
                                        const DataLayout &DL,
                                        MachineRegisterInfo &MRI,
                                        SplitArgTy PerformArgSplit) const {

  const X86TargetLowering &TLI = *getTLI<X86TargetLowering>();
  LLVMContext &Context = OrigArg.Ty->getContext();
  EVT VT = TLI.getValueType(DL, OrigArg.Ty);
  unsigned NumParts = TLI.getNumRegisters(Context, VT);

  if (NumParts == 1) {
    SplitArgs.push_back(OrigArg);
    return;
  }

  SmallVector<uint64_t, 4> BitOffsets;
  SmallVector<unsigned, 8> SplitRegs;

  EVT PartVT = TLI.getRegisterType(Context, VT);
  Type *PartTy = PartVT.getTypeForEVT(Context);

  for (unsigned i = 0; i < NumParts; ++i) {
    ArgInfo Info = ArgInfo{MRI.createGenericVirtualRegister(LLT{*PartTy, DL}),
                           PartTy, OrigArg.Flags};
    SplitArgs.push_back(Info);
    PerformArgSplit(Info.Reg, PartVT.getSizeInBits() * i);
  }
}

namespace {
struct FuncReturnHandler : public CallLowering::ValueHandler {
  FuncReturnHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                    MachineInstrBuilder &MIB, CCAssignFn *AssignFn)
      : ValueHandler(MIRBuilder, MRI, AssignFn), MIB(MIB) {}

  unsigned getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override {
    llvm_unreachable("Don't know how to get a stack address yet");
  }

  void assignValueToReg(unsigned ValVReg, unsigned PhysReg,
                        CCValAssign &VA) override {
    MIB.addUse(PhysReg, RegState::Implicit);
    unsigned ExtReg = extendRegister(ValVReg, VA);
    MIRBuilder.buildCopy(PhysReg, ExtReg);
  }

  void assignValueToAddress(unsigned ValVReg, unsigned Addr, uint64_t Size,
                            MachinePointerInfo &MPO, CCValAssign &VA) override {
    llvm_unreachable("Don't know how to assign a value to an address yet");
  }

  MachineInstrBuilder &MIB;
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
    setArgFlags(OrigArg, AttributeSet::ReturnIndex, DL, F);

    SmallVector<ArgInfo, 8> SplitArgs;
    splitToValueTypes(OrigArg, SplitArgs, DL, MRI,
                      [&](unsigned Reg, uint64_t Offset) {
                        MIRBuilder.buildExtract(Reg, VReg, Offset);
                      });

    FuncReturnHandler Handler(MIRBuilder, MRI, MIB, RetCC_X86);
    if(!handleAssignments(MIRBuilder, SplitArgs, Handler))
      return false;
  }

  MIRBuilder.insertInstr(MIB);
  return true;
}

namespace {
struct FormalArgHandler : public CallLowering::ValueHandler {
  FormalArgHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                   CCAssignFn *AssignFn, const DataLayout &DL)
      : ValueHandler(MIRBuilder, MRI, AssignFn), DL(DL) {}

  unsigned getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override {

    auto &MFI = MIRBuilder.getMF().getFrameInfo();
    int FI = MFI.CreateFixedObject(Size, Offset, true);
    MPO = MachinePointerInfo::getFixedStack(MIRBuilder.getMF(), FI);

    unsigned AddrReg =
        MRI.createGenericVirtualRegister(LLT::pointer(0,
                                         DL.getPointerSizeInBits(0)));
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

  void assignValueToReg(unsigned ValVReg, unsigned PhysReg,
                        CCValAssign &VA) override {
    MIRBuilder.getMBB().addLiveIn(PhysReg);
    MIRBuilder.buildCopy(ValVReg, PhysReg);
  }

  const DataLayout &DL;
};
}

bool X86CallLowering::lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                           const Function &F,
                                           ArrayRef<unsigned> VRegs) const {
  if (F.arg_empty())
    return true;

  //TODO: handle variadic function
  if (F.isVarArg())
    return false;

  MachineFunction &MF = MIRBuilder.getMF();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  auto DL = MF.getDataLayout();

  SmallVector<ArgInfo, 8> SplitArgs;
  unsigned Idx = 0;
  for (auto &Arg : F.getArgumentList()) {
    ArgInfo OrigArg(VRegs[Idx], Arg.getType());
    setArgFlags(OrigArg, Idx + 1, DL, F);
    LLT Ty = MRI.getType(VRegs[Idx]);
    unsigned Dst = VRegs[Idx];
    bool Split = false;
    splitToValueTypes(OrigArg, SplitArgs, DL, MRI,
                      [&](unsigned Reg, uint64_t Offset) {
                        if (!Split) {
                          Split = true;
                          Dst = MRI.createGenericVirtualRegister(Ty);
                          MIRBuilder.buildUndef(Dst);
                        }
                        unsigned Tmp = MRI.createGenericVirtualRegister(Ty);
                        MIRBuilder.buildInsert(Tmp, Dst, Reg, Offset);
                        Dst = Tmp;
                      });
    if (Dst != VRegs[Idx])
      MIRBuilder.buildCopy(VRegs[Idx], Dst);
    Idx++;
  }

  MachineBasicBlock &MBB = MIRBuilder.getMBB();
  if (!MBB.empty())
     MIRBuilder.setInstr(*MBB.begin());

  FormalArgHandler Handler(MIRBuilder, MRI, CC_X86, DL);
  if (!handleAssignments(MIRBuilder, SplitArgs, Handler))
    return false;

  // Move back to the end of the basic block.
  MIRBuilder.setMBB(MBB);

  return true;
}
