//===-- PISACallLowering.cpp - Call lowering ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISACallLowering.h"
#include "MCTargetDesc/PISABaseInfo.h"
#include "PISA.h"
#include "PISAISelLowering.h"
#include "PISAMachineFunctionInfo.h"
#include "PISARegisterInfo.h"
#include "PISASubtarget.h"
#include "PISAUtils.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/ModRef.h"

using namespace llvm;

PISACallLowering::PISACallLowering(const PISATargetLowering &TLI)
    : CallLowering(&TLI) {}

bool PISACallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                   const Value *Val, ArrayRef<Register> VRegs,
                                   FunctionLoweringInfo &FLI,
                                   Register SwiftErrorVReg) const {
  // FIXME: Currently the return support is only for registers.
  // Pending:
  //  - return immediates: fold immediates to return operand
  if (VRegs.size() > 1)
    return false;
  if (Val) {
    auto &DL = MIRBuilder.getDataLayout();
    const auto &STI = MIRBuilder.getMF().getSubtarget();
    unsigned Op = 0;
    auto *Ty = Val->getType();
    auto VReg = VRegs[0];
    if (Ty->isVectorTy()) {
      auto *VTy = cast<FixedVectorType>(Ty);
      unsigned NumElts = VTy->getNumElements();
      unsigned EltSize = DL.getTypeSizeInBits(Ty->getScalarType());
      switch (EltSize) {
      case 8:
        switch (NumElts) {
        case 2:
          Op = PISA::retValue_v2i8_r;
          break;
        case 3:
          Op = PISA::retValue_v3i8_r;
          break;
        case 4:
          Op = PISA::retValue_v4i8_r;
          break;
        default:
          llvm_unreachable("Unknown return vector size!");
          break;
        }
        break;
      case 16:
        switch (NumElts) {
        case 2:
          Op = PISA::retValue_v2i16_r;
          break;
        case 3:
          Op = PISA::retValue_v3i16_r;
          break;
        case 4:
          Op = PISA::retValue_v4i16_r;
          break;
        default:
          llvm_unreachable("Unknown return vector size!");
          break;
        }
        break;
      case 32:
        switch (NumElts) {
        case 2:
          Op = PISA::retValue_v2i32_r;
          break;
        case 3:
          Op = PISA::retValue_v3i32_r;
          break;
        case 4:
          Op = PISA::retValue_v4i32_r;
          break;
        case 5:
          Op = PISA::retValue_v5i32_r;
          break;
        case 6:
          Op = PISA::retValue_v6i32_r;
          break;
        case 7:
          Op = PISA::retValue_v7i32_r;
          break;
        case 8:
          Op = PISA::retValue_v8i32_r;
          break;
        case 16:
          Op = PISA::retValue_v16i32_r;
          break;
        case 32:
          Op = PISA::retValue_v32i32_r;
          break;
        case 64:
          Op = PISA::retValue_v64i32_r;
          break;
        default:
          llvm_unreachable("Unknown return vector size!");
          break;
        }
        break;
      case 64:
        switch (NumElts) {
        case 2:
          Op = PISA::retValue_v2i64_r;
          break;
        case 3:
          Op = PISA::retValue_v3i64_r;
          break;
        case 4:
          Op = PISA::retValue_v4i64_r;
          break;
        default:
          llvm_unreachable("Unknown return vector size!");
          break;
        }
        break;
      default:
        llvm_unreachable("Unknown return size!");
        break;
      }
    } else {
      unsigned BitSize = DL.getTypeSizeInBits(Ty);
      switch (BitSize) {
      case 1: // change i1 to i16 (see lowerCall())
      {
        const LLT I16 = LLT::integer(16);
        auto Dst = MIRBuilder.getMRI()->createGenericVirtualRegister(I16);
        auto &MF = MIRBuilder.getMF();
        auto &F = MF.getFunction();
        const DataLayout &DL = MF.getDataLayout();
        ArgInfo RetInfo(VReg, *Val, 0);
        setArgFlags(RetInfo, AttributeList::ReturnIndex, DL, F);
        auto Sext = llvm::any_of(
            RetInfo.Flags, [](const auto &Flag) { return Flag.isSExt(); });
        auto Zext = llvm::any_of(
            RetInfo.Flags, [](const auto &Flag) { return Flag.isZExt(); });
        if (Sext) {
          MIRBuilder.buildSExt(Dst, VReg);
        } else if (Zext) {
          MIRBuilder.buildZExt(Dst, VReg);
        } else {
          MIRBuilder.buildAnyExt(Dst, VReg);
        }
        VReg = Dst;
      }
        Op = PISA::retValue_i16_r;
        break;
      case 8:
        Op = PISA::retValue_i8_r;
        break;
      case 16:
        Op = PISA::retValue_i16_r;
        break;
      case 32:
        Op = PISA::retValue_i32_r;
        break;
      case 64:
        Op = PISA::retValue_i64_r;
        break;
      default:
        llvm_unreachable("Unknown return size!");
        break;
      }
    }
    // Backend-defined opcodes, e.g. retValue* must have a register
    // class assigned to their 'source' register. During instruction
    // combine, a preceeding instructions may be combined, with a new
    // 'dest' register being assigned. Attempt to replace 'source'
    // with 'dest' will trigger an assertion in canReplaceReg(), since
    // there is an expectation of no register class being assigned.
    // Having an extra copy here eliminates the problem; copy itself
    // will be removed during instruction selection.
    auto *MRI = MIRBuilder.getMRI();
    auto Tmp = MRI->createGenericVirtualRegister(MRI->getType(VReg));
    MIRBuilder.buildCopy(Tmp, VReg);
    MIRBuilder.buildInstr(Op).addUse(Tmp).constrainAllUses(
        MIRBuilder.getTII(), *STI.getRegisterInfo(), *STI.getRegBankInfo());
    return true;
  }
  MIRBuilder.buildInstr(PISA::ret);
  return true;
}

bool PISACallLowering::lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                            const Function &F,
                                            ArrayRef<ArrayRef<Register>> VRegs,
                                            FunctionLoweringInfo &FLI) const {
  auto *MRI = MIRBuilder.getMRI();
  auto &MF = MIRBuilder.getMF();
  auto &Ctx = F.getContext();
  auto *MFInfo = MF.getInfo<PISAMachineFunctionInfo>();
  auto &DL = F.getParent()->getDataLayout();
  bool IsKernel = (F.getCallingConv() == CallingConv::PISA_KERNEL);
  for (const auto [i, Arg] : llvm::enumerate(F.args())) {
    assert(VRegs[i].size() == 1 && "Formal arg has multiple vregs");

    ArgInfo OrigArg{VRegs[i], Arg, static_cast<unsigned>(i)};
    setArgFlags(OrigArg, i + AttributeList::FirstArgIndex, DL, F);
    auto *ArgType = OrigArg.OrigValue->getType();
    const bool IsByRef = ArgType->isPointerTy() && OrigArg.Flags[0].isByRef();
    const unsigned ArgSize = IsByRef
                                 ? OrigArg.Flags[0].getByRefSize()
                                 : MRI->getType(VRegs[i][0]).getSizeInBytes();
    MFInfo->setArgInfo(i, ArgSize, IsByRef);

    if (IsKernel && Arg.use_empty())
      continue;

    unsigned Op = 0;
    if (IsByRef) {
      assert(IsKernel && "'byref' is only used in kernel!");
      Op = PISA::G_PISA_PARAM_SLOT;
      loadParamWithOpcode(MIRBuilder, F, VRegs[i][0], ArgType, Op,
                          Arg.getArgNo(), 0);
    } else if (IsKernel && ArgType->isVectorTy()) {
      auto *VectorTy = cast<FixedVectorType>(ArgType);
      auto NumElts = VectorTy->getNumElements();
      auto *EltTy = VectorTy->getElementType();
      auto EltSize = DL.getTypeSizeInBits(EltTy);
      auto Split = (NumElts > 4) || ((NumElts == 3) && (EltSize != 32)) ||
                   ((NumElts == 4) && (EltSize == 64)) || (NumElts == 1);
      if (Split) {
        // handle odd-sized and large kernel args, e.g.
        //   <3 x i8>                    <16 x i16>
        //     loadParam_8b @[arg+0]       loadParam_v4_32b @[arg+0]
        //     loadParam_8b @[arg+1]       loadParam_v4_32b @[arg+16]
        //     loadParam_8b @[arg+2]       buildVector(<8 x i32>)
        //     buildVector(<3 x i8>)       bitcast(<16 x i16)
        auto TargetReg = VRegs[i][0];

        const auto *EltRegClass =
            (EltSize == 8
                 ? &PISA::Reg8bRegClass
                 : (EltSize == 16 ? &PISA::Reg16bRegClass
                                  : (EltSize == 32 ? &PISA::Reg32bRegClass
                                                   : &PISA::Reg64bRegClass)));
        auto TotalSize = NumElts * EltSize;
        auto EltLLT = LLT::integer(EltSize);
        auto I32 = LLT::integer(32);
        if (NumElts <= 4) {
          // do not group
        } else if (TotalSize % 128 == 0) { // 4 x i32
          EltTy = FixedVectorType::get(Type::getInt32Ty(Ctx), 4);
          EltRegClass = &PISA::RegV4_32bRegClass;
          EltLLT = LLT::vector(ElementCount::getFixed(4), I32);
          TargetReg = MRI->createGenericVirtualRegister(
              LLT::vector(ElementCount::getFixed(TotalSize / 32), I32));
          NumElts = TotalSize / 128;
        } else if (TotalSize % 64 == 0) { // 2 x i32
          EltTy = FixedVectorType::get(Type::getInt32Ty(Ctx), 2);
          EltRegClass = &PISA::RegV2_32bRegClass;
          EltLLT = LLT::vector(ElementCount::getFixed(2), I32);
          TargetReg = MRI->createGenericVirtualRegister(
              LLT::vector(ElementCount::getFixed(TotalSize / 32), I32));
          NumElts = TotalSize / 64;
        } else if (TotalSize % 32 == 0) { // 1 x i32
          EltTy = Type::getInt32Ty(Ctx);
          EltRegClass = &PISA::Reg32bRegClass;
          EltLLT = LLT::integer(32);
          TargetReg = MRI->createGenericVirtualRegister(
              LLT::vector(ElementCount::getFixed(TotalSize / 32), I32));
          NumElts = TotalSize / 32;
        }

        SmallVector<Register, 4> Regs;
        for (unsigned I = 0; I < NumElts; I++) {
          auto Reg = MRI->createGenericVirtualRegister(EltLLT);
          MRI->setRegClass(Reg, EltRegClass);
          Op = getLoadParamOpcode(MIRBuilder, F, Reg, EltTy);
          loadParamWithOpcode(MIRBuilder, F, Reg, EltTy, Op, Arg.getArgNo(),
                              I * EltLLT.getSizeInBytes());
          if (EltTy->isPointerTy()) {
            // ld.param loads a scalar value, so convert to ptr here
            auto AS = cast<PointerType>(EltTy)->getAddressSpace();
            auto PtrLLT = LLT::pointer(AS, EltSize);
            auto CastReg = MRI->createGenericVirtualRegister(PtrLLT);
            MRI->setRegClass(CastReg, EltRegClass);
            MIRBuilder.buildIntToPtr(CastReg, Reg);
            Regs.push_back(CastReg);
          } else if (EltTy->isFloatingPointTy()) {
            // ld.param loads an integer value; bitcast to float so that
            // G_BUILD_VECTOR element types match the result vector element
            // type.
            auto FloatLLT = EltTy->isBFloatTy() ? LLT::bfloat16()
                            : EltSize == 16     ? LLT::float16()
                            : EltSize == 32     ? LLT::float32()
                                                : LLT::float64();
            auto CastReg = MRI->createGenericVirtualRegister(FloatLLT);
            MRI->setRegClass(CastReg, EltRegClass);
            MIRBuilder.buildBitcast(CastReg, Reg);
            Regs.push_back(CastReg);
          } else {
            Regs.push_back(Reg);
          }
        }
        if (NumElts == 1)
          MIRBuilder.buildCopy(TargetReg, Regs[0]);
        else if (EltLLT.isVector())
          MIRBuilder.buildConcatVectors(TargetReg, Regs);
        else
          MIRBuilder.buildBuildVector(TargetReg, Regs);
        if (TargetReg != VRegs[i][0]) {
          if (MRI->getType(TargetReg) != MRI->getType(VRegs[i][0]))
            MIRBuilder.buildBitcast(VRegs[i][0], TargetReg);
          else
            MIRBuilder.buildCopy(VRegs[i][0], TargetReg);
        }
      } else {
        Op = getLoadParamOpcode(MIRBuilder, F, VRegs[i][0], Arg.getType());
        loadParamWithOpcode(MIRBuilder, F, VRegs[i][0], ArgType, Op,
                            Arg.getArgNo(), 0);
      }
    } else {
      Op = getLoadParamOpcode(MIRBuilder, F, VRegs[i][0], Arg.getType());
      loadParamWithOpcode(MIRBuilder, F, VRegs[i][0], ArgType, Op,
                          Arg.getArgNo(), 0);
    }
  }
  return true;
}

void PISACallLowering::loadParamWithOpcode(MachineIRBuilder &MIRBuilder,
                                           const Function &F,
                                           const Register &VReg, Type *ArgType,
                                           unsigned Opcode, unsigned ArgNo,
                                           unsigned Offset) const {
  auto *MRI = MIRBuilder.getMRI();
  auto &DL = F.getParent()->getDataLayout();
  bool IsKernel = (F.getCallingConv() == CallingConv::PISA_KERNEL);
  const auto BitSize = DL.getTypeSizeInBits(ArgType->getScalarType());

  auto VReg16 = VReg;
  if (BitSize == 1) { // load arg into i16
    VReg16 = MRI->createGenericVirtualRegister(LLT::integer(16));
    MRI->setRegClass(VReg16, &PISA::Reg16bRegClass);
  }

  auto MIB = MIRBuilder.buildInstr(Opcode).addDef(VReg16).addImm(ArgNo);
  if (IsKernel)
    MIB.addImm(Offset);

  // Attach the kernel argument name from !kernel_arg_name metadata as an
  // extra symbol operand on the loadParam instruction. This allows
  // PISAInstPrinter to print the actual argument name (e.g., [%input])
  // instead of the generic [%argN].
  if (IsKernel)
    if (MDNode *MD = F.getMetadata("kernel_arg_name"))
      if (ArgNo < MD->getNumOperands())
        if (auto *S = dyn_cast<MDString>(MD->getOperand(ArgNo)))
          if (!S->getString().empty())
            MIB.addExternalSymbol(
                MIRBuilder.getMF().createExternalSymbolName(S->getString()));

  if (BitSize == 1) { // convert i16 into i1
    MIRBuilder.buildTrunc(VReg, VReg16);
  }
}

unsigned PISACallLowering::getLoadParamOpcode(MachineIRBuilder &MIRBuilder,
                                              const Function &F,
                                              const Register &VReg,
                                              Type *ArgType) const {
  auto *MRI = MIRBuilder.getMRI();
  auto &MF = MIRBuilder.getMF();
  const auto *TRI = static_cast<const PISARegisterInfo *>(
      MF.getSubtarget().getRegisterInfo());

  bool IsKernel = (F.getCallingConv() == CallingConv::PISA_KERNEL);
  unsigned Op = 0;
  auto &DL = F.getParent()->getDataLayout();

  const unsigned ParamScalar[2][4] = {
      // [isKernel][8/16/32/64]
      {PISA::functionParameter_i8, PISA::functionParameter_i16,
       PISA::functionParameter_i32, PISA::functionParameter_i64},
      {PISA::loadParam_i8, PISA::loadParam_i16, PISA::loadParam_i32,
       PISA::loadParam_i64}};
  const unsigned ParamVector[2][4][3] = {
      // [isKernel][8/16/32/64][v2/v3/v4]
      {
          {PISA::functionParameter_v2i8, PISA::functionParameter_v3i8,
           PISA::functionParameter_v4i8},
          {PISA::functionParameter_v2i16, PISA::functionParameter_v3i16,
           PISA::functionParameter_v4i16},
          {PISA::functionParameter_v2i32, PISA::functionParameter_v3i32,
           PISA::functionParameter_v4i32},
          {PISA::functionParameter_v2i64, PISA::functionParameter_v3i64,
           PISA::functionParameter_v4i64},
      },
      {{PISA::loadParam_v2i8, 0, PISA::loadParam_v4i8},
       {PISA::loadParam_v2i16, 0, PISA::loadParam_v4i16},
       {PISA::loadParam_v2i32, PISA::loadParam_v3i32, PISA::loadParam_v4i32},
       {PISA::loadParam_v2i64, PISA::loadParam_v3i64, 0}}};

  const auto BitSize = DL.getTypeSizeInBits(ArgType->getScalarType());
  // Calculate the argument size in bytes.
  if (ArgType->isIntegerTy()) {
    switch (BitSize) {
    case 1: // i1 args are loaded via i16 register
    case 16:
      MRI->setRegClass(VReg, &PISA::Reg16bRegClass);
      Op = ParamScalar[IsKernel][1];
      break;
    case 8:
      MRI->setRegClass(VReg, &PISA::Reg8bRegClass);
      Op = ParamScalar[IsKernel][0];
      break;
    case 32:
      MRI->setRegClass(VReg, &PISA::Reg32bRegClass);
      Op = ParamScalar[IsKernel][2];
      break;
    case 64:
      MRI->setRegClass(VReg, &PISA::Reg64bRegClass);
      Op = ParamScalar[IsKernel][3];
      break;
    default:
      assert(false && "Bit size for call arg not supported");
    }
  } else if (ArgType->isPointerTy()) {
    if (BitSize == 64) {
      MRI->setRegClass(VReg, &PISA::Reg64bRegClass);
      Op = ParamScalar[IsKernel][3]; // 64bit
    } else if (BitSize == 32) {
      MRI->setRegClass(VReg, &PISA::Reg32bRegClass);
      Op = ParamScalar[IsKernel][2]; // 32bit
    } else {
      llvm_unreachable("unsupported pointer size");
    }
  } else if (ArgType->isHalfTy()) {
    MRI->setRegClass(VReg, &PISA::Reg16bRegClass);
    Op = ParamScalar[IsKernel][1];
  } else if (ArgType->isBFloatTy()) {
    MRI->setRegClass(VReg, &PISA::Reg16bRegClass);
    Op = ParamScalar[IsKernel][1];
  } else if (ArgType->isFloatTy()) {
    MRI->setRegClass(VReg, &PISA::Reg32bRegClass);
    Op = ParamScalar[IsKernel][2];
  } else if (ArgType->isDoubleTy()) {
    MRI->setRegClass(VReg, &PISA::Reg64bRegClass);
    Op = ParamScalar[IsKernel][3];
  } else if (ArgType->isVectorTy()) {
    auto *VectorTy = cast<FixedVectorType>(ArgType);
    auto NumElts = VectorTy->getNumElements();
    assert(((((BitSize == 8) || (BitSize == 16) || (BitSize == 64)) &&
             ((NumElts >= 2) && (NumElts <= 4))) ||
            ((BitSize == 32) &&
             (((NumElts >= 2) && (NumElts <= 8)) || (NumElts == 16) ||
              (NumElts == 32) || (NumElts == 64)))) &&
           "unsupported vector size");
    MRI->setRegClass(VReg, TRI->getVectorRegClass(NumElts, BitSize));
    switch (BitSize) {
    case 8:
      Op = ParamVector[IsKernel][0][NumElts - 2];
      break;
    case 16:
      Op = ParamVector[IsKernel][1][NumElts - 2];
      break;
    case 32: {
      if (NumElts > 4) {
        assert(!IsKernel && "large vector arg in kernel is not supported");
        switch (NumElts) {
        default:
          llvm_unreachable("unsupported number of elements in large vector");
          break;
        case 5:
          Op = PISA::functionParameter_v5i32;
          break;
        case 6:
          Op = PISA::functionParameter_v6i32;
          break;
        case 7:
          Op = PISA::functionParameter_v7i32;
          break;
        case 8:
          Op = PISA::functionParameter_v8i32;
          break;
        case 16:
          Op = PISA::functionParameter_v16i32;
          break;
        case 32:
          Op = PISA::functionParameter_v32i32;
          break;
        case 64:
          Op = PISA::functionParameter_v64i32;
          break;
        }
      } else {
        Op = ParamVector[IsKernel][2][NumElts - 2];
      }
    } break;
    case 64:
      Op = ParamVector[IsKernel][3][NumElts - 2];
      break;
    default:
      assert(false && "Bit size for call arg not supported");
    }
    assert(Op && "argument type is not supported");
  } else {
    report_fatal_error("Argument type not supported");
  }
  return Op;
}

bool PISACallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                 CallLoweringInfo &Info) const {
  // Currently call returns should have single vregs.
  // TODO: handle the case of multiple registers.
  if (Info.OrigRet.Regs.size() > 1)
    return false;

  bool IsIndirectCall = Info.Callee.isReg();
  if (!IsIndirectCall) {
    assert(Info.Callee.isGlobal());
    const Function *CF =
        dyn_cast_or_null<const Function>(Info.Callee.getGlobal());
    if (CF == nullptr)
      return false;
  }

  MachineInstrBuilder MIB;
  const auto *TRI = static_cast<const PISARegisterInfo *>(
      MIRBuilder.getMF().getSubtarget().getRegisterInfo());

  if (IsIndirectCall) {
    Register CalleeReg = Info.Callee.getReg();
    auto *MRI = MIRBuilder.getMRI();
    LLT CalleeTy = MRI->getType(CalleeReg);
    Register CalleeI64Reg = MRI->createGenericVirtualRegister(LLT::integer(64));

    if (CalleeTy.isPointer())
      MIRBuilder.buildPtrToInt(CalleeI64Reg, CalleeReg);
    else
      llvm_unreachable("Unexpected indirect callee register type");

    MRI->setRegClass(CalleeI64Reg, TRI->getRegClassFromLLT(LLT::integer(64)));
    Info.Callee = MachineOperand::CreateReg(CalleeI64Reg, false);
  }

  // promote i1 args to i16
  SmallVector<Register, 8> ArgRegs;
  for (const auto &Arg : Info.OrigArgs) {
    // Currently call args should have single vregs.
    if (Arg.Regs.size() > 1)
      return false;
    if (Arg.Ty->getScalarSizeInBits() == 1) {
      const LLT I16 = LLT::integer(16);
      auto Reg = MIRBuilder.getMRI()->createGenericVirtualRegister(I16);
      MIRBuilder.getMRI()->setRegClass(Reg, TRI->getRegClassFromLLT(I16));
      auto Sext = llvm::any_of(Arg.Flags,
                               [](const auto &Flag) { return Flag.isSExt(); });
      auto Zext = llvm::any_of(Arg.Flags,
                               [](const auto &Flag) { return Flag.isZExt(); });
      if (Sext) {
        MIRBuilder.buildSExt(Reg, Arg.Regs[0]);
      } else if (Zext) {
        MIRBuilder.buildZExt(Reg, Arg.Regs[0]);
      } else {
        MIRBuilder.buildAnyExt(Reg, Arg.Regs[0]);
      }
      ArgRegs.push_back(Reg);
    } else {
      ArgRegs.push_back(Arg.Regs[0]);
    }
  }

  auto MutateI1 = false;
  if (!Info.OrigRet.Ty->isVoidTy()) {
    // select the call op according to return type and build MI
    auto RetLLT =
        llvm::getLLTForType(*Info.OrigRet.Ty, MIRBuilder.getDataLayout());
    unsigned CallOp = 0;
    if (RetLLT.isScalar() || RetLLT.isPointer()) {
      switch (RetLLT.getSizeInBits()) {
      case 1:
        MutateI1 = true;
        CallOp = IsIndirectCall ? PISA::indirectFunctionCall_i16_r_i64_r
                                : PISA::functionCall_i16_r;
        break;
      case 8:
        CallOp = IsIndirectCall ? PISA::indirectFunctionCall_i8_r_i64_r
                                : PISA::functionCall_i8_r;
        break;
      case 16:
        CallOp = IsIndirectCall ? PISA::indirectFunctionCall_i16_r_i64_r
                                : PISA::functionCall_i16_r;
        break;
      case 32:
        CallOp = IsIndirectCall ? PISA::indirectFunctionCall_i32_r_i64_r
                                : PISA::functionCall_i32_r;
        break;
      case 64:
        CallOp = IsIndirectCall ? PISA::indirectFunctionCall_i64_r_i64_r
                                : PISA::functionCall_i64_r;
        break;
      default:
        llvm_unreachable("Unsupported function return type");
        break;
      }
    } else if (RetLLT.isVector()) {
      auto TypeBitSize = RetLLT.getElementType().getSizeInBits();
      auto NumElts = RetLLT.getNumElements();
      switch (TypeBitSize) {
      case 8:
        switch (NumElts) {
        case 2:
          CallOp = IsIndirectCall ? PISA::indirectFunctionCall_v2i8_r_i64_r
                                  : PISA::functionCall_v2i8_r;
          break;
        case 3:
          CallOp = IsIndirectCall ? PISA::indirectFunctionCall_v3i8_r_i64_r
                                  : PISA::functionCall_v3i8_r;
          break;
        case 4:
          CallOp = IsIndirectCall ? PISA::indirectFunctionCall_v4i8_r_i64_r
                                  : PISA::functionCall_v4i8_r;
          break;
        default:
          llvm_unreachable("Vector size not supported");
        }
        break;
      case 16:
        switch (NumElts) {
        case 2:
          CallOp = IsIndirectCall ? PISA::indirectFunctionCall_v2i16_r_i64_r
                                  : PISA::functionCall_v2i16_r;
          break;
        case 3:
          CallOp = IsIndirectCall ? PISA::indirectFunctionCall_v3i16_r_i64_r
                                  : PISA::functionCall_v3i16_r;
          break;
        case 4:
          CallOp = IsIndirectCall ? PISA::indirectFunctionCall_v4i16_r_i64_r
                                  : PISA::functionCall_v4i16_r;
          break;
        default:
          llvm_unreachable("Vector size not supported");
        }
        break;
      case 32:
        switch (NumElts) {
        case 2:
          CallOp = IsIndirectCall ? PISA::indirectFunctionCall_v2i32_r_i64_r
                                  : PISA::functionCall_v2i32_r;
          break;
        case 3:
          CallOp = IsIndirectCall ? PISA::indirectFunctionCall_v3i32_r_i64_r
                                  : PISA::functionCall_v3i32_r;
          break;
        case 4:
          CallOp = IsIndirectCall ? PISA::indirectFunctionCall_v4i32_r_i64_r
                                  : PISA::functionCall_v4i32_r;
          break;
        case 5:
          CallOp = IsIndirectCall ? PISA::indirectFunctionCall_v5i32_r_i64_r
                                  : PISA::functionCall_v5i32_r;
          break;
        case 6:
          CallOp = IsIndirectCall ? PISA::indirectFunctionCall_v6i32_r_i64_r
                                  : PISA::functionCall_v6i32_r;
          break;
        case 7:
          CallOp = IsIndirectCall ? PISA::indirectFunctionCall_v7i32_r_i64_r
                                  : PISA::functionCall_v7i32_r;
          break;
        case 8:
          CallOp = IsIndirectCall ? PISA::indirectFunctionCall_v8i32_r_i64_r
                                  : PISA::functionCall_v8i32_r;
          break;
        case 16:
          CallOp = IsIndirectCall ? PISA::indirectFunctionCall_v16i32_r_i64_r
                                  : PISA::functionCall_v16i32_r;
          break;
        case 32:
          CallOp = IsIndirectCall ? PISA::indirectFunctionCall_v32i32_r_i64_r
                                  : PISA::functionCall_v32i32_r;
          break;
        case 64:
          CallOp = IsIndirectCall ? PISA::indirectFunctionCall_v64i32_r_i64_r
                                  : PISA::functionCall_v64i32_r;
          break;
        default:
          llvm_unreachable("Vector size not supported");
        }
        break;
      case 64:
        switch (NumElts) {
        case 2:
          CallOp = IsIndirectCall ? PISA::indirectFunctionCall_v2i64_r_i64_r
                                  : PISA::functionCall_v2i64_r;
          break;
        case 3:
          CallOp = IsIndirectCall ? PISA::indirectFunctionCall_v3i64_r_i64_r
                                  : PISA::functionCall_v3i64_r;
          break;
        case 4:
          CallOp = IsIndirectCall ? PISA::indirectFunctionCall_v4i64_r_i64_r
                                  : PISA::functionCall_v4i64_r;
          break;
        default:
          llvm_unreachable("Vector size not supported");
        }
        break;
      default:
        llvm_unreachable("Unsupported function return type");
        break;
      }
    } else {
      llvm_unreachable("Unsupported call return type");
    }

    // set Ret register class
    assert(!Info.OrigRet.Regs.empty());
    Register ResVReg = Info.OrigRet.Regs[0];
    auto OrigRetLLT = RetLLT;
    if (MutateI1) { // will return i16 (see lowerReturn())
      RetLLT = LLT::integer(16);
      ResVReg = MIRBuilder.getMRI()->createGenericVirtualRegister(RetLLT);
    }

    MIRBuilder.getMRI()->setRegClass(ResVReg, TRI->getRegClassFromLLT(RetLLT));
    MIB = MIRBuilder.buildInstr(CallOp).addDef(ResVReg).add(Info.Callee);

    if (MutateI1) { // change i16 back to i1
      Register VReg = Info.OrigRet.Regs[0];
      MIRBuilder.getMRI()->setRegClass(VReg,
                                       TRI->getRegClassFromLLT(OrigRetLLT));
      MIRBuilder.buildTrunc(VReg, ResVReg);
    }
  } else {
    // void return
    MIB = MIRBuilder
              .buildInstr(IsIndirectCall ? PISA::indirectFunctionCall_void_r
                                         : PISA::functionCall_void)
              .add(Info.Callee);
  }

  // add function args into MI if any
  for (auto Reg : ArgRegs) {
    MIB.addUse(Reg);
  }

  return true;
}
