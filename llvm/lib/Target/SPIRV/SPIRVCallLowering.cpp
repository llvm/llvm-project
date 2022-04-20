//===--- SPIRVCallLowering.cpp - Call lowering ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the lowering of LLVM calls to machine code calls for
// GlobalISel.
//
//===----------------------------------------------------------------------===//

#include "SPIRVCallLowering.h"
#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "SPIRV.h"
#include "SPIRVGlobalRegistry.h"
#include "SPIRVISelLowering.h"
#include "SPIRVRegisterInfo.h"
#include "SPIRVSubtarget.h"
#include "SPIRVUtils.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"

using namespace llvm;

SPIRVCallLowering::SPIRVCallLowering(const SPIRVTargetLowering &TLI,
                                     const SPIRVSubtarget &ST,
                                     SPIRVGlobalRegistry *GR)
    : CallLowering(&TLI), ST(ST), GR(GR) {}

bool SPIRVCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                    const Value *Val, ArrayRef<Register> VRegs,
                                    FunctionLoweringInfo &FLI,
                                    Register SwiftErrorVReg) const {
  // Currently all return types should use a single register.
  // TODO: handle the case of multiple registers.
  if (VRegs.size() > 1)
    return false;
  if (Val)
    return MIRBuilder.buildInstr(SPIRV::OpReturnValue)
        .addUse(VRegs[0])
        .constrainAllUses(MIRBuilder.getTII(), *ST.getRegisterInfo(),
                          *ST.getRegBankInfo());
  MIRBuilder.buildInstr(SPIRV::OpReturn);
  return true;
}

// Based on the LLVM function attributes, get a SPIR-V FunctionControl.
static uint32_t getFunctionControl(const Function &F) {
  uint32_t FuncControl = static_cast<uint32_t>(SPIRV::FunctionControl::None);
  if (F.hasFnAttribute(Attribute::AttrKind::AlwaysInline)) {
    FuncControl |= static_cast<uint32_t>(SPIRV::FunctionControl::Inline);
  }
  if (F.hasFnAttribute(Attribute::AttrKind::ReadNone)) {
    FuncControl |= static_cast<uint32_t>(SPIRV::FunctionControl::Pure);
  }
  if (F.hasFnAttribute(Attribute::AttrKind::ReadOnly)) {
    FuncControl |= static_cast<uint32_t>(SPIRV::FunctionControl::Const);
  }
  if (F.hasFnAttribute(Attribute::AttrKind::NoInline)) {
    FuncControl |= static_cast<uint32_t>(SPIRV::FunctionControl::DontInline);
  }
  return FuncControl;
}

bool SPIRVCallLowering::lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                             const Function &F,
                                             ArrayRef<ArrayRef<Register>> VRegs,
                                             FunctionLoweringInfo &FLI) const {
  assert(GR && "Must initialize the SPIRV type registry before lowering args.");

  // Assign types and names to all args, and store their types for later.
  SmallVector<Register, 4> ArgTypeVRegs;
  if (VRegs.size() > 0) {
    unsigned i = 0;
    for (const auto &Arg : F.args()) {
      // Currently formal args should use single registers.
      // TODO: handle the case of multiple registers.
      if (VRegs[i].size() > 1)
        return false;
      auto *SpirvTy =
          GR->assignTypeToVReg(Arg.getType(), VRegs[i][0], MIRBuilder);
      ArgTypeVRegs.push_back(GR->getSPIRVTypeID(SpirvTy));

      if (Arg.hasName())
        buildOpName(VRegs[i][0], Arg.getName(), MIRBuilder);
      if (Arg.getType()->isPointerTy()) {
        auto DerefBytes = static_cast<unsigned>(Arg.getDereferenceableBytes());
        if (DerefBytes != 0)
          buildOpDecorate(VRegs[i][0], MIRBuilder,
                          SPIRV::Decoration::MaxByteOffset, {DerefBytes});
      }
      if (Arg.hasAttribute(Attribute::Alignment)) {
        buildOpDecorate(VRegs[i][0], MIRBuilder, SPIRV::Decoration::Alignment,
                        {static_cast<unsigned>(Arg.getParamAlignment())});
      }
      if (Arg.hasAttribute(Attribute::ReadOnly)) {
        auto Attr =
            static_cast<unsigned>(SPIRV::FunctionParameterAttribute::NoWrite);
        buildOpDecorate(VRegs[i][0], MIRBuilder,
                        SPIRV::Decoration::FuncParamAttr, {Attr});
      }
      if (Arg.hasAttribute(Attribute::ZExt)) {
        auto Attr =
            static_cast<unsigned>(SPIRV::FunctionParameterAttribute::Zext);
        buildOpDecorate(VRegs[i][0], MIRBuilder,
                        SPIRV::Decoration::FuncParamAttr, {Attr});
      }
      ++i;
    }
  }

  // Generate a SPIR-V type for the function.
  auto MRI = MIRBuilder.getMRI();
  Register FuncVReg = MRI->createGenericVirtualRegister(LLT::scalar(32));
  MRI->setRegClass(FuncVReg, &SPIRV::IDRegClass);

  auto *FTy = F.getFunctionType();
  auto FuncTy = GR->assignTypeToVReg(FTy, FuncVReg, MIRBuilder);

  // Build the OpTypeFunction declaring it.
  Register ReturnTypeID = FuncTy->getOperand(1).getReg();
  uint32_t FuncControl = getFunctionControl(F);

  MIRBuilder.buildInstr(SPIRV::OpFunction)
      .addDef(FuncVReg)
      .addUse(ReturnTypeID)
      .addImm(FuncControl)
      .addUse(GR->getSPIRVTypeID(FuncTy));

  // Add OpFunctionParameters.
  const unsigned NumArgs = ArgTypeVRegs.size();
  for (unsigned i = 0; i < NumArgs; ++i) {
    assert(VRegs[i].size() == 1 && "Formal arg has multiple vregs");
    MRI->setRegClass(VRegs[i][0], &SPIRV::IDRegClass);
    MIRBuilder.buildInstr(SPIRV::OpFunctionParameter)
        .addDef(VRegs[i][0])
        .addUse(ArgTypeVRegs[i]);
  }
  // Name the function.
  if (F.hasName())
    buildOpName(FuncVReg, F.getName(), MIRBuilder);

  // Handle entry points and function linkage.
  if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
    auto MIB = MIRBuilder.buildInstr(SPIRV::OpEntryPoint)
                   .addImm(static_cast<uint32_t>(SPIRV::ExecutionModel::Kernel))
                   .addUse(FuncVReg);
    addStringImm(F.getName(), MIB);
  } else if (F.getLinkage() == GlobalValue::LinkageTypes::ExternalLinkage ||
             F.getLinkage() == GlobalValue::LinkOnceODRLinkage) {
    auto LnkTy = F.isDeclaration() ? SPIRV::LinkageType::Import
                                   : SPIRV::LinkageType::Export;
    buildOpDecorate(FuncVReg, MIRBuilder, SPIRV::Decoration::LinkageAttributes,
                    {static_cast<uint32_t>(LnkTy)}, F.getGlobalIdentifier());
  }

  return true;
}

bool SPIRVCallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                  CallLoweringInfo &Info) const {
  // Currently call returns should have single vregs.
  // TODO: handle the case of multiple registers.
  if (Info.OrigRet.Regs.size() > 1)
    return false;

  Register ResVReg =
      Info.OrigRet.Regs.empty() ? Register(0) : Info.OrigRet.Regs[0];
  // Emit a regular OpFunctionCall. If it's an externally declared function,
  // be sure to emit its type and function declaration here. It will be
  // hoisted globally later.
  if (Info.Callee.isGlobal()) {
    auto *CF = dyn_cast_or_null<const Function>(Info.Callee.getGlobal());
    // TODO: support constexpr casts and indirect calls.
    if (CF == nullptr)
      return false;
    if (CF->isDeclaration()) {
      // Emit the type info and forward function declaration to the first MBB
      // to ensure VReg definition dependencies are valid across all MBBs.
      MachineBasicBlock::iterator OldII = MIRBuilder.getInsertPt();
      MachineBasicBlock &OldBB = MIRBuilder.getMBB();
      MachineBasicBlock &FirstBB = *MIRBuilder.getMF().getBlockNumbered(0);
      MIRBuilder.setInsertPt(FirstBB, FirstBB.instr_end());

      SmallVector<ArrayRef<Register>, 8> VRegArgs;
      SmallVector<SmallVector<Register, 1>, 8> ToInsert;
      for (const Argument &Arg : CF->args()) {
        if (MIRBuilder.getDataLayout().getTypeStoreSize(Arg.getType()).isZero())
          continue; // Don't handle zero sized types.
        ToInsert.push_back({MIRBuilder.getMRI()->createGenericVirtualRegister(
            LLT::scalar(32))});
        VRegArgs.push_back(ToInsert.back());
      }
      // TODO: Reuse FunctionLoweringInfo.
      FunctionLoweringInfo FuncInfo;
      lowerFormalArguments(MIRBuilder, *CF, VRegArgs, FuncInfo);
      MIRBuilder.setInsertPt(OldBB, OldII);
    }
  }

  // Make sure there's a valid return reg, even for functions returning void.
  if (!ResVReg.isValid()) {
    ResVReg = MIRBuilder.getMRI()->createVirtualRegister(&SPIRV::IDRegClass);
  }
  SPIRVType *RetType =
      GR->assignTypeToVReg(Info.OrigRet.Ty, ResVReg, MIRBuilder);

  // Emit the OpFunctionCall and its args.
  auto MIB = MIRBuilder.buildInstr(SPIRV::OpFunctionCall)
                 .addDef(ResVReg)
                 .addUse(GR->getSPIRVTypeID(RetType))
                 .add(Info.Callee);

  for (const auto &Arg : Info.OrigArgs) {
    // Currently call args should have single vregs.
    if (Arg.Regs.size() > 1)
      return false;
    MIB.addUse(Arg.Regs[0]);
  }
  return MIB.constrainAllUses(MIRBuilder.getTII(), *ST.getRegisterInfo(),
                              *ST.getRegBankInfo());
}
