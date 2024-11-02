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
#include "SPIRVBuiltins.h"
#include "SPIRVGlobalRegistry.h"
#include "SPIRVISelLowering.h"
#include "SPIRVRegisterInfo.h"
#include "SPIRVSubtarget.h"
#include "SPIRVUtils.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"

using namespace llvm;

SPIRVCallLowering::SPIRVCallLowering(const SPIRVTargetLowering &TLI,
                                     SPIRVGlobalRegistry *GR)
    : CallLowering(&TLI), GR(GR) {}

bool SPIRVCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                    const Value *Val, ArrayRef<Register> VRegs,
                                    FunctionLoweringInfo &FLI,
                                    Register SwiftErrorVReg) const {
  // Currently all return types should use a single register.
  // TODO: handle the case of multiple registers.
  if (VRegs.size() > 1)
    return false;
  if (Val) {
    const auto &STI = MIRBuilder.getMF().getSubtarget();
    return MIRBuilder.buildInstr(SPIRV::OpReturnValue)
        .addUse(VRegs[0])
        .constrainAllUses(MIRBuilder.getTII(), *STI.getRegisterInfo(),
                          *STI.getRegBankInfo());
  }
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

static ConstantInt *getConstInt(MDNode *MD, unsigned NumOp) {
  if (MD->getNumOperands() > NumOp) {
    auto *CMeta = dyn_cast<ConstantAsMetadata>(MD->getOperand(NumOp));
    if (CMeta)
      return dyn_cast<ConstantInt>(CMeta->getValue());
  }
  return nullptr;
}

// This code restores function args/retvalue types for composite cases
// because the final types should still be aggregate whereas they're i32
// during the translation to cope with aggregate flattening etc.
static FunctionType *getOriginalFunctionType(const Function &F) {
  auto *NamedMD = F.getParent()->getNamedMetadata("spv.cloned_funcs");
  if (NamedMD == nullptr)
    return F.getFunctionType();

  Type *RetTy = F.getFunctionType()->getReturnType();
  SmallVector<Type *, 4> ArgTypes;
  for (auto &Arg : F.args())
    ArgTypes.push_back(Arg.getType());

  auto ThisFuncMDIt =
      std::find_if(NamedMD->op_begin(), NamedMD->op_end(), [&F](MDNode *N) {
        return isa<MDString>(N->getOperand(0)) &&
               cast<MDString>(N->getOperand(0))->getString() == F.getName();
      });
  // TODO: probably one function can have numerous type mutations,
  // so we should support this.
  if (ThisFuncMDIt != NamedMD->op_end()) {
    auto *ThisFuncMD = *ThisFuncMDIt;
    MDNode *MD = dyn_cast<MDNode>(ThisFuncMD->getOperand(1));
    assert(MD && "MDNode operand is expected");
    ConstantInt *Const = getConstInt(MD, 0);
    if (Const) {
      auto *CMeta = dyn_cast<ConstantAsMetadata>(MD->getOperand(1));
      assert(CMeta && "ConstantAsMetadata operand is expected");
      assert(Const->getSExtValue() >= -1);
      // Currently -1 indicates return value, greater values mean
      // argument numbers.
      if (Const->getSExtValue() == -1)
        RetTy = CMeta->getType();
      else
        ArgTypes[Const->getSExtValue()] = CMeta->getType();
    }
  }

  return FunctionType::get(RetTy, ArgTypes, F.isVarArg());
}

static MDString *getKernelArgAttribute(const Function &KernelFunction,
                                       unsigned ArgIdx,
                                       const StringRef AttributeName) {
  assert(KernelFunction.getCallingConv() == CallingConv::SPIR_KERNEL &&
         "Kernel attributes are attached/belong only to kernel functions");

  // Lookup the argument attribute in metadata attached to the kernel function.
  MDNode *Node = KernelFunction.getMetadata(AttributeName);
  if (Node && ArgIdx < Node->getNumOperands())
    return cast<MDString>(Node->getOperand(ArgIdx));

  // Sometimes metadata containing kernel attributes is not attached to the
  // function, but can be found in the named module-level metadata instead.
  // For example:
  //   !opencl.kernels = !{!0}
  //   !0 = !{void ()* @someKernelFunction, !1, ...}
  //   !1 = !{!"kernel_arg_addr_space", ...}
  // In this case the actual index of searched argument attribute is ArgIdx + 1,
  // since the first metadata node operand is occupied by attribute name
  // ("kernel_arg_addr_space" in the example above).
  unsigned MDArgIdx = ArgIdx + 1;
  NamedMDNode *OpenCLKernelsMD =
      KernelFunction.getParent()->getNamedMetadata("opencl.kernels");
  if (!OpenCLKernelsMD || OpenCLKernelsMD->getNumOperands() == 0)
    return nullptr;

  // KernelToMDNodeList contains kernel function declarations followed by
  // corresponding MDNodes for each attribute. Search only MDNodes "belonging"
  // to the currently lowered kernel function.
  MDNode *KernelToMDNodeList = OpenCLKernelsMD->getOperand(0);
  bool FoundLoweredKernelFunction = false;
  for (const MDOperand &Operand : KernelToMDNodeList->operands()) {
    ValueAsMetadata *MaybeValue = dyn_cast<ValueAsMetadata>(Operand);
    if (MaybeValue && dyn_cast<Function>(MaybeValue->getValue())->getName() ==
                          KernelFunction.getName()) {
      FoundLoweredKernelFunction = true;
      continue;
    }
    if (MaybeValue && FoundLoweredKernelFunction)
      return nullptr;

    MDNode *MaybeNode = dyn_cast<MDNode>(Operand);
    if (FoundLoweredKernelFunction && MaybeNode &&
        cast<MDString>(MaybeNode->getOperand(0))->getString() ==
            AttributeName &&
        MDArgIdx < MaybeNode->getNumOperands())
      return cast<MDString>(MaybeNode->getOperand(MDArgIdx));
  }
  return nullptr;
}

static SPIRV::AccessQualifier::AccessQualifier
getArgAccessQual(const Function &F, unsigned ArgIdx) {
  if (F.getCallingConv() != CallingConv::SPIR_KERNEL)
    return SPIRV::AccessQualifier::ReadWrite;

  MDString *ArgAttribute =
      getKernelArgAttribute(F, ArgIdx, "kernel_arg_access_qual");
  if (!ArgAttribute)
    return SPIRV::AccessQualifier::ReadWrite;

  if (ArgAttribute->getString().compare("read_only") == 0)
    return SPIRV::AccessQualifier::ReadOnly;
  if (ArgAttribute->getString().compare("write_only") == 0)
    return SPIRV::AccessQualifier::WriteOnly;
  return SPIRV::AccessQualifier::ReadWrite;
}

static std::vector<SPIRV::Decoration::Decoration>
getKernelArgTypeQual(const Function &KernelFunction, unsigned ArgIdx) {
  MDString *ArgAttribute =
      getKernelArgAttribute(KernelFunction, ArgIdx, "kernel_arg_type_qual");
  if (ArgAttribute && ArgAttribute->getString().compare("volatile") == 0)
    return {SPIRV::Decoration::Volatile};
  return {};
}

static Type *getArgType(const Function &F, unsigned ArgIdx) {
  Type *OriginalArgType = getOriginalFunctionType(F)->getParamType(ArgIdx);
  if (F.getCallingConv() != CallingConv::SPIR_KERNEL ||
      isSpecialOpaqueType(OriginalArgType))
    return OriginalArgType;

  MDString *MDKernelArgType =
      getKernelArgAttribute(F, ArgIdx, "kernel_arg_type");
  if (!MDKernelArgType || !MDKernelArgType->getString().endswith("_t"))
    return OriginalArgType;

  std::string KernelArgTypeStr = "opencl." + MDKernelArgType->getString().str();
  Type *ExistingOpaqueType =
      StructType::getTypeByName(F.getContext(), KernelArgTypeStr);
  return ExistingOpaqueType
             ? ExistingOpaqueType
             : StructType::create(F.getContext(), KernelArgTypeStr);
}

bool SPIRVCallLowering::lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                             const Function &F,
                                             ArrayRef<ArrayRef<Register>> VRegs,
                                             FunctionLoweringInfo &FLI) const {
  assert(GR && "Must initialize the SPIRV type registry before lowering args.");
  GR->setCurrentFunc(MIRBuilder.getMF());

  // Assign types and names to all args, and store their types for later.
  FunctionType *FTy = getOriginalFunctionType(F);
  SmallVector<SPIRVType *, 4> ArgTypeVRegs;
  if (VRegs.size() > 0) {
    unsigned i = 0;
    for (const auto &Arg : F.args()) {
      // Currently formal args should use single registers.
      // TODO: handle the case of multiple registers.
      if (VRegs[i].size() > 1)
        return false;
      SPIRV::AccessQualifier::AccessQualifier ArgAccessQual =
          getArgAccessQual(F, i);
      auto *SpirvTy = GR->assignTypeToVReg(getArgType(F, i), VRegs[i][0],
                                           MIRBuilder, ArgAccessQual);
      ArgTypeVRegs.push_back(SpirvTy);

      if (Arg.hasName())
        buildOpName(VRegs[i][0], Arg.getName(), MIRBuilder);
      if (Arg.getType()->isPointerTy()) {
        auto DerefBytes = static_cast<unsigned>(Arg.getDereferenceableBytes());
        if (DerefBytes != 0)
          buildOpDecorate(VRegs[i][0], MIRBuilder,
                          SPIRV::Decoration::MaxByteOffset, {DerefBytes});
      }
      if (Arg.hasAttribute(Attribute::Alignment)) {
        auto Alignment = static_cast<unsigned>(
            Arg.getAttribute(Attribute::Alignment).getValueAsInt());
        buildOpDecorate(VRegs[i][0], MIRBuilder, SPIRV::Decoration::Alignment,
                        {Alignment});
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
      if (Arg.hasAttribute(Attribute::NoAlias)) {
        auto Attr =
            static_cast<unsigned>(SPIRV::FunctionParameterAttribute::NoAlias);
        buildOpDecorate(VRegs[i][0], MIRBuilder,
                        SPIRV::Decoration::FuncParamAttr, {Attr});
      }

      if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
        std::vector<SPIRV::Decoration::Decoration> ArgTypeQualDecs =
            getKernelArgTypeQual(F, i);
        for (SPIRV::Decoration::Decoration Decoration : ArgTypeQualDecs)
          buildOpDecorate(VRegs[i][0], MIRBuilder, Decoration, {});
      }

      MDNode *Node = F.getMetadata("spirv.ParameterDecorations");
      if (Node && i < Node->getNumOperands() &&
          isa<MDNode>(Node->getOperand(i))) {
        MDNode *MD = cast<MDNode>(Node->getOperand(i));
        for (const MDOperand &MDOp : MD->operands()) {
          MDNode *MD2 = dyn_cast<MDNode>(MDOp);
          assert(MD2 && "Metadata operand is expected");
          ConstantInt *Const = getConstInt(MD2, 0);
          assert(Const && "MDOperand should be ConstantInt");
          auto Dec =
              static_cast<SPIRV::Decoration::Decoration>(Const->getZExtValue());
          std::vector<uint32_t> DecVec;
          for (unsigned j = 1; j < MD2->getNumOperands(); j++) {
            ConstantInt *Const = getConstInt(MD2, j);
            assert(Const && "MDOperand should be ConstantInt");
            DecVec.push_back(static_cast<uint32_t>(Const->getZExtValue()));
          }
          buildOpDecorate(VRegs[i][0], MIRBuilder, Dec, DecVec);
        }
      }
      ++i;
    }
  }

  // Generate a SPIR-V type for the function.
  auto MRI = MIRBuilder.getMRI();
  Register FuncVReg = MRI->createGenericVirtualRegister(LLT::scalar(32));
  MRI->setRegClass(FuncVReg, &SPIRV::IDRegClass);
  if (F.isDeclaration())
    GR->add(&F, &MIRBuilder.getMF(), FuncVReg);
  SPIRVType *RetTy = GR->getOrCreateSPIRVType(FTy->getReturnType(), MIRBuilder);
  SPIRVType *FuncTy = GR->getOrCreateOpTypeFunctionWithArgs(
      FTy, RetTy, ArgTypeVRegs, MIRBuilder);

  // Build the OpTypeFunction declaring it.
  uint32_t FuncControl = getFunctionControl(F);

  MIRBuilder.buildInstr(SPIRV::OpFunction)
      .addDef(FuncVReg)
      .addUse(GR->getSPIRVTypeID(RetTy))
      .addImm(FuncControl)
      .addUse(GR->getSPIRVTypeID(FuncTy));

  // Add OpFunctionParameters.
  int i = 0;
  for (const auto &Arg : F.args()) {
    assert(VRegs[i].size() == 1 && "Formal arg has multiple vregs");
    MRI->setRegClass(VRegs[i][0], &SPIRV::IDRegClass);
    MIRBuilder.buildInstr(SPIRV::OpFunctionParameter)
        .addDef(VRegs[i][0])
        .addUse(GR->getSPIRVTypeID(ArgTypeVRegs[i]));
    if (F.isDeclaration())
      GR->add(&Arg, &MIRBuilder.getMF(), VRegs[i][0]);
    i++;
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
  MachineFunction &MF = MIRBuilder.getMF();
  GR->setCurrentFunc(MF);
  FunctionType *FTy = nullptr;
  const Function *CF = nullptr;

  // Emit a regular OpFunctionCall. If it's an externally declared function,
  // be sure to emit its type and function declaration here. It will be hoisted
  // globally later.
  if (Info.Callee.isGlobal()) {
    CF = dyn_cast_or_null<const Function>(Info.Callee.getGlobal());
    // TODO: support constexpr casts and indirect calls.
    if (CF == nullptr)
      return false;
    FTy = getOriginalFunctionType(*CF);
  }

  Register ResVReg =
      Info.OrigRet.Regs.empty() ? Register(0) : Info.OrigRet.Regs[0];
  std::string FuncName = Info.Callee.getGlobal()->getName().str();
  std::string DemangledName = getOclOrSpirvBuiltinDemangledName(FuncName);
  const auto *ST = static_cast<const SPIRVSubtarget *>(&MF.getSubtarget());
  // TODO: check that it's OCL builtin, then apply OpenCL_std.
  if (!DemangledName.empty() && CF && CF->isDeclaration() &&
      ST->canUseExtInstSet(SPIRV::InstructionSet::OpenCL_std)) {
    const Type *OrigRetTy = Info.OrigRet.Ty;
    if (FTy)
      OrigRetTy = FTy->getReturnType();
    SmallVector<Register, 8> ArgVRegs;
    for (auto Arg : Info.OrigArgs) {
      assert(Arg.Regs.size() == 1 && "Call arg has multiple VRegs");
      ArgVRegs.push_back(Arg.Regs[0]);
      SPIRVType *SPIRVTy = GR->getOrCreateSPIRVType(Arg.Ty, MIRBuilder);
      GR->assignSPIRVTypeToVReg(SPIRVTy, Arg.Regs[0], MIRBuilder.getMF());
    }
    if (auto Res = SPIRV::lowerBuiltin(
            DemangledName, SPIRV::InstructionSet::OpenCL_std, MIRBuilder,
            ResVReg, OrigRetTy, ArgVRegs, GR))
      return *Res;
  }
  if (CF && CF->isDeclaration() &&
      !GR->find(CF, &MIRBuilder.getMF()).isValid()) {
    // Emit the type info and forward function declaration to the first MBB
    // to ensure VReg definition dependencies are valid across all MBBs.
    MachineIRBuilder FirstBlockBuilder;
    FirstBlockBuilder.setMF(MF);
    FirstBlockBuilder.setMBB(*MF.getBlockNumbered(0));

    SmallVector<ArrayRef<Register>, 8> VRegArgs;
    SmallVector<SmallVector<Register, 1>, 8> ToInsert;
    for (const Argument &Arg : CF->args()) {
      if (MIRBuilder.getDataLayout().getTypeStoreSize(Arg.getType()).isZero())
        continue; // Don't handle zero sized types.
      ToInsert.push_back(
          {MIRBuilder.getMRI()->createGenericVirtualRegister(LLT::scalar(32))});
      VRegArgs.push_back(ToInsert.back());
    }
    // TODO: Reuse FunctionLoweringInfo
    FunctionLoweringInfo FuncInfo;
    lowerFormalArguments(FirstBlockBuilder, *CF, VRegArgs, FuncInfo);
  }

  // Make sure there's a valid return reg, even for functions returning void.
  if (!ResVReg.isValid())
    ResVReg = MIRBuilder.getMRI()->createVirtualRegister(&SPIRV::IDRegClass);
  SPIRVType *RetType =
      GR->assignTypeToVReg(FTy->getReturnType(), ResVReg, MIRBuilder);

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
  return MIB.constrainAllUses(MIRBuilder.getTII(), *ST->getRegisterInfo(),
                              *ST->getRegBankInfo());
}
