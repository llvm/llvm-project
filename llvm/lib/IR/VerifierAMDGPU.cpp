//===-- VerifierAMDGPU.cpp - AMDGPU-specific IR verification ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains AMDGPU-specific IR verification logic that was extracted
// from Verifier.cpp for code organization purposes only. These checks are
// always compiled and linked as part of LLVMCore — this is not a target-
// dependent IR verifier, which would require a different design.
//
// This file should only contain checks for AMDGPU-specific IR constructs
// (e.g. amdgcn intrinsics, AMDGPU address spaces). It must not contain
// checks for generic IR that might behave differently under AMDGPU.
//
//===----------------------------------------------------------------------===//

#include "VerifierInternal.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/AMDGPUAddrSpace.h"

using namespace llvm;

#define Check(C, ...)                                                          \
  do {                                                                         \
    if (!(C)) {                                                                \
      VS.CheckFailed(__VA_ARGS__);                                             \
      return;                                                                  \
    }                                                                          \
  } while (false)

void llvm::verifyAMDGPUModuleFlag(VerifierSupport &VS, const MDString *ID,
                                  Module::ModFlagBehavior MFB,
                                  const MDNode *Op) {
  if (ID->getString() != "amdgpu.buffer.oob.mode" &&
      ID->getString() != "amdgpu.tbuffer.oob.mode")
    return;

  Check(MFB == Module::Max,
        "'" + ID->getString() + "' module flag must use 'max' merge behaviour");
  ConstantInt *Value =
      mdconst::dyn_extract_or_null<ConstantInt>(Op->getOperand(2));
  Check(Value, "'" + ID->getString() +
                   "' module flag must have a constant integer value");
  Check(Value->getZExtValue() <= 2,
        "'" + ID->getString() + "' module flag must be 0, 1, or 2");
}

// Verify that when a function has !reqd_work_group_size metadata, it also has
// an amdgpu-flat-work-group-size attribute that matches the product of the
// reqd_work_group_size operands.
static void verifyAMDGPUReqdWorkGroupSize(VerifierSupport &VS,
                                          const Function &F) {
  // This is not required for other targets so we only check for AMDGPU.
  if (!VS.TT.isAMDGPU())
    return;

  MDNode *ReqdWorkGroupSize = F.getMetadata("reqd_work_group_size");
  if (!ReqdWorkGroupSize || ReqdWorkGroupSize->getNumOperands() != 3)
    return;

  uint64_t Product = 1;
  for (const MDOperand &Op : ReqdWorkGroupSize->operands()) {
    ConstantInt *C = mdconst::dyn_extract<ConstantInt>(Op);
    if (!C || C->getValue().getActiveBits() > 64)
      return;
    uint64_t Dim = C->getZExtValue();
    if (Dim != 0 && Product > std::numeric_limits<uint64_t>::max() / Dim)
      return;
    Product *= Dim;
  }

  Attribute FlatWorkGroupSize = F.getFnAttribute("amdgpu-flat-work-group-size");
  if (!FlatWorkGroupSize.isValid()) {
    VS.CheckFailed("reqd_work_group_size requires amdgpu-flat-work-group-size",
                   &F, ReqdWorkGroupSize);
    return;
  }

  if (!FlatWorkGroupSize.isStringAttribute()) {
    VS.CheckFailed("amdgpu-flat-work-group-size must be a string attribute",
                   &F);
    return;
  }

  StringRef AttrValue = FlatWorkGroupSize.getValueAsString();
  std::pair<StringRef, StringRef> Values = AttrValue.split(',');
  uint64_t Min = 0;
  uint64_t Max = 0;
  bool Parsed = !Values.second.contains(',') &&
                llvm::to_integer(Values.first.trim(), Min) &&
                llvm::to_integer(Values.second.trim(), Max);
  if (!Parsed) {
    VS.CheckFailed("amdgpu-flat-work-group-size must be a pair of unsigned "
                   "integers",
                   &F);
    return;
  }

  if (Min != Product || Max != Product) {
    VS.CheckFailed("amdgpu-flat-work-group-size must equal the product of "
                   "reqd_work_group_size operands",
                   &F, ReqdWorkGroupSize);
  }
}

void llvm::verifyAMDGPUFunctionMetadata(VerifierSupport &VS,
                                        const Function &F) {
  verifyAMDGPUReqdWorkGroupSize(VS, F);
}

void llvm::verifyAMDGPUAlloca(VerifierSupport &VS, const AllocaInst &AI) {
  // This is not required for other targets so we only check for AMDGPU.
  if (!VS.TT.isAMDGPU())
    return;

  if (AI.getAddressSpace() != AMDGPUAS::PRIVATE_ADDRESS)
    VS.CheckFailed("alloca on amdgpu must be in addrspace(5)", &AI);
}

bool llvm::isAMDGPUCallBrIntrinsic(Intrinsic::ID ID) {
  switch (ID) {
  default:
    return false;
  case Intrinsic::amdgcn_kill:
    return true;
  }
}

void llvm::verifyAMDGPUIntrinsicCall(VerifierSupport &VS, Intrinsic::ID ID,
                                     CallBase &Call) {
  switch (ID) {
  default:
    return;
  case Intrinsic::amdgcn_kill: {
    if (auto *CBI = dyn_cast<CallBrInst>(&Call)) {
      Check(CBI->getNumIndirectDests() == 1,
            "callbr amdgcn_kill only supports one indirect dest");
      bool Unreachable = isa<UnreachableInst>(CBI->getIndirectDest(0)->begin());
      CallInst *CI = dyn_cast<CallInst>(CBI->getIndirectDest(0)->begin());
      Check(Unreachable ||
                (CI && CI->getIntrinsicID() == Intrinsic::amdgcn_unreachable),
            "callbr amdgcn_kill indirect dest needs to be unreachable");
    }
    break;
  }
  case Intrinsic::amdgcn_cs_chain: {
    CallingConv::ID CallerCC = Call.getCaller()->getCallingConv();
    switch (CallerCC) {
    case CallingConv::AMDGPU_CS:
    case CallingConv::AMDGPU_CS_Chain:
    case CallingConv::AMDGPU_CS_ChainPreserve:
    case CallingConv::AMDGPU_ES:
    case CallingConv::AMDGPU_GS:
    case CallingConv::AMDGPU_HS:
    case CallingConv::AMDGPU_LS:
    case CallingConv::AMDGPU_VS:
      break;
    default:
      VS.CheckFailed("Intrinsic cannot be called from functions with this "
                     "calling convention",
                     &Call);
      break;
    }

    Check(Call.paramHasAttr(2, Attribute::InReg),
          "SGPR arguments must have the `inreg` attribute", &Call);
    Check(!Call.paramHasAttr(3, Attribute::InReg),
          "VGPR arguments must not have the `inreg` attribute", &Call);

    ConstantInt *FlagsArg = cast<ConstantInt>(Call.getArgOperand(4));
    Check(FlagsArg->getValue().ult(2),
          "flags must be 0 or 1 for llvm.amdgcn.cs.chain", &Call);

    Instruction *Next = Call.getNextNode();
    bool IsAMDUnreachable = isa_and_nonnull<IntrinsicInst>(Next) &&
                            cast<IntrinsicInst>(Next)->getIntrinsicID() ==
                                Intrinsic::amdgcn_unreachable;
    Check(Next && (isa<UnreachableInst>(Next) || IsAMDUnreachable),
          "llvm.amdgcn.cs.chain must be followed by unreachable", &Call);
    break;
  }
  case Intrinsic::amdgcn_init_exec_from_input: {
    const Argument *Arg = dyn_cast<Argument>(Call.getOperand(0));
    Check(Arg && Arg->hasInRegAttr(),
          "only inreg arguments to the parent function are valid as inputs to "
          "this intrinsic",
          &Call);
    break;
  }
  case Intrinsic::amdgcn_set_inactive_chain_arg: {
    CallingConv::ID CallerCC = Call.getCaller()->getCallingConv();
    switch (CallerCC) {
    case CallingConv::AMDGPU_CS_Chain:
    case CallingConv::AMDGPU_CS_ChainPreserve:
      break;
    default:
      VS.CheckFailed("Intrinsic can only be used from functions with the "
                     "amdgpu_cs_chain or amdgpu_cs_chain_preserve "
                     "calling conventions",
                     &Call);
      break;
    }

    unsigned InactiveIdx = 1;
    Check(!Call.paramHasAttr(InactiveIdx, Attribute::InReg),
          "Value for inactive lanes must not have the `inreg` attribute",
          &Call);
    Check(isa<Argument>(Call.getArgOperand(InactiveIdx)),
          "Value for inactive lanes must be a function argument", &Call);
    Check(!cast<Argument>(Call.getArgOperand(InactiveIdx))->hasInRegAttr(),
          "Value for inactive lanes must be a VGPR function argument", &Call);
    break;
  }
  case Intrinsic::amdgcn_call_whole_wave: {
    Function *F = dyn_cast<Function>(Call.getArgOperand(0));
    Check(F, "Indirect whole wave calls are not allowed", &Call);

    CallingConv::ID CC = F->getCallingConv();
    Check(CC == CallingConv::AMDGPU_Gfx_WholeWave,
          "Callee must have the amdgpu_gfx_whole_wave calling convention",
          &Call);

    Check(!F->isVarArg(), "Variadic whole wave calls are not allowed", &Call);

    Check(Call.arg_size() == F->arg_size(),
          "Call argument count must match callee argument count", &Call);

    Check(F->arg_begin()->getType()->isIntegerTy(1),
          "Callee must have i1 as its first argument", &Call);
    for (auto [CallArg, FuncArg] :
         drop_begin(zip_equal(Call.args(), F->args()))) {
      Check(CallArg->getType() == FuncArg.getType(),
            "Argument types must match", &Call);

      Check(Call.paramHasAttr(FuncArg.getArgNo(), Attribute::InReg) ==
                FuncArg.hasInRegAttr(),
            "Argument inreg attributes must match", &Call);
    }
    break;
  }
  case Intrinsic::amdgcn_s_prefetch_data: {
    Check(
        AMDGPU::isFlatGlobalAddrSpace(
            Call.getArgOperand(0)->getType()->getPointerAddressSpace()),
        "llvm.amdgcn.s.prefetch.data only supports global or constant memory");
    break;
  }
  case Intrinsic::amdgcn_load_to_lds:
  case Intrinsic::amdgcn_load_async_to_lds:
  case Intrinsic::amdgcn_global_load_lds:
  case Intrinsic::amdgcn_global_load_async_lds:
  case Intrinsic::amdgcn_raw_buffer_load_lds:
  case Intrinsic::amdgcn_raw_buffer_load_async_lds:
  case Intrinsic::amdgcn_raw_ptr_buffer_load_lds:
  case Intrinsic::amdgcn_raw_ptr_buffer_load_async_lds:
  case Intrinsic::amdgcn_struct_buffer_load_lds:
  case Intrinsic::amdgcn_struct_buffer_load_async_lds:
  case Intrinsic::amdgcn_struct_ptr_buffer_load_lds:
  case Intrinsic::amdgcn_struct_ptr_buffer_load_async_lds: {
    uint64_t Size = cast<ConstantInt>(Call.getArgOperand(2))->getZExtValue();
    Check(Size == 1 || Size == 2 || Size == 4 || Size == 12 || Size == 16,
          "invalid data size for load-to-LDS intrinsic; must be 1, 2, 4, 12, "
          "or 16",
          &Call);
    break;
  }
  case Intrinsic::amdgcn_mfma_scale_f32_16x16x128_f8f6f4:
  case Intrinsic::amdgcn_mfma_scale_f32_32x32x64_f8f6f4: {
    Value *Src0 = Call.getArgOperand(0);
    Value *Src1 = Call.getArgOperand(1);

    uint64_t CBSZ = cast<ConstantInt>(Call.getArgOperand(3))->getZExtValue();
    uint64_t BLGP = cast<ConstantInt>(Call.getArgOperand(4))->getZExtValue();
    Check(CBSZ <= 4, "invalid value for cbsz format", Call,
          Call.getArgOperand(3));
    Check(BLGP <= 4, "invalid value for blgp format", Call,
          Call.getArgOperand(4));

    auto GetFormatNumRegs = [](unsigned FormatVal) {
      switch (FormatVal) {
      case 0:
      case 1:
        return 8u;
      case 2:
      case 3:
        return 6u;
      case 4:
        return 4u;
      default:
        llvm_unreachable("invalid format value");
      }
    };

    auto IsValidSrcASrcBVector = [](FixedVectorType *Ty) {
      if (!Ty || !Ty->getElementType()->isIntegerTy(32))
        return false;
      unsigned NumElts = Ty->getNumElements();
      return NumElts == 4 || NumElts == 6 || NumElts == 8;
    };

    FixedVectorType *Src0Ty = dyn_cast<FixedVectorType>(Src0->getType());
    FixedVectorType *Src1Ty = dyn_cast<FixedVectorType>(Src1->getType());
    Check(IsValidSrcASrcBVector(Src0Ty),
          "operand 0 must be 4, 6 or 8 element i32 vector", &Call, Src0);
    Check(IsValidSrcASrcBVector(Src1Ty),
          "operand 1 must be 4, 6 or 8 element i32 vector", &Call, Src1);

    Check(Src0Ty->getNumElements() >= GetFormatNumRegs(CBSZ),
          "invalid vector type for format", &Call, Src0, Call.getArgOperand(3));
    Check(Src1Ty->getNumElements() >= GetFormatNumRegs(BLGP),
          "invalid vector type for format", &Call, Src1, Call.getArgOperand(5));
    break;
  }
  case Intrinsic::amdgcn_wmma_f32_16x16x128_f8f6f4:
  case Intrinsic::amdgcn_wmma_scale_f32_16x16x128_f8f6f4:
  case Intrinsic::amdgcn_wmma_scale16_f32_16x16x128_f8f6f4: {
    Value *Src0 = Call.getArgOperand(1);
    Value *Src1 = Call.getArgOperand(3);

    unsigned FmtA = cast<ConstantInt>(Call.getArgOperand(0))->getZExtValue();
    unsigned FmtB = cast<ConstantInt>(Call.getArgOperand(2))->getZExtValue();
    Check(FmtA <= 4, "invalid value for matrix format", Call,
          Call.getArgOperand(0));
    Check(FmtB <= 4, "invalid value for matrix format", Call,
          Call.getArgOperand(2));

    auto GetFormatNumRegs = [](unsigned FormatVal) {
      switch (FormatVal) {
      case 0:
      case 1:
        return 16u;
      case 2:
      case 3:
        return 12u;
      case 4:
        return 8u;
      default:
        llvm_unreachable("invalid format value");
      }
    };

    auto IsValidSrcASrcBVector = [](FixedVectorType *Ty) {
      if (!Ty || !Ty->getElementType()->isIntegerTy(32))
        return false;
      unsigned NumElts = Ty->getNumElements();
      return NumElts == 16 || NumElts == 12 || NumElts == 8;
    };

    FixedVectorType *Src0Ty = dyn_cast<FixedVectorType>(Src0->getType());
    FixedVectorType *Src1Ty = dyn_cast<FixedVectorType>(Src1->getType());
    Check(IsValidSrcASrcBVector(Src0Ty),
          "operand 1 must be 8, 12 or 16 element i32 vector", &Call, Src0);
    Check(IsValidSrcASrcBVector(Src1Ty),
          "operand 3 must be 8, 12 or 16 element i32 vector", &Call, Src1);

    Check(Src0Ty->getNumElements() >= GetFormatNumRegs(FmtA),
          "invalid vector type for format", &Call, Src0, Call.getArgOperand(0));
    Check(Src1Ty->getNumElements() >= GetFormatNumRegs(FmtB),
          "invalid vector type for format", &Call, Src1, Call.getArgOperand(2));
    break;
  }
  case Intrinsic::amdgcn_cooperative_atomic_load_32x4B:
  case Intrinsic::amdgcn_cooperative_atomic_load_16x8B:
  case Intrinsic::amdgcn_cooperative_atomic_load_8x16B:
  case Intrinsic::amdgcn_cooperative_atomic_store_32x4B:
  case Intrinsic::amdgcn_cooperative_atomic_store_16x8B:
  case Intrinsic::amdgcn_cooperative_atomic_store_8x16B: {
    Value *PtrArg = Call.getArgOperand(0);
    const unsigned AS = PtrArg->getType()->getPointerAddressSpace();
    Check(AS == AMDGPUAS::FLAT_ADDRESS || AS == AMDGPUAS::GLOBAL_ADDRESS,
          "cooperative atomic intrinsics require a generic or global pointer",
          &Call, PtrArg);

    MetadataAsValue *Op =
        cast<MetadataAsValue>(Call.getArgOperand(Call.arg_size() - 1));
    MDNode *MD = cast<MDNode>(Op->getMetadata());
    Check((MD->getNumOperands() == 1) && isa<MDString>(MD->getOperand(0)),
          "cooperative atomic intrinsics require that the last argument is a "
          "metadata string",
          &Call, Op);
    break;
  }
  case Intrinsic::amdgcn_av_load_b128:
  case Intrinsic::amdgcn_av_store_b128: {
    MetadataAsValue *Op =
        cast<MetadataAsValue>(Call.getArgOperand(Call.arg_size() - 1));
    MDNode *MD = dyn_cast<MDNode>(Op->getMetadata());
    Check(MD && (MD->getNumOperands() == 1) && isa<MDString>(MD->getOperand(0)),
          "the last argument to av load/store intrinsics must be a "
          "metadata string",
          &Call, Op);
    break;
  }
  }
}

#undef Check
