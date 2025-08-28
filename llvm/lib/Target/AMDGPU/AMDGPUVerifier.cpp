//===- AMDGPUVerifier.h ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// IR verifier plugin for AMDGPU intrinsics.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Verifier.h"

using namespace llvm;

namespace {

#define Check(C, ...)                                                          \
  do {                                                                         \
    if (!(C)) {                                                                \
      VS.CheckFailed(__VA_ARGS__);                                             \
      return;                                                                  \
    }                                                                          \
  } while (false)

class AMDGPUVerifier : public VerifierPlugin {
public:
  void verifyIntrinsicCall(CallBase &Call, VerifierSupport &VS) const override {
    switch (Call.getIntrinsicID()) {
    default:
      break;
    case Intrinsic::amdgcn_cs_chain: {
      auto CallerCC = Call.getCaller()->getCallingConv();
      switch (CallerCC) {
      case CallingConv::AMDGPU_CS:
      case CallingConv::AMDGPU_CS_Chain:
      case CallingConv::AMDGPU_CS_ChainPreserve:
        break;
      default:
        VS.CheckFailed("Intrinsic can only be used from functions with the "
                       "amdgpu_cs, amdgpu_cs_chain or amdgpu_cs_chain_preserve "
                       "calling conventions",
                       &Call);
        break;
      }

      Check(Call.paramHasAttr(2, Attribute::InReg),
            "SGPR arguments must have the `inreg` attribute", &Call);
      Check(!Call.paramHasAttr(3, Attribute::InReg),
            "VGPR arguments must not have the `inreg` attribute", &Call);

      auto *Next = Call.getNextNode();
      bool IsAMDUnreachable = Next && isa<IntrinsicInst>(Next) &&
                              cast<IntrinsicInst>(Next)->getIntrinsicID() ==
                                  Intrinsic::amdgcn_unreachable;
      Check(Next && (isa<UnreachableInst>(Next) || IsAMDUnreachable),
            "llvm.amdgcn.cs.chain must be followed by unreachable", &Call);
      break;
    }
    case Intrinsic::amdgcn_init_exec_from_input: {
      const Argument *Arg = dyn_cast<Argument>(Call.getOperand(0));
      Check(
          Arg && Arg->hasInRegAttr(),
          "only inreg arguments to the parent function are valid as inputs to "
          "this intrinsic",
          &Call);
      break;
    }
    case Intrinsic::amdgcn_set_inactive_chain_arg: {
      auto CallerCC = Call.getCaller()->getCallingConv();
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
      auto F = dyn_cast<Function>(Call.getArgOperand(0));
      Check(F, "Indirect whole wave calls are not allowed", &Call);

      CallingConv::ID CC = F->getCallingConv();
      Check(CC == CallingConv::AMDGPU_Gfx_WholeWave,
            "Callee must have the amdgpu_gfx_whole_wave calling convention",
            &Call);

      Check(!F->isVarArg(), "Variadic whole wave calls are not allowed", &Call);

      Check(Call.arg_size() == F->arg_size(),
            "Call argument count must match callee argument count", &Call);

      // The first argument of the call is the callee, and the first argument of
      // the callee is the active mask. The rest of the arguments must match.
      Check(F->arg_begin()->getType()->isIntegerTy(1),
            "Callee must have i1 as its first argument", &Call);
      for (auto [CallArg, FuncArg] :
           drop_begin(zip_equal(Call.args(), F->args()))) {
        Check(CallArg->getType() == FuncArg.getType(),
              "Argument types must match", &Call);

        // Check that inreg attributes match between call site and function
        Check(Call.paramHasAttr(FuncArg.getArgNo(), Attribute::InReg) ==
                  FuncArg.hasInRegAttr(),
              "Argument inreg attributes must match", &Call);
      }
      break;
    }
    case Intrinsic::amdgcn_s_prefetch_data: {
      Check(AMDGPU::isFlatGlobalAddrSpace(
                Call.getArgOperand(0)->getType()->getPointerAddressSpace()),
            "llvm.amdgcn.s.prefetch.data only supports global or constant "
            "memory");
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

      // AMDGPU::MFMAScaleFormats values
      auto getFormatNumRegs = [](unsigned FormatVal) {
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

      auto isValidSrcASrcBVector = [](FixedVectorType *Ty) {
        if (!Ty || !Ty->getElementType()->isIntegerTy(32))
          return false;
        unsigned NumElts = Ty->getNumElements();
        return NumElts == 4 || NumElts == 6 || NumElts == 8;
      };

      auto *Src0Ty = dyn_cast<FixedVectorType>(Src0->getType());
      auto *Src1Ty = dyn_cast<FixedVectorType>(Src1->getType());
      Check(isValidSrcASrcBVector(Src0Ty),
            "operand 0 must be 4, 6 or 8 element i32 vector", &Call, Src0);
      Check(isValidSrcASrcBVector(Src1Ty),
            "operand 1 must be 4, 6 or 8 element i32 vector", &Call, Src1);

      // Permit excess registers for the format.
      Check(Src0Ty->getNumElements() >= getFormatNumRegs(CBSZ),
            "invalid vector type for format", &Call, Src0,
            Call.getArgOperand(3));
      Check(Src1Ty->getNumElements() >= getFormatNumRegs(BLGP),
            "invalid vector type for format", &Call, Src1,
            Call.getArgOperand(5));
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

      // AMDGPU::MatrixFMT values
      auto getFormatNumRegs = [](unsigned FormatVal) {
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

      auto isValidSrcASrcBVector = [](FixedVectorType *Ty) {
        if (!Ty || !Ty->getElementType()->isIntegerTy(32))
          return false;
        unsigned NumElts = Ty->getNumElements();
        return NumElts == 16 || NumElts == 12 || NumElts == 8;
      };

      auto *Src0Ty = dyn_cast<FixedVectorType>(Src0->getType());
      auto *Src1Ty = dyn_cast<FixedVectorType>(Src1->getType());
      Check(isValidSrcASrcBVector(Src0Ty),
            "operand 1 must be 8, 12 or 16 element i32 vector", &Call, Src0);
      Check(isValidSrcASrcBVector(Src1Ty),
            "operand 3 must be 8, 12 or 16 element i32 vector", &Call, Src1);

      // Permit excess registers for the format.
      Check(Src0Ty->getNumElements() >= getFormatNumRegs(FmtA),
            "invalid vector type for format", &Call, Src0,
            Call.getArgOperand(0));
      Check(Src1Ty->getNumElements() >= getFormatNumRegs(FmtB),
            "invalid vector type for format", &Call, Src1,
            Call.getArgOperand(2));
      break;
    }
    case Intrinsic::amdgcn_cooperative_atomic_load_32x4B:
    case Intrinsic::amdgcn_cooperative_atomic_load_16x8B:
    case Intrinsic::amdgcn_cooperative_atomic_load_8x16B:
    case Intrinsic::amdgcn_cooperative_atomic_store_32x4B:
    case Intrinsic::amdgcn_cooperative_atomic_store_16x8B:
    case Intrinsic::amdgcn_cooperative_atomic_store_8x16B: {
      // Check we only use this intrinsic on the FLAT or GLOBAL address spaces.
      Value *PtrArg = Call.getArgOperand(0);
      const unsigned AS = PtrArg->getType()->getPointerAddressSpace();
      Check(AS == AMDGPUAS::FLAT_ADDRESS || AS == AMDGPUAS::GLOBAL_ADDRESS,
            "cooperative atomic intrinsics require a generic or global pointer",
            &Call, PtrArg);

      // Last argument must be a MD string
      auto *Op = cast<MetadataAsValue>(Call.getArgOperand(Call.arg_size() - 1));
      MDNode *MD = cast<MDNode>(Op->getMetadata());
      Check((MD->getNumOperands() == 1) && isa<MDString>(MD->getOperand(0)),
            "cooperative atomic intrinsics require that the last argument is a "
            "metadata string",
            &Call, Op);
      break;
    }
    }
  }
};

} // anonymous namespace

void llvm::initializeAMDGPUVerifier() { static AMDGPUVerifier TheVerifier; }
