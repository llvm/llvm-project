//===-- R600TextureIntrinsicsReplacer.cpp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass translates tgsi-like texture intrinsics into R600 texture
/// closer to hardware intrinsics.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"

using namespace llvm;

namespace {
class R600TextureIntrinsicsReplacer :
    public FunctionPass, public InstVisitor<R600TextureIntrinsicsReplacer> {
  static char ID;

  Module *Mod;
  Type *FloatType;
  Type *Int32Type;
  Type *V4f32Type;
  Type *V4i32Type;
  FunctionType *TexSign;
  FunctionType *TexQSign;

  void getAdjustmentFromTextureTarget(unsigned TextureType, bool hasLOD,
                                      unsigned SrcSelect[4], unsigned CT[4],
                                      bool &useShadowVariant) {
    enum TextureTypes {
      TEXTURE_1D = 1,
      TEXTURE_2D,
      TEXTURE_3D,
      TEXTURE_CUBE,
      TEXTURE_RECT,
      TEXTURE_SHADOW1D,
      TEXTURE_SHADOW2D,
      TEXTURE_SHADOWRECT,
      TEXTURE_1D_ARRAY,
      TEXTURE_2D_ARRAY,
      TEXTURE_SHADOW1D_ARRAY,
      TEXTURE_SHADOW2D_ARRAY,
      TEXTURE_SHADOWCUBE,
      TEXTURE_2D_MSAA,
      TEXTURE_2D_ARRAY_MSAA,
      TEXTURE_CUBE_ARRAY,
      TEXTURE_SHADOWCUBE_ARRAY
    };

    switch (TextureType) {
    case 0:
      useShadowVariant = false;
      return;
    case TEXTURE_RECT:
    case TEXTURE_1D:
    case TEXTURE_2D:
    case TEXTURE_3D:
    case TEXTURE_CUBE:
    case TEXTURE_1D_ARRAY:
    case TEXTURE_2D_ARRAY:
    case TEXTURE_CUBE_ARRAY:
    case TEXTURE_2D_MSAA:
    case TEXTURE_2D_ARRAY_MSAA:
      useShadowVariant = false;
      break;
    case TEXTURE_SHADOW1D:
    case TEXTURE_SHADOW2D:
    case TEXTURE_SHADOWRECT:
    case TEXTURE_SHADOW1D_ARRAY:
    case TEXTURE_SHADOW2D_ARRAY:
    case TEXTURE_SHADOWCUBE:
    case TEXTURE_SHADOWCUBE_ARRAY:
      useShadowVariant = true;
      break;
    default:
      llvm_unreachable("Unknow Texture Type");
    }

    if (TextureType == TEXTURE_RECT ||
        TextureType == TEXTURE_SHADOWRECT) {
      CT[0] = 0;
      CT[1] = 0;
    }

    if (TextureType == TEXTURE_CUBE_ARRAY ||
        TextureType == TEXTURE_SHADOWCUBE_ARRAY)
      CT[2] = 0;

    if (TextureType == TEXTURE_1D_ARRAY ||
        TextureType == TEXTURE_SHADOW1D_ARRAY) {
      if (hasLOD && useShadowVariant) {
        CT[1] = 0;
      } else {
        CT[2] = 0;
        SrcSelect[2] = 1;
      }
    } else if (TextureType == TEXTURE_2D_ARRAY ||
        TextureType == TEXTURE_SHADOW2D_ARRAY) {
      CT[2] = 0;
    }

    if ((TextureType == TEXTURE_SHADOW1D ||
        TextureType == TEXTURE_SHADOW2D ||
        TextureType == TEXTURE_SHADOWRECT ||
        TextureType == TEXTURE_SHADOW1D_ARRAY) &&
        !(hasLOD && useShadowVariant))
      SrcSelect[3] = 2;
  }

  void ReplaceCallInst(CallInst &I, FunctionType *FT, const char *Name,
                       unsigned SrcSelect[4], Value *Offset[3], Value *Resource,
                       Value *Sampler, unsigned CT[4], Value *Coord) {
    IRBuilder<> Builder(&I);
    Constant *Mask[] = {
      ConstantInt::get(Int32Type, SrcSelect[0]),
      ConstantInt::get(Int32Type, SrcSelect[1]),
      ConstantInt::get(Int32Type, SrcSelect[2]),
      ConstantInt::get(Int32Type, SrcSelect[3])
    };
    Value *SwizzleMask = ConstantVector::get(Mask);
    Value *SwizzledCoord =
        Builder.CreateShuffleVector(Coord, Coord, SwizzleMask);

    Value *Args[] = {
      SwizzledCoord,
      Offset[0],
      Offset[1],
      Offset[2],
      Resource,
      Sampler,
      ConstantInt::get(Int32Type, CT[0]),
      ConstantInt::get(Int32Type, CT[1]),
      ConstantInt::get(Int32Type, CT[2]),
      ConstantInt::get(Int32Type, CT[3])
    };

    Function *F = Mod->getFunction(Name);
    if (!F) {
      F = Function::Create(FT, GlobalValue::ExternalLinkage, Name, Mod);
      F->addFnAttr(Attribute::ReadNone);
    }
    I.replaceAllUsesWith(Builder.CreateCall(F, Args));
    I.eraseFromParent();
  }

  void ReplaceTexIntrinsic(CallInst &I, bool hasLOD, FunctionType *FT,
                           const char *VanillaInt,
                           const char *ShadowInt) {
    Value *Coord = I.getArgOperand(0);
    Value *ResourceId = I.getArgOperand(1);
    Value *SamplerId = I.getArgOperand(2);

    unsigned TextureType =
        cast<ConstantInt>(I.getArgOperand(3))->getZExtValue();

    unsigned SrcSelect[4] = { 0, 1, 2, 3 };
    unsigned CT[4] = {1, 1, 1, 1};
    Value *Offset[3] = {
      ConstantInt::get(Int32Type, 0),
      ConstantInt::get(Int32Type, 0),
      ConstantInt::get(Int32Type, 0)
    };
    bool useShadowVariant;

    getAdjustmentFromTextureTarget(TextureType, hasLOD, SrcSelect, CT,
                                   useShadowVariant);

    ReplaceCallInst(I, FT, useShadowVariant?ShadowInt:VanillaInt, SrcSelect,
                    Offset, ResourceId, SamplerId, CT, Coord);
  }

  void ReplaceTXF(CallInst &I) {
    Value *Coord = I.getArgOperand(0);
    Value *ResourceId = I.getArgOperand(4);
    Value *SamplerId = I.getArgOperand(5);

    unsigned TextureType =
        cast<ConstantInt>(I.getArgOperand(6))->getZExtValue();

    unsigned SrcSelect[4] = { 0, 1, 2, 3 };
    unsigned CT[4] = {1, 1, 1, 1};
    Value *Offset[3] = {
      I.getArgOperand(1),
      I.getArgOperand(2),
      I.getArgOperand(3),
    };
    bool useShadowVariant;

    getAdjustmentFromTextureTarget(TextureType, false, SrcSelect, CT,
                                   useShadowVariant);

    ReplaceCallInst(I, TexQSign, "llvm.R600.txf", SrcSelect,
                    Offset, ResourceId, SamplerId, CT, Coord);
  }

public:
  R600TextureIntrinsicsReplacer():
    FunctionPass(ID) {
  }

  bool doInitialization(Module &M) override {
    LLVMContext &Ctx = M.getContext();
    Mod = &M;
    FloatType = Type::getFloatTy(Ctx);
    Int32Type = Type::getInt32Ty(Ctx);
    V4f32Type = VectorType::get(FloatType, 4);
    V4i32Type = VectorType::get(Int32Type, 4);
    Type *ArgsType[] = {
      V4f32Type,
      Int32Type,
      Int32Type,
      Int32Type,
      Int32Type,
      Int32Type,
      Int32Type,
      Int32Type,
      Int32Type,
      Int32Type,
    };
    TexSign = FunctionType::get(V4f32Type, ArgsType, /*isVarArg=*/false);
    Type *ArgsQType[] = {
      V4i32Type,
      Int32Type,
      Int32Type,
      Int32Type,
      Int32Type,
      Int32Type,
      Int32Type,
      Int32Type,
      Int32Type,
      Int32Type,
    };
    TexQSign = FunctionType::get(V4f32Type, ArgsQType, /*isVarArg=*/false);
    return false;
  }

  bool runOnFunction(Function &F) override {
    visit(F);
    return false;
  }

  const char *getPassName() const override {
    return "R600 Texture Intrinsics Replacer";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
  }

  void visitCallInst(CallInst &I) {
    if (!I.getCalledFunction())
      return;

    StringRef Name = I.getCalledFunction()->getName();
    if (Name == "llvm.AMDGPU.tex") {
      ReplaceTexIntrinsic(I, false, TexSign, "llvm.r600.tex", "llvm.r600.texc");
      return;
    }
    if (Name == "llvm.AMDGPU.txl") {
      ReplaceTexIntrinsic(I, true, TexSign, "llvm.r600.txl", "llvm.r600.txlc");
      return;
    }
    if (Name == "llvm.AMDGPU.txb") {
      ReplaceTexIntrinsic(I, true, TexSign, "llvm.r600.txb", "llvm.r600.txbc");
      return;
    }
    if (Name == "llvm.AMDGPU.txf") {
      ReplaceTXF(I);
      return;
    }
    if (Name == "llvm.AMDGPU.txq") {
      ReplaceTexIntrinsic(I, false, TexQSign, "llvm.r600.txq", "llvm.r600.txq");
      return;
    }
    if (Name == "llvm.AMDGPU.ddx") {
      ReplaceTexIntrinsic(I, false, TexSign, "llvm.r600.ddx", "llvm.r600.ddx");
      return;
    }
    if (Name == "llvm.AMDGPU.ddy") {
      ReplaceTexIntrinsic(I, false, TexSign, "llvm.r600.ddy", "llvm.r600.ddy");
      return;
    }
  }

};

char R600TextureIntrinsicsReplacer::ID = 0;

}

FunctionPass *llvm::createR600TextureIntrinsicsReplacer() {
  return new R600TextureIntrinsicsReplacer();
}
