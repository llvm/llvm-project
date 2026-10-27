//===-- AGXRaiseToAIR.cpp - AGX kernels as upstream AIR -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The inverse of agx-lower-air, for an alternative backend: a module of AGX
// kernels (as ZLUDA's PTX front end writes them) becomes the input of the AIR
// target's portable passes (addAIRPortablePasses), whose output is upstream
// AIR. agx-lower-air compiles that for AGX again; the AIR target's Metal
// pipeline compiles it for Metal. See llvm/docs/AIR/AIRHints.rst.
//
// What AIR has is written as AIR: kernels (the "air-kernel" attribute and the
// C calling convention), their parameters (a byref parameter in AGX's
// constant address space becomes a constant buffer), AIR's address spaces,
// system values, SIMD-group functions and barriers.
//
// What AIR does not have is written as a hint (llvm/IR/AIR.h): a call to an
// internal noinline function that computes the operation in AIR and names it
// in the "air-hint" attribute, so that agx-lower-air can restore the AGX
// operation and any other consumer runs the emulation. Double precision is
// such an emulation: the doubles become i64 and their operations hints whose
// bodies call arith-emulate-f64's library (soft_f64.c, rounded to nearest
// even), so the AIR this pass writes has no double. Generic pointers are an
// address-space hint (air::GenericHintAS).
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/AIR.h"
#include "llvm/IR/Constants.h"
#include <functional>
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#define DEBUG_TYPE "agx-raise-to-air"

#include "AGXAIRSoftF64.inc"

namespace {
Constant *doubleBits(Constant *C);

/// The data layout of the AIR target's modules.
constexpr StringLiteral AIRDataLayout =
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:"
    "64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:"
    "128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

/// AGX's address spaces and double precision become AIR's in every type.
class AGXToAIRTypes : public ValueMapTypeRemapper {
  DenseMap<Type *, Type *> Cache;

public:
  static unsigned addrSpace(unsigned AS) {
    switch (AS) {
    case AGX::GENERIC_ADDRESS:
      return air::GenericHintAS;
    case AGX::PRIVATE_ADDRESS:
      return air::AS::Default;
    case AGX::CONSTANT_ADDRESS:
      return air::AS::Constant;
    default:
      return AS;
    }
  }

  Type *remapType(Type *T) override {
    auto It = Cache.find(T);
    if (It != Cache.end())
      return It->second;
    Type *R = T;
    if (T->isDoubleTy()) {
      R = Type::getInt64Ty(T->getContext());
    } else if (auto *PT = dyn_cast<PointerType>(T)) {
      R = PointerType::get(T->getContext(), addrSpace(PT->getAddressSpace()));
    } else if (auto *VT = dyn_cast<VectorType>(T)) {
      R = VectorType::get(remapType(VT->getElementType()),
                          VT->getElementCount());
    } else if (auto *AT = dyn_cast<ArrayType>(T)) {
      R = ArrayType::get(remapType(AT->getElementType()),
                         AT->getNumElements());
    } else if (auto *FT = dyn_cast<FunctionType>(T)) {
      SmallVector<Type *, 8> Params;
      for (Type *P : FT->params())
        Params.push_back(remapType(P));
      R = FunctionType::get(remapType(FT->getReturnType()), Params,
                            FT->isVarArg());
    } else if (auto *ST = dyn_cast<StructType>(T); ST && !ST->isOpaque()) {
      SmallVector<Type *, 8> Elts;
      for (Type *E : ST->elements())
        Elts.push_back(remapType(E));
      if (!equal(Elts, ST->elements()))
        R = ST->isLiteral()
                ? StructType::get(T->getContext(), Elts, ST->isPacked())
                : StructType::create(T->getContext(), Elts,
                                     ST->getName().str() + ".air",
                                     ST->isPacked());
    }
    Cache[T] = R;
    return R;
  }
};

/// Constants with doubles, wherever the value mapper meets them (operands of
/// constant expressions, metadata): their bits.
class DoubleMaterializer final : public ValueMaterializer {
public:
  Value *materialize(Value *V) override {
    auto *C = dyn_cast<Constant>(V);
    if (!C || isa<GlobalValue>(C))
      return nullptr;
    Constant *Bits = doubleBits(C);
    LLVM_DEBUG(if (!Bits && C->getType()->getScalarType()->isDoubleTy())
                   dbgs() << "agx-raise-to-air: unmapped " << *C << "\n");
    return Bits;
  }
};

class RaiseToAIR {
  Module &M;
  LLVMContext &Ctx;
  bool Failed = false;
  bool NeedSoftF64 = false;
  /// Hint functions by key and AGX type, before the types change.
  std::map<std::pair<std::string, FunctionType *>, Function *> Hints;

public:
  explicit RaiseToAIR(Module &M) : M(M), Ctx(M.getContext()) {}
  bool run();

private:
  void error(const Function *F, const Twine &Msg) {
    Ctx.diagnose(DiagnosticInfoUnsupported(
        F ? *F : *M.begin(), "agx-raise-to-air: " + Msg));
    Failed = true;
  }
  Function *hint(StringRef Key, FunctionType *FT);
  void hintCall(CallInst &CI, StringRef Key, ArrayRef<Value *> Args);
  void hintInst(Instruction &I);
  bool raiseIntrinsic(IntrinsicInst &II);
  void raiseFunction(Function &F);
  void emulate(Function &F, StringRef Key);
  bool linkSoftF64();
  Function *byValueToByRef(Function &K);
};

/// A key's double mask: one character for the result and each operand of the
/// operation, 'd' where it is a double.
static std::string doubleMask(Type *Ret, ArrayRef<Type *> Ops) {
  std::string S(1, Ret->getScalarType()->isDoubleTy() ? 'd' : '-');
  for (Type *T : Ops)
    S += T->getScalarType()->isDoubleTy() ? 'd' : '-';
  return S;
}

Function *RaiseToAIR::hint(StringRef Key, FunctionType *FT) {
  Function *&F = Hints[{Key.str(), FT}];
  if (F)
    return F;
  std::string Name = (air::HintPrefix + Key).str();
  for (char &C : Name)
    if (!isAlnum(C) && C != '.' && C != '_')
      C = '_';
  F = Function::Create(FT, GlobalValue::ExternalLinkage, Name, &M);
  F->addFnAttr(air::HintAttr, Key);
  F->addFnAttr(Attribute::NoInline);
  return F;
}

void RaiseToAIR::hintCall(CallInst &CI, StringRef Key, ArrayRef<Value *> Args) {
  SmallVector<Type *, 4> Tys;
  for (Value *A : Args)
    Tys.push_back(A->getType());
  FunctionType *FT = FunctionType::get(CI.getType(), Tys, false);
  CallInst *New = CallInst::Create(hint(Key, FT), Args, "", CI.getIterator());
  New->takeName(&CI);
  New->setDebugLoc(CI.getDebugLoc());
  CI.replaceAllUsesWith(New);
  CI.eraseFromParent();
}

/// A double precision instruction as a hint: "inst:<opcode>[:<predicate>]".
void RaiseToAIR::hintInst(Instruction &I) {
  SmallVector<Value *, 3> Ops(I.operands());
  SmallVector<Type *, 3> Tys;
  for (Value *O : Ops)
    Tys.push_back(O->getType());
  std::string Key = ("inst:" + Twine(I.getOpcodeName())).str();
  if (auto *C = dyn_cast<FCmpInst>(&I))
    Key += ":" + CmpInst::getPredicateName(C->getPredicate()).str();
  Key += "|" + doubleMask(I.getType(), Tys);
  FunctionType *FT = FunctionType::get(I.getType(), Tys, false);
  CallInst *New = CallInst::Create(hint(Key, FT), Ops, "", I.getIterator());
  New->takeName(&I);
  I.replaceAllUsesWith(New);
  I.eraseFromParent();
}

static bool isDoubleOp(const Instruction &I) {
  auto Dbl = [](const Value *V) {
    return V->getType()->getScalarType()->isDoubleTy();
  };
  switch (I.getOpcode()) {
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::FNeg:
    return Dbl(&I);
  case Instruction::FCmp:
  case Instruction::FPToSI:
  case Instruction::FPToUI:
  case Instruction::FPTrunc:
    return Dbl(I.getOperand(0));
  case Instruction::SIToFP:
  case Instruction::UIToFP:
  case Instruction::FPExt:
    return Dbl(&I);
  default:
    return false;
  }
}

// System values: an AGX special register and the AIR function and element
// that give it.
struct SysVal {
  Intrinsic::ID ID;
  StringLiteral AIR;
  int Element; // -1: a scalar
};
static const SysVal SysVals[] = {
    {Intrinsic::agx_thread_position_in_threadgroup_x,
     "air.thread_position_in_threadgroup", 0},
    {Intrinsic::agx_thread_position_in_threadgroup_y,
     "air.thread_position_in_threadgroup", 1},
    {Intrinsic::agx_thread_position_in_threadgroup_z,
     "air.thread_position_in_threadgroup", 2},
    {Intrinsic::agx_threadgroup_position_in_grid_x,
     "air.threadgroup_position_in_grid", 0},
    {Intrinsic::agx_threadgroup_position_in_grid_y,
     "air.threadgroup_position_in_grid", 1},
    {Intrinsic::agx_threadgroup_position_in_grid_z,
     "air.threadgroup_position_in_grid", 2},
    {Intrinsic::agx_threads_per_threadgroup_x, "air.threads_per_threadgroup",
     0},
    {Intrinsic::agx_threads_per_threadgroup_y, "air.threads_per_threadgroup",
     1},
    {Intrinsic::agx_threads_per_threadgroup_z, "air.threads_per_threadgroup",
     2},
    {Intrinsic::agx_threadgroups_per_grid_x, "air.threadgroups_per_grid", 0},
    {Intrinsic::agx_threadgroups_per_grid_y, "air.threadgroups_per_grid", 1},
    {Intrinsic::agx_threadgroups_per_grid_z, "air.threadgroups_per_grid", 2},
    {Intrinsic::agx_thread_position_in_grid_x, "air.thread_position_in_grid",
     0},
    {Intrinsic::agx_thread_position_in_grid_y, "air.thread_position_in_grid",
     1},
    {Intrinsic::agx_thread_position_in_grid_z, "air.thread_position_in_grid",
     2},
    {Intrinsic::agx_thread_index_in_simdgroup, "air.thread_index_in_simdgroup",
     -1},
    {Intrinsic::agx_simdgroup_index_in_threadgroup,
     "air.simdgroup_index_in_threadgroup", -1},
    {Intrinsic::agx_thread_index_in_threadgroup,
     "air.thread_index_in_threadgroup", -1},
};

/// The AIR type suffix of a SIMD-group function's operand type.
static std::string simdSuffix(Type *T, bool Signed) {
  if (T->isFloatTy())
    return "f32";
  if (T->isHalfTy())
    return "f16";
  if (auto *IT = dyn_cast<IntegerType>(T))
    return (Twine(Signed ? "s.i" : "u.i") + Twine(IT->getBitWidth())).str();
  return "";
}

/// Rewrites an AGX intrinsic that AIR has; false if it has none.
bool RaiseToAIR::raiseIntrinsic(IntrinsicInst &II) {
  using namespace Intrinsic;
  IRBuilder<> B(&II);
  Type *I32 = B.getInt32Ty(), *I16 = B.getInt16Ty();
  Intrinsic::ID ID = II.getIntrinsicID();
  for (const SysVal &S : SysVals) {
    if (S.ID != ID)
      continue;
    Value *V;
    if (S.Element < 0) {
      V = B.CreateCall(M.getOrInsertFunction(S.AIR, I32));
    } else {
      auto *V3 = FixedVectorType::get(I32, 3);
      V = B.CreateExtractElement(
          B.CreateCall(M.getOrInsertFunction((S.AIR + ".v3i32").str(), V3)),
          S.Element);
    }
    II.replaceAllUsesWith(V);
    II.eraseFromParent();
    return true;
  }
  auto Call = [&](StringRef Name, Type *Ret, ArrayRef<Value *> Args) {
    SmallVector<Type *, 4> Tys;
    for (Value *A : Args)
      Tys.push_back(A->getType());
    return B.CreateCall(
        M.getOrInsertFunction(Name, FunctionType::get(Ret, Tys, false)), Args);
  };
  auto Finish = [&](Value *V) {
    if (V)
      II.replaceAllUsesWith(V);
    II.eraseFromParent();
    return true;
  };
  Type *Ty = II.getType();
  switch (ID) {
  case agx_threadgroup_barrier:
    // The fences around it, if any, stay LLVM fences.
    Call("air.wg.barrier", B.getVoidTy(), {B.getInt32(0), B.getInt32(1)});
    return Finish(nullptr);
  case agx_simd_shuffle:
  case agx_simd_shuffle_xor:
  case agx_simd_shuffle_up:
  case agx_simd_shuffle_down:
  case agx_quad_shuffle:
  case agx_quad_shuffle_xor:
  case agx_quad_shuffle_up:
  case agx_quad_shuffle_down: {
    std::string Sfx = simdSuffix(Ty, false);
    if (Sfx.empty())
      return false;
    bool Quad = ID == agx_quad_shuffle || ID == agx_quad_shuffle_xor ||
                ID == agx_quad_shuffle_up || ID == agx_quad_shuffle_down;
    StringRef Op = (ID == agx_simd_shuffle || ID == agx_quad_shuffle) ? ""
                   : (ID == agx_simd_shuffle_xor || ID == agx_quad_shuffle_xor)
                       ? "_xor"
                   : (ID == agx_simd_shuffle_up || ID == agx_quad_shuffle_up)
                       ? "_up"
                       : "_down";
    std::string Name =
        (Twine(Quad ? "air.quad_shuffle" : "air.simd_shuffle") + Op + "." +
         Sfx)
            .str();
    Value *Lane = B.CreateZExtOrTrunc(II.getArgOperand(1), I16);
    return Finish(Call(Name, Ty, {II.getArgOperand(0), Lane}));
  }
  case agx_simd_ballot:
    return Finish(B.CreateTrunc(
        Call("air.simd_ballot.i64", B.getInt64Ty(), {II.getArgOperand(0)}),
        Ty));
  case agx_quad_ballot:
    return Finish(B.CreateZExtOrTrunc(
        Call("air.quad_ballot", I16, {II.getArgOperand(0)}), Ty));
  case agx_simd_reduce:
  case agx_simd_prefix:
  case agx_quad_reduce:
  case agx_quad_prefix: {
    auto *OpC = dyn_cast<ConstantInt>(II.getArgOperand(1));
    if (!OpC)
      return false;
    bool Float = Ty->getScalarType()->isFloatingPointTy();
    StringRef Op;
    bool Signed = false;
    switch (OpC->getZExtValue()) {
    case AGX::SIMD_AND: Op = "and"; break;
    case AGX::SIMD_OR: Op = "or"; break;
    case AGX::SIMD_XOR: Op = "xor"; break;
    case AGX::SIMD_FADD: Op = "sum"; break;
    case AGX::SIMD_FMUL: Op = "product"; break;
    case AGX::SIMD_FMIN: Op = "min"; break;
    case AGX::SIMD_FMAX: Op = "max"; break;
    case AGX::SIMD_IADD: Op = "sum"; Signed = true; break;
    case AGX::SIMD_SMIN: Op = "min"; Signed = true; break;
    case AGX::SIMD_SMAX: Op = "max"; Signed = true; break;
    case AGX::SIMD_UMIN: Op = "min"; break;
    case AGX::SIMD_UMAX: Op = "max"; break;
    default: return false;
    }
    bool Prefix = ID == agx_simd_prefix || ID == agx_quad_prefix;
    // The hardware scans are exclusive; AIR scans only sums and products.
    if (Prefix && Op != "sum" && Op != "product")
      return false;
    if (Float && (Op == "and" || Op == "or" || Op == "xor"))
      return false;
    std::string Sfx = simdSuffix(Ty, Signed);
    if (Sfx.empty())
      return false;
    bool Quad = ID == agx_quad_reduce || ID == agx_quad_prefix;
    std::string Name = (Twine(Quad ? "air.quad_" : "air.simd_") +
                        (Prefix ? "prefix_exclusive_" : "") + Op + "." + Sfx)
                           .str();
    return Finish(Call(Name, Ty, {II.getArgOperand(0)}));
  }
  default:
    return false;
  }
}

/// C with its doubles as i64 bit patterns, or null if it has none.
Constant *doubleBits(Constant *C) {
  Type *T = C->getType();
  LLVMContext &Ctx = T->getContext();
  // (a vector ConstantFP is a splat)
  if (auto *FP = dyn_cast<ConstantFP>(C); FP && T->getScalarType()->isDoubleTy()) {
    Type *I64 = Type::getInt64Ty(Ctx);
    if (auto *VT = dyn_cast<VectorType>(T))
      I64 = VectorType::get(I64, VT);
    return ConstantInt::get(I64, FP->getValueAPF().bitcastToAPInt());
  }
  if (auto *CDS = dyn_cast<ConstantDataSequential>(C)) {
    if (!CDS->getElementType()->isDoubleTy())
      return nullptr;
    SmallVector<uint64_t, 8> Bits;
    for (unsigned I = 0; I < CDS->getNumElements(); ++I)
      Bits.push_back(
          CDS->getElementAsAPFloat(I).bitcastToAPInt().getZExtValue());
    return isa<VectorType>(T) ? ConstantDataVector::get(Ctx, Bits)
                              : ConstantDataArray::get(Ctx, Bits);
  }
  if (auto *CA = dyn_cast<ConstantAggregate>(C)) {
    SmallVector<Constant *, 8> Ops;
    bool Changed = false;
    for (Use &U : CA->operands()) {
      Constant *Op = cast<Constant>(U);
      if (Constant *Bits = doubleBits(Op)) {
        Op = Bits;
        Changed = true;
      } else if (Op->getType()->getScalarType()->isDoubleTy()) {
        // undef, poison or zero doubles
        Type *To = Type::getInt64Ty(Ctx);
        if (auto *VT = dyn_cast<VectorType>(Op->getType()))
          To = VectorType::get(To, VT);
        Op = isa<PoisonValue>(Op)  ? PoisonValue::get(To)
             : isa<UndefValue>(Op) ? UndefValue::get(To)
                                   : Constant::getNullValue(To);
        Changed = true;
      }
      Ops.push_back(Op);
    }
    if (!Changed)
      return nullptr;
    if (isa<ConstantVector>(CA))
      return ConstantVector::get(Ops);
    if (auto *ST = dyn_cast<StructType>(T)) {
      SmallVector<Type *, 8> Tys;
      for (Constant *Op : Ops)
        Tys.push_back(Op->getType());
      return ConstantStruct::get(
          StructType::get(Ctx, Tys, ST->isPacked()), Ops);
    }
    return ConstantArray::get(ArrayType::get(Ops[0]->getType(), Ops.size()),
                              Ops);
  }
  return nullptr;
}

/// A vector operation on doubles as one per element (the soft-f64 library is
/// scalar). False if I is not one this can split.
static bool scalarizeDoubles(Instruction &I) {
  auto *VT = dyn_cast<FixedVectorType>(I.getType());
  auto *OpVT = I.getNumOperands()
                   ? dyn_cast<FixedVectorType>(I.getOperand(0)->getType())
                   : nullptr;
  auto *Any = VT ? VT : OpVT;
  if (!Any || !(isDoubleOp(I) ||
                (isa<IntrinsicInst>(I) &&
                 (I.getType()->getScalarType()->isDoubleTy()))))
    return false;
  auto *II = dyn_cast<IntrinsicInst>(&I);
  if (II && (isa<ConstrainedFPIntrinsic>(II) || II->arg_size() > 3 ||
             !all_of(II->args(), [&](const Use &U) {
               return U->getType() == I.getType();
             })))
    return false;
  IRBuilder<> B(&I);
  Value *R = PoisonValue::get(I.getType());
  for (unsigned E = 0; E < Any->getNumElements(); ++E) {
    SmallVector<Value *, 3> Ops;
    for (Value *Op : (II ? II->args() : I.operands()))
      Ops.push_back(B.CreateExtractElement(Op, E));
    Value *X;
    Type *Ty = I.getType()->getScalarType();
    if (II)
      X = B.CreateIntrinsic(II->getIntrinsicID(), {Ty}, Ops);
    else if (auto *C = dyn_cast<FCmpInst>(&I))
      X = B.CreateFCmp(C->getPredicate(), Ops[0], Ops[1]);
    else if (auto *Cast = dyn_cast<CastInst>(&I))
      X = B.CreateCast(Cast->getOpcode(), Ops[0], Ty);
    else if (I.getOpcode() == Instruction::FNeg)
      X = B.CreateFNeg(Ops[0]);
    else
      X = B.CreateBinOp(cast<BinaryOperator>(I).getOpcode(), Ops[0], Ops[1]);
    if (auto *XI = dyn_cast<Instruction>(X))
      XI->copyIRFlags(&I);
    R = B.CreateInsertElement(R, X, E);
  }
  I.replaceAllUsesWith(R);
  I.eraseFromParent();
  return true;
}

/// The operations of F that AIR does not have become AIR or hints.
void RaiseToAIR::raiseFunction(Function &F) {
  SmallVector<Instruction *, 32> Work;
  for (Instruction &I : make_early_inc_range(instructions(F)))
    scalarizeDoubles(I);
  for (Instruction &I : instructions(F))
    Work.push_back(&I);
  for (Instruction *I : Work) {
    if (auto *II = dyn_cast<IntrinsicInst>(I)) {
      Function *Callee = II->getCalledFunction();
      StringRef Name = Callee->getName();
      bool AGXOp = Name.starts_with("llvm.agx.");
      if (AGXOp && raiseIntrinsic(*II))
        continue;
      bool Double = II->getType()->getScalarType()->isDoubleTy() ||
                    any_of(II->args(), [](const Use &U) {
                      return U->getType()->getScalarType()->isDoubleTy();
                    });
      bool Constrained = isa<ConstrainedFPIntrinsic>(II);
      // Intrinsics whose AIR forms are less exact (air-nan-min-max makes
      // minimum and maximum Metal's fmin and fmax with NaN checks, which
      // differ in signed zeros and denormals).
      bool Exact = is_contained({Intrinsic::minimum, Intrinsic::maximum,
                                 Intrinsic::minimumnum, Intrinsic::maximumnum},
                                II->getIntrinsicID());
      if (!AGXOp && !Double && !Constrained && !Exact)
        continue;
      // "call:<intrinsic>|<double mask>[|<metadata operand>...]"
      SmallVector<Value *, 4> Args;
      SmallVector<Type *, 4> Tys;
      std::string MD;
      for (Value *A : II->args()) {
        if (auto *MV = dyn_cast<MetadataAsValue>(A)) {
          if (auto *S = dyn_cast<MDString>(MV->getMetadata()))
            MD += "|" + S->getString().str();
          continue;
        }
        Args.push_back(A);
        Tys.push_back(A->getType());
      }
      std::string Key =
          ("call:" + Name + "|" + doubleMask(II->getType(), Tys) + MD).str();
      if (II->getIntrinsicID() == Intrinsic::agx_simdgroup_matrix_fma) {
        // The emulation needs the thread's lane, which only a kernel can
        // read (a system value): a trailing operand, "+lane".
        IRBuilder<> B(II);
        Args.push_back(B.CreateCall(
            M.getOrInsertFunction("air.thread_index_in_simdgroup",
                                  B.getInt32Ty())));
        Key += "|+lane";
      }
      hintCall(*II, Key, Args);
      continue;
    }
    if (auto *RMW = dyn_cast<AtomicRMWInst>(I);
        RMW && RMW->getValOperand()->getType()->isDoubleTy()) {
      // "atomic:<operation>|<doubles>|<ordering>|<sync scope>|<align>", with
      // no emulation (AIR has no 64-bit atomics to build one from).
      SmallVector<StringRef, 4> Scopes;
      Ctx.getSyncScopeNames(Scopes);
      std::string Key =
          ("atomic:" + AtomicRMWInst::getOperationName(RMW->getOperation()) +
           "|d-d|" + Twine(unsigned(RMW->getOrdering())) + "|" +
           Scopes[RMW->getSyncScopeID()] + "|" +
           Twine(RMW->getAlign().value()))
              .str();
      Value *Ptr = RMW->getPointerOperand(), *Val = RMW->getValOperand();
      FunctionType *FT = FunctionType::get(
          RMW->getType(), {Ptr->getType(), Val->getType()}, false);
      CallInst *New = CallInst::Create(hint(Key, FT), {Ptr, Val}, "",
                                       RMW->getIterator());
      New->takeName(RMW);
      RMW->replaceAllUsesWith(New);
      RMW->eraseFromParent();
      continue;
    }
    if (isDoubleOp(*I))
      hintInst(*I);
  }
}

bool RaiseToAIR::linkSoftF64() {
  SMDiagnostic Err;
  std::unique_ptr<Module> Lib =
      parseAssemblyString(AIRSoftF64IR, Err, Ctx);
  if (!Lib) {
    error(nullptr, "soft-f64 library: " + Err.getMessage());
    return false;
  }
  Lib->setDataLayout(M.getDataLayout());
  Lib->setTargetTriple(M.getTargetTriple());
  if (Linker::linkModules(M, std::move(Lib), Linker::Flags::LinkOnlyNeeded)) {
    error(nullptr, "cannot link the soft-f64 library");
    return false;
  }
  for (Function &F : M)
    if (F.getName().starts_with("__air_f64_"))
      F.setLinkage(GlobalValue::InternalLinkage);
  return true;
}

/// The AIR body of hint F, whose operation Key names.
void RaiseToAIR::emulate(Function &F, StringRef Key) {
  F.setLinkage(GlobalValue::InternalLinkage);
  BasicBlock *BB = BasicBlock::Create(Ctx, "", &F);
  IRBuilder<> B(BB);
  SmallVector<Value *, 4> A;
  for (Argument &Arg : F.args())
    A.push_back(&Arg);
  Type *Ret = F.getReturnType();
  Type *I32 = B.getInt32Ty(), *I64 = B.getInt64Ty(), *F32 = B.getFloatTy();
  auto Lib = [&](StringRef Name, Type *R, ArrayRef<Value *> Args) -> Value * {
    // A declaration, which linkSoftF64 defines once the bodies are written.
    SmallVector<Type *, 3> Tys;
    for (Value *A : Args)
      Tys.push_back(A->getType());
    NeedSoftF64 = true;
    return B.CreateCall(
        M.getOrInsertFunction(Name, FunctionType::get(R, Tys, false)), Args);
  };
  auto Fin = [&](Value *V) {
    if (Ret->isVoidTy())
      B.CreateRetVoid();
    else
      B.CreateRet(V ? V : PoisonValue::get(Ret));
  };
  // A hint without an emulation: only a consumer that has the operation can
  // compile it (agx-lower-air); for any other it is unreachable.
  auto NoEmulation = [&] {
    while (!BB->empty())
      BB->back().eraseFromParent();
    B.SetInsertPoint(BB);
    B.CreateUnreachable();
    F.addFnAttr("air-hint-no-emulation");
  };

  SmallVector<StringRef, 4> Parts;
  Key.split(Parts, '|');
  StringRef Op = Parts[0], Mask = Parts.size() > 1 ? Parts[1] : "";
  if (Op.starts_with("atomic:"))
    return NoEmulation();
  bool DblIn = Mask.size() > 1 && Mask[1] == 'd';
  bool DblOut = !Mask.empty() && Mask[0] == 'd';
  // The operation of a constrained intrinsic, or of an instruction.
  StringRef Base = Op;
  if (Base.consume_front("call:llvm.experimental.constrained."))
    Base = Base.take_until([](char C) { return C == '.'; });
  else if (Base.consume_front("inst:"))
    ;
  else if (Base.consume_front("call:llvm."))
    Base = Base.take_until([](char C) { return C == '.'; });

  auto Pred = [&]() -> CmpInst::Predicate {
    // fcmp/fcmps metadata operand ("oeq"), or the instruction's predicate
    StringRef P = Op.contains("constrained.fcmp")
                      ? (Parts.size() > 2 ? Parts[2] : StringRef())
                      : Op.rsplit(':').second;
    return StringSwitch<CmpInst::Predicate>(P)
        .Case("false", CmpInst::FCMP_FALSE).Case("oeq", CmpInst::FCMP_OEQ)
        .Case("ogt", CmpInst::FCMP_OGT).Case("oge", CmpInst::FCMP_OGE)
        .Case("olt", CmpInst::FCMP_OLT).Case("ole", CmpInst::FCMP_OLE)
        .Case("one", CmpInst::FCMP_ONE).Case("ord", CmpInst::FCMP_ORD)
        .Case("uno", CmpInst::FCMP_UNO).Case("ueq", CmpInst::FCMP_UEQ)
        .Case("ugt", CmpInst::FCMP_UGT).Case("uge", CmpInst::FCMP_UGE)
        .Case("ult", CmpInst::FCMP_ULT).Case("ule", CmpInst::FCMP_ULE)
        .Case("une", CmpInst::FCMP_UNE).Case("true", CmpInst::FCMP_TRUE)
        .Default(CmpInst::BAD_FCMP_PREDICATE);
  };

  if (DblIn || DblOut) {
    // Double precision, in the soft-f64 library (rounded to nearest even).
    StringRef B2 = Base.starts_with("fcmp") ? "fcmp" : Base;
    Value *R = nullptr;
    auto Bool = [&](Value *V) { return B.CreateICmpNE(V, B.getInt32(0)); };
    if (B2 == "fadd")
      R = Lib("__air_f64_add", I64, {A[0], A[1]});
    else if (B2 == "fsub")
      R = Lib("__air_f64_sub", I64, {A[0], A[1]});
    else if (B2 == "fmul")
      R = Lib("__air_f64_mul", I64, {A[0], A[1]});
    else if (B2 == "fdiv")
      R = Lib("__air_f64_div", I64, {A[0], A[1]});
    else if (B2 == "fma" || B2 == "fmuladd")
      R = Lib("__air_f64_fma", I64, {A[0], A[1], A[2]});
    else if (B2 == "sqrt")
      R = Lib("__air_f64_sqrt", I64, {A[0]});
    else if (B2 == "fneg")
      R = B.CreateXor(A[0], B.getInt64(0x8000000000000000ull));
    else if (B2 == "fabs")
      R = B.CreateAnd(A[0], B.getInt64(0x7fffffffffffffffull));
    else if (B2 == "copysign")
      R = B.CreateOr(B.CreateAnd(A[0], B.getInt64(0x7fffffffffffffffull)),
                     B.CreateAnd(A[1], B.getInt64(0x8000000000000000ull)));
    else if (B2 == "floor")
      R = Lib("__air_f64_floor", I64, {A[0]});
    else if (B2 == "ceil") {
      Value *N = B.CreateXor(A[0], B.getInt64(0x8000000000000000ull));
      R = B.CreateXor(Lib("__air_f64_floor", I64, {N}),
                      B.getInt64(0x8000000000000000ull));
    } else if (B2 == "trunc") {
      Value *Abs = B.CreateAnd(A[0], B.getInt64(0x7fffffffffffffffull));
      R = B.CreateOr(Lib("__air_f64_floor", I64, {Abs}),
                     B.CreateAnd(A[0], B.getInt64(0x8000000000000000ull)));
    } else if (B2 == "roundeven" || B2 == "rint" || B2 == "nearbyint")
      R = Lib("__air_f64_roundeven", I64, {A[0]});
    else if (B2 == "minnum" || B2 == "maxnum") {
      // The non-NaN operand when one is NaN.
      bool Min = B2 == "minnum";
      Value *Lt = Bool(Lib("__air_f64_lt", I32,
                           Min ? ArrayRef<Value *>{A[0], A[1]}
                               : ArrayRef<Value *>{A[1], A[0]}));
      Value *ANaN = Bool(Lib("__air_f64_unord", I32, {A[0], A[0]}));
      R = B.CreateSelect(ANaN, A[1], B.CreateSelect(Lt, A[0], A[1]));
    } else if (B2 == "minimum" || B2 == "maximum" || B2 == "minimumnum" ||
               B2 == "maximumnum") {
      // -0 orders below +0; minimum / maximum return a NaN operand (quiet),
      // the *num forms the other operand.
      bool Min = B2.starts_with("min");
      Value *X = A[0], *Y = A[1];
      Value *Lt = Bool(Lib("__air_f64_lt", I32,
                           Min ? ArrayRef<Value *>{X, Y}
                               : ArrayRef<Value *>{Y, X}));
      Value *Eq = Bool(Lib("__air_f64_eq", I32, {X, Y}));
      // equal values: the one with the sign bit for min, without for max
      Value *XNeg = B.CreateICmpSLT(X, B.getInt64(0));
      Value *PickX = B.CreateOr(
          Lt, B.CreateAnd(Eq, Min ? XNeg : B.CreateNot(XNeg)));
      Value *XNaN = Bool(Lib("__air_f64_unord", I32, {X, X}));
      Value *YNaN = Bool(Lib("__air_f64_unord", I32, {Y, Y}));
      R = B.CreateSelect(PickX, X, Y);
      Value *Quiet = B.getInt64(0x0008000000000000ull);
      if (B2.ends_with("num")) {
        R = B.CreateSelect(YNaN, X, R);
        R = B.CreateSelect(XNaN, Y, R);
      } else {
        R = B.CreateSelect(YNaN, B.CreateOr(Y, Quiet), R);
        R = B.CreateSelect(XNaN, B.CreateOr(X, Quiet), R);
      }
    } else if (B2 == "fcmp" || B2 == "fcmps") {
      Value *Uno = Bool(Lib("__air_f64_unord", I32, {A[0], A[1]}));
      Value *Eq = Bool(Lib("__air_f64_eq", I32, {A[0], A[1]}));
      Value *Lt = Bool(Lib("__air_f64_lt", I32, {A[0], A[1]}));
      Value *Gt = Bool(Lib("__air_f64_lt", I32, {A[1], A[0]}));
      switch (Pred()) {
      case CmpInst::FCMP_FALSE: R = B.getFalse(); break;
      case CmpInst::FCMP_OEQ: R = Eq; break;
      case CmpInst::FCMP_OGT: R = Gt; break;
      case CmpInst::FCMP_OGE: R = B.CreateOr(Gt, Eq); break;
      case CmpInst::FCMP_OLT: R = Lt; break;
      case CmpInst::FCMP_OLE: R = B.CreateOr(Lt, Eq); break;
      case CmpInst::FCMP_ONE: R = B.CreateOr(Lt, Gt); break;
      case CmpInst::FCMP_ORD: R = B.CreateNot(Uno); break;
      case CmpInst::FCMP_UNO: R = Uno; break;
      case CmpInst::FCMP_UEQ: R = B.CreateOr(Uno, Eq); break;
      case CmpInst::FCMP_UGT: R = B.CreateOr(Uno, Gt); break;
      case CmpInst::FCMP_UGE: R = B.CreateNot(Lt); break;
      case CmpInst::FCMP_ULT: R = B.CreateOr(Uno, Lt); break;
      case CmpInst::FCMP_ULE: R = B.CreateNot(Gt); break;
      case CmpInst::FCMP_UNE: R = B.CreateNot(Eq); break;
      case CmpInst::FCMP_TRUE: R = B.getTrue(); break;
      default: break;
      }
    } else if (B2 == "fptosi")
      R = B.CreateTrunc(Lib("__air_f64_to_i64", I64, {A[0]}), Ret);
    else if (B2 == "fptoui")
      R = B.CreateTrunc(Lib("__air_f64_to_u64", I64, {A[0]}), Ret);
    else if (B2 == "sitofp")
      R = Lib("__air_f64_from_i64", I64, {B.CreateSExt(A[0], I64)});
    else if (B2 == "uitofp")
      R = Lib("__air_f64_from_u64", I64, {B.CreateZExt(A[0], I64)});
    else if (B2 == "fpext") {
      Value *X = A[0];
      if (!X->getType()->isFloatTy())
        X = B.CreateFPExt(X, F32);
      R = Lib("__air_f64_from_f32", I64, {B.CreateBitCast(X, I32)});
    } else if (B2 == "fptrunc") {
      R = B.CreateBitCast(Lib("__air_f64_to_f32", I32, {A[0]}), F32);
      if (!Ret->isFloatTy())
        R = B.CreateFPTrunc(R, Ret);
    }
    if (!R)
      return NoEmulation();
    return Fin(R);
  }

  if (Intrinsic::ID ID = Intrinsic::lookupIntrinsicID(
          Op.starts_with("call:") ? Op.drop_front(5) : StringRef());
      is_contained({Intrinsic::minimum, Intrinsic::maximum,
                    Intrinsic::minimumnum, Intrinsic::maximumnum},
                   ID)) {
    // AIR lowers the intrinsic itself, as well as it can.
    return Fin(B.CreateIntrinsic(ID, {Ret}, A));
  }
  if (Op.starts_with("call:llvm.experimental.constrained.")) {
    // Rounded to nearest: the rounding direction is the hint's.
    Value *R = nullptr;
    if (Base == "fadd") R = B.CreateFAdd(A[0], A[1]);
    else if (Base == "fsub") R = B.CreateFSub(A[0], A[1]);
    else if (Base == "fmul") R = B.CreateFMul(A[0], A[1]);
    else if (Base == "fdiv") R = B.CreateFDiv(A[0], A[1]);
    else if (Base == "frem") R = B.CreateFRem(A[0], A[1]);
    else if (Base == "fma" || Base == "fmuladd" || Base == "sqrt" ||
             Base == "floor" || Base == "ceil" || Base == "trunc" ||
             Base == "rint" || Base == "nearbyint" || Base == "roundeven" ||
             Base == "minnum" || Base == "maxnum") {
      Intrinsic::ID IID = StringSwitch<Intrinsic::ID>(Base)
                              .Case("fma", Intrinsic::fma)
                              .Case("fmuladd", Intrinsic::fma)
                              .Case("sqrt", Intrinsic::sqrt)
                              .Case("floor", Intrinsic::floor)
                              .Case("ceil", Intrinsic::ceil)
                              .Case("trunc", Intrinsic::trunc)
                              .Case("rint", Intrinsic::roundeven)
                              .Case("nearbyint", Intrinsic::roundeven)
                              .Case("roundeven", Intrinsic::roundeven)
                              .Case("minnum", Intrinsic::minnum)
                              .Case("maxnum", Intrinsic::maxnum);
      R = B.CreateIntrinsic(IID, {Ret}, A);
    } else if (Base == "fptosi") R = B.CreateFPToSI(A[0], Ret);
    else if (Base == "fptoui") R = B.CreateFPToUI(A[0], Ret);
    else if (Base == "sitofp") R = B.CreateSIToFP(A[0], Ret);
    else if (Base == "uitofp") R = B.CreateUIToFP(A[0], Ret);
    else if (Base == "fpext") R = B.CreateFPExt(A[0], Ret);
    else if (Base == "fptrunc") R = B.CreateFPTrunc(A[0], Ret);
    else if (Base == "fcmp" || Base == "fcmps") R = B.CreateFCmp(Pred(), A[0], A[1]);
    if (!R)
      return NoEmulation();
    return Fin(R);
  }

  // AGX operations.
  StringRef Intr = Op;
  Intr.consume_front("call:");
  auto Afn = [&](Value *V) {
    if (auto *I = dyn_cast<Instruction>(V))
      I->setFast(true);
    return V;
  };
  Value *R = nullptr;
  if (Intr == "llvm.agx.get.fpmode")
    R = B.getInt32(0); // round to nearest, no flushing
  else if (Intr == "llvm.agx.set.fpmode")
    R = nullptr;
  else if (Intr == "llvm.agx.core.id")
    R = B.getInt32(0);
  else if (Intr.starts_with("llvm.agx.rcp."))
    R = Afn(B.CreateFDiv(ConstantFP::get(Ret, 1.0), A[0]));
  else if (Intr.starts_with("llvm.agx.rsqrt.") ||
           Intr.starts_with("llvm.agx.rsqrt.special.")) {
    Value *S = Afn(B.CreateIntrinsic(Intrinsic::sqrt, {Ret}, {A[0]}));
    R = Afn(B.CreateFDiv(ConstantFP::get(Ret, 1.0), S));
    if (Intr.starts_with("llvm.agx.rsqrt.special.")) {
      // x * rsqrt_special(x) == sqrt(x): 0 and infinity take 1.
      Value *Abs = B.CreateIntrinsic(Intrinsic::fabs, {Ret}, {A[0]});
      Value *Edge = B.CreateOr(
          B.CreateFCmpOEQ(A[0], ConstantFP::get(Ret, 0.0)),
          B.CreateFCmpOEQ(Abs, ConstantFP::getInfinity(Ret)));
      R = B.CreateSelect(Edge, ConstantFP::get(Ret, 1.0), R);
    }
  } else if (Intr.starts_with("llvm.agx.exp2."))
    R = Afn(B.CreateIntrinsic(Intrinsic::exp2, {Ret}, {A[0]}));
  else if (Intr.starts_with("llvm.agx.log2."))
    R = Afn(B.CreateIntrinsic(Intrinsic::log2, {Ret}, {A[0]}));
  else if (Intr.starts_with("llvm.agx.sin.") ||
           Intr.starts_with("llvm.agx.cos.")) {
    // The argument is in quadrants.
    Value *X = Afn(B.CreateFMul(
        A[0], ConstantFP::get(Ret, numbers::pi / 2)));
    R = Afn(B.CreateIntrinsic(Intr.starts_with("llvm.agx.sin.")
                                  ? Intrinsic::sin
                                  : Intrinsic::cos,
                              {Ret}, {X}));
  } else if (Intr.starts_with("llvm.agx.simdgroup.matrix.fma.") &&
             A.size() == 4) {
    // D = A * B + C on 8x8 matrices; thread t holds (r, c) and (r, c + 1),
    // r = 4 * (t / 16) + (t / 2) % 4, c = 4 * ((t / 8) % 2) + 2 * (t % 2)
    // (AGXUsage.rst). Each thread gathers its row of A and its columns of B
    // and accumulates in order of k from C, as the AGX expansion does.
    auto *VT = cast<FixedVectorType>(Ret);
    Type *ET = VT->getElementType();
    Type *I16 = B.getInt16Ty();
    Value *T = A[3];
    auto U = [&](unsigned V) { return B.getInt32(V); };
    Value *R = B.CreateAdd(
        B.CreateMul(B.CreateLShr(T, U(4)), U(4)),
        B.CreateAnd(B.CreateLShr(T, U(1)), U(3)));
    Value *C = B.CreateAdd(
        B.CreateMul(B.CreateAnd(B.CreateLShr(T, U(3)), U(1)), U(4)),
        B.CreateMul(B.CreateAnd(T, U(1)), U(2)));
    // The lane holding (row, col), and which of its two elements.
    auto LaneOf = [&](Value *Row, Value *Col) {
      Value *L = B.CreateMul(B.CreateLShr(Row, U(2)), U(16));
      L = B.CreateAdd(L, B.CreateMul(B.CreateLShr(Col, U(2)), U(8)));
      L = B.CreateAdd(L, B.CreateMul(B.CreateAnd(Row, U(3)), U(2)));
      L = B.CreateAdd(L, B.CreateLShr(B.CreateAnd(Col, U(3)), U(1)));
      return B.CreateTrunc(L, I16);
    };
    // Shuffles of bfloat go through their bits.
    Type *ShTy = ET->isBFloatTy() ? I16 : ET;
    std::string Sfx = ET->isFloatTy()  ? "f32"
                      : ET->isHalfTy() ? "f16"
                                       : "u.i16";
    FunctionCallee Shuffle = M.getOrInsertFunction(
        "air.simd_shuffle." + Sfx, FunctionType::get(ShTy, {ShTy, I16}, false));
    auto Read = [&](Value *Vec, unsigned Elt, Value *Lane) {
      Value *X = B.CreateExtractElement(Vec, Elt);
      if (ShTy != ET)
        X = B.CreateBitCast(X, ShTy);
      Value *V = B.CreateCall(Shuffle, {X, Lane});
      return ShTy != ET ? B.CreateBitCast(V, ET) : V;
    };
    Value *D0 = B.CreateExtractElement(A[2], uint64_t(0));
    Value *D1 = B.CreateExtractElement(A[2], uint64_t(1));
    for (unsigned K = 0; K < 8; ++K) {
      Value *Av = Read(A[0], K % 2, LaneOf(R, U(K)));
      Value *BLane = LaneOf(U(K), C);
      Value *B0 = Read(A[1], 0, BLane), *B1 = Read(A[1], 1, BLane);
      D0 = B.CreateIntrinsic(Intrinsic::fma, {ET}, {Av, B0, D0});
      D1 = B.CreateIntrinsic(Intrinsic::fma, {ET}, {Av, B1, D1});
    }
    Value *Out = B.CreateInsertElement(PoisonValue::get(Ret), D0, uint64_t(0));
    R = B.CreateInsertElement(Out, D1, uint64_t(1));
    return Fin(R);
  } else {
    return NoEmulation();
  }
  Fin(R);
}

Function *RaiseToAIR::byValueToByRef(Function &K) {
  if (all_of(K.args(), [](Argument &A) { return A.getType()->isPointerTy(); }))
    return &K;
  SmallVector<Type *, 8> Params;
  for (Argument &A : K.args())
    Params.push_back(A.getType()->isPointerTy()
                         ? A.getType()
                         : PointerType::get(Ctx, AGX::CONSTANT_ADDRESS));
  auto *FT = FunctionType::get(K.getReturnType(), Params, false);
  Function *New = Function::Create(FT, K.getLinkage(), K.getAddressSpace(),
                                   K.getName(), &M);
  New->copyAttributesFrom(&K);
  New->setAttributes(AttributeList::get(Ctx, K.getAttributes().getFnAttrs(),
                                        K.getAttributes().getRetAttrs(), {}));
  New->splice(New->begin(), &K);
  IRBuilder<> B(&*New->getEntryBlock().getFirstInsertionPt());
  for (auto [Old, NewA] : zip(K.args(), New->args())) {
    NewA.takeName(&Old);
    AttributeSet Attrs = K.getAttributes().getParamAttrs(Old.getArgNo());
    AttrBuilder AB(Ctx);
    if (Old.getType()->isPointerTy()) {
      AB.merge(AttrBuilder(Ctx, Attrs));
      Old.replaceAllUsesWith(&NewA);
    } else {
      // Only the string attributes apply to the buffer.
      for (Attribute At : Attrs)
        if (At.isStringAttribute())
          AB.addAttribute(At);
      AB.addByRefAttr(Old.getType());
      Old.replaceAllUsesWith(B.CreateLoad(Old.getType(), &NewA));
    }
    New->addParamAttrs(NewA.getArgNo(), AB);
  }
  std::string Name = K.getName().str();
  K.replaceAllUsesWith(New);
  K.eraseFromParent();
  New->setName(Name);
  return New;
}

bool RaiseToAIR::run() {
  if (M.getTargetTriple().getArch() != Triple::agx)
    return false;
  // Kernels: agx_kernel functions and, as AGX compiles every function as a
  // kernel, the other void functions nothing calls.
  SmallVector<Function *, 8> Kernels;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    bool Called = any_of(F.users(), [](const User *U) {
      return isa<CallBase>(U);
    });
    if (F.getCallingConv() == CallingConv::AGX_KERNEL ||
        (!F.hasLocalLinkage() && !Called && F.getReturnType()->isVoidTy()))
      Kernels.push_back(&F);
  }
  if (Kernels.empty())
    return false;
  // A kernel parameter passed by value (an AGX hidden argument such as the
  // threadgroup memory size) becomes a byref constant buffer like the others,
  // with its attributes ("agx-hidden").
  for (Function *&K : Kernels)
    K = byValueToByRef(*K);

  for (Function &F : make_early_inc_range(M))
    if (!F.isDeclaration())
      raiseFunction(F);
  // The intrinsics the hints took over.
  for (Function &F : make_early_inc_range(M))
    if (F.isIntrinsic() && F.use_empty())
      F.eraseFromParent();

  // Every global value takes AIR's types, as in agx-lower-air.
  AGXToAIRTypes Types;
  ValueToValueMapTy VMap;
  // Double constants map to their bits (the value map comes before the type
  // change).
  for (Function &F : M)
    for (Instruction &I : instructions(F))
      for (Use &U : I.operands())
        if (auto *C = dyn_cast<Constant>(U); C && !isa<GlobalValue>(C))
          if (Constant *Bits = doubleBits(C))
            VMap[C] = Bits;
  SmallVector<std::pair<GlobalVariable *, GlobalVariable *>, 8> Globals;
  SmallVector<std::pair<Function *, Function *>, 8> Functions;
  // The module's own globals and functions, not those created below.
  SmallVector<GlobalVariable *, 8> OldGlobals;
  for (GlobalVariable &GV : M.globals())
    OldGlobals.push_back(&GV);
  SmallVector<Function *, 16> OldFunctions;
  for (Function &F : M)
    OldFunctions.push_back(&F);
  for (GlobalVariable *GVP : OldGlobals) {
    GlobalVariable &GV = *GVP;
    Type *T = Types.remapType(GV.getValueType());
    unsigned AS = AGXToAIRTypes::addrSpace(GV.getAddressSpace());
    if (T == GV.getValueType() && AS == GV.getAddressSpace())
      continue;
    auto *NewGV = new GlobalVariable(
        M, T, GV.isConstant(), GV.getLinkage(), nullptr, GV.getName() + ".air",
        nullptr, GV.getThreadLocalMode(), AS);
    NewGV->copyAttributesFrom(&GV);
    VMap[&GV] = NewGV;
    Globals.push_back({&GV, NewGV});
  }
  for (Function *FP : OldFunctions) {
    Function &F = *FP;
    auto *FT = cast<FunctionType>(Types.remapType(F.getFunctionType()));
    if (F.isIntrinsic()) {
      if (FT == F.getFunctionType())
        continue;
      SmallVector<Type *, 4> Overloads;
      if (!Intrinsic::isSignatureValid(F.getIntrinsicID(), FT, Overloads)) {
        error(&F, "no AIR form of " + F.getName());
        continue;
      }
      VMap[&F] =
          Intrinsic::getOrInsertDeclaration(&M, F.getIntrinsicID(), Overloads);
      continue;
    }
    if (F.isDeclaration() && FT == F.getFunctionType())
      continue;
    Function *NewF = Function::Create(FT, F.getLinkage(), F.getAddressSpace(),
                                      F.getName() + ".air", &M);
    NewF->copyAttributesFrom(&F);
    VMap[&F] = NewF;
    Functions.push_back({&F, NewF});
  }
  if (Failed)
    return false;
  // Initializers, with their doubles as bits.
  std::function<Constant *(Constant *)> MapInit = [&](Constant *C) {
    if (Constant *Bits = doubleBits(C))
      return Bits;
    if (isa<ConstantAggregate>(C)) {
      SmallVector<Constant *, 8> Ops;
      for (Use &U : C->operands())
        Ops.push_back(MapInit(cast<Constant>(U)));
      Type *T = Types.remapType(C->getType());
      if (auto *AT = dyn_cast<ArrayType>(T))
        return ConstantArray::get(AT, Ops);
      if (auto *ST = dyn_cast<StructType>(T))
        return ConstantStruct::get(ST, Ops);
      return ConstantVector::get(Ops);
    }
    return MapValue(C, VMap, RF_None, &Types);
  };
  for (auto [Old, New] : Globals)
    if (Old->hasInitializer())
      New->setInitializer(MapInit(Old->getInitializer()));
  // The globals that keep their types may still refer to the functions and
  // globals that change (a table of function addresses).
  for (GlobalVariable *GV : OldGlobals)
    if (!is_contained(make_first_range(Globals), GV) && GV->hasInitializer())
      GV->setInitializer(MapInit(GV->getInitializer()));
  for (auto [Old, New] : Functions) {
    if (Old->isDeclaration())
      continue;
    auto NewArg = New->arg_begin();
    for (Argument &A : Old->args()) {
      NewArg->setName(A.getName());
      VMap[&A] = &*NewArg++;
    }
    SmallVector<ReturnInst *, 4> Returns;
    DoubleMaterializer Materializer;
    CloneFunctionInto(New, Old, VMap, CloneFunctionChangeType::LocalChangesOnly,
                      Returns, "", nullptr, &Types, &Materializer);
  }
  SmallVector<GlobalVariable *, 8> Order;
  for (GlobalVariable *GV : OldGlobals) {
    auto It = find_if(Globals, [&](auto &P) { return P.first == GV; });
    Order.push_back(It != Globals.end() ? It->second : GV);
  }
  for (auto [Old, New] : Functions)
    Old->dropAllReferences();
  for (auto [Old, New] : Globals)
    Old->setInitializer(nullptr);
  DenseMap<Function *, Function *> NewOf;
  for (auto [Old, New] : Functions) {
    std::string Name = Old->getName().str();
    bool Kernel = is_contained(Kernels, Old);
    NewOf[Old] = New;
    Old->replaceAllUsesWith(PoisonValue::get(Old->getType()));
    Old->eraseFromParent();
    New->setName(Name);
    if (Kernel) {
      New->setCallingConv(CallingConv::C);
      New->addFnAttr("air-kernel");
      // A byref parameter (in the constant address space) is a constant
      // buffer that the kernel reads. Its byref type stays, as a hint of its
      // exact size (AIR describes a buffer by the type its uses load).
    }
  }
  for (auto [Old, New] : Globals) {
    std::string Name = Old->getName().str();
    Old->replaceAllUsesWith(PoisonValue::get(Old->getType()));
    Old->eraseFromParent();
    New->setName(Name);
  }
  // The globals in their order (it decides the layout of the global data).
  for (GlobalVariable *GV : Order) {
    GV->removeFromParent();
    M.insertGlobalVariable(GV);
  }
  for (Function &F : make_early_inc_range(M))
    if (F.isIntrinsic() && F.use_empty())
      F.eraseFromParent();

  M.setTargetTriple(Triple("air"));
  M.setDataLayout(AIRDataLayout);
  // Bodies for the hints, now with AIR's types.
  for (Function &F : make_early_inc_range(M))
    if (F.isDeclaration() && F.hasFnAttribute(air::HintAttr))
      emulate(F, F.getFnAttribute(air::HintAttr).getValueAsString());
  if (NeedSoftF64)
    linkSoftF64();
  return !Failed;
}

} // namespace

PreservedAnalyses AGXRaiseToAIRPass::run(Module &M, ModuleAnalysisManager &) {
  return RaiseToAIR(M).run() ? PreservedAnalyses::none()
                             : PreservedAnalyses::all();
}
