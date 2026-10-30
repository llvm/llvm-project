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
#include "AGXSubtarget.h"
#include "AGXTargetMachine.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/AIR.h"
#include "llvm/IR/Constants.h"
#include <functional>
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
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
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

// Hints are kept only where AGX has an operation that is worth much more
// than its AIR emulation and that it cannot recover from the emulation: the
// SIMD-group matrix FMA (G14's tile GEMMs are 15% faster with it). Everything
// else is plain AIR (llvm/docs/AIR/AIRHints.rst).
static cl::opt<std::string> KeepHints(
    "agx-raise-hints", cl::init("call:llvm.agx.simdgroup.matrix"), cl::Hidden,
    cl::desc("Comma-separated prefixes of the keys of the hints that stay "
             "hints (the others become their AIR emulations, inlined); "
             "\"all\" keeps every one, \"\" none"));


#define DEBUG_TYPE "agx-raise-to-air"

// The soft-f64 library, which the AIR target owns (air-emulate-f64).
#include "../AIR/AIRSoftF64.inc"

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
  /// Where AGX's generic pointers go: AIR's device address space when
  /// nothing in the module makes a generic pointer to threadgroup or
  /// private memory (then they are device pointers, as AGX treats them), the
  /// address-space hint otherwise.
  unsigned GenericAS = air::GenericHintAS;

  unsigned addrSpace(unsigned AS) const {
    switch (AS) {
    case AGX::GENERIC_ADDRESS:
      return GenericAS;
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
  ValueToValueMapTy &VMap;
  AGXToAIRTypes &Types;

public:
  DoubleMaterializer(ValueToValueMapTy &VMap, AGXToAIRTypes &Types)
      : VMap(VMap), Types(Types) {}

  Value *materialize(Value *V) override {
    auto *C = dyn_cast<Constant>(V);
    if (!C || isa<GlobalValue>(C))
      return nullptr;
    // A cast between generic and device pointers, the same address space in
    // AIR when generic pointers are device pointers there.
    if (auto *CE = dyn_cast<ConstantExpr>(C);
        CE && CE->getOpcode() == Instruction::AddrSpaceCast &&
        Types.remapType(CE->getType()) ==
            Types.remapType(CE->getOperand(0)->getType()))
      return MapValue(CE->getOperand(0), VMap, RF_None, &Types, this);
    Constant *Bits = doubleBits(C);
    LLVM_DEBUG(if (!Bits && C->getType()->getScalarType()->isDoubleTy())
                   dbgs() << "agx-raise-to-air: unmapped " << *C << "\n");
    return Bits;
  }
};

/// Whether M makes generic pointers to memory other than device memory, or
/// reads such memory through generic ones (PTX's cvta.shared / .local /
/// .param), or calls through pointers (function addresses are generic):
/// AGX tags those generic pointers (AGXLowerGenericAddresses); all others are
/// device pointers.
static bool makesLocalGeneric(Module &M) {
  // Any memory but device memory (threadgroup, private, the constant
  // memory of kernel parameters: cvta.param).
  auto Local = [](unsigned AS) { return AS != AGX::DEVICE_ADDRESS; };
  auto Cast = [&](unsigned Src, unsigned Dst) {
    return (Dst == AGX::GENERIC_ADDRESS && Local(Src)) ||
           (Src == AGX::GENERIC_ADDRESS && Local(Dst));
  };
  SmallPtrSet<const Constant *, 16> Seen;
  std::function<bool(const Constant *)> InConstant = [&](const Constant *C) {
    if (!Seen.insert(C).second)
      return false;
    if (auto *CE = dyn_cast<ConstantExpr>(C);
        CE && CE->getOpcode() == Instruction::AddrSpaceCast &&
        Cast(CE->getOperand(0)->getType()->getPointerAddressSpace(),
             CE->getType()->getPointerAddressSpace()))
      return true;
    return any_of(C->operands(), [&](const Use &U) {
      auto *Op = dyn_cast<Constant>(U.get());
      return Op && !isa<GlobalValue>(Op) && InConstant(Op);
    });
  };
  for (GlobalVariable &GV : M.globals())
    if (GV.hasInitializer() && InConstant(GV.getInitializer()))
      return true;
  for (Function &F : M)
    for (Instruction &I : instructions(F)) {
      if (auto *C = dyn_cast<AddrSpaceCastInst>(&I);
          C && Cast(C->getSrcAddressSpace(), C->getDestAddressSpace()))
        return true;
      // Calls through pointers: functions are generic (code) addresses.
      if (auto *CB = dyn_cast<CallBase>(&I);
          CB && !CB->getCalledFunction() && !CB->isInlineAsm())
        return true;
      for (const Use &U : I.operands())
        if (auto *C = dyn_cast<Constant>(U.get());
            C && !isa<GlobalValue>(C) && InConstant(C))
          return true;
    }
  return false;
}

class RaiseToAIR {
  Module &M;
  LLVMContext &Ctx;
  /// Double precision NaN results are the canonical NaN (the AGX subtarget's
  /// canonical-nan, as AGXSoftF64 does).
  bool CanonicalNaN = false;
  bool Failed = false;
  bool NeedSoftF64 = false;
  /// Hint functions by key and AGX type, before the types change.
  std::map<std::pair<std::string, FunctionType *>, Function *> Hints;

public:
  RaiseToAIR(Module &M, const TargetMachine *TM)
      : M(M), Ctx(M.getContext()) {
    for (Function &F : M)
      if (TM && !F.isDeclaration()) {
        CanonicalNaN =
            static_cast<const AGXTargetMachine *>(TM)->getSubtarget<AGXSubtarget>(F)
                .hasCanonicalNaN();
        break;
      }
  }
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
  void inlineEmulations();
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
      // AIR passes a scalar system value as a parameter, which keeps no
      // place in the code: a hint "+value" at the read, whose emulation is
      // the AIR value and which AGX reads here again (as the direct path).
      V = B.CreateCall(M.getOrInsertFunction(S.AIR, I32));
      std::string Key =
          ("call:" + II.getCalledFunction()->getName() + "|-|+value").str();
      V = B.CreateCall(hint(Key, FunctionType::get(I32, {I32}, false)), {V});
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

/// The floating point mode register (llvm.agx.set.fpmode) as a value at each
/// operation that reads it: llvm.agx.get.fpmode, the constrained intrinsics
/// and the exact minimum / maximum hints (the mode has their denormal modes,
/// and the rounding of dynamic ones), for their emulations. The values are
/// freezes of the mode, which keep them through the promotion; the mode
/// starts as AGX's lowering (AGXLowerFPEnv) assumes, rounding to nearest
/// even with the function's denormal modes.
static DenseMap<Instruction *, FreezeInst *> trackMode(Function &F) {
  DenseMap<Instruction *, FreezeInst *> ModeAt;
  SmallVector<IntrinsicInst *, 8> Sets, Reads;
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II)
      continue;
    if (II->getIntrinsicID() == Intrinsic::agx_set_fpmode)
      Sets.push_back(II);
    else if (II->getIntrinsicID() == Intrinsic::agx_get_fpmode)
      Reads.push_back(II);
    else if (isa<ConstrainedFPIntrinsic>(II) ||
             is_contained({Intrinsic::minimum, Intrinsic::maximum,
                           Intrinsic::minimumnum, Intrinsic::maximumnum},
                          II->getIntrinsicID()))
      Reads.push_back(II);
  }
  if (Reads.empty())
    return ModeAt;
  auto Flushes = [&](const fltSemantics &Sem) {
    DenormalMode DM = F.getDenormalMode(Sem);
    return DM.Output != DenormalMode::IEEE && DM.Output != DenormalMode::Dynamic;
  };
  // Rounding to nearest even (1) for float and half, the flush bits.
  uint32_t Initial = 1 | (1 << 2) | (Flushes(APFloat::IEEEsingle()) ? 0x10 : 0) |
                     (Flushes(APFloat::IEEEhalf()) ? 0x20 : 0);
  IRBuilder<> Entry(&*F.getEntryBlock().getFirstInsertionPt());
  AllocaInst *Mode = Entry.CreateAlloca(Entry.getInt32Ty());
  Entry.CreateStore(Entry.getInt32(Initial), Mode);
  for (IntrinsicInst *II : Sets)
    IRBuilder<>(II).CreateStore(II->getArgOperand(0), Mode);
  for (IntrinsicInst *II : Reads) {
    IRBuilder<> B(II);
    ModeAt[II] = cast<FreezeInst>(
        B.CreateFreeze(B.CreateLoad(B.getInt32Ty(), Mode)));
  }
  DominatorTree DT(F);
  PromoteMemToReg({Mode}, DT);
  return ModeAt;
}

/// The operations of F that AIR does not have become AIR or hints.
void RaiseToAIR::raiseFunction(Function &F) {
  SmallVector<Instruction *, 32> Work;
  for (Instruction &I : make_early_inc_range(instructions(F)))
    scalarizeDoubles(I);
  DenseMap<Instruction *, FreezeInst *> ModeAt = trackMode(F);
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
      // (minimum, maximum and the *num forms are AIR's intrinsics: AIR's
      // fmin / fmax with NaN checks.)
      bool Exact = false;
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
      if (FreezeInst *Mode = ModeAt.lookup(II)) {
        // The mode, which only the emulation reads: "+mode".
        Args.push_back(Mode);
        Key += "|+mode";
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
  for (auto &[I, Fr] : ModeAt) {
    Fr->replaceAllUsesWith(Fr->getOperand(0));
    Fr->eraseFromParent();
  }
}

bool RaiseToAIR::linkSoftF64() {
  SMDiagnostic Err;
  // (names kept while the textual IR is read: a release build's context
  // discards them, and textual IR cannot be read without)
  bool Discards = Ctx.shouldDiscardValueNames();
  Ctx.setDiscardValueNames(false);
  std::unique_ptr<Module> Lib =
      parseAssemblyString(AIRSoftF64IR, Err, Ctx);
  Ctx.setDiscardValueNames(Discards);
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

using LibFn = function_ref<Value *(StringRef, Type *, ArrayRef<Value *>)>;

/// The constrained operation Base of a format other than double, on A, in
/// the directed rounding Rounding: the operands as doubles (exact), the
/// operation in soft-f64 rounded in that direction, then rounded to Ret in
/// it (rounding twice in the same direction rounds once, as the values of
/// the format are doubles). Null for an operation that does not round.
static Value *directed(IRBuilderBase &B, StringRef Base, ArrayRef<Value *> A,
                       Type *Ret, Value *Rounding, LibFn Lib,
                       bool CanonicalNaN) {
  bool Arith = is_contained({"fadd", "fsub", "fmul", "fdiv", "fma",
                             "fmuladd", "sqrt"},
                            Base);
  bool FromInt = Base == "sitofp" || Base == "uitofp";
  bool Integral = Base == "rint" || Base == "nearbyint";
  Type *ST = Ret->getScalarType();
  if ((!Arith && !FromInt && !Integral && Base != "fptrunc") ||
      !(ST->isFloatTy() || ST->isHalfTy() || ST->isBFloatTy()) ||
      (FromInt && A[0]->getType()->getScalarSizeInBits() > 64))
    return nullptr;
  if (auto *VT = dyn_cast<FixedVectorType>(Ret)) {
    // Lane by lane.
    Value *R = PoisonValue::get(Ret);
    for (unsigned L = 0, E = VT->getNumElements(); L < E; ++L) {
      SmallVector<Value *, 4> Lane;
      for (Value *X : A)
        Lane.push_back(X->getType()->isVectorTy()
                           ? B.CreateExtractElement(X, L)
                           : X);
      R = B.CreateInsertElement(
          R, directed(B, Base, Lane, ST, Rounding, Lib, CanonicalNaN), L);
    }
    return R;
  }
  Type *I16 = B.getInt16Ty(), *I32 = B.getInt32Ty(), *I64 = B.getInt64Ty();
  auto Is = [&](unsigned C) { return B.CreateICmpEQ(Rounding, B.getInt32(C)); };

  if (Integral) {
    auto Call = [&](Intrinsic::ID ID) {
      return B.CreateUnaryIntrinsic(ID, A[0]);
    };
    Value *R = Call(Intrinsic::roundeven);
    R = B.CreateSelect(Is(0), Call(Intrinsic::trunc), R);
    R = B.CreateSelect(Is(2), Call(Intrinsic::ceil), R);
    return B.CreateSelect(Is(3), Call(Intrinsic::floor), R);
  }
  // A value of the format as the bits of the double.
  auto Ext = [&](Value *X) {
    Type *T = X->getType();
    Value *F;
    if (T->isFloatTy())
      F = B.CreateBitCast(X, I32);
    else if (T->isHalfTy())
      F = B.CreateBitCast(B.CreateFPExt(X, B.getFloatTy()), I32);
    else // bfloat: the high half of a float
      F = B.CreateShl(B.CreateZExt(B.CreateBitCast(X, I16), I32), 16);
    return Lib("__air_f64_from_f32", I64, {F});
  };
  // add, sub and mul of floats, by error-free transformations (Knuth's
  // TwoSum, Dekker's product with an fma: Graillat, Lamotte, Nguyen Hong):
  // the result rounded to nearest and the exact sign of its error, then one
  // unit in the direction where the error says so. Exact while operands and
  // result are zero or at least 2^-70 in magnitude and the result is finite
  // (the error, at least about 2^-47 of the result, is then normal, not
  // flushed by an ALU that flushes); the others take the double precision
  // path below. An exact sum of zero is +0, -0 rounding downward, unless both
  // operands are zeros of one sign (that sign).
  BasicBlock *FastBB = nullptr, *Join = nullptr;
  Value *Fast = nullptr;
  if (ST->isFloatTy() && (Base == "fadd" || Base == "fsub" || Base == "fmul")) {
    Value *X = A[0], *Y = Base == "fsub" ? B.CreateFNeg(A[1]) : A[1];
    Value *S, *Err;
    if (Base == "fmul") {
      S = B.CreateFMul(X, Y);
      Err = B.CreateIntrinsic(Intrinsic::fma, {ST}, {X, Y, B.CreateFNeg(S)});
    } else {
      S = B.CreateFAdd(X, Y);
      Value *BV = B.CreateFSub(S, X);
      Value *AV = B.CreateFSub(S, BV);
      Err = B.CreateFAdd(B.CreateFSub(X, AV), B.CreateFSub(Y, BV));
    }
    auto Safe = [&](Value *V, bool Zero) {
      Value *Abs = B.CreateUnaryIntrinsic(Intrinsic::fabs, V);
      Value *InRange =
          B.CreateAnd(B.CreateFCmpOGE(Abs, ConstantFP::get(ST, 0x1p-70)),
                      B.CreateFCmpOLT(Abs, ConstantFP::get(ST, 0x1p127)));
      return Zero ? B.CreateOr(B.CreateFCmpOEQ(V, ConstantFP::get(ST, 0.0)),
                               InRange)
                  : InRange;
    };
    Value *Ok = B.CreateAnd(B.CreateAnd(Safe(X, true), Safe(Y, true)),
                            Safe(S, true));
    // The step: +1 or -1 on the bits (away from or toward zero), by the
    // signs of the result and of the error.
    Value *SBits = B.CreateBitCast(S, I32);
    Value *Neg = B.CreateICmpSLT(SBits, B.getInt32(0));
    Value *Above = B.CreateFCmpOGT(Err, ConstantFP::get(ST, 0.0)); // exact > S
    Value *Below = B.CreateFCmpOLT(Err, ConstantFP::get(ST, 0.0));
    Value *Toward = B.CreateSelect(Neg, Above, Below); // |exact| < |S|
    // rz: toward zero when |exact| < |S|; ru: up when exact > S; rd: down.
    Value *Dec = B.CreateSelect(Is(0), Toward,
                                B.CreateSelect(Is(2), B.CreateAnd(Above, Neg),
                                               B.CreateAnd(Below, B.CreateNot(Neg))));
    Value *Inc = B.CreateSelect(Is(0), B.getFalse(),
                                B.CreateSelect(Is(2), B.CreateAnd(Above, B.CreateNot(Neg)),
                                               B.CreateAnd(Below, Neg)));
    Value *Step = B.CreateSub(B.CreateZExt(Inc, I32), B.CreateZExt(Dec, I32));
    Fast = B.CreateBitCast(B.CreateAdd(SBits, Step), ST);
    if (Base != "fmul") {
      Value *Zero = ConstantFP::get(ST, 0.0);
      Value *SameZeros = B.CreateAnd(
          B.CreateAnd(B.CreateFCmpOEQ(X, Zero), B.CreateFCmpOEQ(Y, Zero)),
          B.CreateICmpEQ(B.CreateICmpSLT(B.CreateBitCast(X, I32), B.getInt32(0)),
                         B.CreateICmpSLT(B.CreateBitCast(Y, I32), B.getInt32(0))));
      Value *MinusZero = B.CreateAnd(
          B.CreateAnd(B.CreateFCmpOEQ(S, Zero), Is(3)), B.CreateNot(SameZeros));
      Fast = B.CreateSelect(MinusZero, ConstantFP::getNegativeZero(ST), Fast);
    }
    FastBB = B.GetInsertBlock();
    Function *Fn = FastBB->getParent();
    BasicBlock *Slow = BasicBlock::Create(B.getContext(), "directed.slow", Fn);
    Join = BasicBlock::Create(B.getContext(), "directed.join", Fn);
    // (Rounding to nearest takes the AIR operation instead, in the caller:
    // the branch then folds away with the rest where the mode is known.)
    B.CreateCondBr(B.CreateOr(Ok, Is(1)), Join, Slow);
    B.SetInsertPoint(Slow);
  }
  Value *D;
  if (FromInt) {
    bool Signed = Base == "sitofp";
    Value *X = Signed ? B.CreateSExt(A[0], I64) : B.CreateZExt(A[0], I64);
    D = Lib(Signed ? "__air_f64_from_i64_r" : "__air_f64_from_u64_r", I64,
            {X, Rounding});
  } else if (Base == "fptrunc") {
    D = Ext(A[0]);
  } else {
    auto [Name, N] = StringSwitch<std::pair<StringRef, unsigned>>(Base)
                         .Case("fadd", {"add", 2})
                         .Case("fsub", {"sub", 2})
                         .Case("fmul", {"mul", 2})
                         .Case("fdiv", {"div", 2})
                         .Case("sqrt", {"sqrt", 1})
                         .Default({"fma", 3});
    SmallVector<Value *, 4> Args;
    for (Value *X : A.take_front(N))
      Args.push_back(Ext(X));
    Args.push_back(Rounding);
    D = Lib(("__air_f64_" + Name + "_r").str(), I64, Args);
  }
  StringRef To = ST->isFloatTy()  ? "__air_f64_to_f32_r"
                 : ST->isHalfTy() ? "__air_f64_to_f16_r"
                                  : "__air_f64_to_bf16_r";
  Value *Bits = Lib(To, I32, {D, Rounding});
  if (!ST->isFloatTy())
    Bits = B.CreateTrunc(Bits, I16);
  Value *R = B.CreateBitCast(Bits, ST);
  if (CanonicalNaN) {
    APFloat NaN(ST->getFltSemantics(),
                APInt::getSignedMaxValue(ST->getPrimitiveSizeInBits()));
    R = B.CreateSelect(B.CreateFCmpUNO(R, R), ConstantFP::get(ST, NaN), R);
  }
  if (Join) {
    BasicBlock *SlowEnd = B.GetInsertBlock();
    B.CreateBr(Join);
    B.SetInsertPoint(Join);
    PHINode *P = B.CreatePHI(ST, 2);
    P->addIncoming(Fast, FastBB);
    P->addIncoming(R, SlowEnd);
    R = P;
  }
  return R;
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
  if (StringRef RMW = Op; RMW.consume_front("atomic:")) {
    // A double precision read-modify-write as a compare-exchange loop of its
    // bits, with the operation in soft-f64 (LLVM's 64-bit cmpxchg: AGX has
    // it, AIR does not).
    unsigned Order = 0, Alignment = 8;
    if (Parts.size() != 5 || Parts[2].getAsInteger(10, Order) ||
        Parts[4].getAsInteger(10, Alignment) || A.size() != 2 ||
        !is_contained({"fadd", "fsub", "fmax", "fmin", "xchg"}, RMW))
      return NoEmulation();
    Value *Ptr = A[0], *X = A[1];
    auto Ordering = AtomicOrdering(Order);
    SyncScope::ID Scope = Ctx.getOrInsertSyncScopeID(Parts[3]);
    if (RMW == "xchg")
      return Fin(B.CreateAtomicRMW(AtomicRMWInst::Xchg, Ptr, X,
                                   Align(Alignment), Ordering, Scope));
    Value *Init = B.CreateAlignedLoad(I64, Ptr, Align(Alignment), true);
    BasicBlock *Loop = BasicBlock::Create(Ctx, "loop", &F);
    BasicBlock *Done = BasicBlock::Create(Ctx, "done", &F);
    B.CreateBr(Loop);
    B.SetInsertPoint(Loop);
    PHINode *Old = B.CreatePHI(I64, 2);
    Old->addIncoming(Init, BB);
    Value *New;
    if (RMW == "fadd" || RMW == "fsub") {
      New = Lib(RMW == "fadd" ? "__air_f64_add" : "__air_f64_sub", I64,
                {Old, X});
    } else {
      // maxnum / minnum: the other operand of a NaN.
      Value *Lt = B.CreateICmpNE(
          Lib("__air_f64_lt", I32,
              RMW == "fmin" ? ArrayRef<Value *>{X, Old}
                            : ArrayRef<Value *>{Old, X}),
          B.getInt32(0));
      Value *OldNaN =
          B.CreateICmpNE(Lib("__air_f64_unord", I32, {Old, Old}), B.getInt32(0));
      New = B.CreateSelect(B.CreateOr(Lt, OldNaN), X, Old);
    }
    AtomicCmpXchgInst *CX = B.CreateAtomicCmpXchg(
        Ptr, Old, New, Align(Alignment), Ordering, AtomicOrdering::Monotonic,
        Scope);
    Value *Loaded = B.CreateExtractValue(CX, 0);
    Old->addIncoming(Loaded, Loop);
    B.CreateCondBr(B.CreateExtractValue(CX, 1), Done, Loop);
    B.SetInsertPoint(Done);
    return Fin(Loaded);
  }
  // A value AIR has (a scalar system value), placed where AGX reads it.
  if (Parts.back() == "+value")
    return Fin(A.back());
  bool DblIn = Mask.size() > 1 && Mask[1] == 'd';
  bool DblOut = !Mask.empty() && Mask[0] == 'd';
  // The floating point mode, a trailing operand.
  Value *Mode = Parts.back() == "+mode" ? A.pop_back_val() : nullptr;
  // The rounding direction, numbered like FLT_ROUNDS (0 toward zero, 1 to
  // nearest even, 2 upward, 3 downward): the metadata's, or the mode's field
  // for the result's format: bits 0-1 for float, 2-3 for the others
  // (AGXLowerFPEnv).
  Value *Rounding = B.getInt32(1);
  for (StringRef P : drop_begin(Parts, 2)) {
    if (!P.consume_front("round."))
      continue;
    if (P == "dynamic") {
      bool Float = Ret->getScalarType()->isFloatTy() && Mask[0] != 'd';
      if (Mode)
        Rounding = B.CreateAnd(Float ? Mode : B.CreateLShr(Mode, 2), 3);
    } else {
      Rounding = B.getInt32(StringSwitch<unsigned>(P)
                                .Case("towardzero", 0)
                                .Case("upward", 2)
                                .Case("downward", 3)
                                .Default(1));
    }
  }
  auto *RoundingC = dyn_cast<ConstantInt>(Rounding);
  bool Nearest = RoundingC && RoundingC->isOne();
  // Denormals of the operands' format (the result's without floating point
  // operands) flush to zero, operands and result, where the mode says so
  // (its bit 4 for float, 5 for half), as AGX does.
  Type *FlushTy = A.empty() ? Ret->getScalarType()
                            : A[0]->getType()->getScalarType();
  if (!FlushTy->isFloatingPointTy())
    FlushTy = Ret->getScalarType();
  Value *Flush = nullptr;
  if (Mode && (FlushTy->isFloatTy() || FlushTy->isHalfTy()) && !DblIn &&
      !DblOut)
    Flush = B.CreateICmpNE(
        B.CreateAnd(Mode, FlushTy->isFloatTy() ? 0x10 : 0x20), B.getInt32(0));
  auto Flushed = [&](Value *X) {
    if (!Flush || X->getType()->getScalarType() != FlushTy)
      return X;
    Value *Tiny = B.CreateFCmpOLT(
        B.CreateUnaryIntrinsic(Intrinsic::fabs, X),
        ConstantFP::get(X->getType(), APFloat::getSmallestNormalized(
                                          FlushTy->getFltSemantics())));
    Value *Zero = B.CreateBinaryIntrinsic(
        Intrinsic::copysign, ConstantFP::get(X->getType(), 0.0), X);
    return B.CreateSelect(Flush, B.CreateSelect(Tiny, Zero, X), X);
  };
  for (Value *&X : A)
    X = Flushed(X);
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
    // The function rounded to nearest, or its _r form in the rounding
    // direction.
    auto RLib = [&](StringRef Name, Type *R, ArrayRef<Value *> Args) {
      if (Nearest)
        return Lib(Name, R, Args);
      SmallVector<Value *, 4> RArgs(Args);
      RArgs.push_back(Rounding);
      return Lib((Name + "_r").str(), R, RArgs);
    };
    if (B2 == "fadd")
      R = RLib("__air_f64_add", I64, {A[0], A[1]});
    else if (B2 == "fsub")
      R = RLib("__air_f64_sub", I64, {A[0], A[1]});
    else if (B2 == "fmul")
      R = RLib("__air_f64_mul", I64, {A[0], A[1]});
    else if (B2 == "fdiv")
      R = RLib("__air_f64_div", I64, {A[0], A[1]});
    else if (B2 == "fma" || B2 == "fmuladd")
      R = RLib("__air_f64_fma", I64, {A[0], A[1], A[2]});
    else if (B2 == "sqrt")
      R = RLib("__air_f64_sqrt", I64, {A[0]});
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
    } else if (B2 == "roundeven")
      R = Lib("__air_f64_roundeven", I64, {A[0]});
    else if (B2 == "rint" || B2 == "nearbyint") {
      // In the rounding direction.
      Value *X = A[0], *S = B.getInt64(0x8000000000000000ull);
      R = Lib("__air_f64_roundeven", I64, {X});
      if (!Nearest) {
        Value *Floor = Lib("__air_f64_floor", I64, {X});
        Value *Ceil =
            B.CreateXor(Lib("__air_f64_floor", I64, {B.CreateXor(X, S)}), S);
        Value *Abs = B.CreateAnd(X, B.getInt64(0x7fffffffffffffffull));
        Value *Trunc = B.CreateOr(Lib("__air_f64_floor", I64, {Abs}),
                                  B.CreateAnd(X, S));
        auto Is = [&](unsigned C) {
          return B.CreateICmpEQ(Rounding, B.getInt32(C));
        };
        R = B.CreateSelect(Is(0), Trunc, R);
        R = B.CreateSelect(Is(2), Ceil, R);
        R = B.CreateSelect(Is(3), Floor, R);
      }
    }
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
    } else if (B2 == "fptosi" || B2 == "fptoui") {
      // Saturating, NaN to 0 (llvm.fpto[su]i.sat; out of range values are
      // poison for the instructions).
      bool Signed = B2 == "fptosi";
      R = Lib(Signed ? "__air_f64_to_i64" : "__air_f64_to_u64", I64, {A[0]});
      unsigned W = Ret->getScalarSizeInBits();
      if (W < 64) {
        R = Signed ? B.CreateBinaryIntrinsic(
                         Intrinsic::smax,
                         B.CreateBinaryIntrinsic(
                             Intrinsic::smin, R,
                             B.getInt64(APInt::getSignedMaxValue(W).getSExtValue())),
                         B.getInt64(APInt::getSignedMinValue(W).getSExtValue()))
                   : B.CreateBinaryIntrinsic(
                         Intrinsic::umin, R,
                         B.getInt64(APInt::getMaxValue(W).getZExtValue()));
        R = B.CreateTrunc(R, Ret);
      }
    }
    else if (B2 == "sitofp")
      R = RLib("__air_f64_from_i64", I64, {B.CreateSExt(A[0], I64)});
    else if (B2 == "uitofp")
      R = RLib("__air_f64_from_u64", I64, {B.CreateZExt(A[0], I64)});
    else if (B2 == "fpext") {
      Value *X = A[0];
      if (!X->getType()->isFloatTy())
        X = B.CreateFPExt(X, F32);
      R = Lib("__air_f64_from_f32", I64, {B.CreateBitCast(X, I32)});
    } else if (B2 == "fptrunc") {
      // Rounded once, to the result's format.
      if (Ret->isFloatTy())
        R = B.CreateBitCast(RLib("__air_f64_to_f32", I32, {A[0]}), F32);
      else if (Ret->isHalfTy() || Ret->isBFloatTy())
        R = B.CreateBitCast(
            B.CreateTrunc(Lib(Ret->isHalfTy() ? "__air_f64_to_f16_r"
                                              : "__air_f64_to_bf16_r",
                              I32, {A[0], Rounding}),
                          B.getInt16Ty()),
            Ret);
    }
    if (!R)
      return NoEmulation();
    // NaN results canonical; f64 results only (not comparisons or integer
    // conversions).
    // A NaN made from operands that are not NaN is the canonical NaN; NaN
    // operands propagate.
    if (CanonicalNaN && DblOut && R->getType()->isIntegerTy(64)) {
      auto NaN = [&](Value *X) {
        return B.CreateICmpUGT(
            B.CreateAnd(X, B.getInt64(0x7fffffffffffffffull)),
            B.getInt64(0x7ff0000000000000ull));
      };
      Value *Made = NaN(R);
      for (auto [I, X] : enumerate(A))
        if (I + 1 < Mask.size() && Mask[I + 1] == 'd')
          Made = B.CreateAnd(Made, B.CreateNot(NaN(X)));
      R = B.CreateSelect(Made, B.getInt64(0x7fffffffffffffffull), R);
    }
    return Fin(R);
  }

  if (Intrinsic::ID ID = Intrinsic::lookupIntrinsicID(
          Op.starts_with("call:") ? Op.drop_front(5) : StringRef());
      is_contained({Intrinsic::minimum, Intrinsic::maximum,
                    Intrinsic::minimumnum, Intrinsic::maximumnum},
                   ID)) {
    // On the bits: AIR's fmin and fmax, which the intrinsics would become,
    // may flush denormals and order zeros either way. The operands are
    // ordered by a key monotonic in their values (-0 below +0); minimum and
    // maximum return a NaN operand (quieted), the *num forms the other one.
    const fltSemantics &Sem = Ret->getScalarType()->getFltSemantics();
    unsigned W = Ret->getScalarSizeInBits();
    Type *IT = Ret->getWithNewType(B.getIntNTy(W));
    auto C = [&](const APInt &V) { return ConstantInt::get(IT, V); };
    APInt Sign = APInt::getSignMask(W);
    APInt Inf = APFloat::getInf(Sem).bitcastToAPInt();
    APInt Quiet = APInt::getOneBitSet(W, APFloat::semanticsPrecision(Sem) - 2);
    Value *X = B.CreateBitCast(A[0], IT), *Y = B.CreateBitCast(A[1], IT);
    auto Key = [&](Value *V) {
      return B.CreateSelect(B.CreateICmpSLT(V, C(APInt(W, 0))), B.CreateNot(V),
                            B.CreateOr(V, C(Sign)));
    };
    auto IsNaN = [&](Value *V) {
      return B.CreateICmpUGT(B.CreateAnd(V, C(~Sign)), C(Inf));
    };
    bool Min = ID == Intrinsic::minimum || ID == Intrinsic::minimumnum;
    Value *PickX = Min ? B.CreateICmpULT(Key(X), Key(Y))
                       : B.CreateICmpUGT(Key(X), Key(Y));
    Value *R = B.CreateSelect(PickX, X, Y);
    if (ID == Intrinsic::minimumnum || ID == Intrinsic::maximumnum) {
      R = B.CreateSelect(IsNaN(Y), X, R);
      R = B.CreateSelect(IsNaN(X), Y, R);
      R = B.CreateSelect(IsNaN(R), B.CreateOr(R, C(Quiet)), R);
    } else {
      R = B.CreateSelect(IsNaN(Y), B.CreateOr(Y, C(Quiet)), R);
      R = B.CreateSelect(IsNaN(X), B.CreateOr(X, C(Quiet)), R);
    }
    if (CanonicalNaN)
      R = B.CreateSelect(IsNaN(R), C(APInt::getSignedMaxValue(W)), R);
    return Fin(Flushed(B.CreateBitCast(R, Ret)));
  }
  if (Op.starts_with("call:llvm.experimental.constrained.")) {
    // Rounded to nearest by the AIR operation, and in a directed rounding
    // computed in double precision in that direction and rounded again in
    // it (exact: the values of the format are doubles).
    Value *Directed = nullptr;
    if (!Nearest)
      Directed = directed(B, Base, A, Ret, Rounding, Lib, CanonicalNaN);
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
    if (Directed)
      R = RoundingC ? Directed
                    : B.CreateSelect(B.CreateICmpEQ(Rounding, B.getInt32(1)),
                                     R, Directed);
    return Fin(Flushed(R));
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
    // the tracked mode, or rounding to nearest even without flushing
    R = Mode ? Mode : B.getInt32(1 | (1 << 2));
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
      // A hint that AGX had it by value.
      AB.addAttribute("air-by-value");
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
  // Kernels: the agx_kernel functions or, in a module without them, the
  // external void functions nothing calls (AGX compiles each external
  // function of such a module as an entry point, AGXUsage.rst).
  SmallVector<Function *, 8> Kernels;
  bool HaveKernelCC = any_of(M, [](const Function &F) {
    return !F.isDeclaration() && F.getCallingConv() == CallingConv::AGX_KERNEL;
  });
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    bool Called = any_of(F.users(), [](const User *U) {
      return isa<CallBase>(U);
    });
    if (HaveKernelCC ? F.getCallingConv() == CallingConv::AGX_KERNEL
                     : !F.hasLocalLinkage() && !Called &&
                           F.getReturnType()->isVoidTy())
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

  // llvm.used and llvm.compiler.used mean nothing in AIR (the globals they
  // list are external, which keeps them), and a list of device pointers is
  // not what they hold.
  for (StringRef Name : {"llvm.used", "llvm.compiler.used"})
    if (GlobalVariable *GV = M.getGlobalVariable(Name))
      GV->eraseFromParent();
  // Every global value takes AIR's types, as in agx-lower-air.
  AGXToAIRTypes Types;
  if (!makesLocalGeneric(M))
    Types.GenericAS = air::AS::Device;
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
    unsigned AS = Types.addrSpace(GV.getAddressSpace());
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
    DoubleMaterializer Materializer(VMap, Types);
    return MapValue(C, VMap, RF_None, &Types, &Materializer);
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
    DoubleMaterializer Materializer(VMap, Types);
    CloneFunctionInto(New, Old, VMap, CloneFunctionChangeType::LocalChangesOnly,
                      Returns, "", nullptr, &Types, &Materializer);
  }
  // Casts between generic and device pointers, the same address space now.
  for (auto [Old, New] : Functions)
    for (Instruction &I : make_early_inc_range(instructions(*New)))
      if (auto *C = dyn_cast<AddrSpaceCastInst>(&I);
          C && C->getSrcTy() == C->getDestTy()) {
        C->replaceAllUsesWith(C->getOperand(0));
        C->eraseFromParent();
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
  inlineEmulations();
  return !Failed;
}

/// The hints that do not stay hints (agx-raise-hints) become their AIR
/// emulations, inlined at each call and simplified there (the FP mode and
/// the rounding are mostly constants at the call), so the AIR has only AIR
/// operations and the soft-f64 library.
void RaiseToAIR::inlineEmulations() {
  SmallVector<StringRef, 4> Prefixes;
  StringRef(KeepHints).split(Prefixes, ',', -1, false);
  auto Keep = [&](StringRef Key) {
    return any_of(Prefixes, [&](StringRef P) {
      return P == "all" || Key.starts_with(P);
    });
  };
  // AIR has no floating point environment: no strictfp (whose callers would
  // turn the inlined emulations into constrained intrinsics).
  for (Function &F : M) {
    F.removeFnAttr(Attribute::StrictFP);
    for (Instruction &I : instructions(F))
      if (auto *CB = dyn_cast<CallBase>(&I))
        CB->removeFnAttr(Attribute::StrictFP);
  }
  SmallPtrSet<Function *, 8> Callers;
  for (Function &F : make_early_inc_range(M)) {
    if (!F.hasFnAttribute(air::HintAttr) || F.isDeclaration())
      continue;
    StringRef Key = F.getFnAttribute(air::HintAttr).getValueAsString();
    if (Keep(Key) || F.hasFnAttribute("air-hint-no-emulation"))
      continue;
    F.setName(""); // (its name would be left in the inlined blocks')
    for (User *U : make_early_inc_range(F.users())) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->getCalledFunction() != &F)
        continue;
      Callers.insert(CI->getFunction());
      if (CI->getFunctionType() != F.getFunctionType()) {
        std::string S;
        raw_string_ostream OS(S);
        OS << "the emulation of " << Key << " is " << *F.getFunctionType()
           << ", its call " << *CI->getFunctionType();
        error(CI->getFunction(), OS.str());
        continue;
      }
      InlineFunctionInfo IFI;
      if (!InlineFunction(*CI, IFI).isSuccess())
        error(CI->getFunction(), "cannot inline the emulation of " + Key);
    }
    if (F.use_empty())
      F.eraseFromParent();
  }
  // Fold what the constants at the calls decide (dead soft-f64 calls are
  // memory(none), so they go too).
  const DataLayout &DL = M.getDataLayout();
  for (Function *F : Callers) {
    for (bool Changed = true; Changed;) {
      Changed = false;
      for (BasicBlock &BB : *F)
        for (Instruction &I : make_early_inc_range(BB)) {
          if (Value *V = simplifyInstruction(&I, SimplifyQuery(DL, &I))) {
            I.replaceAllUsesWith(V);
            Changed = true;
          }
          if (isInstructionTriviallyDead(&I)) {
            I.eraseFromParent();
            Changed = true;
          }
        }
      for (BasicBlock &BB : *F)
        Changed |= ConstantFoldTerminator(&BB, true);
      Changed |= removeUnreachableBlocks(*F);
    }
    for (BasicBlock &BB : make_early_inc_range(*F))
      MergeBlockIntoPredecessor(&BB);
  }
  // Library functions nothing calls any more.
  for (bool Erased = true; Erased;) {
    Erased = false;
    for (Function &F : make_early_inc_range(M))
      if (F.hasLocalLinkage() && F.getName().starts_with("__air_f64_") &&
          F.use_empty()) {
        F.eraseFromParent();
        Erased = true;
      }
  }
}

} // namespace

PreservedAnalyses AGXRaiseToAIRPass::run(Module &M, ModuleAnalysisManager &) {
  return RaiseToAIR(M, TM).run() ? PreservedAnalyses::none()
                             : PreservedAnalyses::all();
}
