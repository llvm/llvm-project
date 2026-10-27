//===-- AGXLowerAIR.cpp - AIR kernels as AGX kernels ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compiles AIR for AGX: a module that describes its kernels in !air.kernel, as
// the AIR target's portable passes leave it (addAIRPortablePasses), becomes an
// AGX module. See llvm/docs/AGXUsage.rst, "AIR input".
//
// The module may only use what AIR has: integers of 8 to 64 bits (i1 only
// outside memory), half, bfloat and float, and pointers in AIR's four address
// spaces, with no casts between them. Anything else is an error.
//
// Address spaces become AGX's: thread (0) becomes private (5) and constant (2)
// becomes AGX's constant (4); device (1) and threadgroup (3) keep their
// numbers. Each kernel becomes an agx_kernel taking its device and constant
// buffers in location order. A constant buffer that the kernel only loads from,
// within the size !air.kernel gives it, is passed by reference (byref), so its
// bytes go in uniform registers. System-value parameters become reads of AGX's
// special registers, and the threadgroup buffer becomes the launch's dynamic
// threadgroup memory. Finally the module takes AGX's triple and data layout.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXTargetMachine.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/AIR.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#define DEBUG_TYPE "agx-lower-air"

namespace {

/// What !air.kernel says a kernel parameter is.
struct Param {
  enum { Undescribed, Buffer, SystemValue } Kind = Undescribed;
  // Buffer.
  unsigned AddrSpace = 0;
  unsigned Location = 0;
  unsigned TypeSize = 0;
  StringRef TypeName;
  // System value: its role, such as air.thread_position_in_grid.
  StringRef Role;
};

struct AIRKernel {
  Function *F;
  SmallVector<Param, 8> Params;
  Function *NewF = nullptr;
};

/// AIR's address spaces become AGX's in every type.
class AIRToAGXTypes : public ValueMapTypeRemapper {
  DenseMap<Type *, Type *> Cache;

public:
  static unsigned addrSpace(unsigned AS) {
    switch (AS) {
    case air::AS::Default:
      return AGX::PRIVATE_ADDRESS;
    case air::AS::Constant:
      return AGX::CONSTANT_ADDRESS;
    default:
      return AS;
    }
  }

  Type *remapType(Type *T) override {
    auto It = Cache.find(T);
    if (It != Cache.end())
      return It->second;
    Type *R = T;
    if (auto *PT = dyn_cast<PointerType>(T)) {
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
                                     ST->getName().str() + ".agx",
                                     ST->isPacked());
    }
    Cache[T] = R;
    return R;
  }
};

class LowerAIR {
  Module &M;
  const TargetMachine &TM;
  LLVMContext &Ctx;
  bool Failed = false;

public:
  LowerAIR(Module &M, const TargetMachine &TM)
      : M(M), TM(TM), Ctx(M.getContext()) {}
  bool run();

private:
  void error(const Function *F, const Twine &Msg) {
    if (F)
      Ctx.diagnose(DiagnosticInfoUnsupported(*F, Msg));
    else
      Ctx.diagnose(DiagnosticInfoGeneric(Msg, DS_Error));
    Failed = true;
  }
  bool parseKernels(NamedMDNode &MD, SmallVectorImpl<AIRKernel> &Kernels);
  void checkTypes();
  Function *newKernel(AIRKernel &K, AIRToAGXTypes &Types);
  void cloneKernel(AIRKernel &K, Function *NewF, ValueToValueMapTy &VMap,
                   AIRToAGXTypes &Types);
  Value *systemValue(IRBuilder<> &B, StringRef Role, Type *Ty, Function *F);
  void passConstantBuffersByReference(AIRKernel &K, Function *NewF);
  void dropAIRMetadata();
};

} // namespace

static unsigned mdInt(const MDOperand &Op) {
  if (auto *C = mdconst::dyn_extract_or_null<ConstantInt>(Op))
    return C->getZExtValue();
  return ~0u;
}

bool LowerAIR::parseKernels(NamedMDNode &MD,
                            SmallVectorImpl<AIRKernel> &Kernels) {
  for (MDNode *N : MD.operands()) {
    auto *F = N->getNumOperands()
                  ? mdconst::dyn_extract_or_null<Function>(N->getOperand(0))
                  : nullptr;
    if (!F) {
      error(nullptr, "an !air.kernel entry names no function");
      continue;
    }
    if (N->getNumOperands() < 3 || !isa<MDNode>(N->getOperand(2))) {
      error(F, "the !air.kernel entry of this kernel describes no parameters");
      continue;
    }
    AIRKernel K{F, SmallVector<Param, 8>(F->arg_size())};
    for (const MDOperand &PO : cast<MDNode>(N->getOperand(2))->operands()) {
      auto *P = dyn_cast<MDNode>(PO);
      unsigned ArgNo = P && P->getNumOperands() >= 2 ? mdInt(P->getOperand(0))
                                                     : ~0u;
      auto *RoleMD = ArgNo != ~0u ? dyn_cast<MDString>(P->getOperand(1))
                                  : nullptr;
      if (!RoleMD || ArgNo >= F->arg_size()) {
        error(F, "a parameter node of !air.kernel is malformed");
        continue;
      }
      Param &Pm = K.Params[ArgNo];
      StringRef Role = RoleMD->getString();
      if (Role == air::IndirectBuffer) {
        error(F, "argument buffers (air.indirect_buffer) are not supported");
        continue;
      }
      if (Role != air::Buffer) {
        Pm.Kind = Param::SystemValue;
        Pm.Role = Role;
        continue;
      }
      Pm.Kind = Param::Buffer;
      for (unsigned I = 2, E = P->getNumOperands(); I < E; ++I) {
        auto *Key = dyn_cast<MDString>(P->getOperand(I));
        if (!Key || I + 1 == E)
          continue;
        StringRef S = Key->getString();
        if (S == air::LocationIndex)
          Pm.Location = mdInt(P->getOperand(++I));
        else if (S == air::AddressSpace)
          Pm.AddrSpace = mdInt(P->getOperand(++I));
        else if (S == air::ArgTypeSize)
          Pm.TypeSize = mdInt(P->getOperand(++I));
        else if (S == air::ArgTypeName)
          if (auto *Name = dyn_cast<MDString>(P->getOperand(++I)))
            Pm.TypeName = Name->getString();
      }
    }
    for (Argument &A : F->args())
      if (K.Params[A.getArgNo()].Kind == Param::Undescribed)
        error(F, "parameter " + Twine(A.getArgNo()) +
                     " is not described in !air.kernel");

    // Device and constant buffers share the location numbers, in parameter
    // order; threadgroup buffers have their own.
    unsigned NextLocation = 0, Threadgroup = 0;
    for (Param &Pm : K.Params) {
      if (Pm.Kind != Param::Buffer)
        continue;
      if (Pm.AddrSpace == air::AS::Threadgroup) {
        if (Threadgroup++)
          error(F, "more than one threadgroup buffer parameter");
      } else if (Pm.AddrSpace != air::AS::Device &&
                 Pm.AddrSpace != air::AS::Constant) {
        error(F, "a buffer in address space " + Twine(Pm.AddrSpace));
      } else if (Pm.Location != NextLocation++) {
        error(F, "buffer locations are not in parameter order");
      }
    }
    Kernels.push_back(std::move(K));
  }
  return !Failed;
}

// The types AIR has (llvm/docs/AGXUsage.rst, "AIR input").
static bool isAIRType(Type *T) {
  if (auto *VT = dyn_cast<VectorType>(T))
    return isAIRType(VT->getElementType());
  if (auto *AT = dyn_cast<ArrayType>(T))
    return isAIRType(AT->getElementType());
  if (auto *ST = dyn_cast<StructType>(T))
    return all_of(ST->elements(), isAIRType);
  if (auto *FT = dyn_cast<FunctionType>(T))
    return isAIRType(FT->getReturnType()) && all_of(FT->params(), isAIRType);
  if (auto *IT = dyn_cast<IntegerType>(T))
    return is_contained({1u, 8u, 16u, 32u, 64u}, IT->getBitWidth());
  if (auto *PT = dyn_cast<PointerType>(T))
    return PT->getAddressSpace() <= air::AS::Threadgroup;
  return T->isHalfTy() || T->isBFloatTy() || T->isFloatTy() ||
         T->isVoidTy() || T->isLabelTy() || T->isMetadataTy() ||
         T->isTokenTy();
}

static bool hasI1(Type *T) { return T->getScalarType()->isIntegerTy(1); }

void LowerAIR::checkTypes() {
  for (GlobalVariable &GV : M.globals())
    if (!GV.getName().starts_with("llvm.") &&
        (!isAIRType(GV.getValueType()) ||
         GV.getAddressSpace() > air::AS::Threadgroup))
      error(nullptr, "global " + GV.getName() + " has a type AIR does not have");
  for (Function &F : M) {
    if (F.isIntrinsic())
      continue;
    if (!isAIRType(F.getFunctionType())) {
      error(&F, "a type AIR does not have in the signature of " + F.getName());
      continue;
    }
    for (Instruction &I : instructions(F)) {
      auto Bad = [&](const Twine &What) {
        error(&F, What + " in " + F.getName() + " (AIR does not have it)");
      };
      if (isa<AddrSpaceCastInst>(I)) {
        Bad("an address space cast");
        break;
      }
      if (!isAIRType(I.getType()) ||
          any_of(I.operands(),
                 [](const Use &U) { return !isAIRType(U->getType()); })) {
        Bad("a value of a type AIR does not have");
        break;
      }
      Type *MemTy = nullptr;
      if (auto *L = dyn_cast<LoadInst>(&I))
        MemTy = L->getType();
      else if (auto *S = dyn_cast<StoreInst>(&I))
        MemTy = S->getValueOperand()->getType();
      else if (auto *A = dyn_cast<AllocaInst>(&I))
        MemTy = A->getAllocatedType();
      if (MemTy && hasI1(MemTy)) {
        Bad("an i1 in memory");
        break;
      }
    }
  }
}

// The type named by air.arg_type_name, or nullptr for a name that is not a
// scalar or vector of AIR.
static Type *namedType(StringRef Name, LLVMContext &Ctx) {
  unsigned N = 1;
  if (!Name.empty() && Name.back() >= '2' && Name.back() <= '4') {
    N = Name.back() - '0';
    Name = Name.drop_back();
  }
  Type *T = StringSwitch<Type *>(Name)
                .Cases({"char", "uchar", "bool"}, Type::getInt8Ty(Ctx))
                .Cases({"short", "ushort"}, Type::getInt16Ty(Ctx))
                .Cases({"int", "uint"}, Type::getInt32Ty(Ctx))
                .Cases({"long", "ulong"}, Type::getInt64Ty(Ctx))
                .Case("half", Type::getHalfTy(Ctx))
                .Case("bfloat", Type::getBFloatTy(Ctx))
                .Case("float", Type::getFloatTy(Ctx))
                .Default(nullptr);
  if (T && N > 1)
    T = FixedVectorType::get(T, N);
  return T;
}

Value *LowerAIR::systemValue(IRBuilder<> &B, StringRef Role, Type *Ty,
                             Function *F) {
  auto Read = [&](Intrinsic::ID ID) { return B.CreateIntrinsic(ID, {}); };
  auto XYZ = [&](Intrinsic::ID X, Intrinsic::ID Y, Intrinsic::ID Z) {
    return SmallVector<Value *, 3>{Read(X), Read(Y), Read(Z)};
  };
  using namespace Intrinsic;
  SmallVector<Value *, 3> C;
  if (Role == air::ThreadPositionInGrid)
    C = XYZ(agx_thread_position_in_grid_x, agx_thread_position_in_grid_y,
            agx_thread_position_in_grid_z);
  else if (Role == air::ThreadgroupPositionInGrid)
    C = XYZ(agx_threadgroup_position_in_grid_x,
            agx_threadgroup_position_in_grid_y,
            agx_threadgroup_position_in_grid_z);
  else if (Role == air::ThreadPositionInThreadgroup)
    C = XYZ(agx_thread_position_in_threadgroup_x,
            agx_thread_position_in_threadgroup_y,
            agx_thread_position_in_threadgroup_z);
  else if (Role == air::ThreadsPerThreadgroup)
    C = XYZ(agx_threads_per_threadgroup_x, agx_threads_per_threadgroup_y,
            agx_threads_per_threadgroup_z);
  else if (Role == air::ThreadgroupsPerGrid)
    C = XYZ(agx_threadgroups_per_grid_x, agx_threadgroups_per_grid_y,
            agx_threadgroups_per_grid_z);
  else if (Role == air::ThreadsPerGrid) {
    // Grids are whole threadgroups on AGX.
    auto G = XYZ(agx_threadgroups_per_grid_x, agx_threadgroups_per_grid_y,
                 agx_threadgroups_per_grid_z);
    auto T = XYZ(agx_threads_per_threadgroup_x, agx_threads_per_threadgroup_y,
                 agx_threads_per_threadgroup_z);
    for (unsigned I = 0; I < 3; ++I)
      C.push_back(B.CreateMul(G[I], T[I]));
  } else if (Role == air::ThreadIndexInThreadgroup)
    C = {Read(agx_thread_index_in_threadgroup)};
  else if (Role == air::ThreadIndexInSimdgroup)
    C = {Read(agx_thread_index_in_simdgroup)};
  else if (Role == air::SimdgroupIndexInThreadgroup)
    C = {Read(agx_simdgroup_index_in_threadgroup)};
  else if (Role == air::ThreadsPerSimdgroup)
    C = {B.getInt32(32)};
  else if (Role == air::SimdgroupsPerThreadgroup) {
    auto T = XYZ(agx_threads_per_threadgroup_x, agx_threads_per_threadgroup_y,
                 agx_threads_per_threadgroup_z);
    Value *N = B.CreateMul(B.CreateMul(T[0], T[1]), T[2]);
    C = {B.CreateLShr(B.CreateAdd(N, B.getInt32(31)), 5)};
  } else {
    error(F, "the system value " + Role + " has no AGX equivalent");
    return PoisonValue::get(Ty);
  }

  // As the parameter has it: a vector (or array) of its components or the
  // first one, in its integer width.
  auto Component = [&](unsigned I, Type *EltTy) -> Value * {
    if (!EltTy->isIntegerTy()) {
      error(F, "the system value " + Role + " is not an integer");
      return PoisonValue::get(EltTy);
    }
    return I < C.size() ? B.CreateZExtOrTrunc(C[I], EltTy)
                        : ConstantInt::get(EltTy, 0);
  };
  if (auto *VT = dyn_cast<FixedVectorType>(Ty)) {
    Value *V = PoisonValue::get(VT);
    for (unsigned I = 0; I < VT->getNumElements(); ++I)
      V = B.CreateInsertElement(V, Component(I, VT->getElementType()), I);
    return V;
  }
  if (auto *AT = dyn_cast<ArrayType>(Ty)) {
    Value *V = PoisonValue::get(AT);
    for (unsigned I = 0; I < AT->getNumElements(); ++I)
      V = B.CreateInsertValue(V, Component(I, AT->getElementType()), I);
    return V;
  }
  return Component(0, Ty);
}

Function *LowerAIR::newKernel(AIRKernel &K, AIRToAGXTypes &Types) {
  Function &F = *K.F;
  SmallVector<Type *, 8> Params;
  for (Argument &A : F.args()) {
    const Param &Pm = K.Params[A.getArgNo()];
    if (Pm.Kind == Param::Buffer && Pm.AddrSpace != air::AS::Threadgroup)
      Params.push_back(Types.remapType(A.getType()));
  }
  auto *FT = FunctionType::get(Type::getVoidTy(Ctx), Params, false);
  Function *NewF = Function::Create(FT, F.getLinkage(), F.getAddressSpace(),
                                    F.getName() + ".agx", &M);
  NewF->setCallingConv(CallingConv::AGX_KERNEL);
  return NewF;
}

void LowerAIR::cloneKernel(AIRKernel &K, Function *NewF,
                           ValueToValueMapTy &VMap, AIRToAGXTypes &Types) {
  Function &F = *K.F;
  if (!F.getReturnType()->isVoidTy())
    error(&F, "a kernel that returns a value");

  // The system values are read in a block ahead of the kernel's body, which
  // the body's entry block is then merged into.
  BasicBlock *Pre = BasicBlock::Create(Ctx, "entry", NewF);
  IRBuilder<> B(Pre);
  auto NewArg = NewF->arg_begin();
  for (Argument &A : F.args()) {
    const Param &Pm = K.Params[A.getArgNo()];
    if (Pm.Kind == Param::SystemValue) {
      VMap[&A] = systemValue(B, Pm.Role, A.getType(), &F);
    } else if (Pm.AddrSpace == air::AS::Threadgroup) {
      // Zero-sized threadgroup variables are the dynamic threadgroup memory
      // of the launch.
      auto *TG = new GlobalVariable(
          M, ArrayType::get(Type::getInt8Ty(Ctx), 0), false,
          GlobalValue::ExternalLinkage, nullptr,
          F.getName() + "." + A.getName(), nullptr,
          GlobalValue::NotThreadLocal, AGX::THREADGROUP_ADDRESS);
      TG->setAlignment(Align(16));
      VMap[&A] = TG;
    } else {
      NewArg->setName(A.getName());
      VMap[&A] = &*NewArg++;
    }
  }

  SmallVector<ReturnInst *, 4> Returns;
  CloneFunctionInto(NewF, &F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns, "", nullptr, &Types);
  NewF->setCallingConv(CallingConv::AGX_KERNEL);
  NewF->removeFnAttr("air-kernel");
  auto *Body = cast<BasicBlock>(VMap[&F.getEntryBlock()]);
  B.CreateBr(Body);
  MergeBlockIntoPredecessor(Body);
}

// Whether every use of Ptr, at Offset bytes into the buffer, is a simple load
// within its first Size bytes, directly or through constant offsets.
static bool onlyLoadsWithin(Value *Ptr, int64_t Offset, uint64_t Size,
                            const DataLayout &DL) {
  for (User *U : Ptr->users()) {
    if (auto *L = dyn_cast<LoadInst>(U)) {
      if (!L->isSimple() || Offset < 0 ||
          uint64_t(Offset) + DL.getTypeStoreSize(L->getType()) > Size)
        return false;
      continue;
    }
    auto *GEP = dyn_cast<GEPOperator>(U);
    APInt Off(DL.getIndexTypeSizeInBits(GEP ? GEP->getType() : Ptr->getType()),
              0);
    if (!GEP || !GEP->accumulateConstantOffset(DL, Off) ||
        !onlyLoadsWithin(GEP, Offset + Off.getSExtValue(), Size, DL))
      return false;
  }
  return true;
}

void LowerAIR::passConstantBuffersByReference(AIRKernel &K, Function *NewF) {
  const DataLayout &DL = M.getDataLayout();
  auto NewArg = NewF->arg_begin();
  for (const Param &Pm : K.Params) {
    if (Pm.Kind != Param::Buffer || Pm.AddrSpace == air::AS::Threadgroup)
      continue;
    Argument &A = *NewArg++;
    if (Pm.AddrSpace != air::AS::Constant)
      continue;
    Type *T = namedType(Pm.TypeName, Ctx);
    if (!T || DL.getTypeAllocSize(T) != Pm.TypeSize) {
      // A type AIR describes by size alone (a structure) is held in bytes.
      if (!Pm.TypeSize)
        continue;
      T = ArrayType::get(Type::getInt8Ty(Ctx), Pm.TypeSize);
    }
    if (onlyLoadsWithin(&A, 0, Pm.TypeSize, DL))
      A.addAttr(Attribute::getWithByRefType(Ctx, T));
  }
}

void LowerAIR::dropAIRMetadata() {
  for (StringRef Name : {"air.kernel", "air.version", "air.language_version",
                         "air.compile_options"})
    if (NamedMDNode *MD = M.getNamedMetadata(Name))
      M.eraseNamedMetadata(MD);
  // The air.max_* limits among the module flags.
  NamedMDNode *Flags = M.getModuleFlagsMetadata();
  if (!Flags)
    return;
  SmallVector<MDNode *, 8> Keep;
  for (MDNode *Flag : Flags->operands()) {
    auto *Key = Flag->getNumOperands() > 1
                    ? dyn_cast<MDString>(Flag->getOperand(1))
                    : nullptr;
    if (!Key || !Key->getString().starts_with("air."))
      Keep.push_back(Flag);
  }
  Flags->clearOperands();
  for (MDNode *Flag : Keep)
    Flags->addOperand(Flag);
  if (Keep.empty())
    M.eraseNamedMetadata(Flags);
}

bool LowerAIR::run() {
  NamedMDNode *KernelMD = M.getNamedMetadata(air::KernelMD);
  if (!KernelMD)
    return false;
  SmallVector<AIRKernel, 4> Kernels;
  parseKernels(*KernelMD, Kernels);
  checkTypes();
  if (Failed)
    return false;

  // Every global value takes the types of AGX's address spaces: definitions
  // and the declarations whose types change are recreated, and the old ones
  // erased once nothing refers to them.
  AIRToAGXTypes Types;
  ValueToValueMapTy VMap;
  SmallVector<std::pair<GlobalVariable *, GlobalVariable *>, 8> Globals;
  SmallVector<std::pair<Function *, Function *>, 8> Functions;
  DenseMap<Function *, AIRKernel *> KernelOf;
  for (AIRKernel &K : Kernels)
    KernelOf[K.F] = &K;

  for (GlobalVariable &GV : M.globals()) {
    Type *T = Types.remapType(GV.getValueType());
    unsigned AS = AIRToAGXTypes::addrSpace(GV.getAddressSpace());
    if (T == GV.getValueType() && AS == GV.getAddressSpace())
      continue;
    auto *NewGV = new GlobalVariable(
        M, T, GV.isConstant(), GV.getLinkage(), nullptr, GV.getName() + ".agx",
        nullptr, GV.getThreadLocalMode(), AS);
    NewGV->copyAttributesFrom(&GV);
    VMap[&GV] = NewGV;
    Globals.push_back({&GV, NewGV});
  }
  for (Function &F : make_early_inc_range(M)) {
    if (AIRKernel *K = KernelOf.lookup(&F)) {
      Function *NewF = K->NewF = newKernel(*K, Types);
      VMap[&F] = NewF;
      Functions.push_back({&F, NewF});
      continue;
    }
    auto *FT = cast<FunctionType>(Types.remapType(F.getFunctionType()));
    if (F.isDeclaration() && FT == F.getFunctionType())
      continue;
    if (F.isIntrinsic()) {
      SmallVector<Type *, 4> Overloads;
      if (!Intrinsic::isSignatureValid(F.getIntrinsicID(), FT, Overloads)) {
        error(nullptr, "no AGX form of " + F.getName());
        continue;
      }
      VMap[&F] =
          Intrinsic::getOrInsertDeclaration(&M, F.getIntrinsicID(), Overloads);
      continue;
    }
    Function *NewF = Function::Create(FT, F.getLinkage(), F.getAddressSpace(),
                                      F.getName() + ".agx", &M);
    NewF->copyAttributesFrom(&F);
    VMap[&F] = NewF;
    Functions.push_back({&F, NewF});
  }
  if (Failed)
    return false;

  for (auto [Old, New] : Globals)
    if (Old->hasInitializer())
      New->setInitializer(
          MapValue(Old->getInitializer(), VMap, RF_None, &Types));
  for (auto [Old, New] : Functions) {
    if (Old->isDeclaration())
      continue;
    if (AIRKernel *K = KernelOf.lookup(Old)) {
      cloneKernel(*K, New, VMap, Types);
      continue;
    }
    auto NewArg = New->arg_begin();
    for (Argument &A : Old->args()) {
      NewArg->setName(A.getName());
      VMap[&A] = &*NewArg++;
    }
    SmallVector<ReturnInst *, 4> Returns;
    CloneFunctionInto(New, Old, VMap, CloneFunctionChangeType::LocalChangesOnly,
                      Returns, "", nullptr, &Types);
  }

  for (auto [Old, New] : Functions)
    Old->dropAllReferences();
  for (auto [Old, New] : Globals)
    Old->setInitializer(nullptr);
  for (auto [Old, New] : Functions) {
    std::string Name = Old->getName().str();
    Old->replaceAllUsesWith(PoisonValue::get(Old->getType()));
    Old->eraseFromParent();
    New->setName(Name);
  }
  for (auto [Old, New] : Globals) {
    std::string Name = Old->getName().str();
    Old->replaceAllUsesWith(PoisonValue::get(Old->getType()));
    Old->eraseFromParent();
    New->setName(Name);
  }
  // Intrinsic declarations left without uses.
  for (Function &F : make_early_inc_range(M))
    if (F.isIntrinsic() && F.use_empty())
      F.eraseFromParent();

  dropAIRMetadata();
  M.setTargetTriple(TM.getTargetTriple());
  M.setDataLayout(TM.createDataLayout());
  for (AIRKernel &K : Kernels)
    passConstantBuffersByReference(K, K.NewF);
  return true;
}

namespace {
class AGXLowerAIRLegacy : public ModulePass {
public:
  static char ID;
  const AGXTargetMachine *TM;
  explicit AGXLowerAIRLegacy(const AGXTargetMachine *TM = nullptr)
      : ModulePass(ID), TM(TM) {}

  StringRef getPassName() const override { return "AGX Lower AIR"; }
  bool runOnModule(Module &M) override {
    return TM && LowerAIR(M, *TM).run();
  }
};
} // namespace

char AGXLowerAIRLegacy::ID = 0;

INITIALIZE_PASS(AGXLowerAIRLegacy, DEBUG_TYPE, "AGX Lower AIR", false, false)

ModulePass *llvm::createAGXLowerAIRPass(const AGXTargetMachine *TM) {
  return new AGXLowerAIRLegacy(TM);
}

PreservedAnalyses AGXLowerAIRPass::run(Module &M, ModuleAnalysisManager &) {
  return LowerAIR(M, TM).run() ? PreservedAnalyses::none()
                               : PreservedAnalyses::all();
}
