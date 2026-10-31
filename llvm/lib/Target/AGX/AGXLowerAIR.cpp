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
// threadgroup memory.
//
// Calls to AIR functions become LLVM intrinsics and instructions: the math
// functions by the inverse of the AIR target's mapping
// (AIRIntrinsicMappings.td, "fast_" functions with afn), atomics and fences
// as atomicrmw, cmpxchg and fence, and barriers as AGX's threadgroup barrier
// between threadgroup-scope fences. An AIR function with no lowering is an
// error. Finally the module takes AGX's triple and data layout.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXSubtarget.h"
#include "AGXTargetMachine.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/ADT/DenseMap.h"
#include <functional>
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringTable.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/AIR.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#define DEBUG_TYPE "agx-lower-air"

static cl::opt<bool> NativeHints(
    "agx-air-native-hints", cl::init(true), cl::Hidden,
    cl::desc("Compile AIR hints as the AGX operations they name; false runs "
             "their AIR emulation, as a consumer without them does"));

static cl::list<std::string> EmulateHints(
    "agx-air-emulate-hints", cl::CommaSeparated, cl::Hidden,
    cl::desc("Run the AIR emulation of the hints whose keys start with one "
             "of these prefixes (measuring what each kind of hint is worth)"));

static cl::opt<bool> NativeF64(
    "agx-air-native-f64", cl::init(false), cl::Hidden,
    cl::desc("Compile calls to AIR's software f64 library (rounded to "
             "nearest) as double operations, lowered as AGX lowers them "
             "(off: the library is as fast; see EXPERIMENTS.md L4-E2)"));

namespace {
// Layout must match the struct emitted by -gen-searchable-tables for the
// AIRFunction class in AIRIntrinsicMappings.td.
struct AIRFunction {
  StringTable::Offset LLVMName;
  StringTable::Offset AIRName;
  uint8_t Signedness;
  bool HasFast;
  uint8_t NumArgs;
  bool Invertible;
};

#define GET_AIRFunctions_IMPL
#include "AGXGenAIRFunctions.inc"

/// What !air.kernel says a kernel parameter is.
struct Param {
  enum { Undescribed, Buffer, SystemValue } Kind = Undescribed;
  // Buffer.
  unsigned AddrSpace = 0;
  unsigned Location = 0;
  unsigned TypeSize = 0;
  StringRef TypeName;
  /// The parameter's byref type, when the AIR kept one (agx-raise-to-air):
  /// the exact size of a constant buffer.
  Type *ByRef = nullptr;
  /// Its string attributes ("agx-hidden" from agx-raise-to-air).
  SmallVector<Attribute, 1> StringAttrs;
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
    case air::GenericHintAS:
      return AGX::GENERIC_ADDRESS;
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
  Value *systemValue(IRBuilder<> &B, StringRef Role, Type *Ty, Function *F,
                     int Only = -1);
  void passConstantBuffersByReference(AIRKernel &K, Function *NewF);
  void lowerMatrices(Function &F);
  void foldOrderedAtomics(Function &F);
  void lowerHints();
  void recoverAGXOperations(Function &F);
  void recoverDoubles();
  void lowerCalls();
  void inlineCalls();
  Function *hiddenByValue(Function &K);
  bool lowerCall(CallInst &CI, StringRef Name, Value *&Result);
  bool lowerAtomic(IRBuilder<> &B, CallInst &CI, StringRef Name,
                   Value *&Result);
  bool lowerMath(IRBuilder<> &B, CallInst &CI, StringRef Name,
                 Value *&Result);
  bool lowerSIMD(IRBuilder<> &B, CallInst &CI, StringRef Name,
                 Value *&Result);
  bool lowerOther(IRBuilder<> &B, CallInst &CI, StringRef Name,
                  Value *&Result);
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
  // Integers of other widths up to 128 bits are a type hint (llvm/IR/AIR.h):
  // AGX legalizes them.
  if (auto *IT = dyn_cast<IntegerType>(T))
    return IT->getBitWidth() <= air::WideIntHintBits;
  if (auto *PT = dyn_cast<PointerType>(T))
    return PT->getAddressSpace() <= air::AS::Threadgroup ||
           PT->getAddressSpace() == air::GenericHintAS;
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
      // Only generic pointers (the address-space hint) are cast.
      if (auto *ASC = dyn_cast<AddrSpaceCastInst>(&I);
          ASC && ASC->getSrcAddressSpace() != air::GenericHintAS &&
          ASC->getDestAddressSpace() != air::GenericHintAS) {
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

// The system value Role as a value of the parameter type Ty, or its
// component Only (of Ty's element type) alone.
Value *LowerAIR::systemValue(IRBuilder<> &B, StringRef Role, Type *Ty,
                             Function *F, int Only) {
  auto Read = [&](Intrinsic::ID ID) { return B.CreateIntrinsic(ID, {}); };
  auto XYZ = [&](Intrinsic::ID X, Intrinsic::ID Y, Intrinsic::ID Z) {
    SmallVector<Value *, 3> V;
    // Not ArrayRef{X, Y, Z}: its array would not outlive the range-init.
    Intrinsic::ID IDs[] = {X, Y, Z};
    for (auto [I, ID] : enumerate(IDs))
      V.push_back(Only < 0 || int(I) == Only ? Read(ID) : nullptr);
    return V;
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
      C.push_back(G[I] ? B.CreateMul(G[I], T[I]) : nullptr);
  } else if (Role == air::ThreadIndexInThreadgroup)
    C = {Read(agx_thread_index_in_threadgroup)};
  else if (Role == air::ThreadIndexInSimdgroup)
    C = {Read(agx_thread_index_in_simdgroup)};
  else if (Role == air::SimdgroupIndexInThreadgroup)
    C = {Read(agx_simdgroup_index_in_threadgroup)};
  else if (Role == air::ThreadsPerSimdgroup)
    C = {B.getInt32(32)};
  else if (Role == air::SimdgroupsPerThreadgroup) {
    Only = -1; // all three, for one value
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
  if (Only >= 0) {
    Type *EltTy = Ty;
    if (auto *VT = dyn_cast<FixedVectorType>(Ty))
      EltTy = VT->getElementType();
    return Component(Only, EltTy);
  }
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
    Param &Pm = K.Params[A.getArgNo()];
    if (Type *T = F.getParamByRefType(A.getArgNo()))
      Pm.ByRef = Types.remapType(T);
    for (Attribute At : F.getAttributes().getParamAttrs(A.getArgNo()))
      if (At.isStringAttribute())
        Pm.StringAttrs.push_back(At);
    if (Pm.Kind == Param::Buffer && Pm.AddrSpace != air::AS::Threadgroup)
      Params.push_back(Types.remapType(A.getType()));
  }
  auto *FT = FunctionType::get(Type::getVoidTy(Ctx), Params, false);
  Function *NewF = Function::Create(FT, F.getLinkage(), F.getAddressSpace(),
                                    F.getName() + ".agx", &M);
  NewF->setCallingConv(CallingConv::AGX_KERNEL);
  // The kernel's own function attributes (AGX ones from agx-raise-to-air:
  // stack size, threadgroup size limits).
  AttrBuilder FnAttrs(Ctx, F.getAttributes().getFnAttrs());
  FnAttrs.removeAttribute("air-kernel");
  NewF->addFnAttrs(FnAttrs);
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

  // AIR passes the system values as parameters, whose components the body
  // extracts where agx-raise-to-air (or AIR's own code) read them. Each
  // component is read at its extraction, as an AGX kernel reads it there.
  // (A scalar one stays at the entry: AIR keeps no place for it, and reading
  // it where its uses meet moved as many kernels the wrong way as the
  // right one.)
  for (Argument &A : F.args()) {
    const Param &Pm = K.Params[A.getArgNo()];
    Value *V = VMap.lookup(&A);
    if (Pm.Kind != Param::SystemValue || !V ||
        !isa<FixedVectorType>(A.getType()))
      continue;
    for (User *U : make_early_inc_range(V->users())) {
      auto *EE = dyn_cast<ExtractElementInst>(U);
      auto *Idx = EE ? dyn_cast<ConstantInt>(EE->getIndexOperand()) : nullptr;
      if (!Idx || Idx->getZExtValue() >= 3)
        continue;
      if (!EE->use_empty()) {
        IRBuilder<> UB(EE);
        Value *C =
            systemValue(UB, Pm.Role, A.getType(), &F, Idx->getZExtValue());
        C->takeName(EE);
        EE->replaceAllUsesWith(C);
      }
      EE->eraseFromParent();
    }
    RecursivelyDeleteTriviallyDeadInstructions(V);
  }
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
    for (Attribute At : Pm.StringAttrs)
      A.addAttr(At);
    if (Pm.AddrSpace != air::AS::Constant)
      continue;
    if (Pm.ByRef) {
      A.addAttr(Attribute::getWithByRefType(Ctx, Pm.ByRef));
      continue;
    }
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

// MSL's memory_order, as AIR's atomic functions take it.
static AtomicOrdering ordering(Value *Order) {
  auto *C = dyn_cast<ConstantInt>(Order);
  switch (C ? C->getZExtValue() : 5) {
  case 0:
    return AtomicOrdering::Monotonic;
  case 1: // consume
  case 2:
    return AtomicOrdering::Acquire;
  case 3:
    return AtomicOrdering::Release;
  case 4:
    return AtomicOrdering::AcquireRelease;
  default:
    return AtomicOrdering::SequentiallyConsistent;
  }
}

// MSL's thread_scope: threadgroup (1) or device (2, and anything wider).
static SyncScope::ID syncScope(LLVMContext &Ctx, Value *Scope) {
  auto *C = dyn_cast<ConstantInt>(Scope);
  if (C && C->getZExtValue() <= 1)
    return Ctx.getOrInsertSyncScopeID("workgroup");
  return SyncScope::System;
}

static bool isVolatile(Value *V) {
  auto *C = dyn_cast<ConstantInt>(V);
  return !C || !C->isZero();
}

// air.atomic.fence(flags, order, scope), the atomic operations
//   air.atomic.<space>.<op>.[s.|u.]<type>(ptr, value, order, scope, volatile)
// and air.atomic.<space>.cmpxchg.weak.i32(ptr, ptr expected, desired,
// success order, failure order, scope, volatile), which returns the value it
// found and, as MSL's compare-exchange does, leaves it in *expected.
bool LowerAIR::lowerAtomic(IRBuilder<> &B, CallInst &CI, StringRef Name,
                           Value *&Result) {
  if (Name == "air.atomic.fence") {
    if (CI.arg_size() != 3)
      return false;
    AtomicOrdering O = ordering(CI.getArgOperand(1));
    if (O != AtomicOrdering::Monotonic)
      B.CreateFence(O, syncScope(Ctx, CI.getArgOperand(2)));
    return true;
  }
  StringRef Rest = Name.drop_front(strlen("air.atomic."));
  if (!Rest.consume_front("global.") && !Rest.consume_front("local."))
    return false;
  Value *Ptr = CI.getArgOperand(0);
  if (Rest == "cmpxchg.weak.i32") {
    // AIR 2.9 adds a memory flags operand before the volatile one.
    if (CI.arg_size() != 7 && CI.arg_size() != 8)
      return false;
    Value *ExpectedPtr = CI.getArgOperand(1);
    Value *Expected = B.CreateLoad(B.getInt32Ty(), ExpectedPtr);
    AtomicOrdering Success = ordering(CI.getArgOperand(3));
    AtomicOrdering Failure = ordering(CI.getArgOperand(4));
    if (Failure == AtomicOrdering::Release)
      Failure = AtomicOrdering::Monotonic;
    else if (Failure == AtomicOrdering::AcquireRelease)
      Failure = AtomicOrdering::Acquire;
    auto *X = B.CreateAtomicCmpXchg(Ptr, Expected, CI.getArgOperand(2),
                                    MaybeAlign(), Success, Failure,
                                    syncScope(Ctx, CI.getArgOperand(5)));
    X->setVolatile(isVolatile(CI.getArgOperand(CI.arg_size() - 1)));
    Result = B.CreateExtractValue(X, 0);
    B.CreateStore(Result, ExpectedPtr);
    return true;
  }
  StringRef Op = Rest.take_until([](char C) { return C == '.'; });
  Rest = Rest.drop_front(Op.size());
  Rest.consume_front(".");
  bool Unsigned = Rest.consume_front("u.");
  Rest.consume_front("s.");
  bool Float = Rest.starts_with("f");
  std::optional<AtomicRMWInst::BinOp> BinOp =
      StringSwitch<std::optional<AtomicRMWInst::BinOp>>(Op)
          .Case("xchg", AtomicRMWInst::Xchg)
          .Case("add", Float ? AtomicRMWInst::FAdd : AtomicRMWInst::Add)
          .Case("sub", Float ? AtomicRMWInst::FSub : AtomicRMWInst::Sub)
          .Case("max", Unsigned ? AtomicRMWInst::UMax : AtomicRMWInst::Max)
          .Case("min", Unsigned ? AtomicRMWInst::UMin : AtomicRMWInst::Min)
          .Case("and", AtomicRMWInst::And)
          .Case("or", AtomicRMWInst::Or)
          .Case("xor", AtomicRMWInst::Xor)
          .Default(std::nullopt);
  if (!BinOp || CI.arg_size() != 5 || (Float && Op != "xchg" && Op != "add" &&
                                       Op != "sub"))
    return false;
  auto *RMW = B.CreateAtomicRMW(*BinOp, Ptr, CI.getArgOperand(1), MaybeAlign(),
                                ordering(CI.getArgOperand(2)),
                                syncScope(Ctx, CI.getArgOperand(3)));
  RMW->setVolatile(isVolatile(CI.getArgOperand(4)));
  Result = RMW;
  return true;
}

// air.<name>.[s.|u.]<type> and air.fast_<name>.<type>, by the inverse of
// AIRIntrinsicMappings.td.
bool LowerAIR::lowerMath(IRBuilder<> &B, CallInst &CI, StringRef Name,
                         Value *&Result) {
  static const StringMap<SmallVector<const AIRFunction *, 2>> ByAIRName = [] {
    StringMap<SmallVector<const AIRFunction *, 2>> Map;
    for (const AIRFunction &F : AIRFunctions)
      if (F.Invertible)
        Map[getAIRFunctionStr(F.AIRName)].push_back(&F);
    return Map;
  }();
  StringRef Rest = Name.drop_front(strlen("air."));
  bool Fast = Rest.consume_front("fast_");
  StringRef Base = Rest.take_until([](char C) { return C == '.'; });
  Rest = Rest.drop_front(Base.size());
  Rest.consume_front(".");
  unsigned Signedness = Rest.consume_front("s.") ? 1
                        : Rest.consume_front("u.") ? 2
                                                   : 0;
  const AIRFunction *Fn = nullptr;
  if (auto It = ByAIRName.find(Base); It != ByAIRName.end())
    for (const AIRFunction *F : It->second)
      if (F->Signedness == Signedness && (!Fast || F->HasFast))
        Fn = F;
  Intrinsic::ID ID =
      Fn ? Intrinsic::lookupIntrinsicID(getAIRFunctionStr(Fn->LLVMName))
         : Intrinsic::not_intrinsic;
  if (ID == Intrinsic::not_intrinsic) {
    // AIR's functions without an LLVM intrinsic (asinh, sinpi, ...).
    if (CI.arg_size() != 1 || CI.getArgOperand(0)->getType() != CI.getType())
      return false;
    B.setFastMathFlags(CI.getFastMathFlags());
    Result = AGX::expandMath(B, Base, CI.getArgOperand(0));
    return Result != nullptr;
  }
  Function *Decl = Intrinsic::getOrInsertDeclaration(&M, ID, {CI.getType()});
  SmallVector<Value *, 4> Args(CI.args());
  // Operands AIR leaves out (llvm.abs's is_int_min_poison) are false.
  while (Args.size() < Decl->arg_size())
    Args.push_back(B.getFalse());
  if (Args.size() != Decl->arg_size())
    return false;
  CallInst *New = B.CreateCall(Decl, Args);
  if (isa<FPMathOperator>(New)) {
    FastMathFlags FMF = CI.getFastMathFlags();
    FMF.setApproxFunc(FMF.approxFunc() || Fast);
    New->setFastMathFlags(FMF);
  }
  Result = New;
  return true;
}

// The SIMD-group and quad-group functions, air.simd_<op>.<type> and
// air.quad_<op>.<type> (signed and unsigned integer types are s.i32 and u.i32):
// shuffles, reductions, prefix scans, ballots and votes.
bool LowerAIR::lowerSIMD(IRBuilder<> &B, CallInst &CI, StringRef Name,
                         Value *&Result) {
  using namespace Intrinsic;
  bool Quad = Name.starts_with("air.quad_");
  StringRef Rest = Name.drop_front(strlen(Quad ? "air.quad_" : "air.simd_"));
  StringRef Op = Rest.take_until([](char C) { return C == '.'; });
  Rest = Rest.drop_front(Op.size());
  Rest.consume_front(".");
  bool Unsigned = Rest.starts_with("u.");
  Type *Ty = CI.getType();
  auto Lane = [&](unsigned I) {
    return B.CreateZExtOrTrunc(CI.getArgOperand(I), B.getInt32Ty());
  };
  auto Active = [&] {
    return B.CreateIntrinsic(Quad ? agx_quad_ballot : agx_simd_ballot,
                             {B.getTrue()});
  };

  ID Shuffle = StringSwitch<ID>(Op)
                   .Cases({"shuffle", "broadcast"},
                          Quad ? agx_quad_shuffle : agx_simd_shuffle)
                   .Case("shuffle_xor",
                         Quad ? agx_quad_shuffle_xor : agx_simd_shuffle_xor)
                   .Case("shuffle_up",
                         Quad ? agx_quad_shuffle_up : agx_simd_shuffle_up)
                   .Case("shuffle_down",
                         Quad ? agx_quad_shuffle_down : agx_simd_shuffle_down)
                   .Default(not_intrinsic);
  if (Shuffle != not_intrinsic && CI.arg_size() == 2) {
    Result = B.CreateIntrinsic(Shuffle, {Ty}, {CI.getArgOperand(0), Lane(1)});
    return true;
  }
  if (Op == "broadcast_first" && CI.arg_size() == 1) {
    Value *First = B.CreateIntrinsic(Intrinsic::cttz, {B.getInt32Ty()},
                                     {Active(), B.getFalse()});
    Result = B.CreateIntrinsic(Quad ? agx_quad_shuffle : agx_simd_shuffle, {Ty},
                               {CI.getArgOperand(0), First});
    return true;
  }
  // shuffle_and_fill_{down,up}(data, fill, delta, modulo): within segments
  // of modulo threads, a thread reads data delta threads away, and the
  // threads that would read outside the segment read fill from the other end.
  if ((Op == "shuffle_and_fill_down" || Op == "shuffle_and_fill_up") &&
      CI.arg_size() == 4) {
    bool Down = Op == "shuffle_and_fill_down";
    Value *Self = B.CreateIntrinsic(agx_thread_index_in_simdgroup, {});
    if (Quad)
      Self = B.CreateAnd(Self, B.getInt32(3));
    Value *Delta = Lane(2), *Mod = Lane(3);
    Value *Pos = B.CreateURem(Self, Mod);
    Value *Base = B.CreateSub(Self, Pos);
    Value *Inside, *Src;
    if (Down) {
      Value *To = B.CreateAdd(Pos, Delta);
      Inside = B.CreateICmpULT(To, Mod);
      Src = B.CreateAdd(Base, B.CreateSelect(Inside, To, B.CreateSub(To, Mod)));
    } else {
      Inside = B.CreateICmpUGE(Pos, Delta);
      Value *From = B.CreateSub(Pos, Delta);
      Src = B.CreateAdd(Base,
                        B.CreateSelect(Inside, From, B.CreateAdd(From, Mod)));
    }
    ID Read = Quad ? agx_quad_shuffle : agx_simd_shuffle;
    Value *Data = B.CreateIntrinsic(Read, {Ty}, {CI.getArgOperand(0), Src});
    Value *Fill = B.CreateIntrinsic(Read, {Ty}, {CI.getArgOperand(1), Src});
    Result = B.CreateSelect(Inside, Data, Fill);
    return true;
  }
  if (Op == "ballot" && CI.arg_size() == 1) {
    Value *Mask = B.CreateIntrinsic(Quad ? agx_quad_ballot : agx_simd_ballot,
                                    {CI.getArgOperand(0)});
    Result = B.CreateZExtOrTrunc(Mask, Ty);
    return true;
  }
  if ((Op == "vote_any" || Op == "vote_all") && CI.arg_size() == 1) {
    // Of a ballot: any thread, or every active thread.
    Value *Mask = CI.getArgOperand(0);
    Result = Op == "vote_any"
                 ? B.CreateIsNotNull(Mask)
                 : B.CreateICmpEQ(Mask,
                                  B.CreateZExtOrTrunc(Active(), Mask->getType()));
    return true;
  }

  // Reductions and scans, by AGX::SIMDOp.
  bool Float = Ty->getScalarType()->isFloatingPointTy();
  bool Prefix = Op.consume_front("prefix_");
  bool Inclusive = Prefix && Op.consume_front("inclusive_");
  if (Prefix && !Inclusive && !Op.consume_front("exclusive_"))
    return false;
  std::optional<unsigned> HWOp =
      StringSwitch<std::optional<unsigned>>(Op)
          .Case("sum", Float ? AGX::SIMD_FADD : AGX::SIMD_IADD)
          .Case("product", Float ? std::optional<unsigned>(AGX::SIMD_FMUL)
                                 : std::nullopt)
          .Case("min", Float      ? AGX::SIMD_FMIN
                       : Unsigned ? AGX::SIMD_UMIN
                                  : AGX::SIMD_SMIN)
          .Case("max", Float      ? AGX::SIMD_FMAX
                       : Unsigned ? AGX::SIMD_UMAX
                                  : AGX::SIMD_SMAX)
          .Case("and", Float ? std::optional<unsigned>() : AGX::SIMD_AND)
          .Case("or", Float ? std::optional<unsigned>() : AGX::SIMD_OR)
          .Case("xor", Float ? std::optional<unsigned>() : AGX::SIMD_XOR)
          .Default(std::nullopt);
  if (!HWOp || CI.arg_size() != 1 || (Prefix && Op != "sum" && Op != "product"))
    return false;
  ID Fn = Prefix ? (Quad ? agx_quad_prefix : agx_simd_prefix)
                 : (Quad ? agx_quad_reduce : agx_simd_reduce);
  Value *X = CI.getArgOperand(0);
  Result = B.CreateIntrinsic(Fn, {Ty}, {X, B.getInt32(*HWOp)});
  // The hardware scans are exclusive.
  if (Inclusive)
    Result = Op == "sum" ? (Float ? B.CreateFAdd(Result, X)
                                  : B.CreateAdd(Result, X))
                         : B.CreateFMul(Result, X);
  return true;
}

// The value of an OpenCL-style work-item function for dimension Dim.
static Value *workItem(IRBuilder<> &B, Intrinsic::ID X, Intrinsic::ID Y,
                       Intrinsic::ID Z, Value *Dim) {
  Value *V[] = {B.CreateIntrinsic(X, {}), B.CreateIntrinsic(Y, {}),
                B.CreateIntrinsic(Z, {})};
  if (auto *C = dyn_cast<ConstantInt>(Dim))
    return C->getZExtValue() < 3 ? V[C->getZExtValue()] : B.getInt32(0);
  Value *D = B.CreateZExtOrTrunc(Dim, B.getInt32Ty());
  return B.CreateSelect(B.CreateICmpEQ(D, B.getInt32(0)), V[0],
                        B.CreateSelect(B.CreateICmpEQ(D, B.getInt32(1)), V[1],
                                       V[2]));
}

// AIR functions that are neither LLVM intrinsics nor SIMD-group functions:
// OpenCL-style work-item functions, conversions, packing, and math with no
// LLVM intrinsic.
bool LowerAIR::lowerOther(IRBuilder<> &B, CallInst &CI, StringRef Name,
                          Value *&Result) {
  using namespace Intrinsic;
  SmallVector<StringRef, 6> Parts;
  Name.drop_front(strlen("air.")).split(Parts, '.');
  StringRef Fn = Parts[0];
  Type *Ty = CI.getType();
  unsigned N = CI.arg_size();
  auto Arg = [&](unsigned I) { return CI.getArgOperand(I); };

  if (Fn.starts_with("get_") && N == 1) {
    Value *D = Arg(0), *V = nullptr;
    if (Fn == "get_global_id")
      V = workItem(B, agx_thread_position_in_grid_x,
                   agx_thread_position_in_grid_y,
                   agx_thread_position_in_grid_z, D);
    else if (Fn == "get_local_id")
      V = workItem(B, agx_thread_position_in_threadgroup_x,
                   agx_thread_position_in_threadgroup_y,
                   agx_thread_position_in_threadgroup_z, D);
    else if (Fn == "get_group_id")
      V = workItem(B, agx_threadgroup_position_in_grid_x,
                   agx_threadgroup_position_in_grid_y,
                   agx_threadgroup_position_in_grid_z, D);
    else if (Fn == "get_local_size")
      V = workItem(B, agx_threads_per_threadgroup_x,
                   agx_threads_per_threadgroup_y,
                   agx_threads_per_threadgroup_z, D);
    else if (Fn == "get_num_groups")
      V = workItem(B, agx_threadgroups_per_grid_x, agx_threadgroups_per_grid_y,
                   agx_threadgroups_per_grid_z, D);
    else if (Fn == "get_global_size")
      V = B.CreateMul(workItem(B, agx_threadgroups_per_grid_x,
                               agx_threadgroups_per_grid_y,
                               agx_threadgroups_per_grid_z, D),
                      workItem(B, agx_threads_per_threadgroup_x,
                               agx_threads_per_threadgroup_y,
                               agx_threads_per_threadgroup_z, D));
    if (!V || !Ty->isIntegerTy())
      return false;
    Result = B.CreateZExtOrTrunc(V, Ty);
    return true;
  }
  // air.convert.<to>.<from>, each a kind (f, s or u) and a type.
  if (Fn == "convert" && Parts.size() == 5 && N == 1) {
    Value *X = Arg(0);
    char To = Parts[1][0], From = Parts[3][0];
    if (To == 'f')
      Result = From == 'f'   ? B.CreateFPCast(X, Ty)
               : From == 's' ? B.CreateSIToFP(X, Ty)
                             : B.CreateUIToFP(X, Ty);
    else if (From == 'f')
      Result = B.CreateIntrinsic(To == 's' ? fptosi_sat : fptoui_sat,
                                 {Ty, X->getType()}, {X});
    else
      Result = B.CreateIntCast(X, Ty, From == 's');
    return true;
  }
  if (Fn == "dot" && N == 2) {
    Value *A = Arg(0), *Bv = Arg(1);
    auto *VT = dyn_cast<FixedVectorType>(A->getType());
    if (!VT)
      return false;
    Value *Sum = nullptr;
    for (unsigned I = 0; I < VT->getNumElements(); ++I) {
      Value *P = B.CreateFMul(B.CreateExtractElement(A, I),
                              B.CreateExtractElement(Bv, I));
      Sum = Sum ? B.CreateFAdd(Sum, P) : P;
    }
    Result = Sum;
    return true;
  }
  // air.pack.{unorm,snorm}{4x8,2x16}.<vector> and air.unpack.<same>.<vector>:
  // normalized integers, rounded to nearest even, the first element lowest.
  if ((Fn == "pack" || Fn == "unpack") && Parts.size() >= 2 && N == 1) {
    StringRef Format = Parts[1];
    bool Signed = Format.starts_with("snorm");
    unsigned Count, Bits;
    if (Format.ends_with("4x8"))
      Count = 4, Bits = 8;
    else if (Format.ends_with("2x16"))
      Count = 2, Bits = 16;
    else
      return false;
    if (!Signed && !Format.starts_with("unorm"))
      return false;
    double Scale = Signed ? (1 << (Bits - 1)) - 1 : (1 << Bits) - 1;
    Type *F32 = B.getFloatTy();
    if (Fn == "pack") {
      Value *Word = B.getInt32(0);
      for (unsigned I = 0; I < Count; ++I) {
        Value *X = B.CreateExtractElement(Arg(0), I);
        X = B.CreateFPCast(X, F32);
        // maxnum first, so that NaN becomes 0.
        X = B.CreateMaxNum(X, ConstantFP::get(F32, Signed ? -1.0 : 0.0));
        X = B.CreateMinNum(X, ConstantFP::get(F32, 1.0));
        X = B.CreateIntrinsic(Intrinsic::rint, {F32},
                              {B.CreateFMul(X, ConstantFP::get(F32, Scale))});
        Value *I32 = B.CreateFPToSI(X, B.getInt32Ty());
        I32 = B.CreateAnd(I32, B.getInt32((1u << Bits) - 1));
        Word = B.CreateOr(Word, B.CreateShl(I32, I * Bits));
      }
      Result = B.CreateZExtOrTrunc(Word, Ty);
      return true;
    }
    auto *VT = dyn_cast<FixedVectorType>(Ty);
    if (!VT || VT->getNumElements() != Count)
      return false;
    Value *Word = B.CreateZExtOrTrunc(Arg(0), B.getInt32Ty());
    Value *V = PoisonValue::get(VT);
    for (unsigned I = 0; I < Count; ++I) {
      Value *E = B.CreateLShr(Word, I * Bits);
      E = B.CreateTrunc(E, B.getIntNTy(Bits));
      Value *F = Signed ? B.CreateSIToFP(E, F32) : B.CreateUIToFP(E, F32);
      F = B.CreateFDiv(F, ConstantFP::get(F32, Scale));
      if (Signed)
        F = B.CreateMaxNum(F, ConstantFP::get(F32, -1.0));
      V = B.CreateInsertElement(V, B.CreateFPCast(F, VT->getElementType()), I);
    }
    Result = V;
    return true;
  }
  // sincos(x, ptr cos) returns the sine and stores the cosine.
  if ((Fn == "sincos" || Fn == "fast_sincos") && N == 2) {
    FastMathFlags FMF = CI.getFastMathFlags();
    FMF.setApproxFunc(FMF.approxFunc() || Fn == "fast_sincos");
    IRBuilder<>::FastMathFlagGuard Guard(B);
    B.setFastMathFlags(FMF);
    Result = B.CreateUnaryIntrinsic(Intrinsic::sin, Arg(0));
    B.CreateStore(B.CreateUnaryIntrinsic(Intrinsic::cos, Arg(0)), Arg(1));
    return true;
  }
  if (Fn == "fract" && N == 1) {
    // x - floor(x), below 1 (fract of tiny negative values rounds up to 1).
    Value *X = Arg(0);
    Value *F = B.CreateFSub(X, B.CreateUnaryIntrinsic(Intrinsic::floor, X));
    APFloat BelowOne = APFloat::getOne(Ty->getScalarType()->getFltSemantics());
    BelowOne.next(/*nextDown=*/true);
    Result = B.CreateMinNum(F, ConstantFP::get(Ty, BelowOne));
    return true;
  }
  if (Fn == "rsqrt" && N == 1) {
    Result = B.CreateFDiv(ConstantFP::get(Ty, 1.0),
                          B.CreateUnaryIntrinsic(Intrinsic::sqrt, Arg(0)));
    return true;
  }
  if (Fn == "fast_rsqrt" && N == 1 && Ty->getScalarType()->isFloatTy() &&
      !Ty->isVectorTy()) {
    Result = B.CreateIntrinsic(agx_rsqrt, {Ty}, {Arg(0)});
    return true;
  }
  if (Fn == "fmod" && N == 2) {
    Result = B.CreateFRem(Arg(0), Arg(1));
    return true;
  }
  if (Fn == "mul_hi" && Parts.size() == 3 && N == 2) {
    bool Signed = Parts[1] == "s";
    unsigned Bits = Ty->getScalarSizeInBits();
    Type *Wide = Ty->getWithNewBitWidth(Bits * 2);
    Value *P = B.CreateMul(B.CreateIntCast(Arg(0), Wide, Signed),
                           B.CreateIntCast(Arg(1), Wide, Signed));
    Result = B.CreateTrunc(B.CreateLShr(P, Bits), Ty);
    return true;
  }
  // Device-coherent accesses are volatile, which the backend encodes
  // coherent.
  if (Fn == "load" && Parts.size() >= 2 && Parts[1] == "device_coherent" &&
      N == 1) {
    Result = B.CreateLoad(Ty, Arg(0), /*isVolatile=*/true);
    return true;
  }
  if (Fn == "store" && Parts.size() >= 2 && Parts[1] == "device_coherent" &&
      N == 2) {
    B.CreateStore(Arg(0), Arg(1), /*isVolatile=*/true);
    return true;
  }
  return false;
}

bool LowerAIR::lowerCall(CallInst &CI, StringRef Name, Value *&Result) {
  IRBuilder<> B(&CI);
  if (Name.starts_with("air.atomic."))
    return lowerAtomic(B, CI, Name, Result);
  if (Name == "air.wg.barrier" || Name == "air.simdgroup.barrier") {
    // The first operand is the memory to order (mem_flags: device 1,
    // threadgroup 2, texture 4). The threads of a threadgroup share a core,
    // where threadgroup-scope fences order device memory too. A SIMD-group
    // runs in lockstep, so its barrier only orders memory.
    auto *Flags = CI.arg_size() ? dyn_cast<ConstantInt>(CI.getArgOperand(0))
                                : nullptr;
    bool Memory = !Flags || !Flags->isZero();
    SyncScope::ID Scope = Ctx.getOrInsertSyncScopeID("workgroup");
    if (Name == "air.simdgroup.barrier") {
      if (Memory)
        B.CreateFence(AtomicOrdering::AcquireRelease, Scope);
      return true;
    }
    if (Memory)
      B.CreateFence(AtomicOrdering::Release, Scope);
    B.CreateIntrinsic(Intrinsic::agx_threadgroup_barrier, {});
    if (Memory)
      B.CreateFence(AtomicOrdering::Acquire, Scope);
    return true;
  }
  if (Name.starts_with("air.simd_") || Name.starts_with("air.quad_"))
    return lowerSIMD(B, CI, Name, Result);
  return lowerOther(B, CI, Name, Result) || lowerMath(B, CI, Name, Result);
}

// AIR's SIMD-group matrix: an 8x8 matrix of half, bfloat or float as one
// <64 x T> value.
static bool isMatrix(Type *T) {
  auto *VT = dyn_cast<FixedVectorType>(T);
  return VT && VT->getNumElements() == 64 &&
         (VT->getElementType()->isFloatTy() ||
          VT->getElementType()->isHalfTy() ||
          VT->getElementType()->isBFloatTy());
}

static bool usesMatrix(const Instruction &I) {
  return isMatrix(I.getType()) || any_of(I.operands(), [](const Use &U) {
           return isMatrix(U->getType());
         });
}

// The SIMD-group matrices of AIR (air.simdgroup_matrix_8x8_*) as AGX's: each
// thread holds the elements (r, c) and (r, c + 1) in a <2 x T>
// (llvm.agx.simdgroup.matrix.fma's layout, AGXUsage.rst "SIMD-group
// matrices"). A load or store moves the thread's two elements, at
// (c + j + origin.x) * stride.x + (r + origin.y) * stride.y; element-wise
// operations, selects and phis apply to the two elements. Anything that
// depends on how AIR orders the 64 elements is an error.
void LowerAIR::lowerMatrices(Function &F) {
  if (F.isDeclaration() || none_of(instructions(F), usesMatrix))
    return;
  if (any_of(F.args(), [](Argument &A) { return isMatrix(A.getType()); })) {
    error(&F, "a SIMD-group matrix argument");
    return;
  }
  IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
  Value *T = B.CreateIntrinsic(Intrinsic::agx_thread_index_in_simdgroup, {});
  // r = 4 * (t / 16) + (t / 2) % 4, c = 4 * ((t / 8) % 2) + 2 * (t % 2)
  Value *Row = B.CreateAdd(B.CreateShl(B.CreateLShr(T, 4), 2),
                           B.CreateAnd(B.CreateLShr(T, 1), 3));
  Value *Col = B.CreateAdd(B.CreateShl(B.CreateAnd(B.CreateLShr(T, 3), 1), 2),
                           B.CreateShl(B.CreateAnd(T, 1), 1));
  Value *Row64 = B.CreateZExt(Row, B.getInt64Ty(), "matrix.row");
  Value *Col64 = B.CreateZExt(Col, B.getInt64Ty(), "matrix.col");

  auto FragTy = [](Type *MT) {
    return FixedVectorType::get(cast<VectorType>(MT)->getElementType(), 2);
  };
  DenseMap<Value *, Value *> Frag;
  SmallVector<Instruction *, 16> Old;
  bool Bad = false;
  auto Get = [&](Value *V) -> Value * {
    if (Value *R = Frag.lookup(V))
      return R;
    if (auto *C = dyn_cast<Constant>(V)) {
      if (isa<UndefValue>(C))
        return isa<PoisonValue>(C) ? PoisonValue::get(FragTy(V->getType()))
                                   : UndefValue::get(FragTy(V->getType()));
      if (Constant *Splat = C->getSplatValue())
        return ConstantVector::getSplat(ElementCount::getFixed(2), Splat);
    }
    Bad = true;
    return PoisonValue::get(FragTy(V->getType()));
  };
  // The addresses of the thread's two elements.
  auto Element = [&](IRBuilder<> &B, Type *EltTy, Value *Ptr, Value *Strides,
                     Value *Origin, unsigned J) {
    Value *R = B.CreateAdd(Row64, B.CreateExtractElement(Origin, 1));
    Value *C = B.CreateAdd(B.CreateAdd(Col64, B.getInt64(J)),
                           B.CreateExtractElement(Origin, uint64_t(0)));
    Value *Idx = B.CreateAdd(B.CreateMul(C, B.CreateExtractElement(Strides, uint64_t(0))),
                             B.CreateMul(R, B.CreateExtractElement(Strides, 1)));
    return B.CreateGEP(EltTy, Ptr, Idx);
  };

  SmallVector<std::pair<PHINode *, PHINode *>, 4> Phis;
  for (Instruction &I : instructions(F))
    if (auto *P = dyn_cast<PHINode>(&I); P && isMatrix(P->getType())) {
      auto *NP = PHINode::Create(FragTy(P->getType()), P->getNumIncomingValues(),
                                 P->getName(), P->getIterator());
      Frag[P] = NP;
      Phis.push_back({P, NP});
    }

  ReversePostOrderTraversal<Function *> RPOT(&F);
  for (BasicBlock *BB : RPOT) {
    for (Instruction &I : *BB) {
      if (!usesMatrix(I) || isa<PHINode>(I))
        continue;
      IRBuilder<> IB(&I);
      Old.push_back(&I);
      if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
        Value *V = IB.CreateBinOp(BO->getOpcode(), Get(BO->getOperand(0)),
                                  Get(BO->getOperand(1)), I.getName());
        if (auto *VI = dyn_cast<Instruction>(V))
          VI->copyIRFlags(&I);
        Frag[&I] = V;
        continue;
      }
      if (auto *UO = dyn_cast<UnaryOperator>(&I)) {
        Value *V = IB.CreateUnOp(UO->getOpcode(), Get(UO->getOperand(0)),
                                 I.getName());
        if (auto *VI = dyn_cast<Instruction>(V))
          VI->copyIRFlags(&I);
        Frag[&I] = V;
        continue;
      }
      if (auto *S = dyn_cast<SelectInst>(&I);
          S && !S->getCondition()->getType()->isVectorTy()) {
        Frag[&I] = IB.CreateSelect(S->getCondition(), Get(S->getTrueValue()),
                                   Get(S->getFalseValue()), I.getName());
        continue;
      }
      // fpext and fptrunc between matrix element types.
      if (auto *C = dyn_cast<CastInst>(&I);
          C && isMatrix(C->getType()) && isMatrix(C->getSrcTy())) {
        Value *V = IB.CreateCast(C->getOpcode(), Get(C->getOperand(0)),
                                 FragTy(C->getType()), I.getName());
        if (auto *VI = dyn_cast<Instruction>(V))
          VI->copyIRFlags(&I);
        Frag[&I] = V;
        continue;
      }
      if (isa<FreezeInst>(I)) {
        Frag[&I] = IB.CreateFreeze(Get(I.getOperand(0)), I.getName());
        continue;
      }
      // An element-wise intrinsic of matrices (llvm.fma.v64f32).
      if (auto *II = dyn_cast<IntrinsicInst>(&I);
          II && isMatrix(II->getType()) &&
          isTriviallyVectorizable(II->getIntrinsicID()) &&
          all_of(II->args(), [&](Value *A) {
            return !A->getType()->isVectorTy() || A->getType() == II->getType();
          })) {
        SmallVector<Value *, 4> Args;
        for (Value *A : II->args())
          Args.push_back(isMatrix(A->getType()) ? Get(A) : A);
        Value *New = IB.CreateIntrinsic(II->getIntrinsicID(),
                                        {FragTy(II->getType())}, Args, nullptr,
                                        I.getName());
        if (auto *NI = dyn_cast<Instruction>(New))
          NI->copyIRFlags(&I);
        Frag[&I] = New;
        continue;
      }
      auto *CI = dyn_cast<CallInst>(&I);
      Function *Callee = CI ? CI->getCalledFunction() : nullptr;
      StringRef Name = Callee ? Callee->getName() : "";
      // An element-wise AIR function of matrices (air.fma.v64f32) applies to
      // the fragments (air.fma.v2f32), lowered with the other AIR calls.
      if (Name.starts_with("air.") &&
          !Name.starts_with("air.simdgroup_matrix_") &&
          isMatrix(CI->getType()) &&
          all_of(CI->args(), [&](Value *A) {
            return !isMatrix(A->getType()) || A->getType() == CI->getType();
          })) {
        std::string FragName = Name.str();
        for (size_t P; (P = FragName.find("v64")) != std::string::npos;)
          FragName.replace(P, 3, "v2");
        SmallVector<Value *, 4> Args;
        SmallVector<Type *, 4> Params;
        for (Value *A : CI->args()) {
          Args.push_back(isMatrix(A->getType()) ? Get(A) : A);
          Params.push_back(Args.back()->getType());
        }
        FunctionCallee Fn = M.getOrInsertFunction(
            FragName,
            FunctionType::get(FragTy(CI->getType()), Params, false));
        CallInst *New = IB.CreateCall(Fn, Args);
        if (isa<FPMathOperator>(New))
          New->setFastMathFlags(CI->getFastMathFlags());
        Frag[&I] = New;
        continue;
      }
      if (!Name.consume_front("air.simdgroup_matrix_8x8_")) {
        error(&F, "a SIMD-group matrix used other than as a matrix");
        return;
      }
      unsigned N = CI->arg_size();
      auto Arg = [&](unsigned K) { return CI->getArgOperand(K); };
      // The dimensions, strides and origin of a load or store: <2 x i64>.
      auto Shape = [&](unsigned First) {
        Type *V2I64 = FixedVectorType::get(IB.getInt64Ty(), 2);
        return N == First + 3 && Arg(First)->getType() == V2I64 &&
               Arg(First + 1)->getType() == V2I64 &&
               Arg(First + 2)->getType() == V2I64;
      };
      if (Name.starts_with("load.") && Shape(1) &&
          Arg(0)->getType()->isPointerTy()) {
        Type *EltTy = cast<VectorType>(CI->getType())->getElementType();
        Value *V = PoisonValue::get(FragTy(CI->getType()));
        for (unsigned J = 0; J < 2; ++J)
          V = IB.CreateInsertElement(
              V,
              IB.CreateLoad(EltTy, Element(IB, EltTy, Arg(0), Arg(2), Arg(3), J)),
              J);
        Frag[&I] = V;
      } else if (Name.starts_with("store.") && Shape(2) &&
                 Arg(1)->getType()->isPointerTy()) {
        Value *V = Get(Arg(0));
        Type *EltTy = cast<VectorType>(Arg(0)->getType())->getElementType();
        for (unsigned J = 0; J < 2; ++J)
          IB.CreateStore(IB.CreateExtractElement(V, J),
                         Element(IB, EltTy, Arg(1), Arg(3), Arg(4), J));
      } else if (Name.starts_with("multiply_accumulate.") && N == 3 &&
                 Arg(0)->getType() == Arg(1)->getType() &&
                 Arg(2)->getType() == CI->getType()) {
        // f16 or bf16 A and B with f32 C: the products are exact in f32.
        Type *FT = FragTy(CI->getType());
        Value *A = Get(Arg(0)), *B = Get(Arg(1));
        if (A->getType() != FT) {
          A = IB.CreateFPExt(A, FT);
          B = IB.CreateFPExt(B, FT);
        }
        Frag[&I] = IB.CreateIntrinsic(Intrinsic::agx_simdgroup_matrix_fma, {FT},
                                      {A, B, Get(Arg(2))});
      } else if (Name.starts_with("multiply.") && N == 2) {
        Type *FT = FragTy(CI->getType());
        Frag[&I] = IB.CreateIntrinsic(Intrinsic::agx_simdgroup_matrix_fma, {FT},
                                      {Get(Arg(0)), Get(Arg(1)),
                                       Constant::getNullValue(FT)});
      } else if (Name.starts_with("init_filled.") && N == 1) {
        Frag[&I] = IB.CreateVectorSplat(2, Arg(0));
      } else if (Name.starts_with("init_diagonal.") && N == 1) {
        // x where r == c + j, else 0.
        Type *FT = FragTy(CI->getType());
        Value *V = PoisonValue::get(FT), *Zero =
            Constant::getNullValue(FT->getScalarType());
        for (unsigned J = 0; J < 2; ++J) {
          Value *OnDiag =
              IB.CreateICmpEQ(Row, IB.CreateAdd(Col, IB.getInt32(J)));
          V = IB.CreateInsertElement(V, IB.CreateSelect(OnDiag, Arg(0), Zero),
                                     J);
        }
        Frag[&I] = V;
      } else {
        error(&F, "the AIR function " + Callee->getName() +
                      " has no AGX lowering");
        return;
      }
    }
  }
  for (auto [P, NP] : Phis)
    for (unsigned K = 0; K < P->getNumIncomingValues(); ++K)
      NP->addIncoming(Get(P->getIncomingValue(K)), P->getIncomingBlock(K));
  if (Bad) {
    error(&F, "a SIMD-group matrix built other than by AIR's matrix "
              "functions or a splat");
    return;
  }
  for (auto [P, NP] : Phis)
    Old.push_back(P);
  for (Instruction *I : Old)
    if (auto *New = dyn_cast_or_null<Instruction>(Frag.lookup(I)))
      New->takeName(I);
  for (Instruction *I : Old)
    I->replaceAllUsesWith(PoisonValue::get(I->getType()));
  for (Instruction *I : Old)
    I->eraseFromParent();
}

// AIR 2.8's atomics are relaxed, so AIR's lowering of an ordered LLVM atomic
// is the relaxed atomic between two seq_cst fences, where the AGX backend
// fences far more than for the ordered atomic. Where the atomic carries the
// order it came from (air::AtomicOrderingMD) and the fences are still around
// it, the atomic takes that order and the fences go: the original atomic.
void LowerAIR::foldOrderedAtomics(Function &F) {
  auto IsFence = [](Instruction *I, Value *Scope) {
    auto *CI = dyn_cast_or_null<CallInst>(I);
    Function *Callee = CI ? CI->getCalledFunction() : nullptr;
    if (!Callee || Callee->getName() != "air.atomic.fence" ||
        CI->arg_size() != 3)
      return false;
    auto *Order = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    return Order && Order->getZExtValue() == 5 &&
           CI->getArgOperand(2) == Scope;
  };
  SmallVector<CallInst *, 8> Atomics;
  for (Instruction &I : instructions(F))
    if (auto *CI = dyn_cast<CallInst>(&I);
        CI && CI->getMetadata(air::AtomicOrderingMD))
      Atomics.push_back(CI);
  for (CallInst *CI : Atomics) {
    MDNode *Hint = CI->getMetadata(air::AtomicOrderingMD);
    CI->setMetadata(air::AtomicOrderingMD, nullptr);
    StringRef Name = CI->getCalledFunction()
                         ? CI->getCalledFunction()->getName()
                         : StringRef();
    bool CmpXchg = Name.ends_with(".cmpxchg.weak.i32");
    // (ptr, value, order, scope, volatile) or (ptr, expected, desired,
    // success, failure, scope, [flags,] volatile).
    unsigned OrderArg = CmpXchg ? 3 : 2;
    unsigned ScopeArg = CmpXchg ? 5 : 3;
    if (!Name.starts_with("air.atomic.") ||
        Hint->getNumOperands() != (CmpXchg ? 2u : 1u) ||
        CI->arg_size() <= ScopeArg)
      continue;
    Value *Scope = CI->getArgOperand(ScopeArg);
    // A compare-exchange stores the expected value just before it.
    Instruction *Before = CI->getPrevNode();
    if (CmpXchg) {
      auto *SI = dyn_cast_or_null<StoreInst>(Before);
      if (!SI || SI->getPointerOperand() != CI->getArgOperand(1))
        continue;
      Before = SI->getPrevNode();
    }
    Instruction *After = CI->getNextNode();
    if (!IsFence(Before, Scope) || !IsFence(After, Scope) ||
        !all_of(Hint->operands(), [](const MDOperand &Op) {
          return mdconst::hasa<ConstantInt>(Op);
        }))
      continue;
    for (unsigned K = 0; K < Hint->getNumOperands(); ++K)
      CI->setArgOperand(OrderArg + K,
                        mdconst::extract<ConstantInt>(Hint->getOperand(K)));
    Before->eraseFromParent();
    After->eraseFromParent();
  }
}

// A hint (llvm/IR/AIR.h) names its operation in its "air-hint" attribute:
// "call:<function>|<doubles>[|<metadata string operand>...]" or
// "inst:<opcode>[:<predicate>]|<doubles>", where <doubles> has a 'd' for the
// result and each operand that is a double (carried as i64 bits in AIR). Each
// call to the hint becomes that operation, and the hint's body (its AIR
// emulation) goes away with the soft-f64 library it calls.
static Value *nativeHint(IRBuilder<> &B, CallInst &CI, StringRef Key) {
  Module &M = *CI.getModule();
  LLVMContext &Ctx = M.getContext();
  SmallVector<StringRef, 4> Parts;
  Key.split(Parts, '|');
  // "+lane", "+mode": trailing operands that only the emulation uses.
  unsigned Extra = 0;
  while (Parts.size() > 2 && Parts.back().starts_with("+")) {
    Parts.pop_back();
    ++Extra;
  }
  if (CI.arg_size() < Extra)
    return nullptr;
  unsigned NumArgs = CI.arg_size() - Extra;
  if (Parts.size() < 2 || Parts[1].size() != NumArgs + 1)
    return nullptr;
  StringRef Op = Parts[0], Mask = Parts[1];
  auto Ty = [&](Type *T, char C) {
    return C == 'd' ? B.getDoubleTy() : T;
  };
  SmallVector<Value *, 4> Args;
  for (auto [I, A] : enumerate(CI.args()))
    if (I < NumArgs)
      Args.push_back(Mask[I + 1] == 'd' ? B.CreateBitCast(A, B.getDoubleTy())
                                        : A.get());
  Type *Ret = Ty(CI.getType(), Mask[0]);
  Value *R = nullptr;
  if (Op.consume_front("call:")) {
    SmallVector<Type *, 6> Params;
    for (Value *A : Args)
      Params.push_back(A->getType());
    for (StringRef MD : drop_begin(Parts, 2)) {
      Args.push_back(MetadataAsValue::get(Ctx, MDString::get(Ctx, MD)));
      Params.push_back(Type::getMetadataTy(Ctx));
    }
    FunctionCallee Fn =
        M.getOrInsertFunction(Op, FunctionType::get(Ret, Params, false));
    R = B.CreateCall(Fn, Args);
  } else if (Op.consume_front("atomic:")) {
    // "atomic:<operation>|<doubles>|<ordering>|<sync scope>|<align>"
    auto RMWOp = StringSwitch<std::optional<AtomicRMWInst::BinOp>>(Op)
                     .Case("fadd", AtomicRMWInst::FAdd)
                     .Case("fsub", AtomicRMWInst::FSub)
                     .Case("fmax", AtomicRMWInst::FMax)
                     .Case("fmin", AtomicRMWInst::FMin)
                     .Case("xchg", AtomicRMWInst::Xchg)
                     .Default(std::nullopt);
    unsigned Order = 0, Align = 0;
    if (!RMWOp || Parts.size() != 5 || Args.size() != 2 ||
        Parts[2].getAsInteger(10, Order) || Parts[4].getAsInteger(10, Align))
      return nullptr;
    R = B.CreateAtomicRMW(*RMWOp, Args[0], Args[1], llvm::Align(Align),
                          AtomicOrdering(Order),
                          Ctx.getOrInsertSyncScopeID(Parts[3]));
  } else if (Op.consume_front("inst:")) {
    auto [Name, Pred] = Op.split(':');
    auto P = StringSwitch<CmpInst::Predicate>(Pred)
                 .Case("false", CmpInst::FCMP_FALSE)
                 .Case("oeq", CmpInst::FCMP_OEQ)
                 .Case("ogt", CmpInst::FCMP_OGT)
                 .Case("oge", CmpInst::FCMP_OGE)
                 .Case("olt", CmpInst::FCMP_OLT)
                 .Case("ole", CmpInst::FCMP_OLE)
                 .Case("one", CmpInst::FCMP_ONE)
                 .Case("ord", CmpInst::FCMP_ORD)
                 .Case("uno", CmpInst::FCMP_UNO)
                 .Case("ueq", CmpInst::FCMP_UEQ)
                 .Case("ugt", CmpInst::FCMP_UGT)
                 .Case("uge", CmpInst::FCMP_UGE)
                 .Case("ult", CmpInst::FCMP_ULT)
                 .Case("ule", CmpInst::FCMP_ULE)
                 .Case("une", CmpInst::FCMP_UNE)
                 .Case("true", CmpInst::FCMP_TRUE)
                 .Default(CmpInst::BAD_FCMP_PREDICATE);
    auto BinOp = StringSwitch<std::optional<Instruction::BinaryOps>>(Name)
                     .Case("fadd", Instruction::FAdd)
                     .Case("fsub", Instruction::FSub)
                     .Case("fmul", Instruction::FMul)
                     .Case("fdiv", Instruction::FDiv)
                     .Case("frem", Instruction::FRem)
                     .Default(std::nullopt);
    auto Cast = StringSwitch<std::optional<Instruction::CastOps>>(Name)
                    .Case("fptosi", Instruction::FPToSI)
                    .Case("fptoui", Instruction::FPToUI)
                    .Case("sitofp", Instruction::SIToFP)
                    .Case("uitofp", Instruction::UIToFP)
                    .Case("fpext", Instruction::FPExt)
                    .Case("fptrunc", Instruction::FPTrunc)
                    .Default(std::nullopt);
    if (BinOp && Args.size() == 2)
      R = B.CreateBinOp(*BinOp, Args[0], Args[1]);
    else if (Name == "fneg" && Args.size() == 1)
      R = B.CreateFNeg(Args[0]);
    else if (Name == "fcmp" && P != CmpInst::BAD_FCMP_PREDICATE &&
             Args.size() == 2)
      R = B.CreateFCmp(P, Args[0], Args[1]);
    else if (Cast && Args.size() == 1)
      R = B.CreateCast(*Cast, Args[0], Ret);
  }
  if (!R)
    return nullptr;
  if (auto *I = dyn_cast<Instruction>(R))
    I->copyMetadata(CI);
  if (Mask[0] == 'd')
    R = B.CreateBitCast(R, CI.getType());
  return R;
}

void LowerAIR::lowerHints() {
  bool Any = false;
  for (Function &F : make_early_inc_range(M)) {
    if (!F.hasFnAttribute(air::HintAttr))
      continue;
    Any = true;
    StringRef Key = F.getFnAttribute(air::HintAttr).getValueAsString();
    for (User *U : make_early_inc_range(F.users())) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->getCalledFunction() != &F)
        continue;
      IRBuilder<> B(CI);
      bool Emulate = !NativeHints || any_of(EmulateHints, [&](StringRef P) {
                       return Key.starts_with(P);
                     });
      if (Value *R = !Emulate ? nativeHint(B, *CI, Key) : nullptr) {
        if (!CI->getType()->isVoidTy()) {
          R->takeName(CI);
          CI->replaceAllUsesWith(R);
        }
        SmallVector<WeakTrackingVH, 2> Ops(CI->args());
        CI->eraseFromParent();
        // The operands only the emulation read (the tracked FP mode, and its
        // phis), now dead.
        for (WeakTrackingVH &V : Ops) {
          if (auto *P = dyn_cast_or_null<PHINode>(V))
            RecursivelyDeleteDeadPHINode(P);
          else if (V)
            RecursivelyDeleteTriviallyDeadInstructions(V);
        }
        continue;
      }
      // Otherwise the hint's AIR emulation runs, inline (AGX has no calls).
      InlineFunctionInfo IFI;
      if (!InlineFunction(*CI, IFI).isSuccess())
        error(CI->getFunction(), "cannot inline the AIR emulation of " + Key);
    }
    if (F.use_empty())
      F.eraseFromParent();
  }
  if (!Any)
    return;
  // The soft-f64 library of the emulations, once nothing calls it.
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

// AIR's portable forms of operations AGX has (as agx-raise-to-air writes
// them, or any AIR producer): the hardware approximations from afn math
// (1 / x, 1 / sqrt(x), sin and cos of x * pi / 2: AGX's take quadrants),
// and the explicit flushes of PTX's .ftz where the f32 ALU flushes anyway.
void LowerAIR::recoverAGXOperations(Function &F) {
  if (F.isDeclaration())
    return;
  const AGXSubtarget &ST = TM.getSubtarget<AGXSubtarget>(F);
  // The f32 ALU flushes denormal operands and results unless the function
  // keeps denormals in software (ieee-fp32).
  bool HardwareFlush =
      !(ST.hasIEEEFP32() &&
        F.getDenormalMode(APFloat::IEEEsingle()).Output == DenormalMode::IEEE);
  using namespace PatternMatch;
  auto IsOne = [](Value *V) {
    auto *C = dyn_cast<ConstantFP>(V);
    return C && C->isExactlyValue(1.0);
  };
  auto Afn = [](Value *V) {
    auto *I = dyn_cast<Instruction>(V);
    return I && isa<FPMathOperator>(I) && I->hasApproxFunc();
  };
  auto FlushingArith = [](Value *V) {
    auto *I = dyn_cast<Instruction>(V);
    if (!I || !I->getType()->isFloatTy())
      return false;
    if (isa<BinaryOperator>(I))
      return I->getOpcode() == Instruction::FAdd ||
             I->getOpcode() == Instruction::FSub ||
             I->getOpcode() == Instruction::FMul;
    auto *II = dyn_cast<IntrinsicInst>(I);
    return II && (II->getIntrinsicID() == Intrinsic::fma ||
                  II->getIntrinsicID() == Intrinsic::fmuladd);
  };
  // Whether a float cannot be denormal where the ALU flushes: made by an
  // operation that flushes, a constant that is not denormal, or a choice
  // among, sign change or absolute value of such values (selects and phis of
  // a run of .ftz operations, which strict mode then flushes on the bits).
  DenseMap<const Value *, bool> NoDenormalMemo;
  std::function<bool(Value *, unsigned)> NoDenormalAt = [&](Value *V,
                                                           unsigned Depth) {
    if (FlushingArith(V))
      return true;
    // (a square root, or its reciprocal, of any float is normal, zero,
    // infinite or NaN: the root of the least denormal is about 2^-74.5)
    if (match(V, m_Intrinsic<Intrinsic::sqrt>(m_Value())) ||
        match(V, m_Intrinsic<Intrinsic::agx_rsqrt>(m_Value())))
      return true;
    if (auto *C = dyn_cast<ConstantFP>(V))
      return !C->getValueAPF().isDenormal();
    auto *I = dyn_cast<Instruction>(V);
    if (!I || !I->getType()->isFloatTy() || Depth > 8)
      return false;
    if (auto It = NoDenormalMemo.find(I); It != NoDenormalMemo.end())
      return It->second;
    // (a cycle through phis is assumed free of denormals while it is checked:
    // every value entering it is checked)
    NoDenormalMemo[I] = true;
    bool R = false;
    Value *X;
    if (auto *Sel = dyn_cast<SelectInst>(I))
      R = NoDenormalAt(Sel->getTrueValue(), Depth + 1) &&
          NoDenormalAt(Sel->getFalseValue(), Depth + 1);
    else if (auto *Phi = dyn_cast<PHINode>(I))
      R = all_of(Phi->incoming_values(),
                 [&](Value *In) { return NoDenormalAt(In, Depth + 1); });
    else if (match(I, m_FNeg(m_Value(X))) ||
             match(I, m_Intrinsic<Intrinsic::fabs>(m_Value(X))))
      R = NoDenormalAt(X, Depth + 1);
    // (what was computed while this one was assumed free of denormals may be
    // wrong when it is not)
    if (!R)
      NoDenormalMemo.clear();
    NoDenormalMemo[I] = R;
    return R;
  };
  auto NoDenormal = [&](Value *V) { return NoDenormalAt(V, 0); };
  for (Instruction &I : make_early_inc_range(instructions(F))) {
    IRBuilder<> B(&I);
    Type *Ty = I.getType();
    bool Float = Ty->isFloatTy();
    Value *R = nullptr;
    if (Float && Afn(&I) && I.getOpcode() == Instruction::FDiv &&
        IsOne(I.getOperand(0))) {
      // 1 / sqrt(x) and 1 / x (R is only set for these: it replaces I)
      R = I.getOperand(1);
      Value *S;
      if (match(R, m_Intrinsic<Intrinsic::sqrt>(m_Value(S))) && Afn(R))
        R = B.CreateIntrinsic(Intrinsic::agx_rsqrt, {Ty}, {S});
      else
        R = B.CreateIntrinsic(Intrinsic::agx_rcp, {Ty}, {R});
    } else if (auto *II = dyn_cast<IntrinsicInst>(&I);
               Float && II && Afn(II) &&
               (II->getIntrinsicID() == Intrinsic::sin ||
                II->getIntrinsicID() == Intrinsic::cos)) {
      // (both in radians: llvm.agx.sin is the code of llvm.sin, which LLVM
      // does not fold on the host)
      R = B.CreateIntrinsic(II->getIntrinsicID() == Intrinsic::sin
                                ? Intrinsic::agx_sin
                                : Intrinsic::agx_cos,
                            {Ty}, {II->getArgOperand(0)});
    } else if (auto *Sel = dyn_cast<SelectInst>(&I);
               Sel && Ty->isFPOrFPVectorTy() &&
               match(Sel->getTrueValue(), m_NaN())) {
      // AIR's minimum / maximum (air-nan-min-max): fmin / fmax with a NaN
      // check, AGX's minimum / maximum.
      Value *A, *C;
      auto *MM = dyn_cast<IntrinsicInst>(Sel->getFalseValue());
      if (MM &&
          (MM->getIntrinsicID() == Intrinsic::minnum ||
           MM->getIntrinsicID() == Intrinsic::maxnum) &&
          match(Sel->getCondition(),
                m_SpecificFCmp(FCmpInst::FCMP_UNO, m_Value(A), m_Value(C))) &&
          ((A == MM->getArgOperand(0) && C == MM->getArgOperand(1)) ||
           (A == MM->getArgOperand(1) && C == MM->getArgOperand(0))))
        R = B.CreateBinaryIntrinsic(MM->getIntrinsicID() == Intrinsic::minnum
                                        ? Intrinsic::minimum
                                        : Intrinsic::maximum,
                                    MM->getArgOperand(0), MM->getArgOperand(1));
    } else if (auto *Tr = dyn_cast<FPTruncInst>(&I)) {
      // The same of half or bfloat values, computed in float: in their type
      // (exact, and without the float ALU's flushing of their denormals).
      auto *MM = dyn_cast<IntrinsicInst>(Tr->getOperand(0));
      Value *A, *C;
      if (MM && MM->hasOneUse() &&
          (MM->getIntrinsicID() == Intrinsic::minimum ||
           MM->getIntrinsicID() == Intrinsic::maximum) &&
          match(MM->getArgOperand(0), m_FPExt(m_Value(A))) &&
          match(MM->getArgOperand(1), m_FPExt(m_Value(C))) &&
          A->getType() == Ty && C->getType() == Ty)
        R = B.CreateBinaryIntrinsic(MM->getIntrinsicID(), A, C);
    } else if (auto *Sel = dyn_cast<SelectInst>(&I); Float && Sel && HardwareFlush) {
      // select(|x| < FLT_MIN, copysign(0, x), x): x, where the operation that
      // makes x or every one that reads the flushed value flushes.
      Value *A = nullptr, *Z;
      const APFloat *Min = nullptr;
      // (or x == 0, which instcombine makes of it where the function's
      // denormal mode compares denormals as zero)
      bool Tiny =
          (match(Sel->getCondition(),
                 m_SpecificFCmp(FCmpInst::FCMP_OLT,
                                m_Intrinsic<Intrinsic::fabs>(m_Value(A)),
                                m_APFloat(Min))) &&
           Min->isSmallestNormalized() && !Min->isNegative()) ||
          (match(Sel->getCondition(),
                 m_SpecificFCmp(FCmpInst::FCMP_OEQ, m_Value(A), m_PosZeroFP())) &&
           F.getDenormalMode(APFloat::IEEEsingle()).Input != DenormalMode::IEEE);
      if (Tiny &&
          match(Sel->getTrueValue(),
                m_Intrinsic<Intrinsic::copysign>(m_Value(Z), m_Specific(A))) &&
          match(Z, m_PosZeroFP()) && Sel->getFalseValue() == A &&
          (FlushingArith(A) ||
           (!Sel->use_empty() && all_of(Sel->users(), FlushingArith))))
        R = A;
    } else if (auto *Sel = dyn_cast<SelectInst>(&I);
               Sel && Ty->isIntegerTy(32) && HardwareFlush) {
      // The same flush on the bits (a front end's strict mode tests them:
      // a flushing GPU compares a denormal as zero):
      // select((x & 0x7f800000) == 0, x & 0x80000000, x), where x is the bits
      // of an operation that flushes, also through the canonical NaN select
      // select((x & 0x7fffffff) > 0x7f800000, 0x7fffffff, x), or every user
      // reads it back as an operand of one.
      Value *X = nullptr;
      if (match(Sel->getCondition(),
                m_SpecificICmp(ICmpInst::ICMP_EQ,
                               m_And(m_Value(X), m_SpecificInt(0x7f800000)),
                               m_Zero())) &&
          Sel->getFalseValue() == X &&
          match(Sel->getTrueValue(),
                m_And(m_Specific(X), m_SpecificInt(0x80000000)))) {
        Value *Bits = X, *Y;
        if (match(Bits, m_Select(m_SpecificICmp(ICmpInst::ICMP_UGT,
                                                m_And(m_Value(Y),
                                                      m_SpecificInt(0x7fffffff)),
                                                m_SpecificInt(0x7f800000)),
                                 m_SpecificInt(0x7fffffff), m_Deferred(Y))))
          Bits = Y;
        Value *A = nullptr;
        bool Made = match(Bits, m_BitCast(m_Value(A))) && NoDenormal(A);
        bool Read = !Sel->use_empty() && all_of(Sel->users(), [&](User *U) {
          auto *BC = dyn_cast<BitCastInst>(U);
          return BC && BC->getType()->isFloatTy() && !BC->use_empty() &&
                 all_of(BC->users(), FlushingArith);
        });
        if (Made || Read)
          R = X;
      }
    }
    if (!R)
      continue;
    R->takeName(&I);
    I.replaceAllUsesWith(R);
    RecursivelyDeleteTriviallyDeadInstructions(&I);
    // (the memo's keys may be among the deleted)
    NoDenormalMemo.clear();
  }
}

// AIR may call its own functions (with return values); AGX inlines every
// function into the kernels (AGXUsage.rst, "Kernel ABI"), so the calls become
// their bodies here, before code generation, unless leaf calls are enabled.
// agx-raise-to-air's software f64 (`__air_f64_*`, double as i64 bits) is the
// portable form of double arithmetic; AGX has its own software f64
// (AGXSoftF64), tuned for its ALU. Calls rounded to nearest (the plain names,
// or the `_r` ones with a constant rounding of 1, FLT_ROUNDS' nearest) become
// the double operations again.
void LowerAIR::recoverDoubles() {
  if (!NativeF64)
    return;
  LLVMContext &Ctx = M.getContext();
  Type *F64 = Type::getDoubleTy(Ctx), *F32 = Type::getFloatTy(Ctx);
  Type *I64 = Type::getInt64Ty(Ctx), *I32 = Type::getInt32Ty(Ctx);
  for (Function &Lib : make_early_inc_range(M)) {
    StringRef Name = Lib.getName();
    if (!Name.consume_front("__air_f64_"))
      continue;
    bool Directed = Name.consume_back("_r");
    for (User *U : make_early_inc_range(Lib.users())) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->getCalledFunction() != &Lib)
        continue;
      unsigned NArgs = CI->arg_size() - Directed;
      if (Directed) {
        auto *RM = dyn_cast<ConstantInt>(CI->getArgOperand(NArgs));
        if (!RM || !RM->isOne())
          continue;
      }
      IRBuilder<> B(CI);
      auto D = [&](unsigned I) {
        return B.CreateBitCast(CI->getArgOperand(I), F64);
      };
      auto Bits = [&](Value *V) { return B.CreateBitCast(V, I64); };
      Value *R = nullptr;
      if (Name == "add" && NArgs == 2)
        R = Bits(B.CreateFAdd(D(0), D(1)));
      else if (Name == "sub" && NArgs == 2)
        R = Bits(B.CreateFSub(D(0), D(1)));
      else if (Name == "mul" && NArgs == 2)
        R = Bits(B.CreateFMul(D(0), D(1)));
      else if (Name == "div" && NArgs == 2)
        R = Bits(B.CreateFDiv(D(0), D(1)));
      else if (Name == "fma" && NArgs == 3)
        R = Bits(B.CreateIntrinsic(Intrinsic::fma, F64, {D(0), D(1), D(2)}));
      else if (Name == "sqrt" && NArgs == 1)
        R = Bits(B.CreateUnaryIntrinsic(Intrinsic::sqrt, D(0)));
      else if (Name == "floor" && NArgs == 1)
        R = Bits(B.CreateUnaryIntrinsic(Intrinsic::floor, D(0)));
      else if (Name == "roundeven" && NArgs == 1)
        R = Bits(B.CreateUnaryIntrinsic(Intrinsic::roundeven, D(0)));
      else if (Name == "from_i64" && NArgs == 1)
        R = Bits(B.CreateSIToFP(CI->getArgOperand(0), F64));
      else if (Name == "from_u64" && NArgs == 1)
        R = Bits(B.CreateUIToFP(CI->getArgOperand(0), F64));
      else if (Name == "from_f32" && NArgs == 1)
        R = Bits(B.CreateFPExt(B.CreateBitCast(CI->getArgOperand(0), F32),
                               F64));
      else if (Name == "to_f32" && NArgs == 1)
        R = B.CreateBitCast(B.CreateFPTrunc(D(0), F32), I32);
      else if (Name == "to_i64" && NArgs == 1)
        R = B.CreateIntrinsic(Intrinsic::fptosi_sat, {I64, F64}, {D(0)});
      else if (Name == "to_u64" && NArgs == 1)
        R = B.CreateIntrinsic(Intrinsic::fptoui_sat, {I64, F64}, {D(0)});
      else if (Name == "unord" && NArgs == 2)
        R = B.CreateZExt(B.CreateFCmpUNO(D(0), D(1)), I32);
      else if (Name == "eq" && NArgs == 2)
        R = B.CreateZExt(B.CreateFCmpOEQ(D(0), D(1)), I32);
      else if (Name == "lt" && NArgs == 2)
        R = B.CreateZExt(B.CreateFCmpOLT(D(0), D(1)), I32);
      if (!R || R->getType() != CI->getType())
        continue;
      CI->replaceAllUsesWith(R);
      CI->eraseFromParent();
    }
  }
}

void LowerAIR::inlineCalls() {
  if (agxExperimentalLeafCallsEnabled())
    return;
  SmallVector<CallBase *, 16> Calls;
  // Recursive calls are left (an error in code generation).
  for (unsigned Round = 0, Again = true; Again && Round < 16; ++Round) {
    Again = false;
    Calls.clear();
    for (Function &F : M)
      for (Instruction &I : instructions(F))
        if (auto *CB = dyn_cast<CallBase>(&I))
          if (Function *Callee = CB->getCalledFunction();
              Callee && !Callee->isDeclaration() && Callee != &F)
            Calls.push_back(CB);
    for (CallBase *CB : Calls) {
      InlineFunctionInfo IFI;
      if (InlineFunction(*CB, IFI).isSuccess())
        Again = true;
    }
  }
  for (Function &F : make_early_inc_range(M))
    if (F.hasLocalLinkage() && F.use_empty() && !F.isDeclaration())
      F.eraseFromParent();
}

// AGX's hidden kernel arguments ("agx-hidden") and by-value parameters are
// passed by value; AIR has them as byref constant buffers ("air-by-value" from
// agx-raise-to-air). A kernel that only loads them whole takes them by value
// again, so that, among others, negations and absolute values of them fold
// into source modifiers.
Function *LowerAIR::hiddenByValue(Function &K) {
  SmallVector<unsigned, 2> Hidden;
  for (Argument &A : K.args()) {
    Type *T = K.getParamByRefType(A.getArgNo());
    if (!T || !(K.getAttributes().hasParamAttr(A.getArgNo(), "agx-hidden") ||
                K.getAttributes().hasParamAttr(A.getArgNo(), "air-by-value")))
      continue;
    if (!all_of(A.users(), [&](User *U) {
          auto *L = dyn_cast<LoadInst>(U);
          return L && L->isSimple() && L->getType() == T;
        }))
      continue;
    Hidden.push_back(A.getArgNo());
  }
  if (Hidden.empty())
    return &K;
  SmallVector<Type *, 8> Params(K.getFunctionType()->params());
  for (unsigned I : Hidden)
    Params[I] = K.getParamByRefType(I);
  auto *FT = FunctionType::get(K.getReturnType(), Params, false);
  Function *New =
      Function::Create(FT, K.getLinkage(), K.getAddressSpace(), "", &M);
  New->copyAttributesFrom(&K);
  New->splice(New->begin(), &K);
  for (auto [Old, NewA] : zip(K.args(), New->args())) {
    NewA.takeName(&Old);
    if (!is_contained(Hidden, Old.getArgNo())) {
      Old.replaceAllUsesWith(&NewA);
      continue;
    }
    for (User *U : make_early_inc_range(Old.users())) {
      U->replaceAllUsesWith(&NewA);
      cast<Instruction>(U)->eraseFromParent();
    }
    AttrBuilder AB(Ctx);
    for (Attribute At : K.getAttributes().getParamAttrs(Old.getArgNo()))
      if (At.isStringAttribute() && At.getKindAsString() != "air-by-value")
        AB.addAttribute(At);
    New->setAttributes(New->getAttributes()
                           .removeParamAttributes(Ctx, NewA.getArgNo())
                           .addParamAttributes(Ctx, NewA.getArgNo(), AB));
  }
  std::string Name = K.getName().str();
  K.replaceAllUsesWith(New);
  K.eraseFromParent();
  New->setName(Name);
  return New;
}

void LowerAIR::lowerCalls() {
  for (Function &F : make_early_inc_range(M)) {
    if (!F.isDeclaration() || !F.getName().starts_with("air."))
      continue;
    std::string Name = F.getName().str();
    for (User *U : make_early_inc_range(F.users())) {
      auto *CI = dyn_cast<CallInst>(U);
      Value *Result = nullptr;
      if (!CI || CI->getCalledFunction() != &F ||
          !lowerCall(*CI, Name, Result)) {
        const Function *In =
            isa<Instruction>(U) ? cast<Instruction>(U)->getFunction() : nullptr;
        error(In, "the AIR function " + Name + " has no AGX lowering");
        continue;
      }
      if (Result) {
        Result->takeName(CI);
        CI->replaceAllUsesWith(Result);
      }
      CI->eraseFromParent();
    }
    if (F.use_empty())
      F.eraseFromParent();
  }
}

/// Whether values of T are laid out differently under the data layouts A and
/// B: AIR's aligns 3-element vectors of 32-bit values to 16 bytes (a
/// <3 x float> takes 16), AGX's to 4 (12).
static bool layoutDiffers(Type *T, const DataLayout &A, const DataLayout &B) {
  if (auto *AT = dyn_cast<ArrayType>(T))
    return layoutDiffers(AT->getElementType(), A, B);
  if (auto *ST = dyn_cast<StructType>(T))
    return any_of(ST->elements(),
                  [&](Type *E) { return layoutDiffers(E, A, B); });
  if (!T->isSized() || isa<ScalableVectorType>(T))
    return false;
  return A.getTypeAllocSize(T) != B.getTypeAllocSize(T) ||
         A.getABITypeAlign(T) != B.getABITypeAlign(T);
}

/// Before the module takes AGX's data layout: the address computations and
/// stack slots whose types AGX lays out differently keep AIR's layout, as
/// byte offsets and byte arrays (a <3 x float> array has a 16-byte stride).
static void keepAIRLayout(Module &M, const DataLayout &AIR,
                          const DataLayout &AGX) {
  for (Function &F : M)
    for (Instruction &I : make_early_inc_range(instructions(F))) {
      if (auto *AI = dyn_cast<AllocaInst>(&I)) {
        Type *T = AI->getAllocatedType();
        if (!layoutDiffers(T, AIR, AGX))
          continue;
        IRBuilder<> B(AI);
        auto *New = B.CreateAlloca(
            ArrayType::get(B.getInt8Ty(), AIR.getTypeAllocSize(T)),
            AI->getAddressSpace(), AI->getArraySize());
        New->setAlignment(std::max(AI->getAlign(), AIR.getABITypeAlign(T)));
        New->takeName(AI);
        AI->replaceAllUsesWith(New);
        AI->eraseFromParent();
        continue;
      }
      auto *GEP = dyn_cast<GetElementPtrInst>(&I);
      if (!GEP || GEP->getType()->isVectorTy() ||
          !layoutDiffers(GEP->getSourceElementType(), AIR, AGX))
        continue;
      unsigned Width = AIR.getIndexTypeSizeInBits(GEP->getType());
      SmallMapVector<Value *, APInt, 4> Variable;
      APInt Constant(Width, 0);
      if (!GEP->collectOffset(AIR, Width, Variable, Constant))
        continue;
      IRBuilder<> B(GEP);
      Type *IntTy = B.getIntNTy(Width);
      Value *Offset = nullptr;
      for (auto &[V, Scale] : Variable) {
        Value *Term = B.CreateMul(B.CreateSExtOrTrunc(V, IntTy),
                                  ConstantInt::get(IntTy, Scale));
        Offset = Offset ? B.CreateAdd(Offset, Term) : Term;
      }
      if (!Offset || !Constant.isZero())
        Offset = Offset ? B.CreateAdd(Offset, ConstantInt::get(IntTy, Constant))
                        : ConstantInt::get(IntTy, Constant);
      Value *New = B.CreateGEP(B.getInt8Ty(), GEP->getPointerOperand(), Offset,
                               "", GEP->getNoWrapFlags());
      New->takeName(GEP);
      GEP->replaceAllUsesWith(New);
      GEP->eraseFromParent();
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

  // The module's own globals and functions: those created below already have
  // AGX's types (AIR's generic hint space is AGX's constant space).
  SmallVector<GlobalVariable *, 8> OldGlobals;
  for (GlobalVariable &GV : M.globals())
    OldGlobals.push_back(&GV);
  SmallVector<Function *, 16> OldFunctions;
  for (Function &F : M)
    OldFunctions.push_back(&F);
  for (GlobalVariable *GVP : OldGlobals) {
    GlobalVariable &GV = *GVP;
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
  for (Function *FP : OldFunctions) {
    Function &F = *FP;
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
  // The globals that keep their types may still refer to the functions and
  // globals that change (a table of function addresses).
  for (GlobalVariable *GV : OldGlobals)
    if (!is_contained(make_first_range(Globals), GV) && GV->hasInitializer())
      GV->setInitializer(MapValue(GV->getInitializer(), VMap, RF_None, &Types));
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

  SmallVector<GlobalVariable *, 8> Order;
  for (GlobalVariable *GV : OldGlobals) {
    auto It = find_if(Globals, [&](auto &P) { return P.first == GV; });
    Order.push_back(It != Globals.end() ? It->second : GV);
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
  // The globals in their order (it decides the layout of the global data).
  for (GlobalVariable *GV : Order) {
    GV->removeFromParent();
    M.insertGlobalVariable(GV);
  }
  // Intrinsic declarations left without uses.
  for (Function &F : make_early_inc_range(M))
    if (F.isIntrinsic() && F.use_empty())
      F.eraseFromParent();

  for (Function &F : M) {
    lowerMatrices(F);
    foldOrderedAtomics(F);
  }
  lowerHints();
  lowerCalls();
  for (Function &F : M)
    recoverAGXOperations(F);
  recoverDoubles();
  dropAIRMetadata();
  M.setTargetTriple(TM.getTargetTriple());
  // (AIR's layout: llc -mtriple=agx has already given the module AGX's)
  keepAIRLayout(M, DataLayout(Triple("air").computeDataLayout()),
                TM.createDataLayout());
  M.setDataLayout(TM.createDataLayout());
  for (AIRKernel &K : Kernels)
    passConstantBuffersByReference(K, K.NewF);
  inlineCalls();
  for (AIRKernel &K : Kernels)
    K.NewF = hiddenByValue(*K.NewF);
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
