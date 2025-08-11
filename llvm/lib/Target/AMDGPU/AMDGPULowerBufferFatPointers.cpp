//===-- AMDGPULowerBufferFatPointers.cpp ---------------------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers operations on buffer fat pointers (addrspace 7) to
// operations on buffer resources (addrspace 8) and is needed for correct
// codegen.
//
// # Background
//
// Address space 7 (the buffer fat pointer) is a 160-bit pointer that consists
// of a 128-bit buffer descriptor and a 32-bit offset into that descriptor.
// The buffer resource part needs to be it needs to be a "raw" buffer resource
// (it must have a stride of 0 and bounds checks must be in raw buffer mode
// or disabled).
//
// When these requirements are met, a buffer resource can be treated as a
// typical (though quite wide) pointer that follows typical LLVM pointer
// semantics. This allows the frontend to reason about such buffers (which are
// often encountered in the context of SPIR-V kernels).
//
// However, because of their non-power-of-2 size, these fat pointers cannot be
// present during translation to MIR (though this restriction may be lifted
// during the transition to GlobalISel). Therefore, this pass is needed in order
// to correctly implement these fat pointers.
//
// The resource intrinsics take the resource part (the address space 8 pointer)
// and the offset part (the 32-bit integer) as separate arguments. In addition,
// many users of these buffers manipulate the offset while leaving the resource
// part alone. For these reasons, we want to typically separate the resource
// and offset parts into separate variables, but combine them together when
// encountering cases where this is required, such as by inserting these values
// into aggretates or moving them to memory.
//
// Therefore, at a high level, `ptr addrspace(7) %x` becomes `ptr addrspace(8)
// %x.rsrc` and `i32 %x.off`, which will be combined into `{ptr addrspace(8),
// i32} %x = {%x.rsrc, %x.off}` if needed. Similarly, `vector<Nxp7>` becomes
// `{vector<Nxp8>, vector<Nxi32 >}` and its component parts.
//
// # Implementation
//
// This pass proceeds in three main phases:
//
// ## Rewriting loads and stores of p7 and memcpy()-like handling
//
// The first phase is to rewrite away all loads and stors of `ptr addrspace(7)`,
// including aggregates containing such pointers, to ones that use `i160`. This
// is handled by `StoreFatPtrsAsIntsAndExpandMemcpyVisitor` , which visits
// loads, stores, and allocas and, if the loaded or stored type contains `ptr
// addrspace(7)`, rewrites that type to one where the p7s are replaced by i160s,
// copying other parts of aggregates as needed. In the case of a store, each
// pointer is `ptrtoint`d to i160 before storing, and load integers are
// `inttoptr`d back. This same transformation is applied to vectors of pointers.
//
// Such a transformation allows the later phases of the pass to not need
// to handle buffer fat pointers moving to and from memory, where we load
// have to handle the incompatibility between a `{Nxp8, Nxi32}` representation
// and `Nxi60` directly. Instead, that transposing action (where the vectors
// of resources and vectors of offsets are concatentated before being stored to
// memory) are handled through implementing `inttoptr` and `ptrtoint` only.
//
// Atomics operations on `ptr addrspace(7)` values are not suppported, as the
// hardware does not include a 160-bit atomic.
//
// In order to save on O(N) work and to ensure that the contents type
// legalizer correctly splits up wide loads, also unconditionally lower
// memcpy-like intrinsics into loops here.
//
// ## Buffer contents type legalization
//
// The underlying buffer intrinsics only support types up to 128 bits long,
// and don't support complex types. If buffer operations were
// standard pointer operations that could be represented as MIR-level loads,
// this would be handled by the various legalization schemes in instruction
// selection. However, because we have to do the conversion from `load` and
// `store` to intrinsics at LLVM IR level, we must perform that legalization
// ourselves.
//
// This involves a combination of
// - Converting arrays to vectors where possible
// - Otherwise, splitting loads and stores of aggregates into loads/stores of
//   each component.
// - Zero-extending things to fill a whole number of bytes
// - Casting values of types that don't neatly correspond to supported machine
// value
//   (for example, an i96 or i256) into ones that would work (
//    like <3 x i32> and <8 x i32>, respectively)
// - Splitting values that are too long (such as aforementioned <8 x i32>) into
//   multiple operations.
//
// ## Type remapping
//
// We use a `ValueMapper` to mangle uses of [vectors of] buffer fat pointers
// to the corresponding struct type, which has a resource part and an offset
// part.
//
// This uses a `BufferFatPtrToStructTypeMap` and a `FatPtrConstMaterializer`
// to, usually by way of `setType`ing values. Constants are handled here
// because there isn't a good way to fix them up later.
//
// This has the downside of leaving the IR in an invalid state (for example,
// the instruction `getelementptr {ptr addrspace(8), i32} %p, ...` will exist),
// but all such invalid states will be resolved by the third phase.
//
// Functions that don't take buffer fat pointers are modified in place. Those
// that do take such pointers have their basic blocks moved to a new function
// with arguments that are {ptr addrspace(8), i32} arguments and return values.
// This phase also records intrinsics so that they can be remangled or deleted
// later.
//
// ## Splitting pointer structs
//
// The meat of this pass consists of defining semantics for operations that
// produce or consume [vectors of] buffer fat pointers in terms of their
// resource and offset parts. This is accomplished throgh the `SplitPtrStructs`
// visitor.
//
// In the first pass through each function that is being lowered, the splitter
// inserts new instructions to implement the split-structures behavior, which is
// needed for correctness and performance. It records a list of "split users",
// instructions that are being replaced by operations on the resource and offset
// parts.
//
// Split users do not necessarily need to produce parts themselves (
// a `load float, ptr addrspace(7)` does not, for example), but, if they do not
// generate fat buffer pointers, they must RAUW in their replacement
// instructions during the initial visit.
//
// When these new instructions are created, they use the split parts recorded
// for their initial arguments in order to generate their replacements, creating
// a parallel set of instructions that does not refer to the original fat
// pointer values but instead to their resource and offset components.
//
// Instructions, such as `extractvalue`, that produce buffer fat pointers from
// sources that do not have split parts, have such parts generated using
// `extractvalue`. This is also the initial handling of PHI nodes, which
// are then cleaned up.
//
// ### Conditionals
//
// PHI nodes are initially given resource parts via `extractvalue`. However,
// this is not an efficient rewrite of such nodes, as, in most cases, the
// resource part in a conditional or loop remains constant throughout the loop
// and only the offset varies. Failing to optimize away these constant resources
// would cause additional registers to be sent around loops and might lead to
// waterfall loops being generated for buffer operations due to the
// "non-uniform" resource argument.
//
// Therefore, after all instructions have been visited, the pointer splitter
// post-processes all encountered conditionals. Given a PHI node or select,
// getPossibleRsrcRoots() collects all values that the resource parts of that
// conditional's input could come from as well as collecting all conditional
// instructions encountered during the search. If, after filtering out the
// initial node itself, the set of encountered conditionals is a subset of the
// potential roots and there is a single potential resource that isn't in the
// conditional set, that value is the only possible value the resource argument
// could have throughout the control flow.
//
// If that condition is met, then a PHI node can have its resource part changed
// to the singleton value and then be replaced by a PHI on the offsets.
// Otherwise, each PHI node is split into two, one for the resource part and one
// for the offset part, which replace the temporary `extractvalue` instructions
// that were added during the first pass.
//
// Similar logic applies to `select`, where
// `%z = select i1 %cond, %cond, ptr addrspace(7) %x, ptr addrspace(7) %y`
// can be split into `%z.rsrc = %x.rsrc` and
// `%z.off = select i1 %cond, ptr i32 %x.off, i32 %y.off`
// if both `%x` and `%y` have the same resource part, but two `select`
// operations will be needed if they do not.
//
// ### Final processing
//
// After conditionals have been cleaned up, the IR for each function is
// rewritten to remove all the old instructions that have been split up.
//
// Any instruction that used to produce a buffer fat pointer (and therefore now
// produces a resource-and-offset struct after type remapping) is
// replaced as follows:
// 1. All debug value annotations are cloned to reflect that the resource part
//    and offset parts are computed separately and constitute different
//    fragments of the underlying source language variable.
// 2. All uses that were themselves split are replaced by a `poison` of the
//    struct type, as they will themselves be erased soon. This rule, combined
//    with debug handling, should leave the use lists of split instructions
//    empty in almost all cases.
// 3. If a user of the original struct-valued result remains, the structure
//    needed for the new types to work is constructed out of the newly-defined
//    parts, and the original instruction is replaced by this structure
//    before being erased. Instructions requiring this construction include
//    `ret` and `insertvalue`.
//
// # Consequences
//
// This pass does not alter the CFG.
//
// Alias analysis information will become coarser, as the LLVM alias analyzer
// cannot handle the buffer intrinsics. Specifically, while we can determine
// that the following two loads do not alias:
// ```
//   %y = getelementptr i32, ptr addrspace(7) %x, i32 1
//   %a = load i32, ptr addrspace(7) %x
//   %b = load i32, ptr addrspace(7) %y
// ```
// we cannot (except through some code that runs during scheduling) determine
// that the rewritten loads below do not alias.
// ```
//   %y.off = add i32 %x.off, 1
//   %a = call @llvm.amdgcn.raw.ptr.buffer.load(ptr addrspace(8) %x.rsrc, i32
//     %x.off, ...)
//   %b = call @llvm.amdgcn.raw.ptr.buffer.load(ptr addrspace(8)
//     %x.rsrc, i32 %y.off, ...)
// ```
// However, existing alias information is preserved.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "SIDefines.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/InstSimplifyFolder.h"
#include "llvm/Analysis/Utils/Local.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/AttributeMask.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#define DEBUG_TYPE "amdgpu-lower-buffer-fat-pointers"

using namespace llvm;

static constexpr unsigned BufferOffsetWidth = 32;

namespace {
/// Recursively replace instances of ptr addrspace(7) and vector<Nxptr
/// addrspace(7)> with some other type as defined by the relevant subclass.
class BufferFatPtrTypeLoweringBase : public ValueMapTypeRemapper {
  DenseMap<Type *, Type *> Map;

  Type *remapTypeImpl(Type *Ty);

protected:
  virtual Type *remapScalar(PointerType *PT) = 0;
  virtual Type *remapVector(VectorType *VT) = 0;

  const DataLayout &DL;

public:
  BufferFatPtrTypeLoweringBase(const DataLayout &DL) : DL(DL) {}
  Type *remapType(Type *SrcTy) override;
  void clear() { Map.clear(); }
};

/// Remap ptr addrspace(7) to i160 and vector<Nxptr addrspace(7)> to
/// vector<Nxi60> in order to correctly handling loading/storing these values
/// from memory.
class BufferFatPtrToIntTypeMap : public BufferFatPtrTypeLoweringBase {
  using BufferFatPtrTypeLoweringBase::BufferFatPtrTypeLoweringBase;

protected:
  Type *remapScalar(PointerType *PT) override { return DL.getIntPtrType(PT); }
  Type *remapVector(VectorType *VT) override { return DL.getIntPtrType(VT); }
};

/// Remap ptr addrspace(7) to {ptr addrspace(8), i32} (the resource and offset
/// parts of the pointer) so that we can easily rewrite operations on these
/// values that aren't loading them from or storing them to memory.
class BufferFatPtrToStructTypeMap : public BufferFatPtrTypeLoweringBase {
  using BufferFatPtrTypeLoweringBase::BufferFatPtrTypeLoweringBase;

protected:
  Type *remapScalar(PointerType *PT) override;
  Type *remapVector(VectorType *VT) override;
};
} // namespace

// This code is adapted from the type remapper in lib/Linker/IRMover.cpp
Type *BufferFatPtrTypeLoweringBase::remapTypeImpl(Type *Ty) {
  Type **Entry = &Map[Ty];
  if (*Entry)
    return *Entry;
  if (auto *PT = dyn_cast<PointerType>(Ty)) {
    if (PT->getAddressSpace() == AMDGPUAS::BUFFER_FAT_POINTER) {
      return *Entry = remapScalar(PT);
    }
  }
  if (auto *VT = dyn_cast<VectorType>(Ty)) {
    auto *PT = dyn_cast<PointerType>(VT->getElementType());
    if (PT && PT->getAddressSpace() == AMDGPUAS::BUFFER_FAT_POINTER) {
      return *Entry = remapVector(VT);
    }
    return *Entry = Ty;
  }
  // Whether the type is one that is structurally uniqued - that is, if it is
  // not a named struct (the only kind of type where multiple structurally
  // identical types that have a distinct `Type*`)
  StructType *TyAsStruct = dyn_cast<StructType>(Ty);
  bool IsUniqued = !TyAsStruct || TyAsStruct->isLiteral();
  // Base case for ints, floats, opaque pointers, and so on, which don't
  // require recursion.
  if (Ty->getNumContainedTypes() == 0 && IsUniqued)
    return *Entry = Ty;
  bool Changed = false;
  SmallVector<Type *> ElementTypes(Ty->getNumContainedTypes(), nullptr);
  for (unsigned int I = 0, E = Ty->getNumContainedTypes(); I < E; ++I) {
    Type *OldElem = Ty->getContainedType(I);
    Type *NewElem = remapTypeImpl(OldElem);
    ElementTypes[I] = NewElem;
    Changed |= (OldElem != NewElem);
  }
  // Recursive calls to remapTypeImpl() may have invalidated pointer.
  Entry = &Map[Ty];
  if (!Changed) {
    return *Entry = Ty;
  }
  if (auto *ArrTy = dyn_cast<ArrayType>(Ty))
    return *Entry = ArrayType::get(ElementTypes[0], ArrTy->getNumElements());
  if (auto *FnTy = dyn_cast<FunctionType>(Ty))
    return *Entry = FunctionType::get(ElementTypes[0],
                                      ArrayRef(ElementTypes).slice(1),
                                      FnTy->isVarArg());
  if (auto *STy = dyn_cast<StructType>(Ty)) {
    // Genuine opaque types don't have a remapping.
    if (STy->isOpaque())
      return *Entry = Ty;
    bool IsPacked = STy->isPacked();
    if (IsUniqued)
      return *Entry = StructType::get(Ty->getContext(), ElementTypes, IsPacked);
    SmallString<16> Name(STy->getName());
    STy->setName("");
    return *Entry = StructType::create(Ty->getContext(), ElementTypes, Name,
                                       IsPacked);
  }
  llvm_unreachable("Unknown type of type that contains elements");
}

Type *BufferFatPtrTypeLoweringBase::remapType(Type *SrcTy) {
  return remapTypeImpl(SrcTy);
}

Type *BufferFatPtrToStructTypeMap::remapScalar(PointerType *PT) {
  LLVMContext &Ctx = PT->getContext();
  return StructType::get(PointerType::get(Ctx, AMDGPUAS::BUFFER_RESOURCE),
                         IntegerType::get(Ctx, BufferOffsetWidth));
}

Type *BufferFatPtrToStructTypeMap::remapVector(VectorType *VT) {
  ElementCount EC = VT->getElementCount();
  LLVMContext &Ctx = VT->getContext();
  Type *RsrcVec =
      VectorType::get(PointerType::get(Ctx, AMDGPUAS::BUFFER_RESOURCE), EC);
  Type *OffVec = VectorType::get(IntegerType::get(Ctx, BufferOffsetWidth), EC);
  return StructType::get(RsrcVec, OffVec);
}

static bool isBufferFatPtrOrVector(Type *Ty) {
  if (auto *PT = dyn_cast<PointerType>(Ty->getScalarType()))
    return PT->getAddressSpace() == AMDGPUAS::BUFFER_FAT_POINTER;
  return false;
}

// True if the type is {ptr addrspace(8), i32} or a struct containing vectors of
// those types. Used to quickly skip instructions we don't need to process.
static bool isSplitFatPtr(Type *Ty) {
  auto *ST = dyn_cast<StructType>(Ty);
  if (!ST)
    return false;
  if (!ST->isLiteral() || ST->getNumElements() != 2)
    return false;
  auto *MaybeRsrc =
      dyn_cast<PointerType>(ST->getElementType(0)->getScalarType());
  auto *MaybeOff =
      dyn_cast<IntegerType>(ST->getElementType(1)->getScalarType());
  return MaybeRsrc && MaybeOff &&
         MaybeRsrc->getAddressSpace() == AMDGPUAS::BUFFER_RESOURCE &&
         MaybeOff->getBitWidth() == BufferOffsetWidth;
}

// True if the result type or any argument types are buffer fat pointers.
static bool isBufferFatPtrConst(Constant *C) {
  Type *T = C->getType();
  return isBufferFatPtrOrVector(T) || any_of(C->operands(), [](const Use &U) {
           return isBufferFatPtrOrVector(U.get()->getType());
         });
}

namespace {
/// Convert [vectors of] buffer fat pointers to integers when they are read from
/// or stored to memory. This ensures that these pointers will have the same
/// memory layout as before they are lowered, even though they will no longer
/// have their previous layout in registers/in the program (they'll be broken
/// down into resource and offset parts). This has the downside of imposing
/// marshalling costs when reading or storing these values, but since placing
/// such pointers into memory is an uncommon operation at best, we feel that
/// this cost is acceptable for better performance in the common case.
class StoreFatPtrsAsIntsAndExpandMemcpyVisitor
    : public InstVisitor<StoreFatPtrsAsIntsAndExpandMemcpyVisitor, bool> {
  BufferFatPtrToIntTypeMap *TypeMap;

  ValueToValueMapTy ConvertedForStore;

  IRBuilder<InstSimplifyFolder> IRB;

  const TargetMachine *TM;

  // Convert all the buffer fat pointers within the input value to inttegers
  // so that it can be stored in memory.
  Value *fatPtrsToInts(Value *V, Type *From, Type *To, const Twine &Name);
  // Convert all the i160s that need to be buffer fat pointers (as specified)
  // by the To type) into those pointers to preserve the semantics of the rest
  // of the program.
  Value *intsToFatPtrs(Value *V, Type *From, Type *To, const Twine &Name);

public:
  StoreFatPtrsAsIntsAndExpandMemcpyVisitor(BufferFatPtrToIntTypeMap *TypeMap,
                                           const DataLayout &DL,
                                           LLVMContext &Ctx,
                                           const TargetMachine *TM)
      : TypeMap(TypeMap), IRB(Ctx, InstSimplifyFolder(DL)), TM(TM) {}
  bool processFunction(Function &F);

  bool visitInstruction(Instruction &I) { return false; }
  bool visitAllocaInst(AllocaInst &I);
  bool visitLoadInst(LoadInst &LI);
  bool visitStoreInst(StoreInst &SI);
  bool visitGetElementPtrInst(GetElementPtrInst &I);

  bool visitMemCpyInst(MemCpyInst &MCI);
  bool visitMemMoveInst(MemMoveInst &MMI);
  bool visitMemSetInst(MemSetInst &MSI);
  bool visitMemSetPatternInst(MemSetPatternInst &MSPI);
};
} // namespace

Value *StoreFatPtrsAsIntsAndExpandMemcpyVisitor::fatPtrsToInts(
    Value *V, Type *From, Type *To, const Twine &Name) {
  if (From == To)
    return V;
  ValueToValueMapTy::iterator Find = ConvertedForStore.find(V);
  if (Find != ConvertedForStore.end())
    return Find->second;
  if (isBufferFatPtrOrVector(From)) {
    Value *Cast = IRB.CreatePtrToInt(V, To, Name + ".int");
    ConvertedForStore[V] = Cast;
    return Cast;
  }
  if (From->getNumContainedTypes() == 0)
    return V;
  // Structs, arrays, and other compound types.
  Value *Ret = PoisonValue::get(To);
  if (auto *AT = dyn_cast<ArrayType>(From)) {
    Type *FromPart = AT->getArrayElementType();
    Type *ToPart = cast<ArrayType>(To)->getElementType();
    for (uint64_t I = 0, E = AT->getArrayNumElements(); I < E; ++I) {
      Value *Field = IRB.CreateExtractValue(V, I);
      Value *NewField =
          fatPtrsToInts(Field, FromPart, ToPart, Name + "." + Twine(I));
      Ret = IRB.CreateInsertValue(Ret, NewField, I);
    }
  } else {
    for (auto [Idx, FromPart, ToPart] :
         enumerate(From->subtypes(), To->subtypes())) {
      Value *Field = IRB.CreateExtractValue(V, Idx);
      Value *NewField =
          fatPtrsToInts(Field, FromPart, ToPart, Name + "." + Twine(Idx));
      Ret = IRB.CreateInsertValue(Ret, NewField, Idx);
    }
  }
  ConvertedForStore[V] = Ret;
  return Ret;
}

Value *StoreFatPtrsAsIntsAndExpandMemcpyVisitor::intsToFatPtrs(
    Value *V, Type *From, Type *To, const Twine &Name) {
  if (From == To)
    return V;
  if (isBufferFatPtrOrVector(To)) {
    Value *Cast = IRB.CreateIntToPtr(V, To, Name + ".ptr");
    return Cast;
  }
  if (From->getNumContainedTypes() == 0)
    return V;
  // Structs, arrays, and other compound types.
  Value *Ret = PoisonValue::get(To);
  if (auto *AT = dyn_cast<ArrayType>(From)) {
    Type *FromPart = AT->getArrayElementType();
    Type *ToPart = cast<ArrayType>(To)->getElementType();
    for (uint64_t I = 0, E = AT->getArrayNumElements(); I < E; ++I) {
      Value *Field = IRB.CreateExtractValue(V, I);
      Value *NewField =
          intsToFatPtrs(Field, FromPart, ToPart, Name + "." + Twine(I));
      Ret = IRB.CreateInsertValue(Ret, NewField, I);
    }
  } else {
    for (auto [Idx, FromPart, ToPart] :
         enumerate(From->subtypes(), To->subtypes())) {
      Value *Field = IRB.CreateExtractValue(V, Idx);
      Value *NewField =
          intsToFatPtrs(Field, FromPart, ToPart, Name + "." + Twine(Idx));
      Ret = IRB.CreateInsertValue(Ret, NewField, Idx);
    }
  }
  return Ret;
}

bool StoreFatPtrsAsIntsAndExpandMemcpyVisitor::processFunction(Function &F) {
  bool Changed = false;
  // Process memcpy-like instructions after the main iteration because they can
  // invalidate iterators.
  SmallVector<WeakTrackingVH> CanBecomeLoops;
  for (Instruction &I : make_early_inc_range(instructions(F))) {
    if (isa<MemTransferInst, MemSetInst, MemSetPatternInst>(I))
      CanBecomeLoops.push_back(&I);
    else
      Changed |= visit(I);
  }
  for (WeakTrackingVH VH : make_early_inc_range(CanBecomeLoops)) {
    Changed |= visit(cast<Instruction>(VH));
  }
  ConvertedForStore.clear();
  return Changed;
}

bool StoreFatPtrsAsIntsAndExpandMemcpyVisitor::visitAllocaInst(AllocaInst &I) {
  Type *Ty = I.getAllocatedType();
  Type *NewTy = TypeMap->remapType(Ty);
  if (Ty == NewTy)
    return false;
  I.setAllocatedType(NewTy);
  return true;
}

bool StoreFatPtrsAsIntsAndExpandMemcpyVisitor::visitGetElementPtrInst(
    GetElementPtrInst &I) {
  Type *Ty = I.getSourceElementType();
  Type *NewTy = TypeMap->remapType(Ty);
  if (Ty == NewTy)
    return false;
  // We'll be rewriting the type `ptr addrspace(7)` out of existence soon, so
  // make sure GEPs don't have different semantics with the new type.
  I.setSourceElementType(NewTy);
  I.setResultElementType(TypeMap->remapType(I.getResultElementType()));
  return true;
}

bool StoreFatPtrsAsIntsAndExpandMemcpyVisitor::visitLoadInst(LoadInst &LI) {
  Type *Ty = LI.getType();
  Type *IntTy = TypeMap->remapType(Ty);
  if (Ty == IntTy)
    return false;

  IRB.SetInsertPoint(&LI);
  auto *NLI = cast<LoadInst>(LI.clone());
  NLI->mutateType(IntTy);
  NLI = IRB.Insert(NLI);
  NLI->takeName(&LI);

  Value *CastBack = intsToFatPtrs(NLI, IntTy, Ty, NLI->getName());
  LI.replaceAllUsesWith(CastBack);
  LI.eraseFromParent();
  return true;
}

bool StoreFatPtrsAsIntsAndExpandMemcpyVisitor::visitStoreInst(StoreInst &SI) {
  Value *V = SI.getValueOperand();
  Type *Ty = V->getType();
  Type *IntTy = TypeMap->remapType(Ty);
  if (Ty == IntTy)
    return false;

  IRB.SetInsertPoint(&SI);
  Value *IntV = fatPtrsToInts(V, Ty, IntTy, V->getName());
  for (auto *Dbg : at::getAssignmentMarkers(&SI))
    Dbg->setValue(IntV);

  SI.setOperand(0, IntV);
  return true;
}

bool StoreFatPtrsAsIntsAndExpandMemcpyVisitor::visitMemCpyInst(
    MemCpyInst &MCI) {
  // TODO: Allow memcpy.p7.p3 as a synonym for the direct-to-LDS copy, which'll
  // need loop expansion here.
  if (MCI.getSourceAddressSpace() != AMDGPUAS::BUFFER_FAT_POINTER &&
      MCI.getDestAddressSpace() != AMDGPUAS::BUFFER_FAT_POINTER)
    return false;
  llvm::expandMemCpyAsLoop(&MCI,
                           TM->getTargetTransformInfo(*MCI.getFunction()));
  MCI.eraseFromParent();
  return true;
}

bool StoreFatPtrsAsIntsAndExpandMemcpyVisitor::visitMemMoveInst(
    MemMoveInst &MMI) {
  if (MMI.getSourceAddressSpace() != AMDGPUAS::BUFFER_FAT_POINTER &&
      MMI.getDestAddressSpace() != AMDGPUAS::BUFFER_FAT_POINTER)
    return false;
  reportFatalUsageError(
      "memmove() on buffer descriptors is not implemented because pointer "
      "comparison on buffer descriptors isn't implemented\n");
}

bool StoreFatPtrsAsIntsAndExpandMemcpyVisitor::visitMemSetInst(
    MemSetInst &MSI) {
  if (MSI.getDestAddressSpace() != AMDGPUAS::BUFFER_FAT_POINTER)
    return false;
  llvm::expandMemSetAsLoop(&MSI);
  MSI.eraseFromParent();
  return true;
}

bool StoreFatPtrsAsIntsAndExpandMemcpyVisitor::visitMemSetPatternInst(
    MemSetPatternInst &MSPI) {
  if (MSPI.getDestAddressSpace() != AMDGPUAS::BUFFER_FAT_POINTER)
    return false;
  llvm::expandMemSetPatternAsLoop(&MSPI);
  MSPI.eraseFromParent();
  return true;
}

namespace {
/// Convert loads/stores of types that the buffer intrinsics can't handle into
/// one ore more such loads/stores that consist of legal types.
///
/// Do this by
/// 1. Recursing into structs (and arrays that don't share a memory layout with
/// vectors) since the intrinsics can't handle complex types.
/// 2. Converting arrays of non-aggregate, byte-sized types into their
/// corresponding vectors
/// 3. Bitcasting unsupported types, namely overly-long scalars and byte
/// vectors, into vectors of supported types.
/// 4. Splitting up excessively long reads/writes into multiple operations.
///
/// Note that this doesn't handle complex data strucures, but, in the future,
/// the aggregate load splitter from SROA could be refactored to allow for that
/// case.
class LegalizeBufferContentTypesVisitor
    : public InstVisitor<LegalizeBufferContentTypesVisitor, bool> {
  friend class InstVisitor<LegalizeBufferContentTypesVisitor, bool>;

  IRBuilder<InstSimplifyFolder> IRB;

  const DataLayout &DL;

  /// If T is [N x U], where U is a scalar type, return the vector type
  /// <N x U>, otherwise, return T.
  Type *scalarArrayTypeAsVector(Type *MaybeArrayType);
  Value *arrayToVector(Value *V, Type *TargetType, const Twine &Name);
  Value *vectorToArray(Value *V, Type *OrigType, const Twine &Name);

  /// Break up the loads of a struct into the loads of its components

  /// Convert a vector or scalar type that can't be operated on by buffer
  /// intrinsics to one that would be legal through bitcasts and/or truncation.
  /// Uses the wider of i32, i16, or i8 where possible.
  Type *legalNonAggregateFor(Type *T);
  Value *makeLegalNonAggregate(Value *V, Type *TargetType, const Twine &Name);
  Value *makeIllegalNonAggregate(Value *V, Type *OrigType, const Twine &Name);

  struct VecSlice {
    uint64_t Index = 0;
    uint64_t Length = 0;
    VecSlice() = delete;
    // Needed for some Clangs
    VecSlice(uint64_t Index, uint64_t Length) : Index(Index), Length(Length) {}
  };
  /// Return the [index, length] pairs into which `T` needs to be cut to form
  /// legal buffer load or store operations. Clears `Slices`. Creates an empty
  /// `Slices` for non-vector inputs and creates one slice if no slicing will be
  /// needed.
  void getVecSlices(Type *T, SmallVectorImpl<VecSlice> &Slices);

  Value *extractSlice(Value *Vec, VecSlice S, const Twine &Name);
  Value *insertSlice(Value *Whole, Value *Part, VecSlice S, const Twine &Name);

  /// In most cases, return `LegalType`. However, when given an input that would
  /// normally be a legal type for the buffer intrinsics to return but that
  /// isn't hooked up through SelectionDAG, return a type of the same width that
  /// can be used with the relevant intrinsics. Specifically, handle the cases:
  /// - <1 x T> => T for all T
  /// - <N x i8> <=> i16, i32, 2xi32, 4xi32 (as needed)
  /// - <N x T> where T is under 32 bits and the total size is 96 bits <=> <3 x
  /// i32>
  Type *intrinsicTypeFor(Type *LegalType);

  bool visitLoadImpl(LoadInst &OrigLI, Type *PartType,
                     SmallVectorImpl<uint32_t> &AggIdxs, uint64_t AggByteOffset,
                     Value *&Result, const Twine &Name);
  /// Return value is (Changed, ModifiedInPlace)
  std::pair<bool, bool> visitStoreImpl(StoreInst &OrigSI, Type *PartType,
                                       SmallVectorImpl<uint32_t> &AggIdxs,
                                       uint64_t AggByteOffset,
                                       const Twine &Name);

  bool visitInstruction(Instruction &I) { return false; }
  bool visitLoadInst(LoadInst &LI);
  bool visitStoreInst(StoreInst &SI);

public:
  LegalizeBufferContentTypesVisitor(const DataLayout &DL, LLVMContext &Ctx)
      : IRB(Ctx, InstSimplifyFolder(DL)), DL(DL) {}
  bool processFunction(Function &F);
};
} // namespace

Type *LegalizeBufferContentTypesVisitor::scalarArrayTypeAsVector(Type *T) {
  ArrayType *AT = dyn_cast<ArrayType>(T);
  if (!AT)
    return T;
  Type *ET = AT->getElementType();
  if (!ET->isSingleValueType() || isa<VectorType>(ET))
    reportFatalUsageError("loading non-scalar arrays from buffer fat pointers "
                          "should have recursed");
  if (!DL.typeSizeEqualsStoreSize(AT))
    reportFatalUsageError(
        "loading padded arrays from buffer fat pinters should have recursed");
  return FixedVectorType::get(ET, AT->getNumElements());
}

Value *LegalizeBufferContentTypesVisitor::arrayToVector(Value *V,
                                                        Type *TargetType,
                                                        const Twine &Name) {
  Value *VectorRes = PoisonValue::get(TargetType);
  auto *VT = cast<FixedVectorType>(TargetType);
  unsigned EC = VT->getNumElements();
  for (auto I : iota_range<unsigned>(0, EC, /*Inclusive=*/false)) {
    Value *Elem = IRB.CreateExtractValue(V, I, Name + ".elem." + Twine(I));
    VectorRes = IRB.CreateInsertElement(VectorRes, Elem, I,
                                        Name + ".as.vec." + Twine(I));
  }
  return VectorRes;
}

Value *LegalizeBufferContentTypesVisitor::vectorToArray(Value *V,
                                                        Type *OrigType,
                                                        const Twine &Name) {
  Value *ArrayRes = PoisonValue::get(OrigType);
  ArrayType *AT = cast<ArrayType>(OrigType);
  unsigned EC = AT->getNumElements();
  for (auto I : iota_range<unsigned>(0, EC, /*Inclusive=*/false)) {
    Value *Elem = IRB.CreateExtractElement(V, I, Name + ".elem." + Twine(I));
    ArrayRes = IRB.CreateInsertValue(ArrayRes, Elem, I,
                                     Name + ".as.array." + Twine(I));
  }
  return ArrayRes;
}

Type *LegalizeBufferContentTypesVisitor::legalNonAggregateFor(Type *T) {
  TypeSize Size = DL.getTypeStoreSizeInBits(T);
  // Implicitly zero-extend to the next byte if needed
  if (!DL.typeSizeEqualsStoreSize(T))
    T = IRB.getIntNTy(Size.getFixedValue());
  Type *ElemTy = T->getScalarType();
  if (isa<PointerType, ScalableVectorType>(ElemTy)) {
    // Pointers are always big enough, and we'll let scalable vectors through to
    // fail in codegen.
    return T;
  }
  unsigned ElemSize = DL.getTypeSizeInBits(ElemTy).getFixedValue();
  if (isPowerOf2_32(ElemSize) && ElemSize >= 16 && ElemSize <= 128) {
    // [vectors of] anything that's 16/32/64/128 bits can be cast and split into
    // legal buffer operations.
    return T;
  }
  Type *BestVectorElemType = nullptr;
  if (Size.isKnownMultipleOf(32))
    BestVectorElemType = IRB.getInt32Ty();
  else if (Size.isKnownMultipleOf(16))
    BestVectorElemType = IRB.getInt16Ty();
  else
    BestVectorElemType = IRB.getInt8Ty();
  unsigned NumCastElems =
      Size.getFixedValue() / BestVectorElemType->getIntegerBitWidth();
  if (NumCastElems == 1)
    return BestVectorElemType;
  return FixedVectorType::get(BestVectorElemType, NumCastElems);
}

Value *LegalizeBufferContentTypesVisitor::makeLegalNonAggregate(
    Value *V, Type *TargetType, const Twine &Name) {
  Type *SourceType = V->getType();
  TypeSize SourceSize = DL.getTypeSizeInBits(SourceType);
  TypeSize TargetSize = DL.getTypeSizeInBits(TargetType);
  if (SourceSize != TargetSize) {
    Type *ShortScalarTy = IRB.getIntNTy(SourceSize.getFixedValue());
    Type *ByteScalarTy = IRB.getIntNTy(TargetSize.getFixedValue());
    Value *AsScalar = IRB.CreateBitCast(V, ShortScalarTy, Name + ".as.scalar");
    Value *Zext = IRB.CreateZExt(AsScalar, ByteScalarTy, Name + ".zext");
    V = Zext;
    SourceType = ByteScalarTy;
  }
  return IRB.CreateBitCast(V, TargetType, Name + ".legal");
}

Value *LegalizeBufferContentTypesVisitor::makeIllegalNonAggregate(
    Value *V, Type *OrigType, const Twine &Name) {
  Type *LegalType = V->getType();
  TypeSize LegalSize = DL.getTypeSizeInBits(LegalType);
  TypeSize OrigSize = DL.getTypeSizeInBits(OrigType);
  if (LegalSize != OrigSize) {
    Type *ShortScalarTy = IRB.getIntNTy(OrigSize.getFixedValue());
    Type *ByteScalarTy = IRB.getIntNTy(LegalSize.getFixedValue());
    Value *AsScalar = IRB.CreateBitCast(V, ByteScalarTy, Name + ".bytes.cast");
    Value *Trunc = IRB.CreateTrunc(AsScalar, ShortScalarTy, Name + ".trunc");
    return IRB.CreateBitCast(Trunc, OrigType, Name + ".orig");
  }
  return IRB.CreateBitCast(V, OrigType, Name + ".real.ty");
}

Type *LegalizeBufferContentTypesVisitor::intrinsicTypeFor(Type *LegalType) {
  auto *VT = dyn_cast<FixedVectorType>(LegalType);
  if (!VT)
    return LegalType;
  Type *ET = VT->getElementType();
  // Explicitly return the element type of 1-element vectors because the
  // underlying intrinsics don't like <1 x T> even though it's a synonym for T.
  if (VT->getNumElements() == 1)
    return ET;
  if (DL.getTypeSizeInBits(LegalType) == 96 && DL.getTypeSizeInBits(ET) < 32)
    return FixedVectorType::get(IRB.getInt32Ty(), 3);
  if (ET->isIntegerTy(8)) {
    switch (VT->getNumElements()) {
    default:
      return LegalType; // Let it crash later
    case 1:
      return IRB.getInt8Ty();
    case 2:
      return IRB.getInt16Ty();
    case 4:
      return IRB.getInt32Ty();
    case 8:
      return FixedVectorType::get(IRB.getInt32Ty(), 2);
    case 16:
      return FixedVectorType::get(IRB.getInt32Ty(), 4);
    }
  }
  return LegalType;
}

void LegalizeBufferContentTypesVisitor::getVecSlices(
    Type *T, SmallVectorImpl<VecSlice> &Slices) {
  Slices.clear();
  auto *VT = dyn_cast<FixedVectorType>(T);
  if (!VT)
    return;

  uint64_t ElemBitWidth =
      DL.getTypeSizeInBits(VT->getElementType()).getFixedValue();

  uint64_t ElemsPer4Words = 128 / ElemBitWidth;
  uint64_t ElemsPer2Words = ElemsPer4Words / 2;
  uint64_t ElemsPerWord = ElemsPer2Words / 2;
  uint64_t ElemsPerShort = ElemsPerWord / 2;
  uint64_t ElemsPerByte = ElemsPerShort / 2;
  // If the elements evenly pack into 32-bit words, we can use 3-word stores,
  // such as for <6 x bfloat> or <3 x i32>, but we can't dot his for, for
  // example, <3 x i64>, since that's not slicing.
  uint64_t ElemsPer3Words = ElemsPerWord * 3;

  uint64_t TotalElems = VT->getNumElements();
  uint64_t Index = 0;
  auto TrySlice = [&](unsigned MaybeLen) {
    if (MaybeLen > 0 && Index + MaybeLen <= TotalElems) {
      VecSlice Slice{/*Index=*/Index, /*Length=*/MaybeLen};
      Slices.push_back(Slice);
      Index += MaybeLen;
      return true;
    }
    return false;
  };
  while (Index < TotalElems) {
    TrySlice(ElemsPer4Words) || TrySlice(ElemsPer3Words) ||
        TrySlice(ElemsPer2Words) || TrySlice(ElemsPerWord) ||
        TrySlice(ElemsPerShort) || TrySlice(ElemsPerByte);
  }
}

Value *LegalizeBufferContentTypesVisitor::extractSlice(Value *Vec, VecSlice S,
                                                       const Twine &Name) {
  auto *VecVT = dyn_cast<FixedVectorType>(Vec->getType());
  if (!VecVT)
    return Vec;
  if (S.Length == VecVT->getNumElements() && S.Index == 0)
    return Vec;
  if (S.Length == 1)
    return IRB.CreateExtractElement(Vec, S.Index,
                                    Name + ".slice." + Twine(S.Index));
  SmallVector<int> Mask = llvm::to_vector(
      llvm::iota_range<int>(S.Index, S.Index + S.Length, /*Inclusive=*/false));
  return IRB.CreateShuffleVector(Vec, Mask, Name + ".slice." + Twine(S.Index));
}

Value *LegalizeBufferContentTypesVisitor::insertSlice(Value *Whole, Value *Part,
                                                      VecSlice S,
                                                      const Twine &Name) {
  auto *WholeVT = dyn_cast<FixedVectorType>(Whole->getType());
  if (!WholeVT)
    return Part;
  if (S.Length == WholeVT->getNumElements() && S.Index == 0)
    return Part;
  if (S.Length == 1) {
    return IRB.CreateInsertElement(Whole, Part, S.Index,
                                   Name + ".slice." + Twine(S.Index));
  }
  int NumElems = cast<FixedVectorType>(Whole->getType())->getNumElements();

  // Extend the slice with poisons to make the main shufflevector happy.
  SmallVector<int> ExtPartMask(NumElems, -1);
  for (auto [I, E] : llvm::enumerate(
           MutableArrayRef<int>(ExtPartMask).take_front(S.Length))) {
    E = I;
  }
  Value *ExtPart = IRB.CreateShuffleVector(Part, ExtPartMask,
                                           Name + ".ext." + Twine(S.Index));

  SmallVector<int> Mask =
      llvm::to_vector(llvm::iota_range<int>(0, NumElems, /*Inclusive=*/false));
  for (auto [I, E] :
       llvm::enumerate(MutableArrayRef<int>(Mask).slice(S.Index, S.Length)))
    E = I + NumElems;
  return IRB.CreateShuffleVector(Whole, ExtPart, Mask,
                                 Name + ".parts." + Twine(S.Index));
}

bool LegalizeBufferContentTypesVisitor::visitLoadImpl(
    LoadInst &OrigLI, Type *PartType, SmallVectorImpl<uint32_t> &AggIdxs,
    uint64_t AggByteOff, Value *&Result, const Twine &Name) {
  if (auto *ST = dyn_cast<StructType>(PartType)) {
    const StructLayout *Layout = DL.getStructLayout(ST);
    bool Changed = false;
    for (auto [I, ElemTy, Offset] :
         llvm::enumerate(ST->elements(), Layout->getMemberOffsets())) {
      AggIdxs.push_back(I);
      Changed |= visitLoadImpl(OrigLI, ElemTy, AggIdxs,
                               AggByteOff + Offset.getFixedValue(), Result,
                               Name + "." + Twine(I));
      AggIdxs.pop_back();
    }
    return Changed;
  }
  if (auto *AT = dyn_cast<ArrayType>(PartType)) {
    Type *ElemTy = AT->getElementType();
    if (!ElemTy->isSingleValueType() || !DL.typeSizeEqualsStoreSize(ElemTy) ||
        ElemTy->isVectorTy()) {
      TypeSize ElemStoreSize = DL.getTypeStoreSize(ElemTy);
      bool Changed = false;
      for (auto I : llvm::iota_range<uint32_t>(0, AT->getNumElements(),
                                               /*Inclusive=*/false)) {
        AggIdxs.push_back(I);
        Changed |= visitLoadImpl(OrigLI, ElemTy, AggIdxs,
                                 AggByteOff + I * ElemStoreSize.getFixedValue(),
                                 Result, Name + Twine(I));
        AggIdxs.pop_back();
      }
      return Changed;
    }
  }

  // Typical case

  Type *ArrayAsVecType = scalarArrayTypeAsVector(PartType);
  Type *LegalType = legalNonAggregateFor(ArrayAsVecType);

  SmallVector<VecSlice> Slices;
  getVecSlices(LegalType, Slices);
  bool HasSlices = Slices.size() > 1;
  bool IsAggPart = !AggIdxs.empty();
  Value *LoadsRes;
  if (!HasSlices && !IsAggPart) {
    Type *LoadableType = intrinsicTypeFor(LegalType);
    if (LoadableType == PartType)
      return false;

    IRB.SetInsertPoint(&OrigLI);
    auto *NLI = cast<LoadInst>(OrigLI.clone());
    NLI->mutateType(LoadableType);
    NLI = IRB.Insert(NLI);
    NLI->setName(Name + ".loadable");

    LoadsRes = IRB.CreateBitCast(NLI, LegalType, Name + ".from.loadable");
  } else {
    IRB.SetInsertPoint(&OrigLI);
    LoadsRes = PoisonValue::get(LegalType);
    Value *OrigPtr = OrigLI.getPointerOperand();
    // If we're needing to spill something into more than one load, its legal
    // type will be a vector (ex. an i256 load will have LegalType = <8 x i32>).
    // But if we're already a scalar (which can happen if we're splitting up a
    // struct), the element type will be the legal type itself.
    Type *ElemType = LegalType->getScalarType();
    unsigned ElemBytes = DL.getTypeStoreSize(ElemType);
    AAMDNodes AANodes = OrigLI.getAAMetadata();
    if (IsAggPart && Slices.empty())
      Slices.push_back(VecSlice{/*Index=*/0, /*Length=*/1});
    for (VecSlice S : Slices) {
      Type *SliceType =
          S.Length != 1 ? FixedVectorType::get(ElemType, S.Length) : ElemType;
      int64_t ByteOffset = AggByteOff + S.Index * ElemBytes;
      // You can't reasonably expect loads to wrap around the edge of memory.
      Value *NewPtr = IRB.CreateGEP(
          IRB.getInt8Ty(), OrigLI.getPointerOperand(), IRB.getInt32(ByteOffset),
          OrigPtr->getName() + ".off.ptr." + Twine(ByteOffset),
          GEPNoWrapFlags::noUnsignedWrap());
      Type *LoadableType = intrinsicTypeFor(SliceType);
      LoadInst *NewLI = IRB.CreateAlignedLoad(
          LoadableType, NewPtr, commonAlignment(OrigLI.getAlign(), ByteOffset),
          Name + ".off." + Twine(ByteOffset));
      copyMetadataForLoad(*NewLI, OrigLI);
      NewLI->setAAMetadata(
          AANodes.adjustForAccess(ByteOffset, LoadableType, DL));
      NewLI->setAtomic(OrigLI.getOrdering(), OrigLI.getSyncScopeID());
      NewLI->setVolatile(OrigLI.isVolatile());
      Value *Loaded = IRB.CreateBitCast(NewLI, SliceType,
                                        NewLI->getName() + ".from.loadable");
      LoadsRes = insertSlice(LoadsRes, Loaded, S, Name);
    }
  }
  if (LegalType != ArrayAsVecType)
    LoadsRes = makeIllegalNonAggregate(LoadsRes, ArrayAsVecType, Name);
  if (ArrayAsVecType != PartType)
    LoadsRes = vectorToArray(LoadsRes, PartType, Name);

  if (IsAggPart)
    Result = IRB.CreateInsertValue(Result, LoadsRes, AggIdxs, Name);
  else
    Result = LoadsRes;
  return true;
}

bool LegalizeBufferContentTypesVisitor::visitLoadInst(LoadInst &LI) {
  if (LI.getPointerAddressSpace() != AMDGPUAS::BUFFER_FAT_POINTER)
    return false;

  SmallVector<uint32_t> AggIdxs;
  Type *OrigType = LI.getType();
  Value *Result = PoisonValue::get(OrigType);
  bool Changed = visitLoadImpl(LI, OrigType, AggIdxs, 0, Result, LI.getName());
  if (!Changed)
    return false;
  Result->takeName(&LI);
  LI.replaceAllUsesWith(Result);
  LI.eraseFromParent();
  return Changed;
}

std::pair<bool, bool> LegalizeBufferContentTypesVisitor::visitStoreImpl(
    StoreInst &OrigSI, Type *PartType, SmallVectorImpl<uint32_t> &AggIdxs,
    uint64_t AggByteOff, const Twine &Name) {
  if (auto *ST = dyn_cast<StructType>(PartType)) {
    const StructLayout *Layout = DL.getStructLayout(ST);
    bool Changed = false;
    for (auto [I, ElemTy, Offset] :
         llvm::enumerate(ST->elements(), Layout->getMemberOffsets())) {
      AggIdxs.push_back(I);
      Changed |= std::get<0>(visitStoreImpl(OrigSI, ElemTy, AggIdxs,
                                            AggByteOff + Offset.getFixedValue(),
                                            Name + "." + Twine(I)));
      AggIdxs.pop_back();
    }
    return std::make_pair(Changed, /*ModifiedInPlace=*/false);
  }
  if (auto *AT = dyn_cast<ArrayType>(PartType)) {
    Type *ElemTy = AT->getElementType();
    if (!ElemTy->isSingleValueType() || !DL.typeSizeEqualsStoreSize(ElemTy) ||
        ElemTy->isVectorTy()) {
      TypeSize ElemStoreSize = DL.getTypeStoreSize(ElemTy);
      bool Changed = false;
      for (auto I : llvm::iota_range<uint32_t>(0, AT->getNumElements(),
                                               /*Inclusive=*/false)) {
        AggIdxs.push_back(I);
        Changed |= std::get<0>(visitStoreImpl(
            OrigSI, ElemTy, AggIdxs,
            AggByteOff + I * ElemStoreSize.getFixedValue(), Name + Twine(I)));
        AggIdxs.pop_back();
      }
      return std::make_pair(Changed, /*ModifiedInPlace=*/false);
    }
  }

  Value *OrigData = OrigSI.getValueOperand();
  Value *NewData = OrigData;

  bool IsAggPart = !AggIdxs.empty();
  if (IsAggPart)
    NewData = IRB.CreateExtractValue(NewData, AggIdxs, Name);

  Type *ArrayAsVecType = scalarArrayTypeAsVector(PartType);
  if (ArrayAsVecType != PartType) {
    NewData = arrayToVector(NewData, ArrayAsVecType, Name);
  }

  Type *LegalType = legalNonAggregateFor(ArrayAsVecType);
  if (LegalType != ArrayAsVecType) {
    NewData = makeLegalNonAggregate(NewData, LegalType, Name);
  }

  SmallVector<VecSlice> Slices;
  getVecSlices(LegalType, Slices);
  bool NeedToSplit = Slices.size() > 1 || IsAggPart;
  if (!NeedToSplit) {
    Type *StorableType = intrinsicTypeFor(LegalType);
    if (StorableType == PartType)
      return std::make_pair(/*Changed=*/false, /*ModifiedInPlace=*/false);
    NewData = IRB.CreateBitCast(NewData, StorableType, Name + ".storable");
    OrigSI.setOperand(0, NewData);
    return std::make_pair(/*Changed=*/true, /*ModifiedInPlace=*/true);
  }

  Value *OrigPtr = OrigSI.getPointerOperand();
  Type *ElemType = LegalType->getScalarType();
  if (IsAggPart && Slices.empty())
    Slices.push_back(VecSlice{/*Index=*/0, /*Length=*/1});
  unsigned ElemBytes = DL.getTypeStoreSize(ElemType);
  AAMDNodes AANodes = OrigSI.getAAMetadata();
  for (VecSlice S : Slices) {
    Type *SliceType =
        S.Length != 1 ? FixedVectorType::get(ElemType, S.Length) : ElemType;
    int64_t ByteOffset = AggByteOff + S.Index * ElemBytes;
    Value *NewPtr =
        IRB.CreateGEP(IRB.getInt8Ty(), OrigPtr, IRB.getInt32(ByteOffset),
                      OrigPtr->getName() + ".part." + Twine(S.Index),
                      GEPNoWrapFlags::noUnsignedWrap());
    Value *DataSlice = extractSlice(NewData, S, Name);
    Type *StorableType = intrinsicTypeFor(SliceType);
    DataSlice = IRB.CreateBitCast(DataSlice, StorableType,
                                  DataSlice->getName() + ".storable");
    auto *NewSI = cast<StoreInst>(OrigSI.clone());
    NewSI->setAlignment(commonAlignment(OrigSI.getAlign(), ByteOffset));
    IRB.Insert(NewSI);
    NewSI->setOperand(0, DataSlice);
    NewSI->setOperand(1, NewPtr);
    NewSI->setAAMetadata(AANodes.adjustForAccess(ByteOffset, StorableType, DL));
  }
  return std::make_pair(/*Changed=*/true, /*ModifiedInPlace=*/false);
}

bool LegalizeBufferContentTypesVisitor::visitStoreInst(StoreInst &SI) {
  if (SI.getPointerAddressSpace() != AMDGPUAS::BUFFER_FAT_POINTER)
    return false;
  IRB.SetInsertPoint(&SI);
  SmallVector<uint32_t> AggIdxs;
  Value *OrigData = SI.getValueOperand();
  auto [Changed, ModifiedInPlace] =
      visitStoreImpl(SI, OrigData->getType(), AggIdxs, 0, OrigData->getName());
  if (Changed && !ModifiedInPlace)
    SI.eraseFromParent();
  return Changed;
}

bool LegalizeBufferContentTypesVisitor::processFunction(Function &F) {
  bool Changed = false;
  // Note, memory transfer intrinsics won't
  for (Instruction &I : make_early_inc_range(instructions(F))) {
    Changed |= visit(I);
  }
  return Changed;
}

/// Return the ptr addrspace(8) and i32 (resource and offset parts) in a lowered
/// buffer fat pointer constant.
static std::pair<Constant *, Constant *>
splitLoweredFatBufferConst(Constant *C) {
  assert(isSplitFatPtr(C->getType()) && "Not a split fat buffer pointer");
  return std::make_pair(C->getAggregateElement(0u), C->getAggregateElement(1u));
}

namespace {
/// Handle the remapping of ptr addrspace(7) constants.
class FatPtrConstMaterializer final : public ValueMaterializer {
  BufferFatPtrToStructTypeMap *TypeMap;
  // An internal mapper that is used to recurse into the arguments of constants.
  // While the documentation for `ValueMapper` specifies not to use it
  // recursively, examination of the logic in mapValue() shows that it can
  // safely be used recursively when handling constants, like it does in its own
  // logic.
  ValueMapper InternalMapper;

  Constant *materializeBufferFatPtrConst(Constant *C);

public:
  // UnderlyingMap is the value map this materializer will be filling.
  FatPtrConstMaterializer(BufferFatPtrToStructTypeMap *TypeMap,
                          ValueToValueMapTy &UnderlyingMap)
      : TypeMap(TypeMap),
        InternalMapper(UnderlyingMap, RF_None, TypeMap, this) {}
  ~FatPtrConstMaterializer() = default;

  Value *materialize(Value *V) override;
};
} // namespace

Constant *FatPtrConstMaterializer::materializeBufferFatPtrConst(Constant *C) {
  Type *SrcTy = C->getType();
  auto *NewTy = dyn_cast<StructType>(TypeMap->remapType(SrcTy));
  if (C->isNullValue())
    return ConstantAggregateZero::getNullValue(NewTy);
  if (isa<PoisonValue>(C)) {
    return ConstantStruct::get(NewTy,
                               {PoisonValue::get(NewTy->getElementType(0)),
                                PoisonValue::get(NewTy->getElementType(1))});
  }
  if (isa<UndefValue>(C)) {
    return ConstantStruct::get(NewTy,
                               {UndefValue::get(NewTy->getElementType(0)),
                                UndefValue::get(NewTy->getElementType(1))});
  }

  if (auto *VC = dyn_cast<ConstantVector>(C)) {
    if (Constant *S = VC->getSplatValue()) {
      Constant *NewS = InternalMapper.mapConstant(*S);
      if (!NewS)
        return nullptr;
      auto [Rsrc, Off] = splitLoweredFatBufferConst(NewS);
      auto EC = VC->getType()->getElementCount();
      return ConstantStruct::get(NewTy, {ConstantVector::getSplat(EC, Rsrc),
                                         ConstantVector::getSplat(EC, Off)});
    }
    SmallVector<Constant *> Rsrcs;
    SmallVector<Constant *> Offs;
    for (Value *Op : VC->operand_values()) {
      auto *NewOp = dyn_cast_or_null<Constant>(InternalMapper.mapValue(*Op));
      if (!NewOp)
        return nullptr;
      auto [Rsrc, Off] = splitLoweredFatBufferConst(NewOp);
      Rsrcs.push_back(Rsrc);
      Offs.push_back(Off);
    }
    Constant *RsrcVec = ConstantVector::get(Rsrcs);
    Constant *OffVec = ConstantVector::get(Offs);
    return ConstantStruct::get(NewTy, {RsrcVec, OffVec});
  }

  if (isa<GlobalValue>(C))
    reportFatalUsageError("global values containing ptr addrspace(7) (buffer "
                          "fat pointer) values are not supported");

  if (isa<ConstantExpr>(C))
    reportFatalUsageError(
        "constant exprs containing ptr addrspace(7) (buffer "
        "fat pointer) values should have been expanded earlier");

  return nullptr;
}

Value *FatPtrConstMaterializer::materialize(Value *V) {
  Constant *C = dyn_cast<Constant>(V);
  if (!C)
    return nullptr;
  // Structs and other types that happen to contain fat pointers get remapped
  // by the mapValue() logic.
  if (!isBufferFatPtrConst(C))
    return nullptr;
  return materializeBufferFatPtrConst(C);
}

using PtrParts = std::pair<Value *, Value *>;
namespace {
// The visitor returns the resource and offset parts for an instruction if they
// can be computed, or (nullptr, nullptr) for cases that don't have a meaningful
// value mapping.
class SplitPtrStructs : public InstVisitor<SplitPtrStructs, PtrParts> {
  ValueToValueMapTy RsrcParts;
  ValueToValueMapTy OffParts;

  // Track instructions that have been rewritten into a user of the component
  // parts of their ptr addrspace(7) input. Instructions that produced
  // ptr addrspace(7) parts should **not** be RAUW'd before being added to this
  // set, as that replacement will be handled in a post-visit step. However,
  // instructions that yield values that aren't fat pointers (ex. ptrtoint)
  // should RAUW themselves with new instructions that use the split parts
  // of their arguments during processing.
  DenseSet<Instruction *> SplitUsers;

  // Nodes that need a second look once we've computed the parts for all other
  // instructions to see if, for example, we really need to phi on the resource
  // part.
  SmallVector<Instruction *> Conditionals;
  // Temporary instructions produced while lowering conditionals that should be
  // killed.
  SmallVector<Instruction *> ConditionalTemps;

  // Subtarget info, needed for determining what cache control bits to set.
  const TargetMachine *TM;
  const GCNSubtarget *ST = nullptr;

  IRBuilder<InstSimplifyFolder> IRB;

  // Copy metadata between instructions if applicable.
  void copyMetadata(Value *Dest, Value *Src);

  // Get the resource and offset parts of the value V, inserting appropriate
  // extractvalue calls if needed.
  PtrParts getPtrParts(Value *V);

  // Given an instruction that could produce multiple resource parts (a PHI or
  // select), collect the set of possible instructions that could have provided
  // its resource parts  that it could have (the `Roots`) and the set of
  // conditional instructions visited during the search (`Seen`). If, after
  // removing the root of the search from `Seen` and `Roots`, `Seen` is a subset
  // of `Roots` and `Roots - Seen` contains one element, the resource part of
  // that element can replace the resource part of all other elements in `Seen`.
  void getPossibleRsrcRoots(Instruction *I, SmallPtrSetImpl<Value *> &Roots,
                            SmallPtrSetImpl<Value *> &Seen);
  void processConditionals();

  // If an instruction hav been split into resource and offset parts,
  // delete that instruction. If any of its uses have not themselves been split
  // into parts (for example, an insertvalue), construct the structure
  // that the type rewrites declared should be produced by the dying instruction
  // and use that.
  // Also, kill the temporary extractvalue operations produced by the two-stage
  // lowering of PHIs and conditionals.
  void killAndReplaceSplitInstructions(SmallVectorImpl<Instruction *> &Origs);

  void setAlign(CallInst *Intr, Align A, unsigned RsrcArgIdx);
  void insertPreMemOpFence(AtomicOrdering Order, SyncScope::ID SSID);
  void insertPostMemOpFence(AtomicOrdering Order, SyncScope::ID SSID);
  Value *handleMemoryInst(Instruction *I, Value *Arg, Value *Ptr, Type *Ty,
                          Align Alignment, AtomicOrdering Order,
                          bool IsVolatile, SyncScope::ID SSID);

public:
  SplitPtrStructs(const DataLayout &DL, LLVMContext &Ctx,
                  const TargetMachine *TM)
      : TM(TM), IRB(Ctx, InstSimplifyFolder(DL)) {}

  void processFunction(Function &F);

  PtrParts visitInstruction(Instruction &I);
  PtrParts visitLoadInst(LoadInst &LI);
  PtrParts visitStoreInst(StoreInst &SI);
  PtrParts visitAtomicRMWInst(AtomicRMWInst &AI);
  PtrParts visitAtomicCmpXchgInst(AtomicCmpXchgInst &AI);
  PtrParts visitGetElementPtrInst(GetElementPtrInst &GEP);

  PtrParts visitPtrToAddrInst(PtrToAddrInst &PA);
  PtrParts visitPtrToIntInst(PtrToIntInst &PI);
  PtrParts visitIntToPtrInst(IntToPtrInst &IP);
  PtrParts visitAddrSpaceCastInst(AddrSpaceCastInst &I);
  PtrParts visitICmpInst(ICmpInst &Cmp);
  PtrParts visitFreezeInst(FreezeInst &I);

  PtrParts visitExtractElementInst(ExtractElementInst &I);
  PtrParts visitInsertElementInst(InsertElementInst &I);
  PtrParts visitShuffleVectorInst(ShuffleVectorInst &I);

  PtrParts visitPHINode(PHINode &PHI);
  PtrParts visitSelectInst(SelectInst &SI);

  PtrParts visitIntrinsicInst(IntrinsicInst &II);
};
} // namespace

void SplitPtrStructs::copyMetadata(Value *Dest, Value *Src) {
  auto *DestI = dyn_cast<Instruction>(Dest);
  auto *SrcI = dyn_cast<Instruction>(Src);

  if (!DestI || !SrcI)
    return;

  DestI->copyMetadata(*SrcI);
}

PtrParts SplitPtrStructs::getPtrParts(Value *V) {
  assert(isSplitFatPtr(V->getType()) && "it's not meaningful to get the parts "
                                        "of something that wasn't rewritten");
  auto *RsrcEntry = &RsrcParts[V];
  auto *OffEntry = &OffParts[V];
  if (*RsrcEntry && *OffEntry)
    return {*RsrcEntry, *OffEntry};

  if (auto *C = dyn_cast<Constant>(V)) {
    auto [Rsrc, Off] = splitLoweredFatBufferConst(C);
    return {*RsrcEntry = Rsrc, *OffEntry = Off};
  }

  IRBuilder<InstSimplifyFolder>::InsertPointGuard Guard(IRB);
  if (auto *I = dyn_cast<Instruction>(V)) {
    LLVM_DEBUG(dbgs() << "Recursing to split parts of " << *I << "\n");
    auto [Rsrc, Off] = visit(*I);
    if (Rsrc && Off)
      return {*RsrcEntry = Rsrc, *OffEntry = Off};
    // We'll be creating the new values after the relevant instruction.
    // This instruction generates a value and so isn't a terminator.
    IRB.SetInsertPoint(*I->getInsertionPointAfterDef());
    IRB.SetCurrentDebugLocation(I->getDebugLoc());
  } else if (auto *A = dyn_cast<Argument>(V)) {
    IRB.SetInsertPointPastAllocas(A->getParent());
    IRB.SetCurrentDebugLocation(DebugLoc());
  }
  Value *Rsrc = IRB.CreateExtractValue(V, 0, V->getName() + ".rsrc");
  Value *Off = IRB.CreateExtractValue(V, 1, V->getName() + ".off");
  return {*RsrcEntry = Rsrc, *OffEntry = Off};
}

/// Returns the instruction that defines the resource part of the value V.
/// Note that this is not getUnderlyingObject(), since that looks through
/// operations like ptrmask which might modify the resource part.
///
/// We can limit ourselves to just looking through GEPs followed by looking
/// through addrspacecasts because only those two operations preserve the
/// resource part, and because operations on an `addrspace(8)` (which is the
/// legal input to this addrspacecast) would produce a different resource part.
static Value *rsrcPartRoot(Value *V) {
  while (auto *GEP = dyn_cast<GEPOperator>(V))
    V = GEP->getPointerOperand();
  while (auto *ASC = dyn_cast<AddrSpaceCastOperator>(V))
    V = ASC->getPointerOperand();
  return V;
}

void SplitPtrStructs::getPossibleRsrcRoots(Instruction *I,
                                           SmallPtrSetImpl<Value *> &Roots,
                                           SmallPtrSetImpl<Value *> &Seen) {
  if (auto *PHI = dyn_cast<PHINode>(I)) {
    if (!Seen.insert(I).second)
      return;
    for (Value *In : PHI->incoming_values()) {
      In = rsrcPartRoot(In);
      Roots.insert(In);
      if (isa<PHINode, SelectInst>(In))
        getPossibleRsrcRoots(cast<Instruction>(In), Roots, Seen);
    }
  } else if (auto *SI = dyn_cast<SelectInst>(I)) {
    if (!Seen.insert(SI).second)
      return;
    Value *TrueVal = rsrcPartRoot(SI->getTrueValue());
    Value *FalseVal = rsrcPartRoot(SI->getFalseValue());
    Roots.insert(TrueVal);
    Roots.insert(FalseVal);
    if (isa<PHINode, SelectInst>(TrueVal))
      getPossibleRsrcRoots(cast<Instruction>(TrueVal), Roots, Seen);
    if (isa<PHINode, SelectInst>(FalseVal))
      getPossibleRsrcRoots(cast<Instruction>(FalseVal), Roots, Seen);
  } else {
    llvm_unreachable("getPossibleRsrcParts() only works on phi and select");
  }
}

void SplitPtrStructs::processConditionals() {
  SmallDenseMap<Value *, Value *> FoundRsrcs;
  SmallPtrSet<Value *, 4> Roots;
  SmallPtrSet<Value *, 4> Seen;
  for (Instruction *I : Conditionals) {
    // These have to exist by now because we've visited these nodes.
    Value *Rsrc = RsrcParts[I];
    Value *Off = OffParts[I];
    assert(Rsrc && Off && "must have visited conditionals by now");

    std::optional<Value *> MaybeRsrc;
    auto MaybeFoundRsrc = FoundRsrcs.find(I);
    if (MaybeFoundRsrc != FoundRsrcs.end()) {
      MaybeRsrc = MaybeFoundRsrc->second;
    } else {
      IRBuilder<InstSimplifyFolder>::InsertPointGuard Guard(IRB);
      Roots.clear();
      Seen.clear();
      getPossibleRsrcRoots(I, Roots, Seen);
      LLVM_DEBUG(dbgs() << "Processing conditional: " << *I << "\n");
#ifndef NDEBUG
      for (Value *V : Roots)
        LLVM_DEBUG(dbgs() << "Root: " << *V << "\n");
      for (Value *V : Seen)
        LLVM_DEBUG(dbgs() << "Seen: " << *V << "\n");
#endif
      // If we are our own possible root, then we shouldn't block our
      // replacement with a valid incoming value.
      Roots.erase(I);
      // We don't want to block the optimization for conditionals that don't
      // refer to themselves but did see themselves during the traversal.
      Seen.erase(I);

      if (set_is_subset(Seen, Roots)) {
        auto Diff = set_difference(Roots, Seen);
        if (Diff.size() == 1) {
          Value *RootVal = *Diff.begin();
          // Handle the case where previous loops already looked through
          // an addrspacecast.
          if (isSplitFatPtr(RootVal->getType()))
            MaybeRsrc = std::get<0>(getPtrParts(RootVal));
          else
            MaybeRsrc = RootVal;
        }
      }
    }

    if (auto *PHI = dyn_cast<PHINode>(I)) {
      Value *NewRsrc;
      StructType *PHITy = cast<StructType>(PHI->getType());
      IRB.SetInsertPoint(*PHI->getInsertionPointAfterDef());
      IRB.SetCurrentDebugLocation(PHI->getDebugLoc());
      if (MaybeRsrc) {
        NewRsrc = *MaybeRsrc;
      } else {
        Type *RsrcTy = PHITy->getElementType(0);
        auto *RsrcPHI = IRB.CreatePHI(RsrcTy, PHI->getNumIncomingValues());
        RsrcPHI->takeName(Rsrc);
        for (auto [V, BB] : llvm::zip(PHI->incoming_values(), PHI->blocks())) {
          Value *VRsrc = std::get<0>(getPtrParts(V));
          RsrcPHI->addIncoming(VRsrc, BB);
        }
        copyMetadata(RsrcPHI, PHI);
        NewRsrc = RsrcPHI;
      }

      Type *OffTy = PHITy->getElementType(1);
      auto *NewOff = IRB.CreatePHI(OffTy, PHI->getNumIncomingValues());
      NewOff->takeName(Off);
      for (auto [V, BB] : llvm::zip(PHI->incoming_values(), PHI->blocks())) {
        assert(OffParts.count(V) && "An offset part had to be created by now");
        Value *VOff = std::get<1>(getPtrParts(V));
        NewOff->addIncoming(VOff, BB);
      }
      copyMetadata(NewOff, PHI);

      // Note: We don't eraseFromParent() the temporaries because we don't want
      // to put the corrections maps in an inconstent state. That'll be handed
      // during the rest of the killing. Also, `ValueToValueMapTy` guarantees
      // that references in that map will be updated as well.
      // Note that if the temporary instruction got `InstSimplify`'d away, it
      // might be something like a block argument.
      if (auto *RsrcInst = dyn_cast<Instruction>(Rsrc)) {
        ConditionalTemps.push_back(RsrcInst);
        RsrcInst->replaceAllUsesWith(NewRsrc);
      }
      if (auto *OffInst = dyn_cast<Instruction>(Off)) {
        ConditionalTemps.push_back(OffInst);
        OffInst->replaceAllUsesWith(NewOff);
      }

      // Save on recomputing the cycle traversals in known-root cases.
      if (MaybeRsrc)
        for (Value *V : Seen)
          FoundRsrcs[V] = NewRsrc;
    } else if (isa<SelectInst>(I)) {
      if (MaybeRsrc) {
        if (auto *RsrcInst = dyn_cast<Instruction>(Rsrc)) {
          ConditionalTemps.push_back(RsrcInst);
          RsrcInst->replaceAllUsesWith(*MaybeRsrc);
        }
        for (Value *V : Seen)
          FoundRsrcs[V] = *MaybeRsrc;
      }
    } else {
      llvm_unreachable("Only PHIs and selects go in the conditionals list");
    }
  }
}

void SplitPtrStructs::killAndReplaceSplitInstructions(
    SmallVectorImpl<Instruction *> &Origs) {
  for (Instruction *I : ConditionalTemps)
    I->eraseFromParent();

  for (Instruction *I : Origs) {
    if (!SplitUsers.contains(I))
      continue;

    SmallVector<DbgVariableRecord *> Dbgs;
    findDbgValues(I, Dbgs);
    for (DbgVariableRecord *Dbg : Dbgs) {
      auto &DL = I->getDataLayout();
      assert(isSplitFatPtr(I->getType()) &&
             "We should've RAUW'd away loads, stores, etc. at this point");
      DbgVariableRecord *OffDbg = Dbg->clone();
      auto [Rsrc, Off] = getPtrParts(I);

      int64_t RsrcSz = DL.getTypeSizeInBits(Rsrc->getType());
      int64_t OffSz = DL.getTypeSizeInBits(Off->getType());

      std::optional<DIExpression *> RsrcExpr =
          DIExpression::createFragmentExpression(Dbg->getExpression(), 0,
                                                 RsrcSz);
      std::optional<DIExpression *> OffExpr =
          DIExpression::createFragmentExpression(Dbg->getExpression(), RsrcSz,
                                                 OffSz);
      if (OffExpr) {
        OffDbg->setExpression(*OffExpr);
        OffDbg->replaceVariableLocationOp(I, Off);
        OffDbg->insertBefore(Dbg);
      } else {
        OffDbg->eraseFromParent();
      }
      if (RsrcExpr) {
        Dbg->setExpression(*RsrcExpr);
        Dbg->replaceVariableLocationOp(I, Rsrc);
      } else {
        Dbg->replaceVariableLocationOp(I, PoisonValue::get(I->getType()));
      }
    }

    Value *Poison = PoisonValue::get(I->getType());
    I->replaceUsesWithIf(Poison, [&](const Use &U) -> bool {
      if (const auto *UI = dyn_cast<Instruction>(U.getUser()))
        return SplitUsers.contains(UI);
      return false;
    });

    if (I->use_empty()) {
      I->eraseFromParent();
      continue;
    }
    IRB.SetInsertPoint(*I->getInsertionPointAfterDef());
    IRB.SetCurrentDebugLocation(I->getDebugLoc());
    auto [Rsrc, Off] = getPtrParts(I);
    Value *Struct = PoisonValue::get(I->getType());
    Struct = IRB.CreateInsertValue(Struct, Rsrc, 0);
    Struct = IRB.CreateInsertValue(Struct, Off, 1);
    copyMetadata(Struct, I);
    Struct->takeName(I);
    I->replaceAllUsesWith(Struct);
    I->eraseFromParent();
  }
}

void SplitPtrStructs::setAlign(CallInst *Intr, Align A, unsigned RsrcArgIdx) {
  LLVMContext &Ctx = Intr->getContext();
  Intr->addParamAttr(RsrcArgIdx, Attribute::getWithAlignment(Ctx, A));
}

void SplitPtrStructs::insertPreMemOpFence(AtomicOrdering Order,
                                          SyncScope::ID SSID) {
  switch (Order) {
  case AtomicOrdering::Release:
  case AtomicOrdering::AcquireRelease:
  case AtomicOrdering::SequentiallyConsistent:
    IRB.CreateFence(AtomicOrdering::Release, SSID);
    break;
  default:
    break;
  }
}

void SplitPtrStructs::insertPostMemOpFence(AtomicOrdering Order,
                                           SyncScope::ID SSID) {
  switch (Order) {
  case AtomicOrdering::Acquire:
  case AtomicOrdering::AcquireRelease:
  case AtomicOrdering::SequentiallyConsistent:
    IRB.CreateFence(AtomicOrdering::Acquire, SSID);
    break;
  default:
    break;
  }
}

Value *SplitPtrStructs::handleMemoryInst(Instruction *I, Value *Arg, Value *Ptr,
                                         Type *Ty, Align Alignment,
                                         AtomicOrdering Order, bool IsVolatile,
                                         SyncScope::ID SSID) {
  IRB.SetInsertPoint(I);

  auto [Rsrc, Off] = getPtrParts(Ptr);
  SmallVector<Value *, 5> Args;
  if (Arg)
    Args.push_back(Arg);
  Args.push_back(Rsrc);
  Args.push_back(Off);
  insertPreMemOpFence(Order, SSID);
  // soffset is always 0 for these cases, where we always want any offset to be
  // part of bounds checking and we don't know which parts of the GEPs is
  // uniform.
  Args.push_back(IRB.getInt32(0));

  uint32_t Aux = 0;
  if (IsVolatile)
    Aux |= AMDGPU::CPol::VOLATILE;
  Args.push_back(IRB.getInt32(Aux));

  Intrinsic::ID IID = Intrinsic::not_intrinsic;
  if (isa<LoadInst>(I))
    IID = Order == AtomicOrdering::NotAtomic
              ? Intrinsic::amdgcn_raw_ptr_buffer_load
              : Intrinsic::amdgcn_raw_ptr_atomic_buffer_load;
  else if (isa<StoreInst>(I))
    IID = Intrinsic::amdgcn_raw_ptr_buffer_store;
  else if (auto *RMW = dyn_cast<AtomicRMWInst>(I)) {
    switch (RMW->getOperation()) {
    case AtomicRMWInst::Xchg:
      IID = Intrinsic::amdgcn_raw_ptr_buffer_atomic_swap;
      break;
    case AtomicRMWInst::Add:
      IID = Intrinsic::amdgcn_raw_ptr_buffer_atomic_add;
      break;
    case AtomicRMWInst::Sub:
      IID = Intrinsic::amdgcn_raw_ptr_buffer_atomic_sub;
      break;
    case AtomicRMWInst::And:
      IID = Intrinsic::amdgcn_raw_ptr_buffer_atomic_and;
      break;
    case AtomicRMWInst::Or:
      IID = Intrinsic::amdgcn_raw_ptr_buffer_atomic_or;
      break;
    case AtomicRMWInst::Xor:
      IID = Intrinsic::amdgcn_raw_ptr_buffer_atomic_xor;
      break;
    case AtomicRMWInst::Max:
      IID = Intrinsic::amdgcn_raw_ptr_buffer_atomic_smax;
      break;
    case AtomicRMWInst::Min:
      IID = Intrinsic::amdgcn_raw_ptr_buffer_atomic_smin;
      break;
    case AtomicRMWInst::UMax:
      IID = Intrinsic::amdgcn_raw_ptr_buffer_atomic_umax;
      break;
    case AtomicRMWInst::UMin:
      IID = Intrinsic::amdgcn_raw_ptr_buffer_atomic_umin;
      break;
    case AtomicRMWInst::FAdd:
      IID = Intrinsic::amdgcn_raw_ptr_buffer_atomic_fadd;
      break;
    case AtomicRMWInst::FMax:
      IID = Intrinsic::amdgcn_raw_ptr_buffer_atomic_fmax;
      break;
    case AtomicRMWInst::FMin:
      IID = Intrinsic::amdgcn_raw_ptr_buffer_atomic_fmin;
      break;
    case AtomicRMWInst::FSub: {
      reportFatalUsageError(
          "atomic floating point subtraction not supported for "
          "buffer resources and should've been expanded away");
      break;
    }
    case AtomicRMWInst::FMaximum: {
      reportFatalUsageError(
          "atomic floating point fmaximum not supported for "
          "buffer resources and should've been expanded away");
      break;
    }
    case AtomicRMWInst::FMinimum: {
      reportFatalUsageError(
          "atomic floating point fminimum not supported for "
          "buffer resources and should've been expanded away");
      break;
    }
    case AtomicRMWInst::Nand:
      reportFatalUsageError(
          "atomic nand not supported for buffer resources and "
          "should've been expanded away");
      break;
    case AtomicRMWInst::UIncWrap:
    case AtomicRMWInst::UDecWrap:
      reportFatalUsageError("wrapping increment/decrement not supported for "
                            "buffer resources and should've ben expanded away");
      break;
    case AtomicRMWInst::BAD_BINOP:
      llvm_unreachable("Not sure how we got a bad binop");
    case AtomicRMWInst::USubCond:
    case AtomicRMWInst::USubSat:
      break;
    }
  }

  auto *Call = IRB.CreateIntrinsic(IID, Ty, Args);
  copyMetadata(Call, I);
  setAlign(Call, Alignment, Arg ? 1 : 0);
  Call->takeName(I);

  insertPostMemOpFence(Order, SSID);
  // The "no moving p7 directly" rewrites ensure that this load or store won't
  // itself need to be split into parts.
  SplitUsers.insert(I);
  I->replaceAllUsesWith(Call);
  return Call;
}

PtrParts SplitPtrStructs::visitInstruction(Instruction &I) {
  return {nullptr, nullptr};
}

PtrParts SplitPtrStructs::visitLoadInst(LoadInst &LI) {
  if (!isSplitFatPtr(LI.getPointerOperandType()))
    return {nullptr, nullptr};
  handleMemoryInst(&LI, nullptr, LI.getPointerOperand(), LI.getType(),
                   LI.getAlign(), LI.getOrdering(), LI.isVolatile(),
                   LI.getSyncScopeID());
  return {nullptr, nullptr};
}

PtrParts SplitPtrStructs::visitStoreInst(StoreInst &SI) {
  if (!isSplitFatPtr(SI.getPointerOperandType()))
    return {nullptr, nullptr};
  Value *Arg = SI.getValueOperand();
  handleMemoryInst(&SI, Arg, SI.getPointerOperand(), Arg->getType(),
                   SI.getAlign(), SI.getOrdering(), SI.isVolatile(),
                   SI.getSyncScopeID());
  return {nullptr, nullptr};
}

PtrParts SplitPtrStructs::visitAtomicRMWInst(AtomicRMWInst &AI) {
  if (!isSplitFatPtr(AI.getPointerOperand()->getType()))
    return {nullptr, nullptr};
  Value *Arg = AI.getValOperand();
  handleMemoryInst(&AI, Arg, AI.getPointerOperand(), Arg->getType(),
                   AI.getAlign(), AI.getOrdering(), AI.isVolatile(),
                   AI.getSyncScopeID());
  return {nullptr, nullptr};
}

// Unlike load, store, and RMW, cmpxchg needs special handling to account
// for the boolean argument.
PtrParts SplitPtrStructs::visitAtomicCmpXchgInst(AtomicCmpXchgInst &AI) {
  Value *Ptr = AI.getPointerOperand();
  if (!isSplitFatPtr(Ptr->getType()))
    return {nullptr, nullptr};
  IRB.SetInsertPoint(&AI);

  Type *Ty = AI.getNewValOperand()->getType();
  AtomicOrdering Order = AI.getMergedOrdering();
  SyncScope::ID SSID = AI.getSyncScopeID();
  bool IsNonTemporal = AI.getMetadata(LLVMContext::MD_nontemporal);

  auto [Rsrc, Off] = getPtrParts(Ptr);
  insertPreMemOpFence(Order, SSID);

  uint32_t Aux = 0;
  if (IsNonTemporal)
    Aux |= AMDGPU::CPol::SLC;
  if (AI.isVolatile())
    Aux |= AMDGPU::CPol::VOLATILE;
  auto *Call =
      IRB.CreateIntrinsic(Intrinsic::amdgcn_raw_ptr_buffer_atomic_cmpswap, Ty,
                          {AI.getNewValOperand(), AI.getCompareOperand(), Rsrc,
                           Off, IRB.getInt32(0), IRB.getInt32(Aux)});
  copyMetadata(Call, &AI);
  setAlign(Call, AI.getAlign(), 2);
  Call->takeName(&AI);
  insertPostMemOpFence(Order, SSID);

  Value *Res = PoisonValue::get(AI.getType());
  Res = IRB.CreateInsertValue(Res, Call, 0);
  if (!AI.isWeak()) {
    Value *Succeeded = IRB.CreateICmpEQ(Call, AI.getCompareOperand());
    Res = IRB.CreateInsertValue(Res, Succeeded, 1);
  }
  SplitUsers.insert(&AI);
  AI.replaceAllUsesWith(Res);
  return {nullptr, nullptr};
}

PtrParts SplitPtrStructs::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  using namespace llvm::PatternMatch;
  Value *Ptr = GEP.getPointerOperand();
  if (!isSplitFatPtr(Ptr->getType()))
    return {nullptr, nullptr};
  IRB.SetInsertPoint(&GEP);

  auto [Rsrc, Off] = getPtrParts(Ptr);
  const DataLayout &DL = GEP.getDataLayout();
  bool IsNUW = GEP.hasNoUnsignedWrap();
  bool IsNUSW = GEP.hasNoUnsignedSignedWrap();

  StructType *ResTy = cast<StructType>(GEP.getType());
  Type *ResRsrcTy = ResTy->getElementType(0);
  VectorType *ResRsrcVecTy = dyn_cast<VectorType>(ResRsrcTy);
  bool BroadcastsPtr = ResRsrcVecTy && !isa<VectorType>(Off->getType());

  // In order to call emitGEPOffset() and thus not have to reimplement it,
  // we need the GEP result to have ptr addrspace(7) type.
  Type *FatPtrTy =
      ResRsrcTy->getWithNewType(IRB.getPtrTy(AMDGPUAS::BUFFER_FAT_POINTER));
  GEP.mutateType(FatPtrTy);
  Value *OffAccum = emitGEPOffset(&IRB, DL, &GEP);
  GEP.mutateType(ResTy);

  if (BroadcastsPtr) {
    Rsrc = IRB.CreateVectorSplat(ResRsrcVecTy->getElementCount(), Rsrc,
                                 Rsrc->getName());
    Off = IRB.CreateVectorSplat(ResRsrcVecTy->getElementCount(), Off,
                                Off->getName());
  }
  if (match(OffAccum, m_Zero())) { // Constant-zero offset
    SplitUsers.insert(&GEP);
    return {Rsrc, Off};
  }

  bool HasNonNegativeOff = false;
  if (auto *CI = dyn_cast<ConstantInt>(OffAccum)) {
    HasNonNegativeOff = !CI->isNegative();
  }
  Value *NewOff;
  if (match(Off, m_Zero())) {
    NewOff = OffAccum;
  } else {
    NewOff = IRB.CreateAdd(Off, OffAccum, "",
                           /*hasNUW=*/IsNUW || (IsNUSW && HasNonNegativeOff),
                           /*hasNSW=*/false);
  }
  copyMetadata(NewOff, &GEP);
  NewOff->takeName(&GEP);
  SplitUsers.insert(&GEP);
  return {Rsrc, NewOff};
}

PtrParts SplitPtrStructs::visitPtrToIntInst(PtrToIntInst &PI) {
  Value *Ptr = PI.getPointerOperand();
  if (!isSplitFatPtr(Ptr->getType()))
    return {nullptr, nullptr};
  IRB.SetInsertPoint(&PI);

  Type *ResTy = PI.getType();
  unsigned Width = ResTy->getScalarSizeInBits();

  auto [Rsrc, Off] = getPtrParts(Ptr);
  const DataLayout &DL = PI.getDataLayout();
  unsigned FatPtrWidth = DL.getPointerSizeInBits(AMDGPUAS::BUFFER_FAT_POINTER);

  Value *Res;
  if (Width <= BufferOffsetWidth) {
    Res = IRB.CreateIntCast(Off, ResTy, /*isSigned=*/false,
                            PI.getName() + ".off");
  } else {
    Value *RsrcInt = IRB.CreatePtrToInt(Rsrc, ResTy, PI.getName() + ".rsrc");
    Value *Shl = IRB.CreateShl(
        RsrcInt,
        ConstantExpr::getIntegerValue(ResTy, APInt(Width, BufferOffsetWidth)),
        "", Width >= FatPtrWidth, Width > FatPtrWidth);
    Value *OffCast = IRB.CreateIntCast(Off, ResTy, /*isSigned=*/false,
                                       PI.getName() + ".off");
    Res = IRB.CreateOr(Shl, OffCast);
  }

  copyMetadata(Res, &PI);
  Res->takeName(&PI);
  SplitUsers.insert(&PI);
  PI.replaceAllUsesWith(Res);
  return {nullptr, nullptr};
}

PtrParts SplitPtrStructs::visitPtrToAddrInst(PtrToAddrInst &PA) {
  Value *Ptr = PA.getPointerOperand();
  if (!isSplitFatPtr(Ptr->getType()))
    return {nullptr, nullptr};
  IRB.SetInsertPoint(&PA);

  auto [Rsrc, Off] = getPtrParts(Ptr);
  Value *Res = IRB.CreateIntCast(Off, PA.getType(), /*isSigned=*/false);
  copyMetadata(Res, &PA);
  Res->takeName(&PA);
  SplitUsers.insert(&PA);
  PA.replaceAllUsesWith(Res);
  return {nullptr, nullptr};
}

PtrParts SplitPtrStructs::visitIntToPtrInst(IntToPtrInst &IP) {
  if (!isSplitFatPtr(IP.getType()))
    return {nullptr, nullptr};
  IRB.SetInsertPoint(&IP);
  const DataLayout &DL = IP.getDataLayout();
  unsigned RsrcPtrWidth = DL.getPointerSizeInBits(AMDGPUAS::BUFFER_RESOURCE);
  Value *Int = IP.getOperand(0);
  Type *IntTy = Int->getType();
  Type *RsrcIntTy = IntTy->getWithNewBitWidth(RsrcPtrWidth);
  unsigned Width = IntTy->getScalarSizeInBits();

  auto *RetTy = cast<StructType>(IP.getType());
  Type *RsrcTy = RetTy->getElementType(0);
  Type *OffTy = RetTy->getElementType(1);
  Value *RsrcPart = IRB.CreateLShr(
      Int,
      ConstantExpr::getIntegerValue(IntTy, APInt(Width, BufferOffsetWidth)));
  Value *RsrcInt = IRB.CreateIntCast(RsrcPart, RsrcIntTy, /*isSigned=*/false);
  Value *Rsrc = IRB.CreateIntToPtr(RsrcInt, RsrcTy, IP.getName() + ".rsrc");
  Value *Off =
      IRB.CreateIntCast(Int, OffTy, /*IsSigned=*/false, IP.getName() + ".off");

  copyMetadata(Rsrc, &IP);
  SplitUsers.insert(&IP);
  return {Rsrc, Off};
}

PtrParts SplitPtrStructs::visitAddrSpaceCastInst(AddrSpaceCastInst &I) {
  // TODO(krzysz00): handle casts from ptr addrspace(7) to global pointers
  // by computing the effective address.
  if (!isSplitFatPtr(I.getType()))
    return {nullptr, nullptr};
  IRB.SetInsertPoint(&I);
  Value *In = I.getPointerOperand();
  // No-op casts preserve parts
  if (In->getType() == I.getType()) {
    auto [Rsrc, Off] = getPtrParts(In);
    SplitUsers.insert(&I);
    return {Rsrc, Off};
  }

  auto *ResTy = cast<StructType>(I.getType());
  Type *RsrcTy = ResTy->getElementType(0);
  Type *OffTy = ResTy->getElementType(1);
  Value *ZeroOff = Constant::getNullValue(OffTy);

  // Special case for null pointers, undef, and poison, which can be created by
  // address space propagation.
  auto *InConst = dyn_cast<Constant>(In);
  if (InConst && InConst->isNullValue()) {
    Value *NullRsrc = Constant::getNullValue(RsrcTy);
    SplitUsers.insert(&I);
    return {NullRsrc, ZeroOff};
  }
  if (isa<PoisonValue>(In)) {
    Value *PoisonRsrc = PoisonValue::get(RsrcTy);
    Value *PoisonOff = PoisonValue::get(OffTy);
    SplitUsers.insert(&I);
    return {PoisonRsrc, PoisonOff};
  }
  if (isa<UndefValue>(In)) {
    Value *UndefRsrc = UndefValue::get(RsrcTy);
    Value *UndefOff = UndefValue::get(OffTy);
    SplitUsers.insert(&I);
    return {UndefRsrc, UndefOff};
  }

  if (I.getSrcAddressSpace() != AMDGPUAS::BUFFER_RESOURCE)
    reportFatalUsageError(
        "only buffer resources (addrspace 8) and null/poison pointers can be "
        "cast to buffer fat pointers (addrspace 7)");
  SplitUsers.insert(&I);
  return {In, ZeroOff};
}

PtrParts SplitPtrStructs::visitICmpInst(ICmpInst &Cmp) {
  Value *Lhs = Cmp.getOperand(0);
  if (!isSplitFatPtr(Lhs->getType()))
    return {nullptr, nullptr};
  Value *Rhs = Cmp.getOperand(1);
  IRB.SetInsertPoint(&Cmp);
  ICmpInst::Predicate Pred = Cmp.getPredicate();

  assert((Pred == ICmpInst::ICMP_EQ || Pred == ICmpInst::ICMP_NE) &&
         "Pointer comparison is only equal or unequal");
  auto [LhsRsrc, LhsOff] = getPtrParts(Lhs);
  auto [RhsRsrc, RhsOff] = getPtrParts(Rhs);
  Value *RsrcCmp =
      IRB.CreateICmp(Pred, LhsRsrc, RhsRsrc, Cmp.getName() + ".rsrc");
  copyMetadata(RsrcCmp, &Cmp);
  Value *OffCmp = IRB.CreateICmp(Pred, LhsOff, RhsOff, Cmp.getName() + ".off");
  copyMetadata(OffCmp, &Cmp);

  Value *Res = nullptr;
  if (Pred == ICmpInst::ICMP_EQ)
    Res = IRB.CreateAnd(RsrcCmp, OffCmp);
  else if (Pred == ICmpInst::ICMP_NE)
    Res = IRB.CreateOr(RsrcCmp, OffCmp);
  copyMetadata(Res, &Cmp);
  Res->takeName(&Cmp);
  SplitUsers.insert(&Cmp);
  Cmp.replaceAllUsesWith(Res);
  return {nullptr, nullptr};
}

PtrParts SplitPtrStructs::visitFreezeInst(FreezeInst &I) {
  if (!isSplitFatPtr(I.getType()))
    return {nullptr, nullptr};
  IRB.SetInsertPoint(&I);
  auto [Rsrc, Off] = getPtrParts(I.getOperand(0));

  Value *RsrcRes = IRB.CreateFreeze(Rsrc, I.getName() + ".rsrc");
  copyMetadata(RsrcRes, &I);
  Value *OffRes = IRB.CreateFreeze(Off, I.getName() + ".off");
  copyMetadata(OffRes, &I);
  SplitUsers.insert(&I);
  return {RsrcRes, OffRes};
}

PtrParts SplitPtrStructs::visitExtractElementInst(ExtractElementInst &I) {
  if (!isSplitFatPtr(I.getType()))
    return {nullptr, nullptr};
  IRB.SetInsertPoint(&I);
  Value *Vec = I.getVectorOperand();
  Value *Idx = I.getIndexOperand();
  auto [Rsrc, Off] = getPtrParts(Vec);

  Value *RsrcRes = IRB.CreateExtractElement(Rsrc, Idx, I.getName() + ".rsrc");
  copyMetadata(RsrcRes, &I);
  Value *OffRes = IRB.CreateExtractElement(Off, Idx, I.getName() + ".off");
  copyMetadata(OffRes, &I);
  SplitUsers.insert(&I);
  return {RsrcRes, OffRes};
}

PtrParts SplitPtrStructs::visitInsertElementInst(InsertElementInst &I) {
  // The mutated instructions temporarily don't return vectors, and so
  // we need the generic getType() here to avoid crashes.
  if (!isSplitFatPtr(cast<Instruction>(I).getType()))
    return {nullptr, nullptr};
  IRB.SetInsertPoint(&I);
  Value *Vec = I.getOperand(0);
  Value *Elem = I.getOperand(1);
  Value *Idx = I.getOperand(2);
  auto [VecRsrc, VecOff] = getPtrParts(Vec);
  auto [ElemRsrc, ElemOff] = getPtrParts(Elem);

  Value *RsrcRes =
      IRB.CreateInsertElement(VecRsrc, ElemRsrc, Idx, I.getName() + ".rsrc");
  copyMetadata(RsrcRes, &I);
  Value *OffRes =
      IRB.CreateInsertElement(VecOff, ElemOff, Idx, I.getName() + ".off");
  copyMetadata(OffRes, &I);
  SplitUsers.insert(&I);
  return {RsrcRes, OffRes};
}

PtrParts SplitPtrStructs::visitShuffleVectorInst(ShuffleVectorInst &I) {
  // Cast is needed for the same reason as insertelement's.
  if (!isSplitFatPtr(cast<Instruction>(I).getType()))
    return {nullptr, nullptr};
  IRB.SetInsertPoint(&I);

  Value *V1 = I.getOperand(0);
  Value *V2 = I.getOperand(1);
  ArrayRef<int> Mask = I.getShuffleMask();
  auto [V1Rsrc, V1Off] = getPtrParts(V1);
  auto [V2Rsrc, V2Off] = getPtrParts(V2);

  Value *RsrcRes =
      IRB.CreateShuffleVector(V1Rsrc, V2Rsrc, Mask, I.getName() + ".rsrc");
  copyMetadata(RsrcRes, &I);
  Value *OffRes =
      IRB.CreateShuffleVector(V1Off, V2Off, Mask, I.getName() + ".off");
  copyMetadata(OffRes, &I);
  SplitUsers.insert(&I);
  return {RsrcRes, OffRes};
}

PtrParts SplitPtrStructs::visitPHINode(PHINode &PHI) {
  if (!isSplitFatPtr(PHI.getType()))
    return {nullptr, nullptr};
  IRB.SetInsertPoint(*PHI.getInsertionPointAfterDef());
  // Phi nodes will be handled in post-processing after we've visited every
  // instruction. However, instead of just returning {nullptr, nullptr},
  // we explicitly create the temporary extractvalue operations that are our
  // temporary results so that they end up at the beginning of the block with
  // the PHIs.
  Value *TmpRsrc = IRB.CreateExtractValue(&PHI, 0, PHI.getName() + ".rsrc");
  Value *TmpOff = IRB.CreateExtractValue(&PHI, 1, PHI.getName() + ".off");
  Conditionals.push_back(&PHI);
  SplitUsers.insert(&PHI);
  return {TmpRsrc, TmpOff};
}

PtrParts SplitPtrStructs::visitSelectInst(SelectInst &SI) {
  if (!isSplitFatPtr(SI.getType()))
    return {nullptr, nullptr};
  IRB.SetInsertPoint(&SI);

  Value *Cond = SI.getCondition();
  Value *True = SI.getTrueValue();
  Value *False = SI.getFalseValue();
  auto [TrueRsrc, TrueOff] = getPtrParts(True);
  auto [FalseRsrc, FalseOff] = getPtrParts(False);

  Value *RsrcRes =
      IRB.CreateSelect(Cond, TrueRsrc, FalseRsrc, SI.getName() + ".rsrc", &SI);
  copyMetadata(RsrcRes, &SI);
  Conditionals.push_back(&SI);
  Value *OffRes =
      IRB.CreateSelect(Cond, TrueOff, FalseOff, SI.getName() + ".off", &SI);
  copyMetadata(OffRes, &SI);
  SplitUsers.insert(&SI);
  return {RsrcRes, OffRes};
}

/// Returns true if this intrinsic needs to be removed when it is
/// applied to `ptr addrspace(7)` values. Calls to these intrinsics are
/// rewritten into calls to versions of that intrinsic on the resource
/// descriptor.
static bool isRemovablePointerIntrinsic(Intrinsic::ID IID) {
  switch (IID) {
  default:
    return false;
  case Intrinsic::amdgcn_make_buffer_rsrc:
  case Intrinsic::ptrmask:
  case Intrinsic::invariant_start:
  case Intrinsic::invariant_end:
  case Intrinsic::launder_invariant_group:
  case Intrinsic::strip_invariant_group:
  case Intrinsic::memcpy:
  case Intrinsic::memcpy_inline:
  case Intrinsic::memmove:
  case Intrinsic::memset:
  case Intrinsic::memset_inline:
  case Intrinsic::experimental_memset_pattern:
  case Intrinsic::amdgcn_load_to_lds:
    return true;
  }
}

PtrParts SplitPtrStructs::visitIntrinsicInst(IntrinsicInst &I) {
  Intrinsic::ID IID = I.getIntrinsicID();
  switch (IID) {
  default:
    break;
  case Intrinsic::amdgcn_make_buffer_rsrc: {
    if (!isSplitFatPtr(I.getType()))
      return {nullptr, nullptr};
    Value *Base = I.getArgOperand(0);
    Value *Stride = I.getArgOperand(1);
    Value *NumRecords = I.getArgOperand(2);
    Value *Flags = I.getArgOperand(3);
    auto *SplitType = cast<StructType>(I.getType());
    Type *RsrcType = SplitType->getElementType(0);
    Type *OffType = SplitType->getElementType(1);
    IRB.SetInsertPoint(&I);
    Value *Rsrc = IRB.CreateIntrinsic(IID, {RsrcType, Base->getType()},
                                      {Base, Stride, NumRecords, Flags});
    copyMetadata(Rsrc, &I);
    Rsrc->takeName(&I);
    Value *Zero = Constant::getNullValue(OffType);
    SplitUsers.insert(&I);
    return {Rsrc, Zero};
  }
  case Intrinsic::ptrmask: {
    Value *Ptr = I.getArgOperand(0);
    if (!isSplitFatPtr(Ptr->getType()))
      return {nullptr, nullptr};
    Value *Mask = I.getArgOperand(1);
    IRB.SetInsertPoint(&I);
    auto [Rsrc, Off] = getPtrParts(Ptr);
    if (Mask->getType() != Off->getType())
      reportFatalUsageError("offset width is not equal to index width of fat "
                            "pointer (data layout not set up correctly?)");
    Value *OffRes = IRB.CreateAnd(Off, Mask, I.getName() + ".off");
    copyMetadata(OffRes, &I);
    SplitUsers.insert(&I);
    return {Rsrc, OffRes};
  }
  // Pointer annotation intrinsics that, given their object-wide nature
  // operate on the resource part.
  case Intrinsic::invariant_start: {
    Value *Ptr = I.getArgOperand(1);
    if (!isSplitFatPtr(Ptr->getType()))
      return {nullptr, nullptr};
    IRB.SetInsertPoint(&I);
    auto [Rsrc, Off] = getPtrParts(Ptr);
    Type *NewTy = PointerType::get(I.getContext(), AMDGPUAS::BUFFER_RESOURCE);
    auto *NewRsrc = IRB.CreateIntrinsic(IID, {NewTy}, {I.getOperand(0), Rsrc});
    copyMetadata(NewRsrc, &I);
    NewRsrc->takeName(&I);
    SplitUsers.insert(&I);
    I.replaceAllUsesWith(NewRsrc);
    return {nullptr, nullptr};
  }
  case Intrinsic::invariant_end: {
    Value *RealPtr = I.getArgOperand(2);
    if (!isSplitFatPtr(RealPtr->getType()))
      return {nullptr, nullptr};
    IRB.SetInsertPoint(&I);
    Value *RealRsrc = getPtrParts(RealPtr).first;
    Value *InvPtr = I.getArgOperand(0);
    Value *Size = I.getArgOperand(1);
    Value *NewRsrc = IRB.CreateIntrinsic(IID, {RealRsrc->getType()},
                                         {InvPtr, Size, RealRsrc});
    copyMetadata(NewRsrc, &I);
    NewRsrc->takeName(&I);
    SplitUsers.insert(&I);
    I.replaceAllUsesWith(NewRsrc);
    return {nullptr, nullptr};
  }
  case Intrinsic::launder_invariant_group:
  case Intrinsic::strip_invariant_group: {
    Value *Ptr = I.getArgOperand(0);
    if (!isSplitFatPtr(Ptr->getType()))
      return {nullptr, nullptr};
    IRB.SetInsertPoint(&I);
    auto [Rsrc, Off] = getPtrParts(Ptr);
    Value *NewRsrc = IRB.CreateIntrinsic(IID, {Rsrc->getType()}, {Rsrc});
    copyMetadata(NewRsrc, &I);
    NewRsrc->takeName(&I);
    SplitUsers.insert(&I);
    return {NewRsrc, Off};
  }
  case Intrinsic::amdgcn_load_to_lds: {
    Value *Ptr = I.getArgOperand(0);
    if (!isSplitFatPtr(Ptr->getType()))
      return {nullptr, nullptr};
    IRB.SetInsertPoint(&I);
    auto [Rsrc, Off] = getPtrParts(Ptr);
    Value *LDSPtr = I.getArgOperand(1);
    Value *LoadSize = I.getArgOperand(2);
    Value *ImmOff = I.getArgOperand(3);
    Value *Aux = I.getArgOperand(4);
    Value *SOffset = IRB.getInt32(0);
    Instruction *NewLoad = IRB.CreateIntrinsic(
        Intrinsic::amdgcn_raw_ptr_buffer_load_lds, {},
        {Rsrc, LDSPtr, LoadSize, Off, SOffset, ImmOff, Aux});
    copyMetadata(NewLoad, &I);
    SplitUsers.insert(&I);
    I.replaceAllUsesWith(NewLoad);
    return {nullptr, nullptr};
  }
  }
  return {nullptr, nullptr};
}

void SplitPtrStructs::processFunction(Function &F) {
  ST = &TM->getSubtarget<GCNSubtarget>(F);
  SmallVector<Instruction *, 0> Originals(
      llvm::make_pointer_range(instructions(F)));
  LLVM_DEBUG(dbgs() << "Splitting pointer structs in function: " << F.getName()
                    << "\n");
  for (Instruction *I : Originals) {
    auto [Rsrc, Off] = visit(I);
    assert(((Rsrc && Off) || (!Rsrc && !Off)) &&
           "Can't have a resource but no offset");
    if (Rsrc)
      RsrcParts[I] = Rsrc;
    if (Off)
      OffParts[I] = Off;
  }
  processConditionals();
  killAndReplaceSplitInstructions(Originals);

  // Clean up after ourselves to save on memory.
  RsrcParts.clear();
  OffParts.clear();
  SplitUsers.clear();
  Conditionals.clear();
  ConditionalTemps.clear();
}

namespace {
class AMDGPULowerBufferFatPointers : public ModulePass {
public:
  static char ID;

  AMDGPULowerBufferFatPointers() : ModulePass(ID) {}

  bool run(Module &M, const TargetMachine &TM);
  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};
} // namespace

/// Returns true if there are values that have a buffer fat pointer in them,
/// which means we'll need to perform rewrites on this function. As a side
/// effect, this will populate the type remapping cache.
static bool containsBufferFatPointers(const Function &F,
                                      BufferFatPtrToStructTypeMap *TypeMap) {
  bool HasFatPointers = false;
  for (const BasicBlock &BB : F)
    for (const Instruction &I : BB)
      HasFatPointers |= (I.getType() != TypeMap->remapType(I.getType()));
  return HasFatPointers;
}

static bool hasFatPointerInterface(const Function &F,
                                   BufferFatPtrToStructTypeMap *TypeMap) {
  Type *Ty = F.getFunctionType();
  return Ty != TypeMap->remapType(Ty);
}

/// Move the body of `OldF` into a new function, returning it.
static Function *moveFunctionAdaptingType(Function *OldF, FunctionType *NewTy,
                                          ValueToValueMapTy &CloneMap) {
  bool IsIntrinsic = OldF->isIntrinsic();
  Function *NewF =
      Function::Create(NewTy, OldF->getLinkage(), OldF->getAddressSpace());
  NewF->copyAttributesFrom(OldF);
  NewF->copyMetadata(OldF, 0);
  NewF->takeName(OldF);
  NewF->updateAfterNameChange();
  NewF->setDLLStorageClass(OldF->getDLLStorageClass());
  OldF->getParent()->getFunctionList().insertAfter(OldF->getIterator(), NewF);

  while (!OldF->empty()) {
    BasicBlock *BB = &OldF->front();
    BB->removeFromParent();
    BB->insertInto(NewF);
    CloneMap[BB] = BB;
    for (Instruction &I : *BB) {
      CloneMap[&I] = &I;
    }
  }

  SmallVector<AttributeSet> ArgAttrs;
  AttributeList OldAttrs = OldF->getAttributes();

  for (auto [I, OldArg, NewArg] : enumerate(OldF->args(), NewF->args())) {
    CloneMap[&NewArg] = &OldArg;
    NewArg.takeName(&OldArg);
    Type *OldArgTy = OldArg.getType(), *NewArgTy = NewArg.getType();
    // Temporarily mutate type of `NewArg` to allow RAUW to work.
    NewArg.mutateType(OldArgTy);
    OldArg.replaceAllUsesWith(&NewArg);
    NewArg.mutateType(NewArgTy);

    AttributeSet ArgAttr = OldAttrs.getParamAttrs(I);
    // Intrinsics get their attributes fixed later.
    if (OldArgTy != NewArgTy && !IsIntrinsic)
      ArgAttr = ArgAttr.removeAttributes(
          NewF->getContext(),
          AttributeFuncs::typeIncompatible(NewArgTy, ArgAttr));
    ArgAttrs.push_back(ArgAttr);
  }
  AttributeSet RetAttrs = OldAttrs.getRetAttrs();
  if (OldF->getReturnType() != NewF->getReturnType() && !IsIntrinsic)
    RetAttrs = RetAttrs.removeAttributes(
        NewF->getContext(),
        AttributeFuncs::typeIncompatible(NewF->getReturnType(), RetAttrs));
  NewF->setAttributes(AttributeList::get(
      NewF->getContext(), OldAttrs.getFnAttrs(), RetAttrs, ArgAttrs));
  return NewF;
}

static void makeCloneInPraceMap(Function *F, ValueToValueMapTy &CloneMap) {
  for (Argument &A : F->args())
    CloneMap[&A] = &A;
  for (BasicBlock &BB : *F) {
    CloneMap[&BB] = &BB;
    for (Instruction &I : BB)
      CloneMap[&I] = &I;
  }
}

bool AMDGPULowerBufferFatPointers::run(Module &M, const TargetMachine &TM) {
  bool Changed = false;
  const DataLayout &DL = M.getDataLayout();
  // Record the functions which need to be remapped.
  // The second element of the pair indicates whether the function has to have
  // its arguments or return types adjusted.
  SmallVector<std::pair<Function *, bool>> NeedsRemap;

  LLVMContext &Ctx = M.getContext();

  BufferFatPtrToStructTypeMap StructTM(DL);
  BufferFatPtrToIntTypeMap IntTM(DL);
  for (const GlobalVariable &GV : M.globals()) {
    if (GV.getAddressSpace() == AMDGPUAS::BUFFER_FAT_POINTER) {
      // FIXME: Use DiagnosticInfo unsupported but it requires a Function
      Ctx.emitError("global variables with a buffer fat pointer address "
                    "space (7) are not supported");
      continue;
    }

    Type *VT = GV.getValueType();
    if (VT != StructTM.remapType(VT)) {
      // FIXME: Use DiagnosticInfo unsupported but it requires a Function
      Ctx.emitError("global variables that contain buffer fat pointers "
                    "(address space 7 pointers) are unsupported. Use "
                    "buffer resource pointers (address space 8) instead");
      continue;
    }
  }

  {
    // Collect all constant exprs and aggregates referenced by any function.
    SmallVector<Constant *, 8> Worklist;
    for (Function &F : M.functions())
      for (Instruction &I : instructions(F))
        for (Value *Op : I.operands())
          if (isa<ConstantExpr, ConstantAggregate>(Op))
            Worklist.push_back(cast<Constant>(Op));

    // Recursively look for any referenced buffer pointer constants.
    SmallPtrSet<Constant *, 8> Visited;
    SetVector<Constant *> BufferFatPtrConsts;
    while (!Worklist.empty()) {
      Constant *C = Worklist.pop_back_val();
      if (!Visited.insert(C).second)
        continue;
      if (isBufferFatPtrOrVector(C->getType()))
        BufferFatPtrConsts.insert(C);
      for (Value *Op : C->operands())
        if (isa<ConstantExpr, ConstantAggregate>(Op))
          Worklist.push_back(cast<Constant>(Op));
    }

    // Expand all constant expressions using fat buffer pointers to
    // instructions.
    Changed |= convertUsersOfConstantsToInstructions(
        BufferFatPtrConsts.getArrayRef(), /*RestrictToFunc=*/nullptr,
        /*RemoveDeadConstants=*/false, /*IncludeSelf=*/true);
  }

  StoreFatPtrsAsIntsAndExpandMemcpyVisitor MemOpsRewrite(&IntTM, DL,
                                                         M.getContext(), &TM);
  LegalizeBufferContentTypesVisitor BufferContentsTypeRewrite(DL,
                                                              M.getContext());
  for (Function &F : M.functions()) {
    bool InterfaceChange = hasFatPointerInterface(F, &StructTM);
    bool BodyChanges = containsBufferFatPointers(F, &StructTM);
    Changed |= MemOpsRewrite.processFunction(F);
    if (InterfaceChange || BodyChanges) {
      NeedsRemap.push_back(std::make_pair(&F, InterfaceChange));
      Changed |= BufferContentsTypeRewrite.processFunction(F);
    }
  }
  if (NeedsRemap.empty())
    return Changed;

  SmallVector<Function *> NeedsPostProcess;
  SmallVector<Function *> Intrinsics;
  // Keep one big map so as to memoize constants across functions.
  ValueToValueMapTy CloneMap;
  FatPtrConstMaterializer Materializer(&StructTM, CloneMap);

  ValueMapper LowerInFuncs(CloneMap, RF_None, &StructTM, &Materializer);
  for (auto [F, InterfaceChange] : NeedsRemap) {
    Function *NewF = F;
    if (InterfaceChange)
      NewF = moveFunctionAdaptingType(
          F, cast<FunctionType>(StructTM.remapType(F->getFunctionType())),
          CloneMap);
    else
      makeCloneInPraceMap(F, CloneMap);
    LowerInFuncs.remapFunction(*NewF);
    if (NewF->isIntrinsic())
      Intrinsics.push_back(NewF);
    else
      NeedsPostProcess.push_back(NewF);
    if (InterfaceChange) {
      F->replaceAllUsesWith(NewF);
      F->eraseFromParent();
    }
    Changed = true;
  }
  StructTM.clear();
  IntTM.clear();
  CloneMap.clear();

  SplitPtrStructs Splitter(DL, M.getContext(), &TM);
  for (Function *F : NeedsPostProcess)
    Splitter.processFunction(*F);
  for (Function *F : Intrinsics) {
    if (isRemovablePointerIntrinsic(F->getIntrinsicID())) {
      F->eraseFromParent();
    } else {
      std::optional<Function *> NewF = Intrinsic::remangleIntrinsicFunction(F);
      if (NewF)
        F->replaceAllUsesWith(*NewF);
    }
  }
  return Changed;
}

bool AMDGPULowerBufferFatPointers::runOnModule(Module &M) {
  TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
  const TargetMachine &TM = TPC.getTM<TargetMachine>();
  return run(M, TM);
}

char AMDGPULowerBufferFatPointers::ID = 0;

char &llvm::AMDGPULowerBufferFatPointersID = AMDGPULowerBufferFatPointers::ID;

void AMDGPULowerBufferFatPointers::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
}

#define PASS_DESC "Lower buffer fat pointer operations to buffer resources"
INITIALIZE_PASS_BEGIN(AMDGPULowerBufferFatPointers, DEBUG_TYPE, PASS_DESC,
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AMDGPULowerBufferFatPointers, DEBUG_TYPE, PASS_DESC, false,
                    false)
#undef PASS_DESC

ModulePass *llvm::createAMDGPULowerBufferFatPointersPass() {
  return new AMDGPULowerBufferFatPointers();
}

PreservedAnalyses
AMDGPULowerBufferFatPointersPass::run(Module &M, ModuleAnalysisManager &MA) {
  return AMDGPULowerBufferFatPointers().run(M, TM) ? PreservedAnalyses::none()
                                                   : PreservedAnalyses::all();
}
