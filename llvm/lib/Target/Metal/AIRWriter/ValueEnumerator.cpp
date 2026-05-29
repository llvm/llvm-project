//===- ValueEnumerator.cpp - AIR value/type/metadata enumerator -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Assigns the contiguous indices the AIR bitcode uses for types, global
// values, and module-level constants. Distinct from LLVM's in-tree
// ValueEnumerator because AIR uses typed POINTER records: the type table
// stores (Type *, pointee) pairs so the same opaque pointer type can take
// several slots, one per inferred pointee.
//
//===----------------------------------------------------------------------===//

#include "ValueEnumerator.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace llvm {
namespace metal {

ValueEnumerator::ValueEnumerator(Module &M, const PointeeTypeMap &PTM)
    : PTM(PTM) {
  auto &Ctx = M.getContext();

  // ── Phase 1: Infer pointee types for pointer Types ─────────────────

  // Function pointer: ptr as0 → kernel's function type
  Type *PtrAs0 = PointerType::get(Ctx, 0);
  for (auto &F : M)
    if (!F.isDeclaration()) {
      inferredPointee[PtrAs0] = F.getFunctionType();
      break;
    }

  // Device/TG pointers: infer from first arg usage
  for (auto &F : M)
    for (auto &Arg : F.args())
      if (Arg.getType()->isPointerTy() && !inferredPointee.count(Arg.getType()))
        if (auto *Ty =
                PointeeTypeMap::inferFromUsage(const_cast<Argument *>(&Arg)))
          inferredPointee[Arg.getType()] = Ty;

  // PTM overrides - but skip global variables (they get separate TypeEntry
  // via globalPtrTypeIdx, not the shared inferredPointee).
  // Skip event_t-typed entries for AS3 - these should NOT set the AS3
  // default because MMA TG pointers need float*3 as default. Event pointers
  // use per-value PTM entries via ptrTypeIdxForValue/funcTypeParamIndices.
  for (auto &[V, T] : PTM)
    if (V->getType()->isPointerTy() && !isa<GlobalVariable>(V) &&
        !inferredPointee.count(V->getType())) {
      // Skip opaque struct types (event_t) from setting TYPE-level defaults
      if (auto *ST = dyn_cast<StructType>(T))
        if (ST->isOpaque())
          continue;
      // Skip ptr-typed pointees (ptr addrspace(3) used for event storage)
      // - these would make AS0 default to ptr*0, which is wrong.
      if (T->isPointerTy())
        continue;
      inferredPointee[V->getType()] = T;
    }

  // ── Phase 2: Enumerate types ───────────────────────────────────────

  addType(Type::getVoidTy(Ctx));
  addType(Type::getFloatTy(Ctx));

  // Pre-create event_t type BEFORE function type processing.
  // Async copy intrinsics return event_t addrspace(3)* and
  // wait_simdgroup_events takes a pointer-to-event-pointer. The event_t type
  // must exist before function types reference it, to avoid forward references
  // in the type table (which crash Metal's LLVM 14-based reader).
  //
  // CRITICAL: Set inferredPointee[PtrAs3] = EventTy so that any bare
  // typeIdx(PtrAs3) call during emission resolves to event_t*3 (not i8*3
  // or float*3). This is needed for wait_simdgroup_events param 1, whose
  // pointee is PtrAs3 - the emission calls typeIdx(PtrAs3) which uses
  // inferredPointee to determine the inner pointer's pointee type.
  // The i8*3 entries for async copy buffer params are handled separately
  // through per-param funcTypeParamIndices, not this TYPE-LEVEL default.
  {
    StructType *EventTy = StructType::getTypeByName(Ctx, "event_t");
    if (EventTy) {
      addType(EventTy);
      auto *PtrAs3 = PointerType::get(Ctx, 3);
      inferredPointee[PtrAs3] = EventTy;
      // Also pre-create the event_t*3 pointer entry so it exists before
      // function type processing (avoids forward references).
      ptrTypeIdx(PtrAs3, EventTy);
    }
  }

  // Enumerate function types - definitions first, then declarations.
  // Must process definitions first so their per-param pointee inference
  // populates funcTypeParamIndices before any recursive addType call
  // from declaration processing caches the function type with wrong params.
  for (auto &F : M)
    if (!F.isDeclaration())
      addFunctionType(F.getFunctionType(), &F);
  for (auto &F : M)
    if (F.isDeclaration())
      addFunctionType(F.getFunctionType(), &F);
  // Create function pointer types for definitions (kernels) only.
  // Declarations (intrinsics) don't need function pointers in Metal v1.
  for (auto &F : M)
    if (!F.isDeclaration())
      ptrTypeIdx(PtrAs0, F.getFunctionType());

  addType(Type::getMetadataTy(Ctx));
  addType(Type::getLabelTy(Ctx));

  // Global variable types - create per-global typed pointer entries
  for (auto &GV : M.globals()) {
    addType(GV.getValueType());
    globalPtrTypeIdx(&GV); // creates ptr(valueType, addrspace) entry
  }

  // Instruction result + operand types - enumerate ALL types used by
  // instructions so the type table is complete before emission.
  // For GEPs into arrays, the result pointer needs a separate typed entry.
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        addType(I.getType());
        // Operand types (including constants like i64 0 in GEPs)
        for (auto &Op : I.operands())
          if (!isa<BasicBlock>(Op))
            addType(Op->getType());
        // Alloca: enumerate the allocated type AND the result's typed-pointer.
        // The alloca result is pointer_to(allocatedType, addrspace=0).
        // Both the allocated type and the result pointer type must exist
        // in the type table for the bitcode reader to materialize correctly.
        if (auto *AI = dyn_cast<AllocaInst>(&I)) {
          addType(AI->getAllocatedType());
          // Create result type entry: ptr(allocatedType, 0)
          ptrTypeIdx(PointerType::get(M.getContext(), 0),
                     AI->getAllocatedType());
        }
        // GEP result: create ptr(elementType, addrspace) entry
        // Use PTM override for device (AS 1) pointers (e.g., store float
        // through i8* GEP should produce float*, not i8*).
        // For TG (AS 3) byte globals, keep GEP's own result element type -
        // the byte global stays as [N x i8] and GEP results must be i8*.
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          if (GEP->getType()->isPointerTy()) {
            Type *ResultPointee = GEP->getResultElementType();
            unsigned AddrSpace = GEP->getType()->getPointerAddressSpace();
            if (AddrSpace != 3 || !ResultPointee->isIntegerTy(8)) {
              if (auto *PtmTy = PTM.get(GEP))
                ResultPointee = PtmTy;
            }
            ptrTypeIdx(PointerType::get(M.getContext(), AddrSpace),
                       ResultPointee);
          }
        }
        // Bitcast ptr→ptr: in Metal v1 these change typed pointer.
        // Create a separate typed pointer entry from PTM.
        if (auto *BC = dyn_cast<BitCastInst>(&I)) {
          if (BC->getType()->isPointerTy() &&
              BC->getSrcTy() == BC->getDestTy()) {
            if (auto *PtmTy = PTM.get(BC)) {
              unsigned AddrSpace = BC->getType()->getPointerAddressSpace();
              ptrTypeIdx(PointerType::get(M.getContext(), AddrSpace), PtmTy);
            }
          }
        }
      }
    }
  }

  // ── Phase 3: Value IDs (globals first, then functions) ─────────────

  for (auto &GV : M.globals()) {
    globalValueMap[&GV] = globalValues.size();
    globalValues.push_back(&GV);
  }
  // Definitions first, then declarations (the order Metal's loader expects)
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    globalValueMap[&F] = globalValues.size();
    globalValues.push_back(&F);
  }
  for (auto &F : M) {
    if (!F.isDeclaration())
      continue;
    globalValueMap[&F] = globalValues.size();
    globalValues.push_back(&F);
  }

  // ── Phase 4: Module constants ──────────────────────────────────────

  for (auto &NMD : M.named_metadata())
    for (unsigned I = 0; I < NMD.getNumOperands(); I++)
      collectMetadataConstants(NMD.getOperand(I));

  for (auto &GV : M.globals())
    if (GV.hasInitializer())
      addModuleConstant(GV.getInitializer());

  // Also collect sub-constants of function-level aggregate constants.
  // ConstantsWriter emits ConstantArray/ConstantStruct/ConstantVector via
  // AGGREGATE records that reference sub-constants by moduleConstIdx.
  // If a function-level constant is a non-data aggregate, its sub-constants
  // must be in the module constant table.
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        for (auto &Op : I.operands()) {
          auto *C = dyn_cast<Constant>(Op);
          if (!C || isa<GlobalValue>(C))
            continue;
          if (isa<ConstantArray>(C) || isa<ConstantStruct>(C) ||
              isa<ConstantVector>(C)) {
            for (unsigned J = 0; J < C->getNumOperands(); J++)
              if (auto *OC = dyn_cast<Constant>(C->getOperand(J)))
                addModuleConstant(OC);
          }
          if (auto *CDA = dyn_cast<ConstantDataArray>(C)) {
            for (unsigned J = 0; J < CDA->getNumElements(); J++)
              addModuleConstant(CDA->getElementAsConstant(J));
          }
        }
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Type queries
// ═══════════════════════════════════════════════════════════════════════

unsigned ValueEnumerator::typeIdx(Type *T) {
  if (isa<PointerType>(T))
    return ptrTypeIdx(T, pointeeType(T));
  TypeEntry E{T, nullptr};
  auto It = typeMap.find(E);
  if (It != typeMap.end())
    return It->second;
  return addType(T);
}

unsigned ValueEnumerator::ptrTypeIdxForValue(const Value *V) {
  Type *Pointee = nullptr;
  if (auto *Ty = PTM.get(const_cast<Value *>(V)))
    Pointee = Ty;
  if (!Pointee)
    Pointee = pointeeType(V->getType());
  return ptrTypeIdx(V->getType(), Pointee);
}

unsigned ValueEnumerator::ptrTypeIdx(Type *PtrTy, Type *Pointee) {
  TypeEntry E{PtrTy, Pointee};
  auto It = typeMap.find(E);
  if (It != typeMap.end())
    return It->second;
  // Ensure pointee is in table first
  addType(Pointee);
  return addEntry(E);
}

unsigned ValueEnumerator::globalPtrTypeIdx(const GlobalVariable *GV) {
  return ptrTypeIdx(GV->getType(), GV->getValueType());
}

unsigned ValueEnumerator::globalIdx(const Value *V) const {
  auto It = globalValueMap.find(V);
  assert(It != globalValueMap.end());
  return It->second;
}

unsigned ValueEnumerator::moduleConstIdx(const Constant *C) const {
  auto It = moduleConstMap.find(C);
  assert(It != moduleConstMap.end());
  return globalValues.size() + It->second;
}

bool ValueEnumerator::hasModuleConst(const Constant *C) const {
  return moduleConstMap.count(C);
}

Type *ValueEnumerator::pointeeType(Type *PtrTy) const {
  auto It = inferredPointee.find(PtrTy);
  if (It != inferredPointee.end())
    return It->second;
  if (auto *PT = dyn_cast<PointerType>(PtrTy)) {
    unsigned AddrSpace = PT->getAddressSpace();
    if (AddrSpace == 1 || AddrSpace == 3)
      return Type::getFloatTy(PtrTy->getContext());
  }
  return Type::getFloatTy(PtrTy->getContext());
}

Type *ValueEnumerator::pointeeTypeForValue(const Value *V) const {
  if (auto *Ty = PTM.get(const_cast<Value *>(V)))
    return Ty;
  if (auto *Ty = PointeeTypeMap::inferFromUsage(const_cast<Value *>(V)))
    return Ty;
  return pointeeType(V->getType());
}

// ═══════════════════════════════════════════════════════════════════════
// Internal
// ═══════════════════════════════════════════════════════════════════════

unsigned ValueEnumerator::addEntry(TypeEntry E) {
  auto It = typeMap.find(E);
  if (It != typeMap.end())
    return It->second;
  unsigned Idx = types.size();
  typeMap[E] = Idx;
  types.push_back(E);
  return Idx;
}

unsigned ValueEnumerator::addType(Type *T) {
  if (isa<PointerType>(T))
    return ptrTypeIdx(T, pointeeType(T));

  // FunctionTypes are handled by addFunctionType for proper per-param
  // pointee tracking. If we get here via a generic path, use the
  // stored indices or fall through to simple entry creation.
  if (auto *FT = dyn_cast<FunctionType>(T)) {
    TypeEntry E{T, nullptr};
    auto It = typeMap.find(E);
    if (It != typeMap.end())
      return It->second;
    // Not yet enumerated - add with default pointees (no Function context)
    return addFunctionType(FT, nullptr);
  }

  TypeEntry E{T, nullptr};
  auto It = typeMap.find(E);
  if (It != typeMap.end())
    return It->second;

  // Add components first (no forward refs)
  if (auto *VT = dyn_cast<VectorType>(T)) {
    addType(VT->getElementType());
  } else if (auto *AT = dyn_cast<ArrayType>(T)) {
    addType(AT->getElementType());
  } else if (auto *ST = dyn_cast<StructType>(T)) {
    if (!ST->isOpaque())
      for (auto *ET : ST->elements())
        addType(ET);
  }

  // Re-check after recursive adds
  It = typeMap.find(E);
  if (It != typeMap.end())
    return It->second;

  return addEntry(E);
}

unsigned ValueEnumerator::addFunctionType(FunctionType *FT, const Function *F) {
  TypeEntry E{FT, nullptr};
  auto It = typeMap.find(E);
  if (It != typeMap.end())
    return It->second;

  // Build per-param type indices with correct pointee types
  SmallVector<unsigned, 8> ParamIndices;

  // Add return type - for pointer returns, infer pointee from call results
  if (FT->getReturnType()->isPointerTy()) {
    Type *RetPointee = nullptr;
    // Async copy intrinsics return event_t addrspace(3)*.
    // Must check function name because async_copy may be declared but never
    // called (no users to infer from via PTM).
    if (F && F->isDeclaration() &&
        F->getName().starts_with("air.simdgroup_async_copy")) {
      auto &Ctx = F->getContext();
      if (auto *EventTy = StructType::getTypeByName(Ctx, "event_t"))
        RetPointee = EventTy;
    }
    // For declarations, infer from how call results are typed in PTM
    if (!RetPointee && F && F->isDeclaration()) {
      for (auto *U : F->users()) {
        if (auto *CI = dyn_cast<CallInst>(U)) {
          if (auto *Ty = PTM.get(const_cast<CallInst *>(CI))) {
            RetPointee = Ty;
            break;
          }
        }
      }
    }
    if (!RetPointee)
      RetPointee = pointeeType(FT->getReturnType());
    unsigned RetIdx = ptrTypeIdx(FT->getReturnType(), RetPointee);
    funcTypeReturnIndex[FT] = RetIdx;
  } else {
    addType(FT->getReturnType());
  }

  // Add each param type - for pointers, use per-param pointee inference
  for (unsigned I = 0; I < FT->getNumParams(); I++) {
    Type *PT = FT->getParamType(I);
    if (!PT->isPointerTy()) {
      ParamIndices.push_back(addType(PT));
      continue;
    }
    // Infer pointee for this specific param
    Type *Pointee = nullptr;

    // For atomic intrinsics, the device pointer param must match the
    // atomic type (i32 or f32) - NOT the kernel buffer's default pointee.
    // E.g., air.atomic.global.cmpxchg.weak.i32 needs i32*, not float*.
    if (F && F->isDeclaration()) {
      StringRef Name = F->getName();
      if (Name.starts_with("air.atomic.")) {
        unsigned AddrSpace = cast<PointerType>(PT)->getAddressSpace();
        if (AddrSpace == 1 || AddrSpace == 3) {
          // Determine pointee from intrinsic name suffix
          if (Name.ends_with(".i32"))
            Pointee = Type::getInt32Ty(F->getContext());
          else if (Name.ends_with(".f32"))
            Pointee = Type::getFloatTy(F->getContext());
        }
      }
      // Async copy intrinsics use i8* for buffer pointer params.
      // The intrinsic name suffix (e.g., .p3i8.p1i8) indicates byte pointers.
      // Use i8* for both AS3 (destination) and AS1 (source) pointer params.
      if (Name.starts_with("air.simdgroup_async_copy")) {
        unsigned AddrSpace = cast<PointerType>(PT)->getAddressSpace();
        if (AddrSpace == 1 || AddrSpace == 3)
          Pointee = Type::getInt8Ty(F->getContext());
      }
      // wait_simdgroup_events param 1: pointer to event_t*3 storage.
      // inferredPointee[PtrAs3] = event_t is set permanently in the
      // constructor when event_t exists, so typeIdx(PtrAs3) at emission
      // time resolves to event_t*3. Just set Pointee = PtrAs3.
      if (Name == "air.wait_simdgroup_events" && I == 1) {
        auto *PtrAs3 = PointerType::get(F->getContext(), 3);
        Pointee = PtrAs3;
        ParamIndices.push_back(ptrTypeIdx(PT, Pointee));
        continue; // Skip the normal param processing below
      }
    }

    if (!Pointee && F && !F->isDeclaration() && I < F->arg_size())
      Pointee = pointeeTypeForValue(F->getArg(I));
    // For declarations, infer from call site arguments
    if (!Pointee && F && F->isDeclaration()) {
      for (auto *U : F->users()) {
        if (auto *CI = dyn_cast<CallInst>(U)) {
          if (I < CI->arg_size()) {
            Pointee = pointeeTypeForValue(CI->getArgOperand(I));
            if (Pointee)
              break;
          }
        }
      }
    }
    if (!Pointee)
      Pointee = pointeeType(PT);
    ParamIndices.push_back(ptrTypeIdx(PT, Pointee));
  }

  // Store per-param indices for TypeTableWriter
  funcTypeParamIndices[FT] = ParamIndices;

  // Re-check (recursive adds may have added this type)
  It = typeMap.find(E);
  if (It != typeMap.end())
    return It->second;

  return addEntry(E);
}

void ValueEnumerator::addModuleConstant(const Constant *C) {
  if (moduleConstMap.count(C) || globalValueMap.count(C))
    return;
  // ConstantDataArray/Vector have packed data with no sub-constant operands.
  // Extract elements as individual constants so they can be referenced by
  // AGGREGATE records (Metal v1 doesn't support DATA for array globals).
  if (auto *CDA = dyn_cast<ConstantDataSequential>(C)) {
    for (unsigned I = 0; I < CDA->getNumElements(); I++)
      addModuleConstant(CDA->getElementAsConstant(I));
  }
  for (unsigned I = 0; I < C->getNumOperands(); I++)
    if (auto *OC = dyn_cast<Constant>(C->getOperand(I)))
      addModuleConstant(OC);
  addType(C->getType());
  moduleConstMap[C] = moduleConstants.size();
  moduleConstants.push_back(C);
}

void ValueEnumerator::collectMetadataConstants(const MDNode *N) {
  for (unsigned I = 0; I < N->getNumOperands(); I++) {
    if (auto *VAM = dyn_cast_or_null<ValueAsMetadata>(N->getOperand(I)))
      if (auto *C = dyn_cast<Constant>(VAM->getValue()))
        addModuleConstant(C);
    if (auto *Sub = dyn_cast_or_null<MDNode>(N->getOperand(I)))
      collectMetadataConstants(Sub);
  }
}

} // namespace metal
} // namespace llvm
