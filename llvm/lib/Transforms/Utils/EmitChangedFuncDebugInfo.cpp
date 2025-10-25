//==- EmitChangedFuncDebugInfoPass - Emit Additional Debug Info -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements emitting debug info for functions with changed
// signatures or new functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/EmitChangedFuncDebugInfo.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

static cl::opt<bool> DisableChangedFuncDBInfo(
    "disable-changed-func-dbinfo", cl::Hidden, cl::init(false),
    cl::desc("Disable debuginfo emission for changed func signatures"));

/// ===================== Small helpers =====================

/// Strip qualifiers through derived types, but stop when we see the first
/// pointer type. Otherwise return the base non-qualified type.
static DIType *stripToBaseOrFirstPointer(DIType *T) {
  while (auto *DT = dyn_cast_or_null<DIDerivedType>(T)) {
    if (DT->getTag() == dwarf::DW_TAG_pointer_type)
      return DT;
    T = DT->getBaseType();
  }
  return T;
}

/// Rebuild a DILocation chain identical to Old, but root it under NewSP via
/// inlinedAt.
static const DILocation *reanchorDILocChain(const DILocation *Old,
                                            DISubprogram *NewSP,
                                            LLVMContext &Ctx) {
  // Root location anchored at the new subprogram.
  DILocation *Root = DILocation::get(Ctx, 0, 0, NewSP);

  SmallVector<const DILocation *, 8> Chain;
  const DILocation *Cur = Old;
  Chain.push_back(Cur);
  while ((Cur = Cur->getInlinedAt()))
    Chain.push_back(Cur);

  DILocation *Prev = Root;
  for (int i = Chain.size() - 1; i >= 0; --i) {
    const DILocation *DL = Chain[i];
    Prev = DILocation::get(Ctx, DL->getLine(), DL->getColumn(), DL->getScope(),
                           Prev, DL->isImplicitCode(), DL->getAtomGroup(),
                           DL->getAtomRank());
  }
  return Prev;
}

/// Recursively transform every DILocation inside an MD tree.
static Metadata *
mapAllDILocs(Metadata *M,
             std::function<const DILocation *(const DILocation *)> X,
             LLVMContext &Ctx) {
  if (auto *DL = dyn_cast<DILocation>(M))
    return const_cast<DILocation *>(X(DL));

  if (auto *N = dyn_cast<MDNode>(M)) {
    SmallVector<Metadata *, 8> NewOps;
    NewOps.reserve(N->getNumOperands());
    for (const MDOperand &Op : N->operands())
      NewOps.push_back(mapAllDILocs(Op.get(), X, Ctx));
    return MDNode::get(Ctx, NewOps); // tag nodes need not be distinct
  }
  return M;
}

/// Clone a loop MD node, rewriting all nested DILocations with X.
static MDNode *cloneLoopIDReplacingAllDILocs(
    MDNode *OldLoopID, std::function<const DILocation *(const DILocation *)> X,
    LLVMContext &Ctx) {
  SmallVector<Metadata *, 8> Ops;
  Ops.reserve(OldLoopID->getNumOperands());
  Ops.push_back(nullptr); // placeholder for self
  for (unsigned i = 1, e = OldLoopID->getNumOperands(); i < e; ++i)
    Ops.push_back(mapAllDILocs(OldLoopID->getOperand(i).get(), X, Ctx));
  MDNode *New = MDNode::getDistinct(Ctx, Ops);
  New->replaceOperandWith(0, New);
  return New;
}

/// ===================== Type utilities =====================

/// For a struct/union parameter split into fragments, derive an integer DIType
/// for the fragment described by Expr. If a member with matching (offset,size)
/// exists, reuse its base type; otherwise synthesize an int type of BitSize.
static DIType *getIntTypeFromExpr(DIBuilder &DIB, DIExpression *Expr,
                                  DICompositeType *DTy) {
  for (auto Op : Expr->expr_ops()) {
    if (Op.getOp() != dwarf::DW_OP_LLVM_fragment)
      continue;

    uint64_t BitOffset = Op.getArg(0);
    uint64_t BitSize = Op.getArg(1);

    for (auto *Element : DTy->getElements()) {
      if (auto *Elem = dyn_cast<DIDerivedType>(Element)) {
        if (Elem->getSizeInBits() == BitSize &&
            Elem->getOffsetInBits() == BitOffset)
          return Elem->getBaseType();
      }
    }
    // No matching member; synthesize.
    return DIB.createBasicType(("int" + std::to_string(BitSize)).c_str(),
                               BitSize, dwarf::DW_ATE_signed);
  }
  return nullptr;
}

/// Compute the DI type to use for a parameter given its IR Type Ty and original
/// DI type Orig (qualifiers stripped / first pointer returned).
/// Sets NeedSuffix when we “coerce” a composite to a fragment or wrap it in a
/// pointer.
static DIType *computeParamDIType(DIBuilder &DIB, Type *Ty, DIType *Orig,
                                  unsigned PointerBitWidth, DIExpression *Expr,
                                  bool &NeedSuffix) {
  NeedSuffix = false;
  DIType *Stripped = stripToBaseOrFirstPointer(Orig);

  if (Ty->isIntegerTy()) {
    if (auto *Comp = dyn_cast_or_null<DICompositeType>(Stripped)) {
      if (!Ty->isIntegerTy(Comp->getSizeInBits())) {
        DIType *Frag = getIntTypeFromExpr(DIB, Expr, Comp);
        if (Frag)
          NeedSuffix = true;
        return Frag;
      }
      // sizes match -> rare; accept fallthrough
    }
    unsigned W = cast<IntegerType>(Ty)->getBitWidth();
    return DIB.createBasicType(("int" + std::to_string(W)).c_str(), W,
                               dwarf::DW_ATE_signed);
  }

  if (Ty->isPointerTy()) {
    if (auto *Comp = dyn_cast_or_null<DICompositeType>(Stripped)) {
      NeedSuffix = true; // struct turned into pointer to struct
      return DIB.createPointerType(Comp, PointerBitWidth);
    }
    if (auto *Der = dyn_cast_or_null<DIDerivedType>(Stripped)) {
      if (Der->getTag() == dwarf::DW_TAG_pointer_type)
        return Der; // already a pointer in DI
    }
    // Generic pointer: synthesize pointer to pointer-sized int
    DIType *Base =
        DIB.createBasicType(("int" + std::to_string(PointerBitWidth)).c_str(),
                            PointerBitWidth, dwarf::DW_ATE_signed);
    return DIB.createPointerType(Base, PointerBitWidth);
  }

  if (Ty->isFloatingPointTy()) {
    unsigned W = Ty->getScalarSizeInBits();
    return DIB.createBasicType(("float" + std::to_string(W)).c_str(), W,
                               dwarf::DW_ATE_float);
  }
  // Default to pointer-sized int
  return DIB.createBasicType(("int" + std::to_string(PointerBitWidth)).c_str(),
                             PointerBitWidth, dwarf::DW_ATE_signed);
}

/// Synthesize a DI type/name for a parameter we failed to match via dbg
/// records.
static std::pair<DIType *, std::string>
fallbackParam(DIBuilder &DIB, Function *F, unsigned Idx, unsigned PtrW) {
  Type *Ty = F->getArg(Idx)->getType();
  unsigned W = Ty->isIntegerTy() ? cast<IntegerType>(Ty)->getBitWidth() : 32;
  DIType *BaseInt = DIB.createBasicType(("int" + std::to_string(W)).c_str(), W,
                                        dwarf::DW_ATE_signed);
  DIType *ParamTy =
      Ty->isIntegerTy() ? BaseInt : DIB.createPointerType(BaseInt, PtrW);
  std::string Name = F->getArg(Idx)->getName().str();
  if (Name.empty())
    Name = "__" + std::to_string(Idx);
  return {ParamTy, Name};
}

/// Aggregate (struct/union) larger than pointer width?
static bool isLargeByValueAggregate(DIType *T, unsigned PtrW) {
  DIType *P = stripToBaseOrFirstPointer(T);
  if (auto *Comp = dyn_cast_or_null<DICompositeType>(P))
    return Comp->getSizeInBits() > PtrW;
  return false;
}

/// ===================== Argument collection =====================

/// Scan the entry block’s dbg records to deduce DI type & name for argument
/// Idx. Handles alloca-based byval lowering and the zext(i1)->i8 adjacent-use
/// pattern. Falls back to a synthesized type if no match.
static bool getOneArgDI(Module &M, unsigned Idx, BasicBlock &Entry,
                        DIBuilder &DIB, Function *F, DISubprogram *OldSP,
                        DISubprogram *NewSP,
                        SmallVectorImpl<Metadata *> &TypeList,
                        SmallVectorImpl<Metadata *> &ArgList,
                        unsigned PointerBitWidth) {
  for (Instruction &I : Entry) {
    for (DbgRecord &DR : I.getDbgRecordRange()) {
      auto *DVR = dyn_cast<DbgVariableRecord>(&DR);
      if (!DVR)
        continue;

      auto *VAM = dyn_cast_or_null<ValueAsMetadata>(DVR->getRawLocation());
      if (!VAM)
        continue;

      Value *LocV = VAM->getValue();
      if (!LocV)
        continue;

      auto *Var = DVR->getVariable();
      if (!Var || !Var->getArg())
        continue;

      // Strip modifiers/pointers as in your original
      DIType *DITy = Var->getType();
      while (auto *DTy = dyn_cast<DIDerivedType>(DITy)) {
        if (DTy->getTag() == dwarf::DW_TAG_pointer_type) {
          DITy = DTy;
          break;
        }
        DITy = DTy->getBaseType();
      }

      // Accept direct match, or special alloca/byval, or the zext(i1)->i8 case.
      bool Matched = (LocV == F->getArg(Idx));
      if (!Matched) {
        if (isa<AllocaInst>(LocV)) {
          if (Var->getName() != F->getArg(Idx)->getName())
            continue;
          Matched = true;
        } else if (Instruction *Prev = I.getPrevNode()) {
          if (auto *ZExt = dyn_cast<ZExtInst>(Prev))
            Matched = (ZExt->getOperand(0) == F->getArg(Idx) && LocV == Prev);
          if (!Matched)
            continue;
        } else {
          continue;
        }
      }

      Type *IRTy = F->getArg(Idx)->getType();
      bool NeedSuffix = false;
      DIType *ParamType =
          computeParamDIType(DIB, IRTy, Var->getType(), PointerBitWidth,
                             DVR->getExpression(), NeedSuffix);
      if (!ParamType)
        return false;

      TypeList.push_back(ParamType);

      std::string ArgName = F->getArg(Idx)->getName().str();
      if (ArgName.empty()) {
        ArgName = Var->getName().str();
        if (NeedSuffix)
          ArgName += "__" + std::to_string(Idx);
      }

      auto *NewVar = DIB.createParameterVariable(
          NewSP, StringRef(ArgName), Idx + 1, OldSP->getUnit()->getFile(),
          OldSP->getLine(), ParamType);
      ArgList.push_back(NewVar);
      return true;
    }
  }

  // Fallback (unused/poison argument)
  auto [ParamTy, Name] = fallbackParam(DIB, F, Idx, PointerBitWidth);
  TypeList.push_back(ParamTy);
  auto *NewVar = DIB.createParameterVariable(NewSP, StringRef(Name), Idx + 1,
                                             OldSP->getUnit()->getFile(),
                                             OldSP->getLine(), ParamTy);
  ArgList.push_back(NewVar);
  return true;
}

static bool collectReturnAndArgs(Module &M, DIBuilder &DIB, Function *F,
                                 DISubprogram *OldSP, DISubprogram *NewSP,
                                 SmallVectorImpl<Metadata *> &TypeList,
                                 SmallVectorImpl<Metadata *> &ArgList,
                                 unsigned PointerBitWidth) {
  FunctionType *FTy = F->getFunctionType();
  Type *RetTy = FTy->getReturnType();

  if (RetTy->isVoidTy()) {
    TypeList.push_back(nullptr);
  } else {
    // Non-void return type is assumed unchanged by optimization.
    DITypeRefArray TyArray = OldSP->getType()->getTypeArray();
    TypeList.push_back(TyArray[0]);
  }

  BasicBlock &Entry = F->getEntryBlock();
  for (unsigned i = 0, n = FTy->getNumParams(); i < n; ++i) {
    if (!getOneArgDI(M, i, Entry, DIB, F, OldSP, NewSP, TypeList, ArgList,
                     PointerBitWidth))
      return false;
  }
  return true;
}

/// ===================== Per-function transform =====================

static void generateDebugInfo(Module &M, Function *F,
                              unsigned PointerBitWidth) {
  DISubprogram *OldSP = F->getSubprogram();
  DICompileUnit *CU = OldSP->getUnit();
  DIBuilder DIB(M, /*AllowUnresolved=*/false, CU);

  SmallVector<Metadata *, 5> TypeList, ArgList;

  // Create a fresh “artificial” subprogram (type/args filled later).
  DISubprogram *NewSP =
      DIB.createFunction(OldSP->getScope(),     // Scope
                         F->getName(),          // Name
                         F->getName(),          // Linkage name
                         CU->getFile(),         // File
                         OldSP->getLine(),      // Line
                         nullptr,               // DISubroutineType
                         OldSP->getScopeLine(), // ScopeLine
                         DINode::FlagZero | DINode::FlagArtificial,
                         DISubprogram::SPFlagDefinition);

  bool Success = collectReturnAndArgs(M, DIB, F, OldSP, NewSP, TypeList,
                                      ArgList, PointerBitWidth);
  if (!Success) {
    // Cannot decide a signature: mark the old one nocall and bail out.
    auto Temp = OldSP->getType()->cloneWithCC(llvm::dwarf::DW_CC_nocall);
    OldSP->replaceType(MDNode::replaceWithPermanent(std::move(Temp)));
    DIB.finalize();
    return;
  }

  // Install new type + retained params.
  DITypeRefArray DITypeArray = DIB.getOrCreateTypeArray(TypeList);
  auto *SubroutineType = DIB.createSubroutineType(DITypeArray);
  NewSP->replaceType(SubroutineType);
  NewSP->replaceRetainedNodes(DIB.getOrCreateArray(ArgList));
  F->setSubprogram(NewSP);

  // Reanchor all DILocations (DbgRecord stream + Instruction DebugLoc +
  // MD_loop).
  LLVMContext &Ctx = M.getContext();
  const auto Reanchor = [&](const DILocation *Old) -> const DILocation * {
    return reanchorDILocChain(Old, NewSP, Ctx);
  };

  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      for (DbgRecord &DR : I.getDbgRecordRange()) {
        if (DebugLoc DL = DR.getDebugLoc())
          DR.setDebugLoc(
              DebugLoc(const_cast<DILocation *>(Reanchor(DL.get()))));
      }
      if (DebugLoc DL = I.getDebugLoc())
        I.setDebugLoc(DebugLoc(const_cast<DILocation *>(Reanchor(DL.get()))));
      if (MDNode *LoopID = I.getMetadata(LLVMContext::MD_loop)) {
        MDNode *New = cloneLoopIDReplacingAllDILocs(LoopID, Reanchor, Ctx);
        I.setMetadata(LLVMContext::MD_loop, New);
      }
    }
  }

  // Insert dbg.values for the real IR arguments at function entry.
  if (unsigned NumArgs = F->getFunctionType()->getNumParams()) {
    auto IP = F->getEntryBlock().getFirstInsertionPt();
    const DILocation *Top = DILocation::get(Ctx, 0, 0, NewSP);
    for (int i = (int)NumArgs - 1; i >= 0; --i) {
      auto *Var = cast<DILocalVariable>(ArgList[i]);
      DIB.insertDbgValueIntrinsic(F->getArg(i), Var, DIB.createExpression(),
                                  Top, IP);
    }
  }

  DIB.finalize();
}

/// ===================== Pass driver =====================

PreservedAnalyses EmitChangedFuncDebugInfoPass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  if (DisableChangedFuncDBInfo)
    return PreservedAnalyses::all();

  // C-only
  for (DICompileUnit *CU : M.debug_compile_units()) {
    auto L = CU->getSourceLanguage().getUnversionedName();
    if (L != dwarf::DW_LANG_C && L != dwarf::DW_LANG_C89 &&
        L != dwarf::DW_LANG_C99 && L != dwarf::DW_LANG_C11 &&
        L != dwarf::DW_LANG_C17)
      return PreservedAnalyses::all();
  }

  Triple T(M.getTargetTriple());
  if (T.isBPF()) // BPF: LLVM emits BTF; skip here for now.
    return PreservedAnalyses::all();

  const unsigned PointerBitWidth = T.getArchPointerBitWidth();

  SmallVector<Function *> ChangedFuncs;
  for (Function &F : M) {
    if (F.isIntrinsic() || F.isDeclaration())
      continue;
    DISubprogram *SP = F.getSubprogram();
    if (!SP)
      continue;

    DITypeRefArray TyArray = SP->getType()->getTypeArray();
    if (TyArray.size() == 0)
      continue;

    // Skip if return is a large aggregate (> pointer size).
    {
      DIType *RetDI = stripToBaseOrFirstPointer(TyArray[0]);
      if (auto *Comp = dyn_cast_or_null<DICompositeType>(RetDI))
        if (Comp->getSizeInBits() > PointerBitWidth)
          continue;
    }

    // Skip varargs originals.
    if (TyArray.size() > 1 && TyArray[TyArray.size() - 1] == nullptr)
      continue;

    // Consider signature "changed" if any arg is a large by-value aggregate.
    bool SigChanged = false;
    if (!F.getName().contains('.')) {
      uint8_t cc = SP->getType()->getCC();
      if (cc != dwarf::DW_CC_nocall) {
        for (unsigned i = 1; i < TyArray.size(); ++i) {
          if (isLargeByValueAggregate(TyArray[i], PointerBitWidth)) {
            SigChanged = true;
            break;
          }
        }
        if (!SigChanged)
          continue;
      }
    }

    // Reset CC to DW_CC_normal; we’ll mark the new SP as Artificial.
    auto Temp = SP->getType()->cloneWithCC(llvm::dwarf::DW_CC_normal);
    SP->replaceType(MDNode::replaceWithPermanent(std::move(Temp)));

    ChangedFuncs.push_back(&F);
  }

  for (Function *F : ChangedFuncs)
    generateDebugInfo(M, F, PointerBitWidth);

  return ChangedFuncs.empty() ? PreservedAnalyses::all()
                              : PreservedAnalyses::none();
}
