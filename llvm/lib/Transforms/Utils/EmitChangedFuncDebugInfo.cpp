//==- EmitChangedFuncDebugInfoPass - Emit Additional Debug Info -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass synthesizes a "shadow" DISubprogram carrying a *possibly changed*
// signature for certain optimized functions. The new subprogram lives in a
// dedicated DICompileUnit whose file name is "<changed_signatures>", and is
// attached to a dummy AvailableExternally function so that the metadata forms
// a valid graph.
//
// When we can recover argument names/types from dbg records in the entry
// block, we do so; otherwise we conservatively fall back to pointer- or
// integer-typed parameters.
//
// We *only* run for C-family source languages, skip BPF targets (BTF is used
// there), skip varargs originals, and skip functions whose return type is a
// large by-value aggregate.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/EmitChangedFuncDebugInfo.h"

#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

/// Disable switch.
static cl::opt<bool> DisableChangedFuncDBInfo(
    "disable-changed-func-dbinfo", cl::Hidden, cl::init(false),
    cl::desc("Disable debuginfo emission for changed func signatures"));

/// Replace all '.' with "__" (stable with opaque-lifetime inputs).
static std::string sanitizeDots(StringRef S) {
  std::string Out = S.str();
  for (size_t pos = 0; (pos = Out.find('.', pos)) != std::string::npos;
       pos += 2)
    Out.replace(pos, 1, "__");
  return Out;
}

/// Return the "basename" (prefix before the first '.') of a name.
static StringRef baseBeforeDot(StringRef S) {
  return S.take_front(S.find('.'));
}

/// Ensure a variable name is unique among previously recorded parameters.
/// If collision, append "__<Idx>".
static std::string uniquifyParamName(StringRef Candidate,
                                     ArrayRef<Metadata *> Existing,
                                     unsigned Idx) {
  for (unsigned i = 0; i < Existing.size(); ++i)
    if (auto *LV = dyn_cast<DILocalVariable>(Existing[i]))
      if (LV->getName() == Candidate)
        return (Twine(Candidate) + "__" + Twine(Idx)).str();
  return Candidate.str();
}

/// Walk backward in the current block to see whether LocV is exactly a
/// zext/trunc of Arg (used by two separate match sites originally).
static bool comesFromArgViaCast(Value *LocV, Argument *Arg, Instruction &At) {
  if (!LocV)
    return false;
  for (Instruction *Prev = At.getPrevNode(); Prev; Prev = Prev->getPrevNode()) {
    // FIXME: maybe some other insns need check as well.
    if (auto *Z = dyn_cast<ZExtInst>(Prev))
      if (Z->getOperand(0) == Arg && LocV == Prev)
        return true;
    if (auto *T = dyn_cast<TruncInst>(Prev))
      if (T->getOperand(0) == Arg && LocV == Prev)
        return true;
  }
  return false;
}

/// Strip qualifiers/typedefs until the first pointer-type (which we keep), or
/// to the base non-derived type if no pointer is found.
static DIType *stripToBaseOrFirstPointer(DIType *T) {
  while (auto *DT = dyn_cast_or_null<DIDerivedType>(T)) {
    if (DT->getTag() == dwarf::DW_TAG_pointer_type)
      return DT;
    T = DT->getBaseType();
  }
  return T;
}

static DIType *createBasicType(DIBuilder &DIB, uint64_t SizeInBits) {
  switch (SizeInBits) {
  case 8:
    return DIB.createBasicType("char", 8, dwarf::DW_ATE_signed);
  case 16:
    return DIB.createBasicType("short", 16, dwarf::DW_ATE_signed);
  case 32:
    return DIB.createBasicType("int", 32, dwarf::DW_ATE_signed);
  case 64:
    return DIB.createBasicType("long long", 64, dwarf::DW_ATE_signed);
  default:
    return DIB.createBasicType("__int128", SizeInBits, dwarf::DW_ATE_signed);
  }
}

static DIType *createFloatType(DIBuilder &DIB, uint64_t SizeInBits) {
  if (SizeInBits == 32)
    return DIB.createBasicType("float", 32, dwarf::DW_ATE_float);
  if (SizeInBits == 64)
    return DIB.createBasicType("double", 64, dwarf::DW_ATE_float);
  return DIB.createBasicType("long double", SizeInBits, dwarf::DW_ATE_float);
}

static DIType *getIntTypeFromExpr(DIBuilder &DIB, DIExpression *Expr,
                                  DICompositeType *DTy, unsigned W) {
  for (auto Op : Expr->expr_ops()) {
    if (Op.getOp() != dwarf::DW_OP_LLVM_fragment)
      break;

    const uint64_t BitOffset = Op.getArg(0);
    const uint64_t BitSize = Op.getArg(1);
    const uint64_t BitUpLimit = BitOffset + BitSize;

    DINodeArray Elems = DTy->getElements();
    unsigned N = Elems.size();

    for (unsigned i = 0; i < N; ++i)
      if (auto *Elem = dyn_cast<DIDerivedType>(Elems[i])) {
        if (N >= 2 && i < N - 1) {
          if (Elem->getOffsetInBits() <= BitOffset &&
              BitUpLimit <= (Elem->getOffsetInBits() + Elem->getSizeInBits()))
            return Elem->getBaseType();
        } else {
          if (Elem->getOffsetInBits() <= BitOffset &&
              BitUpLimit <= DTy->getSizeInBits())
            return Elem->getBaseType();
        }
      }

    return createBasicType(DIB, BitSize);
  }
  return createBasicType(DIB, W);
}

static DIType *computeParamDIType(DIBuilder &DIB, Type *Ty, DIType *Orig,
                                  unsigned PointerBitWidth,
                                  DIExpression *Expr) {
  DIType *Stripped = stripToBaseOrFirstPointer(Orig);

  if (Ty->isIntegerTy()) {
    unsigned W = cast<IntegerType>(Ty)->getBitWidth();
    if (auto *Comp = dyn_cast_or_null<DICompositeType>(Stripped)) {
      if (!Ty->isIntegerTy(Comp->getSizeInBits()))
        return getIntTypeFromExpr(DIB, Expr, Comp, W);
    }
    return createBasicType(DIB, W);
  }

  if (Ty->isFloatingPointTy())
    return createFloatType(DIB, Ty->getScalarSizeInBits());

  // Ty->isPointerTy().
  if (auto *Der = dyn_cast_or_null<DIDerivedType>(Stripped)) {
    assert(Der->getTag() == dwarf::DW_TAG_pointer_type);
    return Der;
  }

  auto *Comp = cast<DICompositeType>(Stripped);
  return DIB.createPointerType(Comp, PointerBitWidth);
}

static bool isLargeByValueAggregate(DIType *T, unsigned PtrW) {
  DIType *P = stripToBaseOrFirstPointer(T);
  if (auto *Comp = dyn_cast_or_null<DICompositeType>(P))
    return Comp->getSizeInBits() > PtrW;
  return false;
}

static void pushParam(DIBuilder &DIB, DISubprogram *OldSP,
                      SmallVectorImpl<Metadata *> &TypeList,
                      SmallVectorImpl<Metadata *> &ArgList, DIType *Ty,
                      StringRef VarName, unsigned Idx) {
  TypeList.push_back(Ty);
  ArgList.push_back(DIB.createParameterVariable(
      OldSP, VarName, Idx + 1, OldSP->getFile(), OldSP->getLine(), Ty));
}

/// Argument collection.
static bool getOneArgDI(unsigned Idx, BasicBlock &Entry, DIBuilder &DIB,
                        Function *F, DISubprogram *OldSP,
                        SmallVectorImpl<Metadata *> &TypeList,
                        SmallVectorImpl<Metadata *> &ArgList,
                        unsigned PointerBitWidth) {
  Argument *Arg = F->getArg(Idx);
  StringRef ArgName = Arg->getName();
  Type *ArgTy = Arg->getType();

  // If byval struct, remember its identified-name and kind to match via dbg.
  StringRef ByValUserName;
  bool IsByValStruct = true;
  if (ArgTy->isPointerTy() && Arg->hasByValAttr()) {
    if (Type *ByValTy = F->getParamByValType(Idx))
      if (auto *ST = dyn_cast<StructType>(ByValTy)) {
        auto [Kind, Name] = ST->getName().split('.');
        ByValUserName = Name;
        IsByValStruct = (Kind == "struct");
      }
  }

  DILocalVariable *DIVar = nullptr;
  DIExpression *DIExpr = nullptr;

  // Scan the entry block for dbg records.
  for (Instruction &I : Entry) {
    bool Final = false;

    for (DbgRecord &DR : I.getDbgRecordRange()) {
      auto *DVR = dyn_cast<DbgVariableRecord>(&DR);
      if (!DVR)
        continue;

      auto *VAM = dyn_cast_or_null<ValueAsMetadata>(DVR->getRawLocation());
      if (!VAM)
        continue;

      Value *LocV = VAM->getValue();
      auto *Var = DVR->getVariable();
      if (!Var || !Var->getArg())
        continue;

      // Canonicalize through derived types stopping at first pointer.
      DIType *DITy = Var->getType();
      while (auto *DTy = dyn_cast<DIDerivedType>(DITy)) {
        if (DTy->getTag() == dwarf::DW_TAG_pointer_type) {
          DITy = DTy;
          break;
        }
        DITy = DTy->getBaseType();
      }

      if (LocV == Arg) {
        DIVar = Var;
        DIExpr = DVR->getExpression();
        Final = true;
        break;
      }

      // Compare base names (before dot) in several cases.
      StringRef ArgBase = baseBeforeDot(ArgName);
      StringRef VarBase = baseBeforeDot(Var->getName());

      if (ArgName.empty()) {
        if (!ByValUserName.empty()) {
          // Match by byval struct DI type’s name/kind.
          DIType *Stripped = stripToBaseOrFirstPointer(Var->getType());
          auto *Comp = dyn_cast<DICompositeType>(Stripped);
          if (!Comp)
            continue;
          bool IsStruct = Comp->getTag() == dwarf::DW_TAG_structure_type;
          if (Comp->getName() != ByValUserName || IsStruct != IsByValStruct)
            continue;
          DIVar = Var;
          DIExpr = DVR->getExpression();
          Final = true;
          break;
        }

        // FIXME: more work is needed to find precise DILocalVariable.
        if (isa<PoisonValue>(LocV) || isa<AllocaInst>(LocV))
          continue;

        if (comesFromArgViaCast(LocV, Arg, I)) {
          DIVar = Var;
          DIExpr = DVR->getExpression();
          Final = true;
          break;
        }
      } else {
        // We do have an IR arg name.
        if (isa<PoisonValue>(LocV)) {
          if (Var->getName() != ArgBase)
            continue;
          DIVar = Var;
          DIExpr = DVR->getExpression();
          // Possibly we may find a non poison value later.
        } else if (isa<AllocaInst>(LocV)) {
          if (Var->getName() != ArgName)
            continue;
          DIVar = Var;
          DIExpr = DVR->getExpression();
          Final = true;
          break;
        } else if (ArgBase == VarBase) {
          DIVar = Var;
          DIExpr = DVR->getExpression();
          Final = true;
          break;
        } else if (comesFromArgViaCast(LocV, Arg, I)) {
          DIVar = Var;
          DIExpr = DVR->getExpression();
          Final = true;
          break;
        }
      }
    }

    if (Final)
      break;
  }

  // Fallback types if we failed to find a dbg match.
  if (!DIVar) {
    // Likely to be a unused parameter.
    if (ArgTy->isIntegerTy()) {
      auto *Ty = createBasicType(DIB, cast<IntegerType>(ArgTy)->getBitWidth());
      pushParam(DIB, OldSP, TypeList, ArgList, Ty,
                (Twine("__") + Twine(Idx)).str(), Idx);
      return true;
    }
    // Pointer: use void *
    // Returning false means the DIType is not precise.
    auto *Ty = DIB.createPointerType(nullptr, PointerBitWidth);
    pushParam(DIB, OldSP, TypeList, ArgList, Ty,
              (Twine("__") + Twine(Idx)).str(), Idx);
    return false;
  }

  // Compute parameter DI type from IR type + original debug type.
  DIType *ParamType =
      computeParamDIType(DIB, ArgTy, DIVar->getType(), PointerBitWidth, DIExpr);

  // Decide the parameter name (sanitize + uniquify).
  std::string VarName;
  if (ArgName.empty()) {
    VarName = sanitizeDots(DIVar->getName());
    VarName = uniquifyParamName(VarName, ArgList, Idx);
  } else {
    VarName = sanitizeDots(ArgName);
  }

  pushParam(DIB, OldSP, TypeList, ArgList, ParamType, VarName, Idx);
  return true;
}

/// Collect return and parameter DI information.
static bool collectReturnAndArgs(DIBuilder &DIB, Function *F,
                                 DISubprogram *OldSP,
                                 SmallVectorImpl<Metadata *> &TypeList,
                                 SmallVectorImpl<Metadata *> &ArgList,
                                 unsigned PointerBitWidth) {
  FunctionType *FTy = F->getFunctionType();
  Type *RetTy = FTy->getReturnType();

  if (RetTy->isVoidTy())
    TypeList.push_back(nullptr);
  else
    TypeList.push_back(OldSP->getType()->getTypeArray()[0]);

  BasicBlock &Entry = F->getEntryBlock();
  for (unsigned i = 0, n = FTy->getNumParams(); i < n; ++i)
    if (!getOneArgDI(i, Entry, DIB, F, OldSP, TypeList, ArgList,
                     PointerBitWidth))
      return false;
  return true;
}

static DICompileUnit *findChangedSigCU(Module &M) {
  if (NamedMDNode *CUs = M.getNamedMetadata("llvm.dbg.cu"))
    for (MDNode *Node : CUs->operands()) {
      auto *CU = cast<DICompileUnit>(Node);
      if (CU->getFile()->getFilename() == "<changed_signatures>")
        return CU;
    }
  return nullptr;
}

static void generateDebugInfo(Module &M, Function *F,
                              unsigned PointerBitWidth) {
  DICompileUnit *NewCU = findChangedSigCU(M);
  DIBuilder DIB(M, /*AllowUnresolved=*/false, NewCU);

  DISubprogram *OldSP = F->getSubprogram();
  DIFile *NewFile;

  if (NewCU) {
    NewFile = NewCU->getFile();
  } else {
    DICompileUnit *OldCU = OldSP->getUnit();
    DIFile *OldFile = OldCU->getFile();
    NewFile = DIB.createFile("<changed_signatures>", OldFile->getDirectory());
    NewCU = DIB.createCompileUnit(
        OldCU->getSourceLanguage(), NewFile, OldCU->getProducer(),
        OldCU->isOptimized(), OldCU->getFlags(), OldCU->getRuntimeVersion());
  }

  SmallVector<Metadata *, 5> TypeList, ArgList;
  bool Success =
      collectReturnAndArgs(DIB, F, OldSP, TypeList, ArgList, PointerBitWidth);

  DITypeRefArray DITypeArray = DIB.getOrCreateTypeArray(
      TypeList.empty() ? ArrayRef<Metadata *>{nullptr}
                       : ArrayRef<Metadata *>{TypeList});
  auto *SubTy = DIB.createSubroutineType(DITypeArray);
  DINodeArray ArgArray = DIB.getOrCreateArray(ArgList);

  DISubprogram *NewSP = DIB.createFunction(
      OldSP, OldSP->getName(), F->getName(), NewFile, OldSP->getLine(), SubTy,
      OldSP->getScopeLine(), DINode::FlagZero, DISubprogram::SPFlagDefinition);
  NewSP->replaceRetainedNodes(ArgArray);

  // No success mean some argument ptr type is not precise.
  if (!Success) {
    auto Temp = NewSP->getType()->cloneWithCC(llvm::dwarf::DW_CC_nocall);
    NewSP->replaceType(MDNode::replaceWithPermanent(std::move(Temp)));
  }

  DIB.finalizeSubprogram(NewSP);

  // Dummy anchor function
  Function *DummyF = Function::Create(F->getFunctionType(),
                                      GlobalValue::AvailableExternallyLinkage,
                                      F->getName() + ".newsig", &M);

  // Provide a trivial body so the SP is marked as "defined".
  BasicBlock *BB = BasicBlock::Create(M.getContext(), "entry", DummyF);
  IRBuilder<> IRB(BB);
  IRB.CreateUnreachable();
  DummyF->setSubprogram(NewSP);

  DIB.finalize();
}

PreservedAnalyses EmitChangedFuncDebugInfoPass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  if (DisableChangedFuncDBInfo)
    return PreservedAnalyses::all();

  // Only C-family
  for (DICompileUnit *CU : M.debug_compile_units()) {
    auto L = CU->getSourceLanguage().getUnversionedName();
    if (L != dwarf::DW_LANG_C && L != dwarf::DW_LANG_C89 &&
        L != dwarf::DW_LANG_C99 && L != dwarf::DW_LANG_C11 &&
        L != dwarf::DW_LANG_C17)
      return PreservedAnalyses::all();
  }

  Triple T(M.getTargetTriple());
  if (T.isBPF()) // BPF uses BTF
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

    // Skip varargs
    if (TyArray.size() > 1 && TyArray[TyArray.size() - 1] == nullptr)
      continue;

    // For C language, only supports int/float/ptr types, no support for vector.
    unsigned i = 0;
    unsigned n = F.getFunctionType()->getNumParams();
    for (i = 0; i < n; ++i) {
      Type *ArgTy = F.getArg(i)->getType();
      if (ArgTy->isVectorTy())
        break;
    }
    if (i != n)
      continue;

    // Skip if large by-value return
    DIType *RetDI = stripToBaseOrFirstPointer(TyArray[0]);
    if (auto *Comp = dyn_cast_or_null<DICompositeType>(RetDI))
      if (Comp->getSizeInBits() > PointerBitWidth)
        continue;

    // Only when signature changed (or any arg is large by-value aggregate)
    if (SP->getType()->getCC() != dwarf::DW_CC_nocall) {
      n = TyArray.size();
      for (i = 1; i < n; ++i)
        if (isLargeByValueAggregate(TyArray[i], PointerBitWidth)) {
          break;
        }
      if (i == n)
        continue;
    }

    ChangedFuncs.push_back(&F);
  }

  for (Function *F : ChangedFuncs)
    generateDebugInfo(M, F, PointerBitWidth);

  return ChangedFuncs.empty() ? PreservedAnalyses::all()
                              : PreservedAnalyses::none();
}
