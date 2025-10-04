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

// A struct param breaks into two actual arguments like
//    static int count(struct user_arg_ptr argv, int max)
// and the actual func signature:
//    i32 @count(i8 range(i8 0, 2) %argv.coerce0, ptr %argv.coerce1)
//    {
//      #dbg_value(i8 %argv.coerce0, !14759,
//      !DIExpression(DW_OP_LLVM_fragment, 0, 8), !14768)
//      #dbg_value(ptr %argv.coerce1, !14759,
//      !DIExpression(DW_OP_LLVM_fragment, 64, 64), !14768)
//      ...
//    }
static DIType *getTypeFromExpr(DIBuilder &DIB, DIExpression *Expr,
                               DICompositeType *DTy) {
  for (auto Op : Expr->expr_ops()) {
    if (Op.getOp() != dwarf::DW_OP_LLVM_fragment)
      continue;

    uint64_t BitOffset = Op.getArg(0);
    uint64_t BitSize = Op.getArg(1);

    for (auto *Element : DTy->getElements()) {
      auto Elem = cast<DIDerivedType>(Element);
      if (Elem->getSizeInBits() == BitSize &&
          Elem->getOffsetInBits() == BitOffset)
        return Elem->getBaseType();
      else
        // Create a new int type. For example, original debuginfo is an array.
        return DIB.createBasicType("int" + std::to_string(BitSize), BitSize,
                                   dwarf::DW_ATE_signed);
    }
  }
  return nullptr;
}

static bool getArg(Module &M, unsigned Idx, BasicBlock &FirstBB, DIBuilder &DIB,
                   Function *F, DISubprogram *OldSP, DISubprogram *NewSP,
                   SmallVector<Metadata *, 5> &TypeList,
                   SmallVector<Metadata *, 5> &ArgList,
                   unsigned PointerBitWidth) {
  for (Instruction &I : FirstBB) {
    for (DbgRecord &DR : I.getDbgRecordRange()) {
      auto *DVR = dyn_cast<DbgVariableRecord>(&DR);
      if (!DVR)
        continue;
      // All of DbgVariableRecord::LocationType::{Value,Assign,Declare}
      // are covered.
      Metadata *Loc = DVR->getRawLocation();
      auto *ValueMDN = dyn_cast<ValueAsMetadata>(Loc);
      if (!ValueMDN)
        continue;

      Value *MDNValue = ValueMDN->getValue();
      if (!MDNValue)
        continue;

      Type *Ty = ValueMDN->getType();
      auto *Var = DVR->getVariable();
      if (!Var->getArg())
        continue;

      // Strip modifiers (const, volatile, etc.)
      DIType *DITy = Var->getType();
      while (auto *DTy = dyn_cast<DIDerivedType>(DITy)) {
        if (DTy->getTag() == dwarf::DW_TAG_pointer_type) {
          DITy = DTy;
          break;
        }
        DITy = DTy->getBaseType();
      }

      if (dyn_cast<AllocaInst>(MDNValue)) {
        // A struct turned into a pointer to struct.
        //   @rhashtable_lookup_fast(ptr noundef %key,
        //        ptr noundef readonly byval(%struct.rhashtable_params)
        //        align 8 captures(none) %params) {
        //      ...
        //      %MyAlloca = alloca [160 x i8], align 32
        //      %0 = ptrtoint ptr %MyAlloca to i64
        //      %1 = add i64 %0, 32
        //      %2 = inttoptr i64 %1 to ptr
        //      ...
        //      call void @llvm.memcpy.p0.p0.i64(ptr align 8 %2, ptr align 8
        //                                       %params, i64 40, i1 false)
        //        #dbg_value(ptr @offdevs, !15308, !DIExpression(), !15312)
        //        #dbg_value(ptr %key, !15309, !DIExpression(), !15312)
        //        #dbg_declare(ptr %MyAlloca, !15310,
        //                     !DIExpression(DW_OP_plus_uconst, 32), !15313)
        //      tail call void @__rcu_read_lock() #14, !dbg !15314
        //   }
        if (Var->getName() != F->getArg(Idx)->getName())
          continue;
      } else if (MDNValue != F->getArg(Idx) &&
                 !(Ty->isIntegerTy() && dyn_cast<DIDerivedType>(DITy))) {
        // Ty->isIntegerTy() && dyn_cast<DIDerivedType>(DITy) means that
        // actual type is integer and debug info is a pointer, so it is
        // likely to be caused by argument promotion and will be processed
        // later.
        //
        // Handle the following pattern:
        //   ... @vgacon_do_font_op(..., i32 noundef, i1 noundef zeroext %ch512)
        //   ... {
        //     ...
        //       #dbg_value(i32 %set, !8568, !DIExpression(), !8589)
        //     %storedv = zext i1 %ch512 to i8
        //       #dbg_value(i8 %storedv, !8569, !DIExpression(), !8589)
        //     ...
        //   }
        Instruction *PrevI = I.getPrevNode();
        if (!PrevI)
          continue;
        if (MDNValue != PrevI)
          continue;
        auto *ZExt = dyn_cast<ZExtInst>(PrevI);
        if (!ZExt)
          continue;
        if (ZExt->getOperand(0) != F->getArg(Idx))
          continue;
      }

      auto *Expr = DVR->getExpression();
      DIType *ParamType = Var->getType();
      bool NeedSuffix = false;
      if (Ty->isIntegerTy()) {
        if (auto *DTy = dyn_cast<DICompositeType>(DITy)) {
          if (!Ty->isIntegerTy(DTy->getSizeInBits())) {
            ParamType = getTypeFromExpr(DIB, Expr, DTy);
            if (!ParamType)
              return false;
            NeedSuffix = true;
          }
        } else if (dyn_cast<DIDerivedType>(DITy)) {
          // For argument promotion case where a pointer argument becomes an
          // int.
          ParamType = DIB.createBasicType(
              "int" + std::to_string(Ty->getIntegerBitWidth()),
              Ty->getIntegerBitWidth(), dwarf::DW_ATE_signed);
        }
      } else if (Ty->isPointerTy()) {
        if (dyn_cast<DICompositeType>(DITy)) {
          ParamType = DIB.createPointerType(DITy, PointerBitWidth);
          NeedSuffix = true;
        } else {
          auto *DTy = dyn_cast<DIDerivedType>(DITy);
          if (!DTy)
            continue;
          if (DTy->getTag() != dwarf::DW_TAG_pointer_type)
            continue;
        }
      }

      TypeList.push_back(ParamType);

      std::string ArgName = F->getArg(Idx)->getName().str();
      if (ArgName.empty()) {
        ArgName = Var->getName().str();
        if (NeedSuffix)
          ArgName += std::string("__") + std::to_string(Idx);
      }
      Var = DIB.createParameterVariable(NewSP, StringRef(ArgName), Idx + 1,
                                        OldSP->getUnit()->getFile(),
                                        OldSP->getLine(), ParamType);
      ArgList.push_back(Var);
      return true;
    }
  }

  /* The parameter is not handled due to poison value. There are two cases here:
   *  - the parameter is not used at all. Somehow the parameter is not removed
   *    with dead argument elimination pass.
   *  - the parameter type changed, e.g., from a pointer type to an integer type
   *    due to argument promotion pass.
   * just create a new int type for the argument.
   */
  Type *Ty = F->getArg(Idx)->getType();
  unsigned IntBitWidth = 32;
  if (Ty->isIntegerTy())
    IntBitWidth = cast<IntegerType>(Ty)->getBitWidth();

  DIType *ParamType = DIB.createBasicType("int" + std::to_string(IntBitWidth),
                                          IntBitWidth, dwarf::DW_ATE_signed);
  if (!Ty->isIntegerTy())
    ParamType = DIB.createPointerType(ParamType, PointerBitWidth);

  std::string ArgName = F->getArg(Idx)->getName().str();
  if (ArgName.empty())
    ArgName += std::string("__") + std::to_string(Idx);
  DILocalVariable *Var = DIB.createParameterVariable(
      NewSP, StringRef(ArgName), Idx + 1, OldSP->getUnit()->getFile(),
      OldSP->getLine(), ParamType);
  TypeList.push_back(ParamType);
  ArgList.push_back(Var);
  return true;
}

static bool getTypeArgList(Module &M, DIBuilder &DIB, Function *F,
                           DISubprogram *OldSP, DISubprogram *NewSP,
                           SmallVector<Metadata *, 5> &TypeList,
                           SmallVector<Metadata *, 5> &ArgList,
                           unsigned PointerBitWidth) {
  FunctionType *FTy = F->getFunctionType();
  Type *RetTy = FTy->getReturnType();
  if (RetTy->isVoidTy()) {
    // Void return type may be due to optimization.
    TypeList.push_back(nullptr);
  } else {
    // Optimization does not change return type from one
    // non-void type to another non-void type.
    DITypeRefArray TyArray = OldSP->getType()->getTypeArray();
    TypeList.push_back(TyArray[0]);
  }

  unsigned NumArgs = FTy->getNumParams();
  if (!NumArgs)
    return true;

  BasicBlock &FirstBB = F->getEntryBlock();
  for (unsigned i = 0; i < NumArgs; ++i) {
    if (!getArg(M, i, FirstBB, DIB, F, OldSP, NewSP, TypeList, ArgList,
                PointerBitWidth))
      return false;
  }

  return true;
}

static Metadata *
mapAllDILocs(Metadata *M,
             std::function<const DILocation *(const DILocation *)> X,
             LLVMContext &Ctx) {
  if (!M)
    return nullptr;

  if (auto *DL = dyn_cast<DILocation>(M))
    return const_cast<DILocation *>(X(DL));

  if (auto *N = dyn_cast<MDNode>(M)) {
    SmallVector<Metadata *, 8> NewOps;
    NewOps.reserve(N->getNumOperands());
    for (const MDOperand &Op : N->operands())
      NewOps.push_back(mapAllDILocs(Op.get(), X, Ctx));
    // Tag nodes need not be distinct.
    return MDNode::get(Ctx, NewOps);
  }

  // MDString / ConstantAsMetadata / etc.
  return M;
}

static MDNode *cloneLoopIDReplacingAllDILocs(
    MDNode *OldLoopID, std::function<const DILocation *(const DILocation *)> X,
    LLVMContext &Ctx) {
  SmallVector<Metadata *, 8> Ops;
  Ops.reserve(OldLoopID->getNumOperands());
  Ops.push_back(nullptr); // placeholder for self

  // Copy/transform operands 1..N (operand 0 is always the self reference)
  for (unsigned i = 1, e = OldLoopID->getNumOperands(); i < e; ++i) {
    Metadata *Old = OldLoopID->getOperand(i).get();
    Ops.push_back(mapAllDILocs(Old, X, Ctx));
  }

  MDNode *NewLoopID = MDNode::getDistinct(Ctx, Ops);
  NewLoopID->replaceOperandWith(0, NewLoopID); // self reference
  return NewLoopID;
}

// For a particular function, we do the following three steps:
// 1. Collect new signatures for the function.
// 2. Go through all function body for all DILocations
//    add inlinedAt() for the new function.
// 3. At the beginning of the function, add dbg_value
//    for all actual arguments.
static void generateDebugInfo(Module &M, Function *F,
                              unsigned PointerBitWidth) {
  DISubprogram *OldSP = F->getSubprogram();
  DICompileUnit *CU = OldSP->getUnit();
  DIBuilder DIB(M, /*AllowUnresolved=*/false, CU);

  SmallVector<Metadata *, 5> TypeList;
  SmallVector<Metadata *, 5> ArgList;

  // Collect new signatures for the function.
  DISubprogram *NewSP =
      DIB.createFunction(OldSP->getScope(),     // Scope
                         F->getName(),          // Name
                         F->getName(),          // Linkage name
                         CU->getFile(),         // File
                         OldSP->getLine(),      // Line
                         nullptr,               // DISubroutineType
                         OldSP->getScopeLine(), // ScopeLine
                         DINode::FlagZero, DISubprogram::SPFlagDefinition);
  NewSP = DIB.createArtificialSubprogram(NewSP);

  bool Success = getTypeArgList(M, DIB, F, OldSP, NewSP, TypeList, ArgList,
                                PointerBitWidth);
  if (!Success) {
    DIB.finalize();
    return;
  }

  DITypeRefArray DITypeArray = DIB.getOrCreateTypeArray(TypeList);
  auto *SubroutineType = DIB.createSubroutineType(DITypeArray);
  DINodeArray ArgArray = DIB.getOrCreateArray(ArgList);

  NewSP->replaceType(SubroutineType);
  NewSP->replaceRetainedNodes(ArgArray);

  F->setSubprogram(NewSP);

  // Go through the function itself to replace DILocations.
  LLVMContext &Ctx = M.getContext();
  DILocation *DL2 = DILocation::get(Ctx, 0, 0, NewSP, nullptr, 0, 0);
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      for (DbgRecord &DR : I.getDbgRecordRange()) {
        DebugLoc DL = DR.getDebugLoc();
        auto *OldDL = DL.get();
        SmallVector<DILocation *, 5> DLlist;

        DLlist.push_back(OldDL);
        while (OldDL->getInlinedAt()) {
          OldDL = OldDL->getInlinedAt();
          DLlist.push_back(OldDL);
        }
        DILocation *PrevLoc = DL2;
        for (int i = DLlist.size() - 1; i >= 0; i--) {
          OldDL = DLlist[i];
          PrevLoc = DILocation::get(
              Ctx, OldDL->getLine(), OldDL->getColumn(), OldDL->getScope(),
              PrevLoc, OldDL->isImplicitCode(), OldDL->getAtomGroup(),
              OldDL->getAtomRank());
        }
        DR.setDebugLoc(DebugLoc(const_cast<DILocation *>(PrevLoc)));
      }
      if (DebugLoc DL = I.getDebugLoc()) {
        auto *OldDL = DL.get();
        SmallVector<DILocation *, 5> DLlist;

        DLlist.push_back(OldDL);
        while (OldDL->getInlinedAt()) {
          OldDL = OldDL->getInlinedAt();
          DLlist.push_back(OldDL);
        }
        DILocation *PrevLoc = DL2;
        for (int i = DLlist.size() - 1; i >= 0; i--) {
          OldDL = DLlist[i];
          PrevLoc = DILocation::get(
              Ctx, OldDL->getLine(), OldDL->getColumn(), OldDL->getScope(),
              PrevLoc, OldDL->isImplicitCode(), OldDL->getAtomGroup(),
              OldDL->getAtomRank());
        }
        I.setDebugLoc(DebugLoc(PrevLoc));
      }
      if (MDNode *LoopID = I.getMetadata(LLVMContext::MD_loop)) {
        auto X = [&](const DILocation *OldDL) -> const DILocation * {
          return DILocation::get(Ctx, OldDL->getLine(), OldDL->getColumn(),
                                 OldDL->getScope(), DL2,
                                 OldDL->isImplicitCode(), OldDL->getAtomGroup(),
                                 OldDL->getAtomRank());
        };
        MDNode *New = cloneLoopIDReplacingAllDILocs(LoopID, X, Ctx);
        I.setMetadata(LLVMContext::MD_loop, New);
      }
    }
  }

  // At the beginning of the function, add dbg_values for true func signatures.
  unsigned NumArgs = F->getFunctionType()->getNumParams();
  if (NumArgs) {
    BasicBlock::iterator InsertPt = F->getEntryBlock().getFirstInsertionPt();
    for (int i = NumArgs - 1; i >= 0; --i) {
      DILocalVariable *Var = cast<DILocalVariable>(ArgList[i]);
      DIB.insertDbgValueIntrinsic(F->getArg(i), Var, DIB.createExpression(),
                                  DL2, InsertPt);
    }
  }

  DIB.finalize();
}

PreservedAnalyses EmitChangedFuncDebugInfoPass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  if (DisableChangedFuncDBInfo)
    return PreservedAnalyses::all();

  // For C only
  for (DICompileUnit *CU : M.debug_compile_units()) {
    auto L = static_cast<llvm::dwarf::SourceLanguage>(CU->getSourceLanguage());
    if (L != dwarf::DW_LANG_C && L != dwarf::DW_LANG_C89 &&
        L != dwarf::DW_LANG_C99 && L != dwarf::DW_LANG_C11 &&
        L != dwarf::DW_LANG_C17)
      return PreservedAnalyses::all();
  }

  llvm::Triple T(M.getTargetTriple());

  // FIXME: Skip if BPF target. Unlike other architectures, BPF target will
  // generate BTF in LLVM. We can tune BPF target later.
  if (T.isBPF())
    return PreservedAnalyses::all();

  unsigned PointerBitWidth = T.getArchPointerBitWidth();

  SmallVector<Function *> ChangedFuncs;
  for (auto &F : M) {
    // Function must already have DebugInfo.
    DISubprogram *SP = F.getSubprogram();
    if (!SP)
      continue;

    // Ignore all intrinsics/declare-only functions.
    if (F.isIntrinsic() || F.isDeclaration())
      continue;

    // Skip if the return value is a DICompositeType and its size is greater
    // than PointerBitWidth.
    DITypeRefArray TyArray = SP->getType()->getTypeArray();
    if (TyArray.size() == 0)
      continue;
    DIType *DITy = TyArray[0];
    while (auto *DTy = dyn_cast_or_null<DIDerivedType>(DITy)) {
      if (DTy->getTag() == dwarf::DW_TAG_pointer_type) {
        DITy = DTy;
        break;
      }
      DITy = DTy->getBaseType();
    }
    if (auto *DTy = dyn_cast_or_null<DICompositeType>(DITy)) {
      if (DTy->getSizeInBits() > PointerBitWidth)
        continue;
    }

    // Skip if the func has variable number of arguments
    if (TyArray.size() > 1 && TyArray[TyArray.size() - 1] == nullptr)
      continue;

    // For original functions with struct/union as the argument and
    // if the argument size is greater than 8 bytes, consider this
    // function as signature changed.
    StringRef FName = F.getName();
    if (!FName.contains('.')) {
      uint8_t cc = SP->getType()->getCC();
      if (cc != llvm::dwarf::DW_CC_nocall) {
        bool SigChanged = false;
        for (unsigned i = 1; i < TyArray.size(); ++i) {
          DITy = TyArray[i];
          while (auto *DTy = dyn_cast<DIDerivedType>(DITy)) {
            if (DTy->getTag() == dwarf::DW_TAG_pointer_type) {
              DITy = DTy;
              break;
            }
            DITy = DTy->getBaseType();
          }
          if (auto *DTy = dyn_cast<DICompositeType>(DITy)) {
            if (DTy->getSizeInBits() <= PointerBitWidth)
              continue;
            SigChanged = true;
            break;
          }
        }
        if (!SigChanged)
          continue;
      }
    }

    // Reset calling convention to DW_CC_normal as later the function will
    // be marked as Artificial.
    auto Temp = SP->getType()->cloneWithCC(llvm::dwarf::DW_CC_normal);
    SP->replaceType(MDNode::replaceWithPermanent(std::move(Temp)));

    ChangedFuncs.push_back(&F);
  }

  bool Changed = ChangedFuncs.size() != 0;
  for (auto *F : ChangedFuncs)
    generateDebugInfo(M, F, PointerBitWidth);

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
