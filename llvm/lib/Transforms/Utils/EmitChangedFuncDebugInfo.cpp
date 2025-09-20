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

static bool getArg(unsigned Idx, BasicBlock &FirstBB, DIBuilder &DIB,
                   DIFile *NewFile, Function *F, DISubprogram *OldSP,
                   SmallVector<Metadata *, 5> &TypeList,
                   SmallVector<Metadata *, 5> &ArgList,
                   unsigned PointerBitWidth) {
  for (Instruction &I : FirstBB) {
    for (const DbgRecord &DR : I.getDbgRecordRange()) {
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
      } else if (MDNValue != F->getArg(Idx)) {
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

      // Strip modifiers (const, volatile, etc.)
      DIType *DITy = Var->getType();
      while (auto *DTy = dyn_cast<DIDerivedType>(DITy)) {
        if (DTy->getTag() == dwarf::DW_TAG_pointer_type) {
          DITy = DTy;
          break;
        }
        DITy = DTy->getBaseType();
      }

      DIType *ParamType = Var->getType();
      if (Ty->isIntegerTy()) {
        if (auto *DTy = dyn_cast<DICompositeType>(DITy)) {
          if (!Ty->isIntegerTy(DTy->getSizeInBits())) {
            ParamType = getTypeFromExpr(DIB, Expr, DTy);
            if (!ParamType)
              return false;
          }
        }
      } else if (Ty->isPointerTy()) {
        if (dyn_cast<DICompositeType>(DITy)) {
          ParamType = DIB.createPointerType(DITy, PointerBitWidth);
        } else {
          auto *DTy = dyn_cast<DIDerivedType>(DITy);
          if (!DTy)
            continue;
          if (DTy->getTag() != dwarf::DW_TAG_pointer_type)
            continue;
        }
      }

      TypeList.push_back(ParamType);
      if (Var->getArg() != (Idx + 1) ||
          Var->getName() != F->getArg(Idx)->getName()) {
        Var = DIB.createParameterVariable(OldSP, F->getArg(Idx)->getName(),
                                          Idx + 1, OldSP->getUnit()->getFile(),
                                          OldSP->getLine(), ParamType);
      }
      ArgList.push_back(Var);
      return true;
    }
  }

  /* The parameter is not handled due to poison value, so just create a new type
   */
  Type *Ty = F->getArg(Idx)->getType();
  unsigned IntBitWidth = 32;
  if (Ty->isIntegerTy())
    IntBitWidth = cast<IntegerType>(Ty)->getBitWidth();

  DIType *ParamType = DIB.createBasicType("int" + std::to_string(IntBitWidth),
                                          IntBitWidth, dwarf::DW_ATE_signed);
  DILocalVariable *Var =
      DIB.createParameterVariable(OldSP, F->getArg(Idx)->getName(), Idx + 1,
                                  NewFile, OldSP->getLine(), ParamType);
  TypeList.push_back(ParamType);
  ArgList.push_back(Var);
  return true;
}

static bool getTypeArgList(DIBuilder &DIB, DIFile *NewFile, Function *F,
                           FunctionType *FTy, DISubprogram *OldSP,
                           SmallVector<Metadata *, 5> &TypeList,
                           SmallVector<Metadata *, 5> &ArgList,
                           unsigned PointerBitWidth) {
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
    if (!getArg(i, FirstBB, DIB, NewFile, F, OldSP, TypeList, ArgList,
                PointerBitWidth))
      return false;
  }

  return true;
}

static void generateDebugInfo(Module &M, Function *F,
                              unsigned PointerBitWidth) {
  // For this CU, we want generate the following three dwarf units:
  // DW_TAG_compile_unit
  //   ...
  //   // New functions with suffix
  //   DW_TAG_inlined_subroutine
  //     DW_AT_name      ("foo.1")
  //     DW_AT_type      (0x0000000000000091 "int")
  //     DW_AT_artificial (true)
  //     DW_AT_specificiation (original DW_TAG_subprogram)
  //
  //     DW_TAG_formal_parameter
  //       DW_AT_name    ("b")
  //       DW_AT_type    (0x0000000000000091 "int")
  //
  //     DW_TAG_formal_parameter
  //       DW_AT_name    ("c")
  //       DW_AT_type    (0x0000000000000095 "long")
  //   ...
  //   // Functions with changed signatures
  //   DW_TAG_inlined_subroutine
  //     DW_AT_name      ("bar")
  //     DW_AT_type      (0x0000000000000091 "int")
  //     DW_AT_artificial (true)
  //     DW_AT_specificiation (original DW_TAG_subprogram)
  //
  //     DW_TAG_formal_parameter
  //       DW_AT_name    ("c")
  //       DW_AT_type    (0x0000000000000095 "unsigned int")
  //   ...
  //   // Functions not obtained function changed signatures yet
  //   // The DW_CC_nocall presence indicates such cases.
  //   DW_TAG_inlined_subroutine
  //     DW_AT_name      ("bar" or "bar.1")
  //     DW_AT_calling_convention        (DW_CC_nocall)
  //     DW_AT_artificial (true)
  //     DW_AT_specificiation (original DW_TAG_subprogram)
  //   ...

  // A new ComputeUnit is created with file name "<artificial>"
  // to host newly-created DISubprogram's.
  DICompileUnit *NewCU = nullptr;
  NamedMDNode *CUs = M.getNamedMetadata("llvm.dbg.cu");
  for (MDNode *Node : CUs->operands()) {
    auto *CU = cast<DICompileUnit>(Node);
    if (CU->getFile()->getFilename() == "<artificial>") {
      NewCU = CU;
      break;
    }
  }

  DISubprogram *OldSP = F->getSubprogram();
  DIBuilder DIB(M, /*AllowUnresolved=*/false, NewCU);
  DIFile *NewFile;

  if (NewCU) {
    NewFile = NewCU->getFile();
  } else {
    DIFile *OldFile = OldSP->getFile();
    NewFile = DIB.createFile("<artificial>", OldFile->getDirectory());
    NewCU = DIB.createCompileUnit(dwarf::DW_LANG_C, NewFile, "", false, "", 0);
  }

  SmallVector<Metadata *, 5> TypeList;
  SmallVector<Metadata *, 5> ArgList;

  FunctionType *FTy = F->getFunctionType();
  bool Success = getTypeArgList(DIB, NewFile, F, FTy, OldSP, TypeList, ArgList,
                                PointerBitWidth);
  if (!Success) {
    fprintf(stderr, "YHS20 ...\n");
    TypeList.clear();
    TypeList.push_back(nullptr);
    ArgList.clear();
  }

  DITypeRefArray DITypeArray = DIB.getOrCreateTypeArray(TypeList);
  auto *SubroutineType = DIB.createSubroutineType(DITypeArray);
  DINodeArray ArgArray = DIB.getOrCreateArray(ArgList);

  Function *DummyF =
      Function::Create(FTy, GlobalValue::AvailableExternallyLinkage,
                       F->getName() + ".newsig", &M);

  DISubprogram *NewSP =
      DIB.createFunction(OldSP,                   // Scope
                         F->getName(),            // Name
                         OldSP->getLinkageName(), // Linkage name
                         NewFile,                 // File
                         OldSP->getLine(),        // Line
                         SubroutineType,          // DISubroutineType
                         OldSP->getScopeLine(),   // ScopeLine
                         DINode::FlagZero, DISubprogram::SPFlagDefinition);
  NewSP->replaceRetainedNodes(ArgArray);

  if (!Success) {
    auto Temp = NewSP->getType()->cloneWithCC(llvm::dwarf::DW_CC_nocall);
    NewSP->replaceType(MDNode::replaceWithPermanent(std::move(Temp)));
  }

  DIB.finalizeSubprogram(NewSP);

  // Add dummy return block
  BasicBlock *BB = BasicBlock::Create(M.getContext(), "entry", DummyF);
  IRBuilder<> IRB(BB);
  IRB.CreateUnreachable();

  DummyF->setSubprogram(NewSP);

  DIB.finalize();
}

PreservedAnalyses EmitChangedFuncDebugInfoPass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  /* For C only */
  for (DICompileUnit *CU : M.debug_compile_units()) {
    auto L = static_cast<llvm::dwarf::SourceLanguage>(CU->getSourceLanguage());
    if (L != dwarf::DW_LANG_C && L != dwarf::DW_LANG_C89 &&
        L != dwarf::DW_LANG_C99 && L != dwarf::DW_LANG_C11 &&
        L != dwarf::DW_LANG_C17)
      return PreservedAnalyses::all();
  }

  llvm::Triple T(M.getTargetTriple());
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
    // FIXME: workaround for some selftests
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

    ChangedFuncs.push_back(&F);
  }

  bool Changed = ChangedFuncs.size() != 0;
  for (auto *F : ChangedFuncs)
    generateDebugInfo(M, F, PointerBitWidth);

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
