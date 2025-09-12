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

using namespace llvm;

static bool getArg(BasicBlock &FirstBB, unsigned Idx, DIBuilder &DIB,
                   DIFile *NewFile, Function *F, DISubprogram *OldSP,
                   SmallVector<Metadata *, 5> &TypeList,
                   SmallVector<Metadata *, 5> &ArgList) {
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

      // A poison value may correspond to a unused argument.
      if (isa<PoisonValue>(MDNValue)) {
        Type *Ty = ValueMDN->getType();
        auto *Var = dyn_cast<DILocalVariable>(DVR->getRawVariable());
        if (!Var || Var->getArg() != (Idx + 1))
          continue;

        // Check for cases like below due to ArgumentPromotion
        //   define internal ... i32 @add42_byref(i32 %p.0.val) ... {
        //     #dbg_value(ptr poison, !17, !DIExpression(), !18)
        //     ...
        //   }
        // TODO: one pointer expands to more than one argument is not
        // supported yet. For example,
        //   define internal ... i32 @add42_byref(i32 %p.0.val, i32 %p.4.val)
        //   ...
        if (Ty->isPointerTy() && F->getArg(Idx)->getType()->isIntegerTy()) {
          // For such cases, a new argument is created.
          auto *IntTy = cast<IntegerType>(F->getArg(Idx)->getType());
          unsigned IntBitWidth = IntTy->getBitWidth();

          DIType *IntDIType =
              DIB.createBasicType("int" + std::to_string(IntBitWidth),
                                  IntBitWidth, dwarf::DW_ATE_signed);
          Var = DIB.createParameterVariable(OldSP, F->getArg(Idx)->getName(),
                                            Idx + 1, NewFile, OldSP->getLine(),
                                            IntDIType);
        }

        TypeList.push_back(Var->getType());
        ArgList.push_back(Var);
        return true;
      }

      // Handle the following pattern:
      //   ... @vgacon_do_font_op(..., i32 noundef, i1 noundef zeroext %ch512)
      //   ... {
      //     ...
      //       #dbg_value(i32 %set, !8568, !DIExpression(), !8589)
      //     %storedv = zext i1 %ch512 to i8
      //       #dbg_value(i8 %storedv, !8569, !DIExpression(), !8589)
      //     ...
      //   }
      if (MDNValue != F->getArg(Idx)) {
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

      auto *Var = cast<DILocalVariable>(DVR->getRawVariable());

      // Even we get dbg_*(...) for arguments, we still need to ensure
      // compatible types between IR func argument types and debugInfo argument
      // types.
      Type *Ty = ValueMDN->getType();
      DIType *DITy = Var->getType();
      while (auto *DTy = dyn_cast<DIDerivedType>(DITy)) {
        if (DTy->getTag() == dwarf::DW_TAG_pointer_type) {
          DITy = DTy;
          break;
        }
        DITy = DTy->getBaseType();
      }

      if (Ty->isIntegerTy()) {
        if (auto *DTy = dyn_cast<DICompositeType>(DITy)) {
          if (!Ty->isIntegerTy(DTy->getSizeInBits())) {
            // TODO: A struct param breaks into two actual arguments like
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
            return false;
          }
        }
      } else if (Ty->isPointerTy()) {
        // TODO: A struct turned into a pointer to struct.
        //   @rhashtable_lookup_fast(ptr noundef %key,
        //      ptr noundef readonly byval(%struct.rhashtable_params)
        //        align 8 captures(none) %params) {
        //      ...
        //      %MyAlloca = alloca [160 x i8], align 32
        //      %0 = ptrtoint ptr %MyAlloca to i64
        //      %1 = add i64 %0, 32
        //      %2 = inttoptr i64 %1 to ptr
        //      ...
        //      call void @llvm.memcpy.p0.p0.i64(ptr align 8 %2, ptr align 8
        //      %params, i64 40, i1 false)
        //        #dbg_value(ptr @offdevs, !15308, !DIExpression(), !15312)
        //        #dbg_value(ptr %key, !15309, !DIExpression(), !15312)
        //        #dbg_declare(ptr %MyAlloca, !15310,
        //        !DIExpression(DW_OP_plus_uconst, 32), !15313)
        //      tail call void @__rcu_read_lock() #14, !dbg !15314
        //   }
        if (dyn_cast<DICompositeType>(DITy))
          return false;

        auto *DTy = dyn_cast<DIDerivedType>(DITy);
        if (!DTy)
          continue;
        if (DTy->getTag() != dwarf::DW_TAG_pointer_type)
          continue;
      }

      TypeList.push_back(Var->getType());
      if (Var->getArg() != (Idx + 1) ||
          Var->getName() != F->getArg(Idx)->getName()) {
        Var = DIB.createParameterVariable(OldSP, F->getArg(Idx)->getName(),
                                          Idx + 1, OldSP->getUnit()->getFile(),
                                          OldSP->getLine(), Var->getType());
      }
      ArgList.push_back(Var);
      return true;
    }
  }

  return false;
}

static bool getTypeArgList(DIBuilder &DIB, DIFile *NewFile, Function *F,
                           FunctionType *FTy, DISubprogram *OldSP,
                           SmallVector<Metadata *, 5> &TypeList,
                           SmallVector<Metadata *, 5> &ArgList) {
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
  BasicBlock &FirstBB = F->getEntryBlock();
  for (unsigned i = 0; i < NumArgs; ++i) {
    if (!getArg(FirstBB, i, DIB, NewFile, F, OldSP, TypeList, ArgList))
      return false;
  }

  return true;
}

static void generateDebugInfo(Module &M, Function *F) {
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
    DICompileUnit *OldCU = OldSP->getUnit();
    DIFile *OldFile = OldCU->getFile();
    NewFile = DIB.createFile("<artificial>", OldFile->getDirectory());
    NewCU = DIB.createCompileUnit(
        OldCU->getSourceLanguage(), NewFile, OldCU->getProducer(),
        OldCU->isOptimized(), OldCU->getFlags(), OldCU->getRuntimeVersion());
  }

  SmallVector<Metadata *, 5> TypeList;
  SmallVector<Metadata *, 5> ArgList;

  FunctionType *FTy = F->getFunctionType();
  bool Success = getTypeArgList(DIB, NewFile, F, FTy, OldSP, TypeList, ArgList);
  if (!Success) {
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

  SmallVector<Function *> ChangedFuncs;
  for (auto &F : M) {
    // Function must already have DebugInfo.
    DISubprogram *SP = F.getSubprogram();
    if (!SP)
      continue;

    // Ignore all intrinsics functions.
    if (F.isIntrinsic())
      continue;

    StringRef FName = F.getName();
    if (!FName.contains('.')) {
      uint8_t cc = SP->getType()->getCC();
      if (cc != llvm::dwarf::DW_CC_nocall)
        continue;
    }

    ChangedFuncs.push_back(&F);
  }

  bool Changed = ChangedFuncs.size() != 0;
  for (auto *F : ChangedFuncs)
    generateDebugInfo(M, F);

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
