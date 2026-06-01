//===-- EJitRegisterBitcode.cpp - EmbeddedJIT Bitcode Extraction ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/EmbeddedJIT/EJitPasses.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;
using namespace llvm::ejit;

#define DEBUG_TYPE "ejit-register-bitcode"

static void collectEntryFunctions(Module &M,
                                  SmallVectorImpl<Function *> &EntryFuncs) {
  for (Function &F : M.functions()) {
    MDNode *MD = F.getMetadata(MD_EJIT_METADATA);
    if (hasMDStringEntry(MD, TAG_EJIT_ENTRY))
      EntryFuncs.push_back(&F);
  }
}

static void collectReferencedGlobals(Function &F,
                                     SetVector<GlobalVariable *> &Globals) {
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      for (Value *Op : I.operands())
        if (auto *GV = dyn_cast<GlobalVariable>(Op->stripPointerCasts()))
          Globals.insert(GV);
}

static void computeTransitiveClosure(
    const SmallVectorImpl<Function *> &EntryFuncs,
    SetVector<Function *> &ClosureFuncs,
    SetVector<GlobalVariable *> &ClosureGlobals) {

  SmallVector<Function *, 16> Worklist(EntryFuncs.begin(), EntryFuncs.end());
  while (!Worklist.empty()) {
    Function *F = Worklist.pop_back_val();
    if (!ClosureFuncs.insert(F))
      continue;
    collectReferencedGlobals(*F, ClosureGlobals);
    for (BasicBlock &BB : *F)
      for (Instruction &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (Function *Callee = CI->getCalledFunction())
            if (!Callee->isDeclaration() && !Callee->isIntrinsic())
              Worklist.push_back(Callee);
  }
}

/// Walk a GEP chain from a load's pointer operand down to the root
/// GlobalVariable, accumulating the total byte offset.
static const GlobalVariable *findRootGV(const Value *V, APInt &Offset,
                                         const DataLayout &DL) {
  Offset = APInt(DL.getPointerSizeInBits(0), 0);
  while (V) {
    V = V->stripPointerCasts();
    if (isa<GlobalVariable>(V))
      return cast<GlobalVariable>(V);
    auto *GEP = dyn_cast<GEPOperator>(V);
    if (!GEP)
      return nullptr;
    SmallVector<Value *, 4> IdxList;
    for (auto I = GEP->idx_begin(), E = GEP->idx_end(); I != E; ++I) {
      if (!isa<ConstantInt>(*I))
        return nullptr;
      IdxList.push_back(*I);
    }
    Offset += DL.getIndexedOffsetInType(GEP->getSourceElementType(), IdxList);
    V = GEP->getPointerOperand();
  }
  return nullptr;
}

/// Re-annotate loads with !ejit.may_const using GV-level offset metadata.
/// Optimization passes may drop per-load metadata; this restores it from
/// the !ejit.may_const_field entries on the GV's !ejit.metadata.
static void reAnnotateMayConst(Module &M) {
  const DataLayout &DL = M.getDataLayout();
  LLVMContext &Ctx = M.getContext();
  auto MayConstKind = Ctx.getMDKindID(MD_EJIT_MAY_CONST);

  // Build offset map from GV metadata
  DenseMap<const GlobalVariable *, SmallVector<uint64_t, 4>> mayConstMap;
  for (GlobalVariable &GV : M.globals()) {
    MDNode *MD = GV.getMetadata(MD_EJIT_METADATA);
    if (!MD)
      continue;
    SmallVector<uint64_t, 4> offsets;
    for (const MDOperand &Op : MD->operands()) {
      auto *Sub = dyn_cast<MDNode>(Op.get());
      if (!Sub || Sub->getNumOperands() < 2)
        continue;
      auto *Tag = dyn_cast<MDString>(Sub->getOperand(0));
      if (!Tag || Tag->getString() != TAG_EJIT_MAY_CONST_FIELD)
        continue;
      if (auto *CI = mdconst::dyn_extract<ConstantInt>(Sub->getOperand(1)))
        offsets.push_back(CI->getZExtValue());
    }
    if (!offsets.empty())
      mayConstMap[&GV] = std::move(offsets);
  }
  if (mayConstMap.empty())
    return;

  // Re-annotate matching loads
  unsigned count = 0;
  for (Function &F : M.functions()) {
    if (F.isDeclaration())
      continue;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto *LI = dyn_cast<LoadInst>(&I);
        if (!LI || LI->hasMetadata(MayConstKind))
          continue;
        APInt Off;
        const GlobalVariable *GV = findRootGV(LI->getPointerOperand(), Off, DL);
        if (!GV)
          continue;
        auto it = mayConstMap.find(GV);
        if (it == mayConstMap.end())
          continue;
        if (is_contained(it->second, Off.getZExtValue())) {
          LI->setMetadata(MayConstKind, MDNode::get(Ctx, {}));
          count++;
        }
      }
    }
  }
  LLVM_DEBUG(dbgs() << "ejit-register-bitcode: re-annotated " << count
                    << " may_const load(s)\n");
}

/// Run pre-optimization on the extracted bitcode at AOT time to reduce
/// JIT compilation pressure. In debug/shared builds this is a no-op
/// (cyclic link dependency: LLVMPasses <-> LLVMEmbeddedJIT).
#ifdef NDEBUG
static void preOptimizeBitcode(Module &M) {
  PassBuilder PB;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerModuleAnalyses(MAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // 1. Inline: AlwaysInline + cost-based inliner for small functions
  {
    ModulePassManager MPM;
    MPM.addPass(AlwaysInlinerPass());
    MPM.addPass(PB.buildModuleInlinerPipeline(
        llvm::OptimizationLevel::O2, ThinOrFullLTOPhase::None));
    MPM.run(M, MAM);
  }

  // 2. Mem2Reg: promote allocas from inlined code to SSA
  {
    FunctionPassManager FPM;
    FPM.addPass(PromotePass());
    for (Function &F : M.functions())
      if (!F.isDeclaration())
        FPM.run(F, FAM);
  }

  // 3. EarlyCSE + InstCombine: simplify and fold redundant computations
  {
    FunctionPassManager FPM;
    FPM.addPass(EarlyCSEPass());
    FPM.addPass(InstCombinePass());
    for (Function &F : M.functions())
      if (!F.isDeclaration())
        FPM.run(F, FAM);
  }

  // 4. SimplifyCFG: flatten branches, merge blocks
  {
    FunctionPassManager FPM;
    FPM.addPass(SimplifyCFGPass());
    for (Function &F : M.functions())
      if (!F.isDeclaration())
        FPM.run(F, FAM);
  }

  // 5. Restore !ejit.may_const metadata that passes may have dropped
  reAnnotateMayConst(M);
}
#else
static void preOptimizeBitcode(Module &) {}
#endif

static std::string extractAndSerialize(Module &M,
    const SetVector<Function *> &Funcs,
    const SetVector<GlobalVariable *> &Globals) {

  auto Extracted = CloneModule(M);

  DenseSet<StringRef> FuncNames;
  for (Function *F : Funcs)
    FuncNames.insert(F->getName());

  DenseSet<StringRef> GlobalNames;
  for (GlobalVariable *GV : Globals)
    GlobalNames.insert(GV->getName());

  SmallVector<Function *, 16> FuncsToDelete;
  for (Function &F : Extracted->functions())
    if (!FuncNames.count(F.getName()))
      FuncsToDelete.push_back(&F);
  for (Function *F : FuncsToDelete) {
    if (F->isDeclaration())
      continue; // Keep declarations (intrinsics, external refs)
    F->replaceAllUsesWith(UndefValue::get(F->getType()));
    F->deleteBody();
    F->eraseFromParent();
  }

  SmallVector<GlobalVariable *, 16> GVToDelete;
  for (GlobalVariable &GV : Extracted->globals())
    if (!GlobalNames.count(GV.getName()))
      GVToDelete.push_back(&GV);
  for (GlobalVariable *GV : GVToDelete) {
    if (GV->isDeclaration())
      continue; // Keep declarations (external refs)
    GV->replaceAllUsesWith(UndefValue::get(GV->getType()));
    GV->eraseFromParent();
  }

  // Pre-optimize the extracted bitcode to reduce JIT compilation pressure.
  // InstCombine + Mem2Reg + SimplifyCFG folds constant chains, promotes
  // allocas, and cleans up dead branches before serialization.
  preOptimizeBitcode(*Extracted);

  // Convert kept non-constant global definitions to external declarations
  // so the JIT linker resolves them from the host process. Constants (e.g.
  // version strings, lookup tables) are kept as-is since they're embedded
  // in the bitcode and don't need external resolution.
  for (GlobalVariable &GV : Extracted->globals()) {
    if (GV.isDeclaration() || GV.isConstant())
      continue;
    GV.setInitializer(nullptr);
    GV.setLinkage(GlobalValue::ExternalLinkage);
  }

  std::string Bitcode;
  raw_string_ostream OS(Bitcode);
  WriteBitcodeToFile(*Extracted, OS);
  OS.flush();
  return Bitcode;
}

static GlobalVariable *embedBitcode(Module &M, const std::string &Bitcode) {
  LLVMContext &Ctx = M.getContext();
  SmallVector<uint8_t, 0> Bytes;
  Bytes.reserve(Bitcode.size());
  for (char C : Bitcode)
    Bytes.push_back(static_cast<uint8_t>(C));

  auto *ArrTy = ArrayType::get(Type::getInt8Ty(Ctx), Bitcode.size());
  auto *Const = ConstantDataArray::get(Ctx, Bytes);
  auto *GV = new GlobalVariable(M, ArrTy, true, GlobalValue::InternalLinkage,
                                Const, GV_EJIT_BITCODE);
  GV->setAlignment(Align(1));
  // Bitcode lives in default section (.rodata for const); no custom section
  // needed — bare-metal environments may not support custom ELF sections.
  return GV;
}

/// Collect external symbols (functions + globals) referenced by the
/// closure and generate ejit_register_symbol calls so the JIT can resolve
/// them without dlsym — suitable for bare-metal embedded environments.
static void generateSymbolRegisters(
    Module &M,
    const SetVector<Function *> &ClosureFuncs,
    Function *AutoReg) {
  LLVMContext &Ctx = M.getContext();
  auto *VoidTy = Type::getVoidTy(Ctx);
  auto *PtrTy = PointerType::getUnqual(Ctx);

  M.getOrInsertFunction("ejit_register_symbol",
      FunctionType::get(VoidTy, {PtrTy, PtrTy}, false));

  std::set<std::string> registered;

  auto isPeriodVar = [&](GlobalVariable &GV) -> bool {
    return GV.hasMetadata(MD_EJIT_METADATA);
  };

  BasicBlock *BB = &AutoReg->getEntryBlock();
  Instruction *InsertBefore = BB->getTerminator();

  for (Function *F : ClosureFuncs) {
    for (BasicBlock &Blk : *F) {
      for (Instruction &I : Blk) {
        // External function calls
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (Function *Callee = CI->getCalledFunction()) {
            if (Callee->isDeclaration() && !Callee->isIntrinsic()) {
              std::string Name = Callee->getName().str();
              if (registered.insert(Name).second) {
                IRBuilder<> Builder(InsertBefore);
                Builder.CreateCall(M.getFunction("ejit_register_symbol"),
                    {Builder.CreateGlobalString(Name),
                     Builder.CreateBitCast(Callee, PtrTy)});
              }
            }
          }
        }
        // External global variable references. Skip constants (compiler-
        // generated strings etc.) — they're embedded in the bitcode.
        for (Use &U : I.operands()) {
          if (auto *GV = dyn_cast<GlobalVariable>(U.get())) {
            if (GV->isConstant())
              continue;
            if (GV->isDeclaration() || !isPeriodVar(*GV)) {
              std::string Name = GV->getName().str();
              if (registered.insert(Name).second) {
                IRBuilder<> Builder(InsertBefore);
                Builder.CreateCall(M.getFunction("ejit_register_symbol"),
                    {Builder.CreateGlobalString(Name),
                     Builder.CreateBitCast(GV, PtrTy)});
              }
            }
          }
        }
      }
    }
  }
}

static void
generateRegistryTable(Module &M, const SmallVectorImpl<Function *> &EntryFuncs,
                      const SetVector<Function *> &ClosureFuncs,
                      GlobalVariable *BitcodeGV);

static void generateRegisterCall(Module &M, GlobalVariable *BitcodeGV,
                                 const SmallVectorImpl<Function *> &EntryFuncs,
                                 const SetVector<Function *> &ClosureFuncs) {
  LLVMContext &Ctx = M.getContext();
  auto *VoidTy = Type::getVoidTy(Ctx);
  auto *PtrTy = PointerType::getUnqual(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  M.getOrInsertFunction(FN_REGISTER_BITCODE,
      FunctionType::get(VoidTy, {PtrTy, PtrTy, I64Ty}, false));

  Function *AutoReg = M.getFunction(FN_AUTO_REGISTER);
  if (!AutoReg) {
    AutoReg = Function::Create(FunctionType::get(VoidTy, false),
                               GlobalValue::InternalLinkage,
                               FN_AUTO_REGISTER, &M);
    BasicBlock::Create(Ctx, "entry", AutoReg);
    ReturnInst::Create(Ctx, &AutoReg->getEntryBlock());
  }

  BasicBlock *EntryBB = &AutoReg->getEntryBlock();
  Instruction *Ret = EntryBB->getTerminator();
  FunctionCallee Callee = M.getFunction(FN_REGISTER_BITCODE);

  for (Function *F : EntryFuncs) {
    IRBuilder<> Builder(Ret);
    Builder.CreateCall(Callee, {
        Builder.CreateGlobalString(F->getName()),
        Builder.CreateBitCast(BitcodeGV, PtrTy),
        ConstantInt::get(I64Ty, BitcodeGV->getValueType()->getArrayNumElements())
    });
  }

  // Auto-register external symbols referenced by the closure so the JIT
  // can resolve them without manual ejit_register_symbol calls.
  generateSymbolRegisters(M, ClosureFuncs, AutoReg);

  appendToGlobalCtors(M, AutoReg, EJIT_CTOR_PRIORITY);

  // Also build the static registry table for bare-metal fallback.
  generateRegistryTable(M, EntryFuncs, ClosureFuncs, BitcodeGV);
}

/// Build a global constant array __ejit_registry_bitcode[] that ejit_init()
/// walks on bare-metal where global constructors are unavailable.
static void
generateRegistryTable(Module &M, const SmallVectorImpl<Function *> &EntryFuncs,
                      const SetVector<Function *> &ClosureFuncs,
                      GlobalVariable *BitcodeGV) {
  LLVMContext &Ctx = M.getContext();
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *PtrTy = PointerType::getUnqual(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  // Struct: { i32 type, ptr name1, ptr name2, ptr data, i64 size }
  StructType *EntryTy = StructType::get(
      Ctx, {I32Ty, PtrTy, PtrTy, PtrTy, I64Ty}, /*isPacked=*/false);

  SmallVector<Constant *, 16> Entries;

  // Bitcode entries
  for (Function *F : EntryFuncs) {
    Entries.push_back(ConstantStruct::get(EntryTy, {
        ConstantInt::get(I32Ty, 0),                          // EJIT_REG_BITCODE
        ConstantExpr::getBitCast(
            M.getOrInsertGlobal(F->getName(), PtrTy), PtrTy),// global string name1
        ConstantPointerNull::get(PtrTy),                     // name2 = NULL
        ConstantExpr::getBitCast(BitcodeGV, PtrTy),          // bitcode data ptr
        ConstantInt::get(I64Ty,
            BitcodeGV->getValueType()->getArrayNumElements()),// bitcode size
    }));
  }

  // Symbol entries for external references
  SmallPtrSet<const Function *, 8> SymbolsDone;
  auto addSymbol = [&](const Function *F) {
    if (F->isIntrinsic() || F->isDeclaration()) {
      if (SymbolsDone.insert(F).second) {
        Entries.push_back(ConstantStruct::get(EntryTy, {
            ConstantInt::get(I32Ty, 3),                      // EJIT_REG_SYMBOL
            ConstantExpr::getBitCast(
                M.getOrInsertGlobal(F->getName(), PtrTy), PtrTy),
            ConstantPointerNull::get(PtrTy),
            ConstantExpr::getBitCast(const_cast<Function *>(F), PtrTy),
            ConstantInt::get(I64Ty, 0),
        }));
      }
    }
  };
  for (Function *F : ClosureFuncs) {
    for (const BasicBlock &BB : *F) {
      for (const Instruction &I : BB) {
        if (const CallBase *CB = dyn_cast<CallBase>(&I))
          addSymbol(const_cast<CallBase *>(CB)->getCalledFunction());
      }
    }
  }

  // Global variable symbol entries
  SmallPtrSet<const GlobalVariable *, 4> GVsDone;
  for (Function *F : ClosureFuncs) {
    for (const BasicBlock &BB : *F) {
      for (const Instruction &I : BB) {
        for (const Value *Op : I.operands()) {
          if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(Op)) {
            if (!GV->isConstant() && !GV->getName().starts_with("llvm.") &&
                GVsDone.insert(GV).second) {
              Entries.push_back(ConstantStruct::get(EntryTy, {
                  ConstantInt::get(I32Ty, 3),                // EJIT_REG_SYMBOL
                  ConstantExpr::getBitCast(
                      M.getOrInsertGlobal(GV->getName(), PtrTy), PtrTy),
                  ConstantPointerNull::get(PtrTy),
                  ConstantExpr::getBitCast(
                      const_cast<GlobalVariable *>(GV), PtrTy),
                  ConstantInt::get(I64Ty, 0),
              }));
            }
          }
        }
      }
    }
  }

  // Sentinel entry
  Entries.push_back(ConstantStruct::get(EntryTy, {
      ConstantInt::get(I32Ty, 4),                            // EJIT_REG_NONE
      ConstantPointerNull::get(PtrTy),
      ConstantPointerNull::get(PtrTy),
      ConstantPointerNull::get(PtrTy),
      ConstantInt::get(I64Ty, 0),
  }));

  ArrayType *ArrayTy = ArrayType::get(EntryTy, Entries.size());
  Constant *ArrayInit = ConstantArray::get(ArrayTy, Entries);

  (void)new GlobalVariable(M, ArrayTy, /*isConstant=*/true,
                           GlobalValue::ExternalLinkage, ArrayInit,
                           "__ejit_registry_bitcode");
}

PreservedAnalyses
EJitRegisterBitcodePass::run(Module &M, ModuleAnalysisManager &) {
  LLVM_DEBUG(dbgs() << "ejit-register-bitcode: running on " << M.getName() << "\n");
  SmallVector<Function *, 4> EntryFuncs;
  collectEntryFunctions(M, EntryFuncs);
  if (EntryFuncs.empty()) {
    LLVM_DEBUG(dbgs() << "ejit-register-bitcode: no entry functions, skip\n");
    return PreservedAnalyses::all();
  }

  SetVector<Function *> ClosureFuncs;
  SetVector<GlobalVariable *> ClosureGlobals;
  computeTransitiveClosure(EntryFuncs, ClosureFuncs, ClosureGlobals);
  LLVM_DEBUG(dbgs() << "ejit-register-bitcode: closure " << ClosureFuncs.size()
                    << " funcs, " << ClosureGlobals.size() << " globals\n");
  if (ClosureFuncs.empty())
    return PreservedAnalyses::all();

  std::string Bitcode = extractAndSerialize(M, ClosureFuncs, ClosureGlobals);
  GlobalVariable *BitcodeGV = embedBitcode(M, Bitcode);
  generateRegisterCall(M, BitcodeGV, EntryFuncs, ClosureFuncs);

  return PreservedAnalyses::none();
}
