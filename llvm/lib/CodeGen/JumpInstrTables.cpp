//===-- JumpInstrTables.cpp: Jump-Instruction Tables ----------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief An implementation of jump-instruction tables.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jt"

#include "llvm/CodeGen/JumpInstrTables.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/JumpInstrTableInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

using namespace llvm;

char JumpInstrTables::ID = 0;

INITIALIZE_PASS_BEGIN(JumpInstrTables, "jump-instr-tables",
                      "Jump-Instruction Tables", true, true)
INITIALIZE_PASS_DEPENDENCY(JumpInstrTableInfo);
INITIALIZE_PASS_END(JumpInstrTables, "jump-instr-tables",
                    "Jump-Instruction Tables", true, true)

STATISTIC(NumJumpTables, "Number of indirect call tables generated");
STATISTIC(NumFuncsInJumpTables, "Number of functions in the jump tables");

ModulePass *llvm::createJumpInstrTablesPass() {
  // The default implementation uses a single table for all functions.
  return new JumpInstrTables(JumpTable::Single);
}

ModulePass *llvm::createJumpInstrTablesPass(JumpTable::JumpTableType JTT) {
  return new JumpInstrTables(JTT);
}

namespace {
static const char jump_func_prefix[] = "__llvm_jump_instr_table_";
static const char jump_section_prefix[] = ".jump.instr.table.text.";

// Checks to see if a given CallSite is making an indirect call, including
// cases where the indirect call is made through a bitcast.
bool isIndirectCall(CallSite &CS) {
  if (CS.getCalledFunction())
    return false;

  // Check the value to see if it is merely a bitcast of a function. In
  // this case, it will translate to a direct function call in the resulting
  // assembly, so we won't treat it as an indirect call here.
  const Value *V = CS.getCalledValue();
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    return !(CE->isCast() && isa<Function>(CE->getOperand(0)));
  }

  // Otherwise, since we know it's a call, it must be an indirect call
  return true;
}

// Replaces Functions and GlobalAliases with a different Value.
bool replaceGlobalValueIndirectUse(GlobalValue *GV, Value *V, Use *U) {
  User *Us = U->getUser();
  if (!Us)
    return false;
  if (Instruction *I = dyn_cast<Instruction>(Us)) {
    CallSite CS(I);

    // Don't do the replacement if this use is a direct call to this function.
    // If the use is not the called value, then replace it.
    if (CS && (isIndirectCall(CS) || CS.isCallee(U))) {
      return false;
    }

    U->set(V);
  } else if (Constant *C = dyn_cast<Constant>(Us)) {
    // Don't replace calls to bitcasts of function symbols, since they get
    // translated to direct calls.
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Us)) {
      if (CE->getOpcode() == Instruction::BitCast) {
        // This bitcast must have exactly one user.
        if (CE->user_begin() != CE->user_end()) {
          User *ParentUs = *CE->user_begin();
          if (CallInst *CI = dyn_cast<CallInst>(ParentUs)) {
            CallSite CS(CI);
            Use &CEU = *CE->use_begin();
            if (CS.isCallee(&CEU)) {
              return false;
            }
          }
        }
      }
    }

    // GlobalAlias doesn't support replaceUsesOfWithOnConstant. And the verifier
    // requires alias to point to a defined function. So, GlobalAlias is handled
    // as a separate case in runOnModule.
    if (!isa<GlobalAlias>(C))
      C->replaceUsesOfWithOnConstant(GV, V, U);
  } else {
    assert(false && "The Use of a Function symbol is neither an instruction nor"
                    " a constant");
  }

  return true;
}

// Replaces all replaceable address-taken uses of GV with a pointer to a
// jump-instruction table entry.
void replaceValueWithFunction(GlobalValue *GV, Function *F) {
  // Go through all uses of this function and replace the uses of GV with the
  // jump-table version of the function. Get the uses as a vector before
  // replacing them, since replacing them changes the use list and invalidates
  // the iterator otherwise.
  for (Value::use_iterator I = GV->use_begin(), E = GV->use_end(); I != E;) {
    Use &U = *I++;

    // Replacement of constants replaces all instances in the constant. So, some
    // uses might have already been handled by the time we reach them here.
    if (U.get() == GV)
      replaceGlobalValueIndirectUse(GV, F, &U);
  }

  return;
}
} // end anonymous namespace

JumpInstrTables::JumpInstrTables()
    : ModulePass(ID), Metadata(), JITI(nullptr), TableCount(0),
      JTType(JumpTable::Single) {
  initializeJumpInstrTablesPass(*PassRegistry::getPassRegistry());
}

JumpInstrTables::JumpInstrTables(JumpTable::JumpTableType JTT)
    : ModulePass(ID), Metadata(), JITI(nullptr), TableCount(0), JTType(JTT) {
  initializeJumpInstrTablesPass(*PassRegistry::getPassRegistry());
}

JumpInstrTables::~JumpInstrTables() {}

void JumpInstrTables::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<JumpInstrTableInfo>();
}

Function *JumpInstrTables::insertEntry(Module &M, Function *Target) {
  FunctionType *OrigFunTy = Target->getFunctionType();
  FunctionType *FunTy = transformType(OrigFunTy);

  JumpMap::iterator it = Metadata.find(FunTy);
  if (Metadata.end() == it) {
    struct TableMeta Meta;
    Meta.TableNum = TableCount;
    Meta.Count = 0;
    Metadata[FunTy] = Meta;
    it = Metadata.find(FunTy);
    ++NumJumpTables;
    ++TableCount;
  }

  it->second.Count++;

  std::string NewName(jump_func_prefix);
  NewName += (Twine(it->second.TableNum) + "_" + Twine(it->second.Count)).str();
  Function *JumpFun =
      Function::Create(OrigFunTy, GlobalValue::ExternalLinkage, NewName, &M);
  // The section for this table
  JumpFun->setSection((jump_section_prefix + Twine(it->second.TableNum)).str());
  JITI->insertEntry(FunTy, Target, JumpFun);

  ++NumFuncsInJumpTables;
  return JumpFun;
}

bool JumpInstrTables::hasTable(FunctionType *FunTy) {
  FunctionType *TransTy = transformType(FunTy);
  return Metadata.end() != Metadata.find(TransTy);
}

FunctionType *JumpInstrTables::transformType(FunctionType *FunTy) {
  // Returning nullptr forces all types into the same table, since all types map
  // to the same type
  Type *VoidPtrTy = Type::getInt8PtrTy(FunTy->getContext());

  // Ignore the return type.
  Type *RetTy = VoidPtrTy;
  bool IsVarArg = FunTy->isVarArg();
  std::vector<Type *> ParamTys(FunTy->getNumParams());
  FunctionType::param_iterator PI, PE;
  int i = 0;

  std::vector<Type *> EmptyParams;
  Type *Int32Ty = Type::getInt32Ty(FunTy->getContext());
  FunctionType *VoidFnTy = FunctionType::get(
      Type::getVoidTy(FunTy->getContext()), EmptyParams, false);
  switch (JTType) {
  case JumpTable::Single:

    return FunctionType::get(RetTy, EmptyParams, false);
  case JumpTable::Arity:
    // Transform all types to void* so that all functions with the same arity
    // end up in the same table.
    for (PI = FunTy->param_begin(), PE = FunTy->param_end(); PI != PE;
         PI++, i++) {
      ParamTys[i] = VoidPtrTy;
    }

    return FunctionType::get(RetTy, ParamTys, IsVarArg);
  case JumpTable::Simplified:
    // Project all parameters types to one of 3 types: composite, integer, and
    // function, matching the three subclasses of Type.
    for (PI = FunTy->param_begin(), PE = FunTy->param_end(); PI != PE;
         ++PI, ++i) {
      assert((isa<IntegerType>(*PI) || isa<FunctionType>(*PI) ||
              isa<CompositeType>(*PI)) &&
             "This type is not an Integer or a Composite or a Function");
      if (isa<CompositeType>(*PI)) {
        ParamTys[i] = VoidPtrTy;
      } else if (isa<FunctionType>(*PI)) {
        ParamTys[i] = VoidFnTy;
      } else if (isa<IntegerType>(*PI)) {
        ParamTys[i] = Int32Ty;
      }
    }

    return FunctionType::get(RetTy, ParamTys, IsVarArg);
  case JumpTable::Full:
    // Don't transform this type at all.
    return FunTy;
  }

  return nullptr;
}

bool JumpInstrTables::runOnModule(Module &M) {
  JITI = &getAnalysis<JumpInstrTableInfo>();

  // Get the set of jumptable-annotated functions.
  DenseMap<Function *, Function *> Functions;
  for (Function &F : M) {
    if (F.hasFnAttribute(Attribute::JumpTable)) {
      assert(F.hasUnnamedAddr() &&
             "Attribute 'jumptable' requires 'unnamed_addr'");
      Functions[&F] = nullptr;
    }
  }

  // Create the jump-table functions.
  for (auto &KV : Functions) {
    Function *F = KV.first;
    KV.second = insertEntry(M, F);
  }

  // GlobalAlias is a special case, because the target of an alias statement
  // must be a defined function. So, instead of replacing a given function in
  // the alias, we replace all uses of aliases that target jumptable functions.
  // Note that there's no need to create these functions, since only aliases
  // that target known jumptable functions are replaced, and there's no way to
  // put the jumptable annotation on a global alias.
  DenseMap<GlobalAlias *, Function *> Aliases;
  for (GlobalAlias &GA : M.aliases()) {
    Constant *Aliasee = GA.getAliasee();
    if (Function *F = dyn_cast<Function>(Aliasee)) {
      auto it = Functions.find(F);
      if (it != Functions.end()) {
        Aliases[&GA] = it->second;
      }
    }
  }

  // Replace each address taken function with its jump-instruction table entry.
  for (auto &KV : Functions)
    replaceValueWithFunction(KV.first, KV.second);

  for (auto &KV : Aliases)
    replaceValueWithFunction(KV.first, KV.second);

  return !Functions.empty();
}
