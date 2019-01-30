//===- ASFixer.cpp - Address spaces fixer pass implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the address spaces fixer pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/ConstantFolder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/SYCL/ASFixer.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>
#include <stack>
#include <vector>

using namespace llvm;

namespace {

typedef std::pair<Value *, Value *> ValueValuePair;
typedef std::stack<ValueValuePair, std::vector<ValueValuePair>> WorkListType;
typedef DenseMap<Value *, Value *> ValueToValueMap;
typedef DenseMap<Type *, Type *> TypeToTypeMap;

typedef SmallVector<std::pair<unsigned, Value *>, 32> OperandsVector;

// Contains Instruction stub and it's future operands
typedef DenseMap<User *, OperandsVector> UserToOperandsMap;

typedef DenseMap<Function *, FunctionType *> FunctionToNewTypeMap;

enum SPIRAddressSpace {
  SPIRAS_Private,
  SPIRAS_Global,
  SPIRAS_Constant,
  SPIRAS_Local,
  SPIRAS_Generic,
};

static CallInst *createCallInstStub(FunctionType *FTy, Function *Func) {
  SmallVector<Value *, 8> Args;
  for (auto &Arg : FTy->params()) {
    Args.push_back(UndefValue::get(Arg));
  }
  CallInst *Result = CallInst::Create(Func, Args, "");
  Result->setCallingConv(Func->getCallingConv());
  return Result;
}

static Function *createNewFunction(Function *ExistingF, FunctionType *NewFTy,
                                   ValueToValueMap &VMap,
                                   UserToOperandsMap &UOpMap) {

  Function *Replacement =
      Function::Create(NewFTy, ExistingF->getLinkage(),
                       "new." + ExistingF->getName(), ExistingF->getParent());
  Replacement->setCallingConv(ExistingF->getCallingConv());
  ValueToValueMapTy CloneMap;
  auto ExistFArgIt = ExistingF->arg_begin();
  auto RArgIt = Replacement->arg_begin();
  auto RArgItE = Replacement->arg_end();
  for (; RArgIt != RArgItE; ++RArgIt, ++ExistFArgIt) {
    CloneMap[&*ExistFArgIt] = &*RArgIt;
    for (auto &ArgUse : ExistFArgIt->uses()) {
      Value *Usr = ArgUse.getUser();
      Value *NewVal = VMap[Usr];
      if (NewVal) {
        UOpMap[cast<User>(NewVal)].push_back(
            std::make_pair(ArgUse.getOperandNo(), RArgIt));
      }
    }
  }

  SmallVector<ReturnInst *, 4> Returns;
  CloneFunctionInto(Replacement, ExistingF, CloneMap, true, Returns, "");
  assert(Replacement && "CloneFunctionInto failed");

  // Remap new instructions to clones.
  SmallVector<Instruction *, 32> NewInsts;
  for (auto It : CloneMap) {
    if (isa<Instruction>(It.first)) {
      Instruction *OldInstClone = cast<Instruction>(It.second);
      Value *NewVal = VMap[const_cast<Value *>(It.first)];
      if (NewVal) {
        VMap[OldInstClone] = NewVal;
        auto NewInst = cast<Instruction>(NewVal);
        if (!NewInst->getParent()) {
          NewInst->insertBefore(OldInstClone);
        }
        NewInsts.push_back(NewInst);
      }
      VMap.erase(const_cast<Value *>(It.first));
    }
  }
  // If new instruction uses some old instructions -
  // it should use old instructions clones
  for (auto NI : NewInsts) {
    for (auto &OpIt : NI->operands()) {
      Value *OpVal = OpIt;
      if (CloneMap[OpVal])
        OpIt.set(CloneMap[OpVal]);
    }
  }
  return Replacement;
}

static Value *createTypeUserStub(Value *OldValUser, Value *NewVal) {
  Type *NewTy = NewVal->getType();
  auto *UndefType = UndefValue::get(NewTy);

  auto *OldInst = cast<Instruction>(OldValUser);

  if (auto *Alloca = dyn_cast<AllocaInst>(OldInst)) {
    return new AllocaInst(NewTy, Alloca->getType()->getAddressSpace(),
                          Alloca->getArraySize(), "new." + Alloca->getName());
  }

  if (auto *Store = dyn_cast<StoreInst>(OldInst)) {
    Value *DestPtr = UndefValue::get(PointerType::getUnqual(NewTy));
    auto SI = new StoreInst(UndefType, DestPtr, Store->isVolatile(),
                            Store->getAlignment(), Store->getOrdering(),
                            Store->getSyncScopeID());
    return SI;
  }

  if (auto *GEP = dyn_cast<GetElementPtrInst>(OldInst)) {
    SmallVector<Value *, 8> IdxList(GEP->idx_begin(), GEP->idx_end());

    return GetElementPtrInst::Create(NewTy->getPointerElementType(), UndefType,
                                     IdxList, OldInst->getName());
  }

  if (auto *Load = dyn_cast<LoadInst>(OldInst)) {
    return new LoadInst(NewTy->getPointerElementType(), UndefType,
                        Load->getName(), Load->isVolatile());
  }
  if (auto *Select = dyn_cast<SelectInst>(OldInst)) {
    return SelectInst::Create(Select->getCondition(), UndefType, UndefType,
                              OldInst->getName());
  }

  if (auto *Phi = dyn_cast<PHINode>(OldInst)) {
    auto *NewPhi =
        PHINode::Create(NewTy, Phi->getNumIncomingValues(), OldInst->getName());
    for (auto *BB : Phi->blocks()) {
      NewPhi->addIncoming(UndefType, BB);
    }
    return NewPhi;
  }

  if (auto *Ret = dyn_cast<ReturnInst>(OldInst)) {
    return ReturnInst::Create(Ret->getContext(), UndefType);
  }

  if (auto *BCast = dyn_cast<BitCastInst>(OldInst)) {
    unsigned AS = NewTy->getPointerAddressSpace();
    return new BitCastInst(
        UndefType,
        BCast->getDestTy()->getPointerElementType()->getPointerTo(AS),
        OldInst->getName());
  }

  if (auto *Shuffle = dyn_cast<ShuffleVectorInst>(OldInst))
    return Shuffle->clone();

  if (auto *BinOp = dyn_cast<BinaryOperator>(OldInst))
    return BinOp->clone();

  llvm_unreachable("Unsupported instruction.");
}

static Value *getAllocaOrArgValue(Function *F, const unsigned ArgNo) {
  assert(ArgNo < F->arg_size() && "Invalid ArgNo");
  auto Arg = F->arg_begin() + ArgNo;
  for (auto *ArgUser : Arg->users()) {
    if (auto *Store = dyn_cast<StoreInst>(ArgUser)) {
      Value *POperand = Store->getPointerOperand();
      assert(isa<AllocaInst>(POperand) && "Expected alloca for argument");
      return POperand;
    }
  }
  return Arg;
}

static bool checkFunctionArgument(FunctionType *FTy, Type *Ty,
                                  const unsigned ArgNo) {
  assert(ArgNo < FTy->getNumParams() && "Invalid ArgNo");
  return (FTy->getParamType(ArgNo) == Ty);
}

static bool checkFunctionRetType(FunctionType *FTy, Type *Ty) {
  return (FTy->getReturnType() == Ty);
}

static FunctionType *createNewFuncReplacementType(FunctionType *FTy, Type *Ty,
                                                  const unsigned ArgNo) {
  SmallVector<Type *, 16> Args;
  for (auto P : FTy->params()) {
    Args.push_back(P);
  }
  Args[ArgNo] = Ty;
  return FunctionType::get(FTy->getReturnType(), Args, FTy->isVarArg());
}

static FunctionType *createNewFuncReplacementType(FunctionType *FTy, Type *Ty) {
  return FunctionType::get(Ty, FTy->params(), FTy->isVarArg());
}

static bool valueIsReplacement(ValueToValueMap &VMap, Value *V) {
  for (auto It : VMap) {
    if (It.second == V)
      return true;
  }
  return false;
}

static void collectTypeReplacementData(Type *OldTy, Type *NewTy,
                                       ValueToValueMap &VMap,
                                       WorkListType &WorkList,
                                       UserToOperandsMap &UOpMap,
                                       FunctionToNewTypeMap &FTyMap) {

  while (!WorkList.empty()) {
    Value *OldVal = WorkList.top().first;
    Value *NewVal = WorkList.top().second;
    WorkList.pop();

    for (const auto &U : OldVal->uses()) {
      auto OpNo = U.getOperandNo();
      Value *OldValUser = U.getUser();

      if (CallInst *Call = dyn_cast<CallInst>(OldValUser)) {
        auto F = Call->getCalledFunction();
        assert(F && "Indirect function call?");
        FunctionType *&FuncReplacementType = FTyMap[Call->getCalledFunction()];
        if (!FuncReplacementType)
          FuncReplacementType = F->getFunctionType();
        if (!checkFunctionArgument(FuncReplacementType, NewVal->getType(),
                                   OpNo)) {
          FuncReplacementType = createNewFuncReplacementType(
              FuncReplacementType, NewVal->getType(), OpNo);
        }

        Function *FuncStub = Function::Create(
            FuncReplacementType, F->getLinkage(), "new." + F->getName());
        Value *&CallInstStub = VMap[Call];
        if (!CallInstStub) {
          CallInstStub = createCallInstStub(FuncReplacementType, FuncStub);
        } else {
          cast<CallInst>(CallInstStub)
              ->setCalledFunction(FuncReplacementType, FuncStub);
        }
        auto Arg = getAllocaOrArgValue(F, OpNo);
        if (isa<AllocaInst>(Arg)) {
          auto *&NewValUser = VMap[Arg];
          if (!NewValUser) {
            NewValUser = createTypeUserStub(Arg, NewVal);
            WorkList.push(std::make_pair(Arg, NewValUser));
          }
        } else {
          WorkList.push(
              std::make_pair(Arg, getAllocaOrArgValue(FuncStub, OpNo)));
        }
        UOpMap[cast<User>(CallInstStub)].push_back(
            std::make_pair(OpNo, NewVal));
        // TODO: Case when call used next
        continue;
      }

      if (auto *Ret = dyn_cast<ReturnInst>(OldValUser)) {
        auto F = Ret->getFunction();
        FunctionType *&FuncReplacementType = FTyMap[F];
        if (!FuncReplacementType)
          FuncReplacementType = F->getFunctionType();
        if (!checkFunctionRetType(FuncReplacementType, NewVal->getType())) {
          FuncReplacementType = createNewFuncReplacementType(
              FuncReplacementType, NewVal->getType());
        }
      }

      // TODO: Don't handle this case separately
      if (auto *Store = dyn_cast<StoreInst>(OldValUser)) {
        if (OpNo == 1) {

          auto *UndefType =
              UndefValue::get(NewVal->getType()->getPointerElementType());
          auto *&NewValUser = VMap[OldValUser];
          if (!NewValUser) {
            NewValUser =
                new StoreInst(UndefType, UndefValue::get(NewVal->getType()),
                              Store->isVolatile(), Store->getAlignment(),
                              Store->getOrdering(), Store->getSyncScopeID());
            WorkList.push(std::make_pair(OldValUser, NewValUser));
          }
          UOpMap[cast<User>(NewValUser)].push_back(
              std::make_pair(OpNo, NewVal));
          continue;
        }
      }

      // We are cloning shuffle vectors and binary operators,
      // so these new instructions contains references to
      // old instructions and we can see new instructions as old instructions
      // here, so we should skip these instructions
      // to don't create clone for clone.
      if (valueIsReplacement(VMap, OldValUser))
        continue;

      auto *&NewValUser = VMap[OldValUser];
      if (!NewValUser) {
        NewValUser = createTypeUserStub(OldValUser, NewVal);
        WorkList.push(std::make_pair(OldValUser, NewValUser));
      }
      UOpMap[cast<User>(NewValUser)].push_back(std::make_pair(OpNo, NewVal));
    }
  }
}

static void traceAddressSpace(AddrSpaceCastInst *AS,
                              ValueToValueMap &VMap, UserToOperandsMap &UOpMap,
                              FunctionToNewTypeMap &FTyMap) {

  Type *OldTy = AS->getType();
  Type *NewTy =
      AS->getSrcTy()->getPointerElementType()->getPointerTo(SPIRAS_Generic);
  AddrSpaceCastInst *NewAS =
      new AddrSpaceCastInst(AS->getPointerOperand(), NewTy, "", AS);
  WorkListType WorkList;
  for (const auto &U : AS->uses()) {
    Value *User = U.getUser();
    if (auto Call = dyn_cast<CallInst>(User)) {
      WorkList.push(std::make_pair(AS, NewAS));
      auto F = Call->getCalledFunction();
      assert(F && "No function info.");
      for (auto &Arg : F->args()) {
        if (Arg.getType() == OldTy) {
          auto ActArg = Call->getArgOperand(Arg.getArgNo());
          assert(ActArg && "No argument info.");
          if (!isa<AddrSpaceCastInst>(ActArg)) {
            AddrSpaceCastInst *AddAS =
                new AddrSpaceCastInst(ActArg, NewTy, "", Call);
            Call->setArgOperand(Arg.getArgNo(), AddAS);
            WorkList.push(std::make_pair(AddAS, AddAS));
          }
        }
      }
    }
  }
  collectTypeReplacementData(OldTy, NewTy, VMap, WorkList, UOpMap, FTyMap);
}

static void doReplace(ValueToValueMap &VMap, UserToOperandsMap &UOpMap,
                      FunctionToNewTypeMap &FTyMap) {

  SmallVector<std::pair<CallInst *, CallInst *>, 32> Calls;
  for (auto It : VMap) {
    if (It.second) {
      if (auto Call = dyn_cast<CallInst>(It.first)) {
        auto NewCall = cast<CallInst>(It.second);
        Calls.push_back(std::make_pair(Call, NewCall));
      }
    }
  }

  // Create all functions
  ValueToValueMap FMap;
  for (auto It : Calls) {
    auto Call = It.first;
    auto NewCall = It.second;
    Function *ExistingF = Call->getCalledFunction();
    FunctionType *NewFTy = FTyMap[ExistingF];
    Value *&NewFunc = FMap[ExistingF];
    if (NewFTy && NewFTy != ExistingF->getFunctionType()) {
      if (!NewFunc) {
        assert(NewFTy && "Forgot function?");
        NewFunc = createNewFunction(ExistingF, NewFTy, VMap, UOpMap);
      }
    } else {
      NewFunc = ExistingF;
    }
    NewCall->setCalledFunction(cast<Function>(NewFunc)->getFunctionType(),
                               NewFunc);
    NewCall->setCallingConv(cast<Function>(NewFunc)->getCallingConv());
    NewCall->setDebugLoc(Call->getDebugLoc());
  }

  // Set all operands
  for (auto it : UOpMap) {
    auto Operands = it.second;
    User *Usr = it.first;
    for (auto OpIt : Operands) {
      Usr->setOperand(OpIt.first, OpIt.second);
    }
  }

  // New instructions can use some old instructions, so
  // we need to set correspondig operands
  for (auto It : VMap) {
    if (It.second) {
      if (auto Usr = dyn_cast<User>(It.first)) {
        auto NewUsr = cast<User>(It.second);
        for (auto &Op : NewUsr->operands()) {
          if (isa<UndefValue>(Op.get())) {
            NewUsr->setOperand(Op.getOperandNo(),
                               Usr->getOperand(Op.getOperandNo()));
          }
        }
      }
    }
  }

  // Replace old instructions
  for (auto it : VMap) {
    if (it.second) {
      if (auto NewInst = dyn_cast<Instruction>(it.second)) {
        auto OldInst = cast<Instruction>(it.first);
        if (!NewInst->getParent()) {
          NewInst->insertBefore(OldInst);
        }
        OldInst->mutateType(NewInst->getType());
        OldInst->replaceAllUsesWith(NewInst);
        if (OldInst->use_empty()) {
          OldInst->eraseFromParent();
        }
      }
    }
  }
}

static Type *createNewStructType(Type *NewElTy, StructType *OldTy) {
  SmallVector<Type *, 16> Elements;
  for (auto T : OldTy->elements()) {
    if (T->isPointerTy() &&
        (T->getPointerElementType() == NewElTy->getPointerElementType()) &&
        (T->getPointerAddressSpace() != NewElTy->getPointerAddressSpace())) {
      Elements.push_back(NewElTy);
    } else {
      Elements.push_back(T);
    }
  }
  auto NewStruct =
      StructType::create(OldTy->getContext(), Elements,
                         "new." + std::string(OldTy->getStructName()));
  return NewStruct;
}

static AllocaInst *createAllocaReplacement(AllocaInst *OldAlloca, Type *NewTy) {
  auto NewAlloca = new AllocaInst(
      NewTy, OldAlloca->getType()->getAddressSpace(), OldAlloca->getArraySize(),
      "new." + OldAlloca->getName(), OldAlloca);
  NewAlloca->setAlignment(OldAlloca->getAlignment());
  return NewAlloca;
}

static bool needToReplaceAlloca(AllocaInst *Alloca,
                                ValueToValueMap &VMap,
                                UserToOperandsMap &UOpMap,
                                TypeToTypeMap &TMap) {
  SmallSet<Value *, 32> Seen;
  std::stack<Value *> WorkList;
  Type *AllocType = Alloca->getAllocatedType();
  WorkList.push(Alloca);
  while (!WorkList.empty()) {
    Value *Val = WorkList.top();
    WorkList.pop();
    for (const auto &U : Val->uses()) {
      Value *Usr = U.getUser();
      Value *NextUsr = nullptr;
      if (CallInst *Call = dyn_cast<CallInst>(Usr)) {
        if (VMap[Call]) {
          auto F = Call->getCalledFunction();
          assert(F && "No function info.");
          NextUsr =
              getAllocaOrArgValue(F, U.getOperandNo());
        }
      } else {
        NextUsr = Usr;
      }
      if (NextUsr && !Seen.count(NextUsr)) {
        WorkList.push(NextUsr);
        Seen.insert(NextUsr);
      }
      // TODO: Try only with stores, maybe add more cases later
      auto *&NewVal = VMap[Usr];
      auto Store = dyn_cast<StoreInst>(Usr);
      if (Store && NewVal) {
        auto NewStore = cast<StoreInst>(NewVal);
        auto StoreValType = NewStore->getValueOperand()->getType();
        auto Operands = UOpMap[NewStore];
        if (Operands.size() == 1 && Operands[0].first == 0) {
          if (StoreValType->isPointerTy()) {
            auto *&NewStructTy = TMap[AllocType];
            if (!NewStructTy) {
              NewStructTy = createNewStructType(StoreValType,
                                                cast<StructType>(AllocType));
            }
            return true;
          }
        }
      }
    }
  }
  return false;
}

static bool structContainsPointers(StructType *Struct) {
  // TODO: more general case, for example when struct
  // contains struct which contains pointers.
  for (auto *T : Struct->elements()) {
    if (T->isPointerTy())
      return true;
  }
  return false;
}

struct ASFixer : public ModulePass {
  static char ID;
  ASFixer() : ModulePass(ID) {}
  bool runOnModule(Module &M) override {
    bool Changed = false;
    ValueToValueMap VMap;
    UserToOperandsMap UOpMap;
    FunctionToNewTypeMap FTyMap;
    // We find and replace all address space casts to private
    // address space
    for (auto &F : M.functions()) {
      for (auto &BB : F) {
        for (auto &I : BB) {
          auto AS = dyn_cast<AddrSpaceCastInst>(&I);
          if (AS && AS->getDestAddressSpace() == SPIRAS_Private &&
              AS->getSrcAddressSpace() != SPIRAS_Generic) {
            traceAddressSpace(AS, VMap, UOpMap, FTyMap);
            Changed = true;
          }
        }
      }
    }
    // Pointer with changed address space can be stored
    // into structure so we need to check structures with
    // pointers and replace it if needed.
    // As described in SYCL spec structures with pointers
    // can't be passed as kernel argument so we check
    // allocas of structures with pointers.
    std::vector<std::pair<Value *, Value *>> BadAllocas;
    TypeToTypeMap TMap;
    if (Changed) {
      for (auto &F : M.functions()) {
        for (auto &BB : F) {
          for (auto &I : BB) {
            if (auto Alloca = dyn_cast<AllocaInst>(&I)) {
              Type *AllocType = Alloca->getAllocatedType();
              if (auto StructTy = dyn_cast<StructType>(AllocType)) {
                if (structContainsPointers(StructTy)) {
                  if (needToReplaceAlloca(Alloca, VMap, UOpMap, TMap)) {
                    auto NewStructTy = TMap[StructTy];
                    AllocaInst *AllocaReplacement =
                        createAllocaReplacement(Alloca, NewStructTy);
                    BadAllocas.push_back(
                        std::make_pair(Alloca, AllocaReplacement));
                  }
                }
              }
            }
          }
        }
      }
    }
    for (auto It : BadAllocas) {
      WorkListType W;
      W.push(It);
      collectTypeReplacementData(It.first->getType(), It.second->getType(),
                                 VMap, W, UOpMap, FTyMap);
    }
    doReplace(VMap, UOpMap, FTyMap);
    return Changed;
  }

  virtual llvm::StringRef getPassName() const { return "ASFixer"; }
};
} // namespace

namespace llvm {
void initializeASFixerPass(PassRegistry &Registry);
}

INITIALIZE_PASS(ASFixer, "asfix", "Fix SYCL address spaces", false, false)
ModulePass *llvm::createASFixerPass() { return new ASFixer(); }

char ASFixer::ID = 0;
