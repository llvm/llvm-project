//===- NVVMReflect.cpp - NVVM Emulate conditional compilation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass replaces occurences of __nvvm_reflect("string") with an
// integer based on -nvvm-reflect-list string=<int> option given to this pass.
// If an undefined string value is seen in a call to __nvvm_reflect("string"),
// a default value of 0 will be used.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include <map>
#include <sstream>
#include <string>
#include <vector>

#define NVVM_REFLECT_FUNCTION "__nvvm_reflect"

using namespace llvm;

namespace llvm { void initializeNVVMReflectPass(PassRegistry &); }

namespace {
class NVVMReflect : public ModulePass {
private:
  StringMap<int> VarMap;
  typedef DenseMap<std::string, int>::iterator VarMapIter;
  Function *ReflectFunction;

public:
  static char ID;
  NVVMReflect() : ModulePass(ID), ReflectFunction(0) {
    initializeNVVMReflectPass(*PassRegistry::getPassRegistry());
    VarMap.clear();
  }

  NVVMReflect(const StringMap<int> &Mapping)
  : ModulePass(ID), ReflectFunction(0) {
    initializeNVVMReflectPass(*PassRegistry::getPassRegistry());
    for (StringMap<int>::const_iterator I = Mapping.begin(), E = Mapping.end();
         I != E; ++I) {
      VarMap[(*I).getKey()] = (*I).getValue();
    }
  }

  void getAnalysisUsage(AnalysisUsage &AU) const { AU.setPreservesAll(); }
  virtual bool runOnModule(Module &);

  void setVarMap();
};
}

ModulePass *llvm::createNVVMReflectPass() {
  return new NVVMReflect();
}

ModulePass *llvm::createNVVMReflectPass(const StringMap<int>& Mapping) {
  return new NVVMReflect(Mapping);
}

static cl::opt<bool>
NVVMReflectEnabled("nvvm-reflect-enable", cl::init(true),
                   cl::desc("NVVM reflection, enabled by default"));

char NVVMReflect::ID = 0;
INITIALIZE_PASS(NVVMReflect, "nvvm-reflect",
                "Replace occurences of __nvvm_reflect() calls with 0/1", false,
                false)

static cl::list<std::string>
ReflectList("nvvm-reflect-list", cl::value_desc("name=<int>"),
            cl::desc("A list of string=num assignments"),
            cl::ValueRequired);

/// The command line can look as follows :
/// -nvvm-reflect-list a=1,b=2 -nvvm-reflect-list c=3,d=0 -R e=2
/// The strings "a=1,b=2", "c=3,d=0", "e=2" are available in the
/// ReflectList vector. First, each of ReflectList[i] is 'split'
/// using "," as the delimiter. Then each of this part is split
/// using "=" as the delimiter.
void NVVMReflect::setVarMap() {
  for (unsigned i = 0, e = ReflectList.size(); i != e; ++i) {
    DEBUG(dbgs() << "Option : "  << ReflectList[i] << "\n");
    SmallVector<StringRef, 4> NameValList;
    StringRef(ReflectList[i]).split(NameValList, ",");
    for (unsigned j = 0, ej = NameValList.size(); j != ej; ++j) {
      SmallVector<StringRef, 2> NameValPair;
      NameValList[j].split(NameValPair, "=");
      assert(NameValPair.size() == 2 && "name=val expected");
      std::stringstream ValStream(NameValPair[1]);
      int Val;
      ValStream >> Val;
      assert((!(ValStream.fail())) && "integer value expected");
      VarMap[NameValPair[0]] = Val;
    }
  }
}

bool NVVMReflect::runOnModule(Module &M) {
  if (!NVVMReflectEnabled)
    return false;

  setVarMap();

  ReflectFunction = M.getFunction(NVVM_REFLECT_FUNCTION);

  // If reflect function is not used, then there will be
  // no entry in the module.
  if (ReflectFunction == 0)
    return false;

  // Validate _reflect function
  assert(ReflectFunction->isDeclaration() &&
         "_reflect function should not have a body");
  assert(ReflectFunction->getReturnType()->isIntegerTy() &&
         "_reflect's return type should be integer");

  std::vector<Instruction *> ToRemove;

  // Go through the uses of ReflectFunction in this Function.
  // Each of them should a CallInst with a ConstantArray argument.
  // First validate that. If the c-string corresponding to the
  // ConstantArray can be found successfully, see if it can be
  // found in VarMap. If so, replace the uses of CallInst with the
  // value found in VarMap. If not, replace the use  with value 0.
  for (Value::use_iterator I = ReflectFunction->use_begin(),
                           E = ReflectFunction->use_end();
       I != E; ++I) {
    assert(isa<CallInst>(*I) && "Only a call instruction can use _reflect");
    CallInst *Reflect = cast<CallInst>(*I);

    assert((Reflect->getNumOperands() == 2) &&
           "Only one operand expect for _reflect function");
    // In cuda, we will have an extra constant-to-generic conversion of
    // the string.
    const Value *conv = Reflect->getArgOperand(0);
    assert(isa<CallInst>(conv) && "Expected a const-to-gen conversion");
    const CallInst *ConvCall = cast<CallInst>(conv);
    const Value *str = ConvCall->getArgOperand(0);
    assert(isa<ConstantExpr>(str) &&
           "Format of _reflect function not recognized");
    const ConstantExpr *GEP = cast<ConstantExpr>(str);

    const Value *Sym = GEP->getOperand(0);
    assert(isa<Constant>(Sym) && "Format of _reflect function not recognized");

    const Constant *SymStr = cast<Constant>(Sym);

    assert(isa<ConstantDataSequential>(SymStr->getOperand(0)) &&
           "Format of _reflect function not recognized");

    assert(cast<ConstantDataSequential>(SymStr->getOperand(0))->isCString() &&
           "Format of _reflect function not recognized");

    std::string ReflectArg =
        cast<ConstantDataSequential>(SymStr->getOperand(0))->getAsString();

    ReflectArg = ReflectArg.substr(0, ReflectArg.size() - 1);
    DEBUG(dbgs() << "Arg of _reflect : " << ReflectArg << "\n");

    int ReflectVal = 0; // The default value is 0
    if (VarMap.find(ReflectArg) != VarMap.end()) {
      ReflectVal = VarMap[ReflectArg];
    }
    Reflect->replaceAllUsesWith(
        ConstantInt::get(Reflect->getType(), ReflectVal));
    ToRemove.push_back(Reflect);
  }
  if (ToRemove.size() == 0)
    return false;

  for (unsigned i = 0, e = ToRemove.size(); i != e; ++i)
    ToRemove[i]->eraseFromParent();
  return true;
}
