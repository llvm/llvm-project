//===- NVVMReflect.cpp - NVVM Emulate conditional compilation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass replaces occurrences of __nvvm_reflect("foo") and llvm.nvvm.reflect
// with an integer.
//
// We choose the value we use by looking, in this order, at:
//
//  * the -nvvm-reflect-list flag, which has the format "foo=1,bar=42",
//  * the StringMap passed to the pass's constructor, and
//  * metadata in the module itself.
//
// If we see an unknown string, we replace its call with 0.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include <map>
#include <sstream>
#include <string>
#include <vector>
#define NVVM_REFLECT_FUNCTION "__nvvm_reflect"

using namespace llvm;

#define DEBUG_TYPE "nvptx-reflect"

namespace llvm { void initializeNVVMReflectPass(PassRegistry &); }

namespace {
class NVVMReflect : public FunctionPass {
private:
  StringMap<int> VarMap;

public:
  static char ID;
  NVVMReflect() : NVVMReflect(StringMap<int>()) {}

  NVVMReflect(const StringMap<int> &Mapping)
      : FunctionPass(ID), VarMap(Mapping) {
    initializeNVVMReflectPass(*PassRegistry::getPassRegistry());
    setVarMap();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
  bool runOnFunction(Function &) override;

private:
  bool handleFunction(Function *ReflectFunction);
  void setVarMap();
};
}

FunctionPass *llvm::createNVVMReflectPass() { return new NVVMReflect(); }
FunctionPass *llvm::createNVVMReflectPass(const StringMap<int> &Mapping) {
  return new NVVMReflect(Mapping);
}

static cl::opt<bool>
NVVMReflectEnabled("nvvm-reflect-enable", cl::init(true), cl::Hidden,
                   cl::desc("NVVM reflection, enabled by default"));

char NVVMReflect::ID = 0;
INITIALIZE_PASS(NVVMReflect, "nvvm-reflect",
                "Replace occurrences of __nvvm_reflect() calls with 0/1", false,
                false)

static cl::list<std::string>
ReflectList("nvvm-reflect-list", cl::value_desc("name=<int>"), cl::Hidden,
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
    StringRef(ReflectList[i]).split(NameValList, ',');
    for (unsigned j = 0, ej = NameValList.size(); j != ej; ++j) {
      SmallVector<StringRef, 2> NameValPair;
      NameValList[j].split(NameValPair, '=');
      assert(NameValPair.size() == 2 && "name=val expected");
      std::stringstream ValStream(NameValPair[1]);
      int Val;
      ValStream >> Val;
      assert((!(ValStream.fail())) && "integer value expected");
      VarMap[NameValPair[0]] = Val;
    }
  }
}

bool NVVMReflect::runOnFunction(Function &F) {
  if (!NVVMReflectEnabled)
    return false;

  if (F.getName() == NVVM_REFLECT_FUNCTION) {
    assert(F.isDeclaration() && "_reflect function should not have a body");
    assert(F.getReturnType()->isIntegerTy() &&
           "_reflect's return type should be integer");
    return false;
  }

  SmallVector<Instruction *, 4> ToRemove;

  // Go through the calls in this function.  Each call to __nvvm_reflect or
  // llvm.nvvm.reflect should be a CallInst with a ConstantArray argument.
  // First validate that. If the c-string corresponding to the ConstantArray can
  // be found successfully, see if it can be found in VarMap. If so, replace the
  // uses of CallInst with the value found in VarMap. If not, replace the use
  // with value 0.

  // The IR for __nvvm_reflect calls differs between CUDA versions.
  //
  // CUDA 6.5 and earlier uses this sequence:
  //    %ptr = tail call i8* @llvm.nvvm.ptr.constant.to.gen.p0i8.p4i8
  //        (i8 addrspace(4)* getelementptr inbounds
  //           ([8 x i8], [8 x i8] addrspace(4)* @str, i32 0, i32 0))
  //    %reflect = tail call i32 @__nvvm_reflect(i8* %ptr)
  //
  // The value returned by Sym->getOperand(0) is a Constant with a
  // ConstantDataSequential operand which can be converted to string and used
  // for lookup.
  //
  // CUDA 7.0 does it slightly differently:
  //   %reflect = call i32 @__nvvm_reflect(i8* addrspacecast
  //        (i8 addrspace(1)* getelementptr inbounds
  //           ([8 x i8], [8 x i8] addrspace(1)* @str, i32 0, i32 0) to i8*))
  //
  // In this case, we get a Constant with a GlobalVariable operand and we need
  // to dig deeper to find its initializer with the string we'll use for lookup.
  for (Instruction &I : instructions(F)) {
    CallInst *Call = dyn_cast<CallInst>(&I);
    if (!Call)
      continue;
    Function *Callee = Call->getCalledFunction();
    if (!Callee || (Callee->getName() != NVVM_REFLECT_FUNCTION &&
                    Callee->getIntrinsicID() != Intrinsic::nvvm_reflect))
      continue;

    // FIXME: Improve error handling here and elsewhere in this pass.
    assert(Call->getNumOperands() == 2 &&
           "Wrong number of operands to __nvvm_reflect function");

    // In cuda 6.5 and earlier, we will have an extra constant-to-generic
    // conversion of the string.
    const Value *Str = Call->getArgOperand(0);
    if (const CallInst *ConvCall = dyn_cast<CallInst>(Str)) {
      // FIXME: Add assertions about ConvCall.
      Str = ConvCall->getArgOperand(0);
    }
    assert(isa<ConstantExpr>(Str) &&
           "Format of __nvvm__reflect function not recognized");
    const ConstantExpr *GEP = cast<ConstantExpr>(Str);

    const Value *Sym = GEP->getOperand(0);
    assert(isa<Constant>(Sym) &&
           "Format of __nvvm_reflect function not recognized");

    const Value *Operand = cast<Constant>(Sym)->getOperand(0);
    if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(Operand)) {
      // For CUDA-7.0 style __nvvm_reflect calls, we need to find the operand's
      // initializer.
      assert(GV->hasInitializer() &&
             "Format of _reflect function not recognized");
      const Constant *Initializer = GV->getInitializer();
      Operand = Initializer;
    }

    assert(isa<ConstantDataSequential>(Operand) &&
           "Format of _reflect function not recognized");
    assert(cast<ConstantDataSequential>(Operand)->isCString() &&
           "Format of _reflect function not recognized");

    StringRef ReflectArg = cast<ConstantDataSequential>(Operand)->getAsString();
    ReflectArg = ReflectArg.substr(0, ReflectArg.size() - 1);
    DEBUG(dbgs() << "Arg of _reflect : " << ReflectArg << "\n");

    int ReflectVal = 0; // The default value is 0
    auto Iter = VarMap.find(ReflectArg);
    if (Iter != VarMap.end())
      ReflectVal = Iter->second;
    else if (ReflectArg == "__CUDA_FTZ") {
      // Try to pull __CUDA_FTZ from the nvvm-reflect-ftz module flag.
      if (auto *Flag = mdconst::extract_or_null<ConstantInt>(
              F.getParent()->getModuleFlag("nvvm-reflect-ftz")))
        ReflectVal = Flag->getSExtValue();
    }
    Call->replaceAllUsesWith(ConstantInt::get(Call->getType(), ReflectVal));
    ToRemove.push_back(Call);
  }

  for (Instruction *I : ToRemove)
    I->eraseFromParent();

  return ToRemove.size() > 0;
}
