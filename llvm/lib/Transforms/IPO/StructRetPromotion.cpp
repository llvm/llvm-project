//===-- StructRetPromotion.cpp - Promote sret arguments ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass finds functions that return a struct (using a pointer to the struct
// as the first argument of the function, marked with the 'sret' attribute) and
// replaces them with a new function that simply returns each of the elements of
// that struct (using multiple return values).
//
// This pass works under a number of conditions:
//  1. The returned struct must not contain other structs
//  2. The returned struct must only be used to load values from
//  3. The placeholder struct passed in is the result of an alloca
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sretpromotion"
#include "llvm/Transforms/IPO.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/CallGraphSCCPass.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

STATISTIC(NumRejectedSRETUses , "Number of sret rejected due to unexpected uses");
STATISTIC(NumSRET , "Number of sret promoted");
namespace {
  /// SRETPromotion - This pass removes sret parameter and updates
  /// function to use multiple return value.
  ///
  struct VISIBILITY_HIDDEN SRETPromotion : public CallGraphSCCPass {
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      CallGraphSCCPass::getAnalysisUsage(AU);
    }

    virtual bool runOnSCC(const std::vector<CallGraphNode *> &SCC);
    static char ID; // Pass identification, replacement for typeid
    SRETPromotion() : CallGraphSCCPass(&ID) {}

  private:
    bool PromoteReturn(CallGraphNode *CGN);
    bool isSafeToUpdateAllCallers(Function *F);
    Function *cloneFunctionBody(Function *F, const StructType *STy);
    void updateCallSites(Function *F, Function *NF);
    bool nestedStructType(const StructType *STy);
  };
}

char SRETPromotion::ID = 0;
static RegisterPass<SRETPromotion>
X("sretpromotion", "Promote sret arguments to multiple ret values");

Pass *llvm::createStructRetPromotionPass() {
  return new SRETPromotion();
}

bool SRETPromotion::runOnSCC(const std::vector<CallGraphNode *> &SCC) {
  bool Changed = false;

  for (unsigned i = 0, e = SCC.size(); i != e; ++i)
    Changed |= PromoteReturn(SCC[i]);

  return Changed;
}

/// PromoteReturn - This method promotes function that uses StructRet paramater 
/// into a function that uses mulitple return value.
bool SRETPromotion::PromoteReturn(CallGraphNode *CGN) {
  Function *F = CGN->getFunction();

  if (!F || F->isDeclaration() || !F->hasLocalLinkage())
    return false;

  // Make sure that function returns struct.
  if (F->arg_size() == 0 || !F->hasStructRetAttr() || F->doesNotReturn())
    return false;

  DOUT << "SretPromotion: Looking at sret function " << F->getNameStart() << "\n";

  assert (F->getReturnType() == Type::VoidTy && "Invalid function return type");
  Function::arg_iterator AI = F->arg_begin();
  const llvm::PointerType *FArgType = dyn_cast<PointerType>(AI->getType());
  assert (FArgType && "Invalid sret parameter type");
  const llvm::StructType *STy = 
    dyn_cast<StructType>(FArgType->getElementType());
  assert (STy && "Invalid sret parameter element type");

  // Check if it is ok to perform this promotion.
  if (isSafeToUpdateAllCallers(F) == false) {
    DOUT << "SretPromotion: Not all callers can be updated\n";
    NumRejectedSRETUses++;
    return false;
  }

  DOUT << "SretPromotion: sret argument will be promoted\n";
  NumSRET++;
  // [1] Replace use of sret parameter 
  AllocaInst *TheAlloca = new AllocaInst (STy, NULL, "mrv", 
                                          F->getEntryBlock().begin());
  Value *NFirstArg = F->arg_begin();
  NFirstArg->replaceAllUsesWith(TheAlloca);

  // [2] Find and replace ret instructions
  for (Function::iterator FI = F->begin(), FE = F->end();  FI != FE; ++FI) 
    for(BasicBlock::iterator BI = FI->begin(), BE = FI->end(); BI != BE; ) {
      Instruction *I = BI;
      ++BI;
      if (isa<ReturnInst>(I)) {
        Value *NV = new LoadInst(TheAlloca, "mrv.ld", I);
        ReturnInst *NR = ReturnInst::Create(NV, I);
        I->replaceAllUsesWith(NR);
        I->eraseFromParent();
      }
    }

  // [3] Create the new function body and insert it into the module.
  Function *NF = cloneFunctionBody(F, STy);

  // [4] Update all call sites to use new function
  updateCallSites(F, NF);

  F->eraseFromParent();
  getAnalysis<CallGraph>().changeFunction(F, NF);
  return true;
}

// Check if it is ok to perform this promotion.
bool SRETPromotion::isSafeToUpdateAllCallers(Function *F) {

  if (F->use_empty())
    // No users. OK to modify signature.
    return true;

  for (Value::use_iterator FnUseI = F->use_begin(), FnUseE = F->use_end();
       FnUseI != FnUseE; ++FnUseI) {
    // The function is passed in as an argument to (possibly) another function,
    // we can't change it!
    CallSite CS = CallSite::get(*FnUseI);
    Instruction *Call = CS.getInstruction();
    // The function is used by something else than a call or invoke instruction,
    // we can't change it!
    if (!Call || !CS.isCallee(FnUseI))
      return false;
    CallSite::arg_iterator AI = CS.arg_begin();
    Value *FirstArg = *AI;

    if (!isa<AllocaInst>(FirstArg))
      return false;

    // Check FirstArg's users.
    for (Value::use_iterator ArgI = FirstArg->use_begin(), 
           ArgE = FirstArg->use_end(); ArgI != ArgE; ++ArgI) {

      // If FirstArg user is a CallInst that does not correspond to current
      // call site then this function F is not suitable for sret promotion.
      if (CallInst *CI = dyn_cast<CallInst>(ArgI)) {
        if (CI != Call)
          return false;
      }
      // If FirstArg user is a GEP whose all users are not LoadInst then
      // this function F is not suitable for sret promotion.
      else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(ArgI)) {
        // TODO : Use dom info and insert PHINodes to collect get results
        // from multiple call sites for this GEP.
        if (GEP->getParent() != Call->getParent())
          return false;
        for (Value::use_iterator GEPI = GEP->use_begin(), GEPE = GEP->use_end();
             GEPI != GEPE; ++GEPI) 
          if (!isa<LoadInst>(GEPI))
            return false;
      } 
      // Any other FirstArg users make this function unsuitable for sret 
      // promotion.
      else
        return false;
    }
  }

  return true;
}

/// cloneFunctionBody - Create a new function based on F and
/// insert it into module. Remove first argument. Use STy as
/// the return type for new function.
Function *SRETPromotion::cloneFunctionBody(Function *F, 
                                           const StructType *STy) {

  const FunctionType *FTy = F->getFunctionType();
  std::vector<const Type*> Params;

  // Attributes - Keep track of the parameter attributes for the arguments.
  SmallVector<AttributeWithIndex, 8> AttributesVec;
  const AttrListPtr &PAL = F->getAttributes();

  // Add any return attributes.
  if (Attributes attrs = PAL.getRetAttributes())
    AttributesVec.push_back(AttributeWithIndex::get(0, attrs));

  // Skip first argument.
  Function::arg_iterator I = F->arg_begin(), E = F->arg_end();
  ++I;
  // 0th parameter attribute is reserved for return type.
  // 1th parameter attribute is for first 1st sret argument.
  unsigned ParamIndex = 2; 
  while (I != E) {
    Params.push_back(I->getType());
    if (Attributes Attrs = PAL.getParamAttributes(ParamIndex))
      AttributesVec.push_back(AttributeWithIndex::get(ParamIndex - 1, Attrs));
    ++I;
    ++ParamIndex;
  }

  // Add any fn attributes.
  if (Attributes attrs = PAL.getFnAttributes())
    AttributesVec.push_back(AttributeWithIndex::get(~0, attrs));


  FunctionType *NFTy = FunctionType::get(STy, Params, FTy->isVarArg());
  Function *NF = Function::Create(NFTy, F->getLinkage());
  NF->takeName(F);
  NF->copyAttributesFrom(F);
  NF->setAttributes(AttrListPtr::get(AttributesVec.begin(), AttributesVec.end()));
  F->getParent()->getFunctionList().insert(F, NF);
  NF->getBasicBlockList().splice(NF->begin(), F->getBasicBlockList());

  // Replace arguments
  I = F->arg_begin();
  E = F->arg_end();
  Function::arg_iterator NI = NF->arg_begin();
  ++I;
  while (I != E) {
      I->replaceAllUsesWith(NI);
      NI->takeName(I);
      ++I;
      ++NI;
  }

  return NF;
}

/// updateCallSites - Update all sites that call F to use NF.
void SRETPromotion::updateCallSites(Function *F, Function *NF) {
  CallGraph &CG = getAnalysis<CallGraph>();
  SmallVector<Value*, 16> Args;

  // Attributes - Keep track of the parameter attributes for the arguments.
  SmallVector<AttributeWithIndex, 8> ArgAttrsVec;

  while (!F->use_empty()) {
    CallSite CS = CallSite::get(*F->use_begin());
    Instruction *Call = CS.getInstruction();

    const AttrListPtr &PAL = F->getAttributes();
    // Add any return attributes.
    if (Attributes attrs = PAL.getRetAttributes())
      ArgAttrsVec.push_back(AttributeWithIndex::get(0, attrs));

    // Copy arguments, however skip first one.
    CallSite::arg_iterator AI = CS.arg_begin(), AE = CS.arg_end();
    Value *FirstCArg = *AI;
    ++AI;
    // 0th parameter attribute is reserved for return type.
    // 1th parameter attribute is for first 1st sret argument.
    unsigned ParamIndex = 2; 
    while (AI != AE) {
      Args.push_back(*AI); 
      if (Attributes Attrs = PAL.getParamAttributes(ParamIndex))
        ArgAttrsVec.push_back(AttributeWithIndex::get(ParamIndex - 1, Attrs));
      ++ParamIndex;
      ++AI;
    }

    // Add any function attributes.
    if (Attributes attrs = PAL.getFnAttributes())
      ArgAttrsVec.push_back(AttributeWithIndex::get(~0, attrs));
    
    AttrListPtr NewPAL = AttrListPtr::get(ArgAttrsVec.begin(), ArgAttrsVec.end());
    
    // Build new call instruction.
    Instruction *New;
    if (InvokeInst *II = dyn_cast<InvokeInst>(Call)) {
      New = InvokeInst::Create(NF, II->getNormalDest(), II->getUnwindDest(),
                               Args.begin(), Args.end(), "", Call);
      cast<InvokeInst>(New)->setCallingConv(CS.getCallingConv());
      cast<InvokeInst>(New)->setAttributes(NewPAL);
    } else {
      New = CallInst::Create(NF, Args.begin(), Args.end(), "", Call);
      cast<CallInst>(New)->setCallingConv(CS.getCallingConv());
      cast<CallInst>(New)->setAttributes(NewPAL);
      if (cast<CallInst>(Call)->isTailCall())
        cast<CallInst>(New)->setTailCall();
    }
    Args.clear();
    ArgAttrsVec.clear();
    New->takeName(Call);

    // Update the callgraph to know that the callsite has been transformed.
    CG[Call->getParent()->getParent()]->replaceCallSite(Call, New);

    // Update all users of sret parameter to extract value using extractvalue.
    for (Value::use_iterator UI = FirstCArg->use_begin(), 
           UE = FirstCArg->use_end(); UI != UE; ) {
      User *U2 = *UI++;
      CallInst *C2 = dyn_cast<CallInst>(U2);
      if (C2 && (C2 == Call))
        continue;
      else if (GetElementPtrInst *UGEP = dyn_cast<GetElementPtrInst>(U2)) {
        ConstantInt *Idx = dyn_cast<ConstantInt>(UGEP->getOperand(2));
        assert (Idx && "Unexpected getelementptr index!");
        Value *GR = ExtractValueInst::Create(New, Idx->getZExtValue(),
                                             "evi", UGEP);
        while(!UGEP->use_empty()) {
          // isSafeToUpdateAllCallers has checked that all GEP uses are
          // LoadInsts
          LoadInst *L = cast<LoadInst>(*UGEP->use_begin());
          L->replaceAllUsesWith(GR);
          L->eraseFromParent();
        }
        UGEP->eraseFromParent();
      }
      else assert( 0 && "Unexpected sret parameter use");
    }
    Call->eraseFromParent();
  }
}

/// nestedStructType - Return true if STy includes any
/// other aggregate types
bool SRETPromotion::nestedStructType(const StructType *STy) {
  unsigned Num = STy->getNumElements();
  for (unsigned i = 0; i < Num; i++) {
    const Type *Ty = STy->getElementType(i);
    if (!Ty->isSingleValueType() && Ty != Type::VoidTy)
      return true;
  }
  return false;
}
