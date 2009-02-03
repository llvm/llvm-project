//===- LowerSetJmp.cpp - Code pertaining to lowering set/long jumps -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the lowering of setjmp and longjmp to use the
//  LLVM invoke and unwind instructions as necessary.
//
//  Lowering of longjmp is fairly trivial. We replace the call with a
//  call to the LLVM library function "__llvm_sjljeh_throw_longjmp()".
//  This unwinds the stack for us calling all of the destructors for
//  objects allocated on the stack.
//
//  At a setjmp call, the basic block is split and the setjmp removed.
//  The calls in a function that have a setjmp are converted to invoke
//  where the except part checks to see if it's a longjmp exception and,
//  if so, if it's handled in the function. If it is, then it gets the
//  value returned by the longjmp and goes to where the basic block was
//  split. Invoke instructions are handled in a similar fashion with the
//  original except block being executed if it isn't a longjmp except
//  that is handled by that function.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FIXME: This pass doesn't deal with PHI statements just yet. That is,
// we expect this to occur before SSAification is done. This would seem
// to make sense, but in general, it might be a good idea to make this
// pass invokable via the "opt" command at will.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lowersetjmp"
#include "llvm/Transforms/IPO.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/VectorExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <map>
using namespace llvm;

STATISTIC(LongJmpsTransformed, "Number of longjmps transformed");
STATISTIC(SetJmpsTransformed , "Number of setjmps transformed");
STATISTIC(CallsTransformed   , "Number of calls invokified");
STATISTIC(InvokesTransformed , "Number of invokes modified");

namespace {
  //===--------------------------------------------------------------------===//
  // LowerSetJmp pass implementation.
  class VISIBILITY_HIDDEN LowerSetJmp : public ModulePass,
                      public InstVisitor<LowerSetJmp> {
    // LLVM library functions...
    Constant *InitSJMap;        // __llvm_sjljeh_init_setjmpmap
    Constant *DestroySJMap;     // __llvm_sjljeh_destroy_setjmpmap
    Constant *AddSJToMap;       // __llvm_sjljeh_add_setjmp_to_map
    Constant *ThrowLongJmp;     // __llvm_sjljeh_throw_longjmp
    Constant *TryCatchLJ;       // __llvm_sjljeh_try_catching_longjmp_exception
    Constant *IsLJException;    // __llvm_sjljeh_is_longjmp_exception
    Constant *GetLJValue;       // __llvm_sjljeh_get_longjmp_value

    typedef std::pair<SwitchInst*, CallInst*> SwitchValuePair;

    // Keep track of those basic blocks reachable via a depth-first search of
    // the CFG from a setjmp call. We only need to transform those "call" and
    // "invoke" instructions that are reachable from the setjmp call site.
    std::set<BasicBlock*> DFSBlocks;

    // The setjmp map is going to hold information about which setjmps
    // were called (each setjmp gets its own number) and with which
    // buffer it was called.
    std::map<Function*, AllocaInst*>            SJMap;

    // The rethrow basic block map holds the basic block to branch to if
    // the exception isn't handled in the current function and needs to
    // be rethrown.
    std::map<const Function*, BasicBlock*>      RethrowBBMap;

    // The preliminary basic block map holds a basic block that grabs the
    // exception and determines if it's handled by the current function.
    std::map<const Function*, BasicBlock*>      PrelimBBMap;

    // The switch/value map holds a switch inst/call inst pair. The
    // switch inst controls which handler (if any) gets called and the
    // value is the value returned to that handler by the call to
    // __llvm_sjljeh_get_longjmp_value.
    std::map<const Function*, SwitchValuePair>  SwitchValMap;

    // A map of which setjmps we've seen so far in a function.
    std::map<const Function*, unsigned>         SetJmpIDMap;

    AllocaInst*     GetSetJmpMap(Function* Func);
    BasicBlock*     GetRethrowBB(Function* Func);
    SwitchValuePair GetSJSwitch(Function* Func, BasicBlock* Rethrow);

    void TransformLongJmpCall(CallInst* Inst);
    void TransformSetJmpCall(CallInst* Inst);

    bool IsTransformableFunction(const std::string& Name);
  public:
    static char ID; // Pass identification, replacement for typeid
    LowerSetJmp() : ModulePass(&ID) {}

    void visitCallInst(CallInst& CI);
    void visitInvokeInst(InvokeInst& II);
    void visitReturnInst(ReturnInst& RI);
    void visitUnwindInst(UnwindInst& UI);

    bool runOnModule(Module& M);
    bool doInitialization(Module& M);
  };
} // end anonymous namespace

char LowerSetJmp::ID = 0;
static RegisterPass<LowerSetJmp> X("lowersetjmp", "Lower Set Jump");

// run - Run the transformation on the program. We grab the function
// prototypes for longjmp and setjmp. If they are used in the program,
// then we can go directly to the places they're at and transform them.
bool LowerSetJmp::runOnModule(Module& M) {
  bool Changed = false;

  // These are what the functions are called.
  Function* SetJmp = M.getFunction("llvm.setjmp");
  Function* LongJmp = M.getFunction("llvm.longjmp");

  // This program doesn't have longjmp and setjmp calls.
  if ((!LongJmp || LongJmp->use_empty()) &&
        (!SetJmp || SetJmp->use_empty())) return false;

  // Initialize some values and functions we'll need to transform the
  // setjmp/longjmp functions.
  doInitialization(M);

  if (SetJmp) {
    for (Value::use_iterator B = SetJmp->use_begin(), E = SetJmp->use_end();
         B != E; ++B) {
      BasicBlock* BB = cast<Instruction>(*B)->getParent();
      for (df_ext_iterator<BasicBlock*> I = df_ext_begin(BB, DFSBlocks),
             E = df_ext_end(BB, DFSBlocks); I != E; ++I)
        /* empty */;
    }

    while (!SetJmp->use_empty()) {
      assert(isa<CallInst>(SetJmp->use_back()) &&
             "User of setjmp intrinsic not a call?");
      TransformSetJmpCall(cast<CallInst>(SetJmp->use_back()));
      Changed = true;
    }
  }

  if (LongJmp)
    while (!LongJmp->use_empty()) {
      assert(isa<CallInst>(LongJmp->use_back()) &&
             "User of longjmp intrinsic not a call?");
      TransformLongJmpCall(cast<CallInst>(LongJmp->use_back()));
      Changed = true;
    }

  // Now go through the affected functions and convert calls and invokes
  // to new invokes...
  for (std::map<Function*, AllocaInst*>::iterator
      B = SJMap.begin(), E = SJMap.end(); B != E; ++B) {
    Function* F = B->first;
    for (Function::iterator BB = F->begin(), BE = F->end(); BB != BE; ++BB)
      for (BasicBlock::iterator IB = BB->begin(), IE = BB->end(); IB != IE; ) {
        visit(*IB++);
        if (IB != BB->end() && IB->getParent() != BB)
          break;  // The next instruction got moved to a different block!
      }
  }

  DFSBlocks.clear();
  SJMap.clear();
  RethrowBBMap.clear();
  PrelimBBMap.clear();
  SwitchValMap.clear();
  SetJmpIDMap.clear();

  return Changed;
}

// doInitialization - For the lower long/setjmp pass, this ensures that a
// module contains a declaration for the intrisic functions we are going
// to call to convert longjmp and setjmp calls.
//
// This function is always successful, unless it isn't.
bool LowerSetJmp::doInitialization(Module& M)
{
  const Type *SBPTy = PointerType::getUnqual(Type::Int8Ty);
  const Type *SBPPTy = PointerType::getUnqual(SBPTy);

  // N.B. See llvm/runtime/GCCLibraries/libexception/SJLJ-Exception.h for
  // a description of the following library functions.

  // void __llvm_sjljeh_init_setjmpmap(void**)
  InitSJMap = M.getOrInsertFunction("__llvm_sjljeh_init_setjmpmap",
                                    Type::VoidTy, SBPPTy, (Type *)0);
  // void __llvm_sjljeh_destroy_setjmpmap(void**)
  DestroySJMap = M.getOrInsertFunction("__llvm_sjljeh_destroy_setjmpmap",
                                       Type::VoidTy, SBPPTy, (Type *)0);

  // void __llvm_sjljeh_add_setjmp_to_map(void**, void*, unsigned)
  AddSJToMap = M.getOrInsertFunction("__llvm_sjljeh_add_setjmp_to_map",
                                     Type::VoidTy, SBPPTy, SBPTy,
                                     Type::Int32Ty, (Type *)0);

  // void __llvm_sjljeh_throw_longjmp(int*, int)
  ThrowLongJmp = M.getOrInsertFunction("__llvm_sjljeh_throw_longjmp",
                                       Type::VoidTy, SBPTy, Type::Int32Ty,
                                       (Type *)0);

  // unsigned __llvm_sjljeh_try_catching_longjmp_exception(void **)
  TryCatchLJ =
    M.getOrInsertFunction("__llvm_sjljeh_try_catching_longjmp_exception",
                          Type::Int32Ty, SBPPTy, (Type *)0);

  // bool __llvm_sjljeh_is_longjmp_exception()
  IsLJException = M.getOrInsertFunction("__llvm_sjljeh_is_longjmp_exception",
                                        Type::Int1Ty, (Type *)0);

  // int __llvm_sjljeh_get_longjmp_value()
  GetLJValue = M.getOrInsertFunction("__llvm_sjljeh_get_longjmp_value",
                                     Type::Int32Ty, (Type *)0);
  return true;
}

// IsTransformableFunction - Return true if the function name isn't one
// of the ones we don't want transformed. Currently, don't transform any
// "llvm.{setjmp,longjmp}" functions and none of the setjmp/longjmp error
// handling functions (beginning with __llvm_sjljeh_...they don't throw
// exceptions).
bool LowerSetJmp::IsTransformableFunction(const std::string& Name) {
  std::string SJLJEh("__llvm_sjljeh");

  if (Name.size() > SJLJEh.size())
    return std::string(Name.begin(), Name.begin() + SJLJEh.size()) != SJLJEh;

  return true;
}

// TransformLongJmpCall - Transform a longjmp call into a call to the
// internal __llvm_sjljeh_throw_longjmp function. It then takes care of
// throwing the exception for us.
void LowerSetJmp::TransformLongJmpCall(CallInst* Inst)
{
  const Type* SBPTy = PointerType::getUnqual(Type::Int8Ty);

  // Create the call to "__llvm_sjljeh_throw_longjmp". This takes the
  // same parameters as "longjmp", except that the buffer is cast to a
  // char*. It returns "void", so it doesn't need to replace any of
  // Inst's uses and doesn't get a name.
  CastInst* CI = 
    new BitCastInst(Inst->getOperand(1), SBPTy, "LJBuf", Inst);
  SmallVector<Value *, 2> Args;
  Args.push_back(CI);
  Args.push_back(Inst->getOperand(2));
  CallInst::Create(ThrowLongJmp, Args.begin(), Args.end(), "", Inst);

  SwitchValuePair& SVP = SwitchValMap[Inst->getParent()->getParent()];

  // If the function has a setjmp call in it (they are transformed first)
  // we should branch to the basic block that determines if this longjmp
  // is applicable here. Otherwise, issue an unwind.
  if (SVP.first)
    BranchInst::Create(SVP.first->getParent(), Inst);
  else
    new UnwindInst(Inst);

  // Remove all insts after the branch/unwind inst.  Go from back to front to
  // avoid replaceAllUsesWith if possible.
  BasicBlock *BB = Inst->getParent();
  Instruction *Removed;
  do {
    Removed = &BB->back();
    // If the removed instructions have any users, replace them now.
    if (!Removed->use_empty())
      Removed->replaceAllUsesWith(UndefValue::get(Removed->getType()));
    Removed->eraseFromParent();
  } while (Removed != Inst);

  ++LongJmpsTransformed;
}

// GetSetJmpMap - Retrieve (create and initialize, if necessary) the
// setjmp map. This map is going to hold information about which setjmps
// were called (each setjmp gets its own number) and with which buffer it
// was called. There can be only one!
AllocaInst* LowerSetJmp::GetSetJmpMap(Function* Func)
{
  if (SJMap[Func]) return SJMap[Func];

  // Insert the setjmp map initialization before the first instruction in
  // the function.
  Instruction* Inst = Func->getEntryBlock().begin();
  assert(Inst && "Couldn't find even ONE instruction in entry block!");

  // Fill in the alloca and call to initialize the SJ map.
  const Type *SBPTy = PointerType::getUnqual(Type::Int8Ty);
  AllocaInst* Map = new AllocaInst(SBPTy, 0, "SJMap", Inst);
  CallInst::Create(InitSJMap, Map, "", Inst);
  return SJMap[Func] = Map;
}

// GetRethrowBB - Only one rethrow basic block is needed per function.
// If this is a longjmp exception but not handled in this block, this BB
// performs the rethrow.
BasicBlock* LowerSetJmp::GetRethrowBB(Function* Func)
{
  if (RethrowBBMap[Func]) return RethrowBBMap[Func];

  // The basic block we're going to jump to if we need to rethrow the
  // exception.
  BasicBlock* Rethrow = BasicBlock::Create("RethrowExcept", Func);

  // Fill in the "Rethrow" BB with a call to rethrow the exception. This
  // is the last instruction in the BB since at this point the runtime
  // should exit this function and go to the next function.
  new UnwindInst(Rethrow);
  return RethrowBBMap[Func] = Rethrow;
}

// GetSJSwitch - Return the switch statement that controls which handler
// (if any) gets called and the value returned to that handler.
LowerSetJmp::SwitchValuePair LowerSetJmp::GetSJSwitch(Function* Func,
                                                      BasicBlock* Rethrow)
{
  if (SwitchValMap[Func].first) return SwitchValMap[Func];

  BasicBlock* LongJmpPre = BasicBlock::Create("LongJmpBlkPre", Func);

  // Keep track of the preliminary basic block for some of the other
  // transformations.
  PrelimBBMap[Func] = LongJmpPre;

  // Grab the exception.
  CallInst* Cond = CallInst::Create(IsLJException, "IsLJExcept", LongJmpPre);

  // The "decision basic block" gets the number associated with the
  // setjmp call returning to switch on and the value returned by
  // longjmp.
  BasicBlock* DecisionBB = BasicBlock::Create("LJDecisionBB", Func);

  BranchInst::Create(DecisionBB, Rethrow, Cond, LongJmpPre);

  // Fill in the "decision" basic block.
  CallInst* LJVal = CallInst::Create(GetLJValue, "LJVal", DecisionBB);
  CallInst* SJNum = CallInst::Create(TryCatchLJ, GetSetJmpMap(Func), "SJNum",
                                     DecisionBB);

  SwitchInst* SI = SwitchInst::Create(SJNum, Rethrow, 0, DecisionBB);
  return SwitchValMap[Func] = SwitchValuePair(SI, LJVal);
}

// TransformSetJmpCall - The setjmp call is a bit trickier to transform.
// We're going to convert all setjmp calls to nops. Then all "call" and
// "invoke" instructions in the function are converted to "invoke" where
// the "except" branch is used when returning from a longjmp call.
void LowerSetJmp::TransformSetJmpCall(CallInst* Inst)
{
  BasicBlock* ABlock = Inst->getParent();
  Function* Func = ABlock->getParent();

  // Add this setjmp to the setjmp map.
  const Type* SBPTy = PointerType::getUnqual(Type::Int8Ty);
  CastInst* BufPtr = 
    new BitCastInst(Inst->getOperand(1), SBPTy, "SBJmpBuf", Inst);
  std::vector<Value*> Args = 
    make_vector<Value*>(GetSetJmpMap(Func), BufPtr,
                        ConstantInt::get(Type::Int32Ty,
                                         SetJmpIDMap[Func]++), 0);
  CallInst::Create(AddSJToMap, Args.begin(), Args.end(), "", Inst);

  // We are guaranteed that there are no values live across basic blocks
  // (because we are "not in SSA form" yet), but there can still be values live
  // in basic blocks.  Because of this, splitting the setjmp block can cause
  // values above the setjmp to not dominate uses which are after the setjmp
  // call.  For all of these occasions, we must spill the value to the stack.
  //
  std::set<Instruction*> InstrsAfterCall;

  // The call is probably very close to the end of the basic block, for the
  // common usage pattern of: 'if (setjmp(...))', so keep track of the
  // instructions after the call.
  for (BasicBlock::iterator I = ++BasicBlock::iterator(Inst), E = ABlock->end();
       I != E; ++I)
    InstrsAfterCall.insert(I);

  for (BasicBlock::iterator II = ABlock->begin();
       II != BasicBlock::iterator(Inst); ++II)
    // Loop over all of the uses of instruction.  If any of them are after the
    // call, "spill" the value to the stack.
    for (Value::use_iterator UI = II->use_begin(), E = II->use_end();
         UI != E; ++UI)
      if (cast<Instruction>(*UI)->getParent() != ABlock ||
          InstrsAfterCall.count(cast<Instruction>(*UI))) {
        DemoteRegToStack(*II);
        break;
      }
  InstrsAfterCall.clear();

  // Change the setjmp call into a branch statement. We'll remove the
  // setjmp call in a little bit. No worries.
  BasicBlock* SetJmpContBlock = ABlock->splitBasicBlock(Inst);
  assert(SetJmpContBlock && "Couldn't split setjmp BB!!");

  SetJmpContBlock->setName(ABlock->getName()+"SetJmpCont");

  // Add the SetJmpContBlock to the set of blocks reachable from a setjmp.
  DFSBlocks.insert(SetJmpContBlock);

  // This PHI node will be in the new block created from the
  // splitBasicBlock call.
  PHINode* PHI = PHINode::Create(Type::Int32Ty, "SetJmpReturn", Inst);

  // Coming from a call to setjmp, the return is 0.
  PHI->addIncoming(ConstantInt::getNullValue(Type::Int32Ty), ABlock);

  // Add the case for this setjmp's number...
  SwitchValuePair SVP = GetSJSwitch(Func, GetRethrowBB(Func));
  SVP.first->addCase(ConstantInt::get(Type::Int32Ty, SetJmpIDMap[Func] - 1),
                     SetJmpContBlock);

  // Value coming from the handling of the exception.
  PHI->addIncoming(SVP.second, SVP.second->getParent());

  // Replace all uses of this instruction with the PHI node created by
  // the eradication of setjmp.
  Inst->replaceAllUsesWith(PHI);
  Inst->eraseFromParent();

  ++SetJmpsTransformed;
}

// visitCallInst - This converts all LLVM call instructions into invoke
// instructions. The except part of the invoke goes to the "LongJmpBlkPre"
// that grabs the exception and proceeds to determine if it's a longjmp
// exception or not.
void LowerSetJmp::visitCallInst(CallInst& CI)
{
  if (CI.getCalledFunction())
    if (!IsTransformableFunction(CI.getCalledFunction()->getName()) ||
        CI.getCalledFunction()->isIntrinsic()) return;

  BasicBlock* OldBB = CI.getParent();

  // If not reachable from a setjmp call, don't transform.
  if (!DFSBlocks.count(OldBB)) return;

  BasicBlock* NewBB = OldBB->splitBasicBlock(CI);
  assert(NewBB && "Couldn't split BB of \"call\" instruction!!");
  DFSBlocks.insert(NewBB);
  NewBB->setName("Call2Invoke");

  Function* Func = OldBB->getParent();

  // Construct the new "invoke" instruction.
  TerminatorInst* Term = OldBB->getTerminator();
  std::vector<Value*> Params(CI.op_begin() + 1, CI.op_end());
  InvokeInst* II =
    InvokeInst::Create(CI.getCalledValue(), NewBB, PrelimBBMap[Func],
                       Params.begin(), Params.end(), CI.getName(), Term);
  II->setCallingConv(CI.getCallingConv());
  II->setAttributes(CI.getAttributes());

  // Replace the old call inst with the invoke inst and remove the call.
  CI.replaceAllUsesWith(II);
  CI.eraseFromParent();

  // The old terminator is useless now that we have the invoke inst.
  Term->eraseFromParent();
  ++CallsTransformed;
}

// visitInvokeInst - Converting the "invoke" instruction is fairly
// straight-forward. The old exception part is replaced by a query asking
// if this is a longjmp exception. If it is, then it goes to the longjmp
// exception blocks. Otherwise, control is passed the old exception.
void LowerSetJmp::visitInvokeInst(InvokeInst& II)
{
  if (II.getCalledFunction())
    if (!IsTransformableFunction(II.getCalledFunction()->getName()) ||
        II.getCalledFunction()->isIntrinsic()) return;

  BasicBlock* BB = II.getParent();

  // If not reachable from a setjmp call, don't transform.
  if (!DFSBlocks.count(BB)) return;

  BasicBlock* ExceptBB = II.getUnwindDest();

  Function* Func = BB->getParent();
  BasicBlock* NewExceptBB = BasicBlock::Create("InvokeExcept", Func);

  // If this is a longjmp exception, then branch to the preliminary BB of
  // the longjmp exception handling. Otherwise, go to the old exception.
  CallInst* IsLJExcept = CallInst::Create(IsLJException, "IsLJExcept",
                                          NewExceptBB);

  BranchInst::Create(PrelimBBMap[Func], ExceptBB, IsLJExcept, NewExceptBB);

  II.setUnwindDest(NewExceptBB);
  ++InvokesTransformed;
}

// visitReturnInst - We want to destroy the setjmp map upon exit from the
// function.
void LowerSetJmp::visitReturnInst(ReturnInst &RI) {
  Function* Func = RI.getParent()->getParent();
  CallInst::Create(DestroySJMap, GetSetJmpMap(Func), "", &RI);
}

// visitUnwindInst - We want to destroy the setjmp map upon exit from the
// function.
void LowerSetJmp::visitUnwindInst(UnwindInst &UI) {
  Function* Func = UI.getParent()->getParent();
  CallInst::Create(DestroySJMap, GetSetJmpMap(Func), "", &UI);
}

ModulePass *llvm::createLowerSetJmpPass() {
  return new LowerSetJmp();
}

