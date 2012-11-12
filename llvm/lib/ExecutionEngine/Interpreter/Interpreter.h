//===-- Interpreter.h ------------------------------------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file defines the interpreter structure
//
//===----------------------------------------------------------------------===//

#ifndef LLI_INTERPRETER_H
#define LLI_INTERPRETER_H

#include "llvm/Function.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/DataLayout.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/raw_ostream.h"
namespace llvm {

class IntrinsicLowering;
struct FunctionInfo;
template<typename T> class generic_gep_type_iterator;
class ConstantExpr;
typedef generic_gep_type_iterator<User::const_op_iterator> gep_type_iterator;


// AllocaHolder - Object to track all of the blocks of memory allocated by
// alloca.  When the function returns, this object is popped off the execution
// stack, which causes the dtor to be run, which frees all the alloca'd memory.
//
class AllocaHolder {
  friend class AllocaHolderHandle;
  std::vector<void*> Allocations;
  unsigned RefCnt;
public:
  AllocaHolder() : RefCnt(0) {}
  void add(void *mem) { Allocations.push_back(mem); }
  ~AllocaHolder() {
    for (unsigned i = 0; i < Allocations.size(); ++i)
      free(Allocations[i]);
  }
};

// AllocaHolderHandle gives AllocaHolder value semantics so we can stick it into
// a vector...
//
class AllocaHolderHandle {
  AllocaHolder *H;
public:
  AllocaHolderHandle() : H(new AllocaHolder()) { H->RefCnt++; }
  AllocaHolderHandle(const AllocaHolderHandle &AH) : H(AH.H) { H->RefCnt++; }
  ~AllocaHolderHandle() { if (--H->RefCnt == 0) delete H; }

  void add(void *mem) { H->add(mem); }
};

typedef std::vector<GenericValue> ValuePlaneTy;

// ExecutionContext struct - This struct represents one stack frame currently
// executing.
//
struct ExecutionContext {
  Function             *CurFunction;// The currently executing function
  BasicBlock           *CurBB;      // The currently executing BB
  BasicBlock::iterator  CurInst;    // The next instruction to execute
  std::map<Value *, GenericValue> Values; // LLVM values used in this invocation
  std::vector<GenericValue>  VarArgs; // Values passed through an ellipsis
  CallSite             Caller;     // Holds the call that called subframes.
                                   // NULL if main func or debugger invoked fn
  AllocaHolderHandle    Allocas;    // Track memory allocated by alloca
};

// Interpreter - This class represents the entirety of the interpreter.
//
class Interpreter : public ExecutionEngine, public InstVisitor<Interpreter> {
  GenericValue ExitValue;          // The return value of the called function
  DataLayout TD;
  IntrinsicLowering *IL;

  // The runtime stack of executing code.  The top of the stack is the current
  // function record.
  std::vector<ExecutionContext> ECStack;

  // AtExitHandlers - List of functions to call when the program exits,
  // registered with the atexit() library function.
  std::vector<Function*> AtExitHandlers;

public:
  explicit Interpreter(Module *M);
  ~Interpreter();

  /// runAtExitHandlers - Run any functions registered by the program's calls to
  /// atexit(3), which we intercept and store in AtExitHandlers.
  ///
  void runAtExitHandlers();

  static void Register() {
    InterpCtor = create;
  }
  
  /// create - Create an interpreter ExecutionEngine. This can never fail.
  ///
  static ExecutionEngine *create(Module *M, std::string *ErrorStr = 0);

  /// run - Start execution with the specified function and arguments.
  ///
  virtual GenericValue runFunction(Function *F,
                                   const std::vector<GenericValue> &ArgValues);

  virtual void *getPointerToNamedFunction(const std::string &Name,
                                          bool AbortOnFailure = true) {
    // FIXME: not implemented.
    return 0;
  }

  /// recompileAndRelinkFunction - For the interpreter, functions are always
  /// up-to-date.
  ///
  virtual void *recompileAndRelinkFunction(Function *F) {
    return getPointerToFunction(F);
  }

  /// freeMachineCodeForFunction - The interpreter does not generate any code.
  ///
  void freeMachineCodeForFunction(Function *F) { }

  // Methods used to execute code:
  // Place a call on the stack
  void callFunction(Function *F, const std::vector<GenericValue> &ArgVals);
  void run();                // Execute instructions until nothing left to do

  // Opcode Implementations
  void visitReturnInst(ReturnInst &I);
  void visitBranchInst(BranchInst &I);
  void visitSwitchInst(SwitchInst &I);
  void visitIndirectBrInst(IndirectBrInst &I);

  void visitBinaryOperator(BinaryOperator &I);
  void visitICmpInst(ICmpInst &I);
  void visitFCmpInst(FCmpInst &I);
  void visitAllocaInst(AllocaInst &I);
  void visitLoadInst(LoadInst &I);
  void visitStoreInst(StoreInst &I);
  void visitGetElementPtrInst(GetElementPtrInst &I);
  void visitPHINode(PHINode &PN) { 
    llvm_unreachable("PHI nodes already handled!"); 
  }
  void visitTruncInst(TruncInst &I);
  void visitZExtInst(ZExtInst &I);
  void visitSExtInst(SExtInst &I);
  void visitFPTruncInst(FPTruncInst &I);
  void visitFPExtInst(FPExtInst &I);
  void visitUIToFPInst(UIToFPInst &I);
  void visitSIToFPInst(SIToFPInst &I);
  void visitFPToUIInst(FPToUIInst &I);
  void visitFPToSIInst(FPToSIInst &I);
  void visitPtrToIntInst(PtrToIntInst &I);
  void visitIntToPtrInst(IntToPtrInst &I);
  void visitBitCastInst(BitCastInst &I);
  void visitSelectInst(SelectInst &I);


  void visitCallSite(CallSite CS);
  void visitCallInst(CallInst &I) { visitCallSite (CallSite (&I)); }
  void visitInvokeInst(InvokeInst &I) { visitCallSite (CallSite (&I)); }
  void visitUnreachableInst(UnreachableInst &I);

  void visitShl(BinaryOperator &I);
  void visitLShr(BinaryOperator &I);
  void visitAShr(BinaryOperator &I);

  void visitVAArgInst(VAArgInst &I);
  void visitInstruction(Instruction &I) {
    errs() << I << "\n";
    llvm_unreachable("Instruction not interpretable yet!");
  }

  GenericValue callExternalFunction(Function *F,
                                    const std::vector<GenericValue> &ArgVals);
  void exitCalled(GenericValue GV);

  void addAtExitHandler(Function *F) {
    AtExitHandlers.push_back(F);
  }

  GenericValue *getFirstVarArg () {
    return &(ECStack.back ().VarArgs[0]);
  }

private:  // Helper functions
  GenericValue executeGEPOperation(Value *Ptr, gep_type_iterator I,
                                   gep_type_iterator E, ExecutionContext &SF);

  // SwitchToNewBasicBlock - Start execution in a new basic block and run any
  // PHI nodes in the top of the block.  This is used for intraprocedural
  // control flow.
  //
  void SwitchToNewBasicBlock(BasicBlock *Dest, ExecutionContext &SF);

  void *getPointerToFunction(Function *F) { return (void*)F; }
  void *getPointerToBasicBlock(BasicBlock *BB) { return (void*)BB; }

  void initializeExecutionEngine() { }
  void initializeExternalFunctions();
  GenericValue getConstantExprValue(ConstantExpr *CE, ExecutionContext &SF);
  GenericValue getOperandValue(Value *V, ExecutionContext &SF);
  GenericValue executeTruncInst(Value *SrcVal, Type *DstTy,
                                ExecutionContext &SF);
  GenericValue executeSExtInst(Value *SrcVal, Type *DstTy,
                               ExecutionContext &SF);
  GenericValue executeZExtInst(Value *SrcVal, Type *DstTy,
                               ExecutionContext &SF);
  GenericValue executeFPTruncInst(Value *SrcVal, Type *DstTy,
                                  ExecutionContext &SF);
  GenericValue executeFPExtInst(Value *SrcVal, Type *DstTy,
                                ExecutionContext &SF);
  GenericValue executeFPToUIInst(Value *SrcVal, Type *DstTy,
                                 ExecutionContext &SF);
  GenericValue executeFPToSIInst(Value *SrcVal, Type *DstTy,
                                 ExecutionContext &SF);
  GenericValue executeUIToFPInst(Value *SrcVal, Type *DstTy,
                                 ExecutionContext &SF);
  GenericValue executeSIToFPInst(Value *SrcVal, Type *DstTy,
                                 ExecutionContext &SF);
  GenericValue executePtrToIntInst(Value *SrcVal, Type *DstTy,
                                   ExecutionContext &SF);
  GenericValue executeIntToPtrInst(Value *SrcVal, Type *DstTy,
                                   ExecutionContext &SF);
  GenericValue executeBitCastInst(Value *SrcVal, Type *DstTy,
                                  ExecutionContext &SF);
  GenericValue executeCastOperation(Instruction::CastOps opcode, Value *SrcVal, 
                                    Type *Ty, ExecutionContext &SF);
  void popStackAndReturnValueToCaller(Type *RetTy, GenericValue Result);

};

} // End llvm namespace

#endif
