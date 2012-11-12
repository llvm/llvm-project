//===-- JIT.h - Class definition for the JIT --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the top-level JIT data structure.
//
//===----------------------------------------------------------------------===//

#ifndef JIT_H
#define JIT_H

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/PassManager.h"
#include "llvm/Support/ValueHandle.h"

namespace llvm {

class Function;
struct JITEvent_EmittedFunctionDetails;
class MachineCodeEmitter;
class MachineCodeInfo;
class TargetJITInfo;
class TargetMachine;

class JITState {
private:
  FunctionPassManager PM;  // Passes to compile a function
  Module *M;               // Module used to create the PM

  /// PendingFunctions - Functions which have not been code generated yet, but
  /// were called from a function being code generated.
  std::vector<AssertingVH<Function> > PendingFunctions;

public:
  explicit JITState(Module *M) : PM(M), M(M) {}

  FunctionPassManager &getPM(const MutexGuard &L) {
    return PM;
  }

  Module *getModule() const { return M; }
  std::vector<AssertingVH<Function> > &getPendingFunctions(const MutexGuard &L){
    return PendingFunctions;
  }
};


class JIT : public ExecutionEngine {
  /// types
  typedef ValueMap<const BasicBlock *, void *>
      BasicBlockAddressMapTy;
  /// data
  TargetMachine &TM;       // The current target we are compiling to
  TargetJITInfo &TJI;      // The JITInfo for the target we are compiling to
  JITCodeEmitter *JCE;     // JCE object
  JITMemoryManager *JMM;
  std::vector<JITEventListener*> EventListeners;

  /// AllocateGVsWithCode - Some applications require that global variables and
  /// code be allocated into the same region of memory, in which case this flag
  /// should be set to true.  Doing so breaks freeMachineCodeForFunction.
  bool AllocateGVsWithCode;

  /// True while the JIT is generating code.  Used to assert against recursive
  /// entry.
  bool isAlreadyCodeGenerating;

  JITState *jitstate;

  /// BasicBlockAddressMap - A mapping between LLVM basic blocks and their
  /// actualized version, only filled for basic blocks that have their address
  /// taken.
  BasicBlockAddressMapTy BasicBlockAddressMap;


  JIT(Module *M, TargetMachine &tm, TargetJITInfo &tji,
      JITMemoryManager *JMM, bool AllocateGVsWithCode);
public:
  ~JIT();

  static void Register() {
    JITCtor = createJIT;
  }

  /// getJITInfo - Return the target JIT information structure.
  ///
  TargetJITInfo &getJITInfo() const { return TJI; }

  /// create - Create an return a new JIT compiler if there is one available
  /// for the current target.  Otherwise, return null.
  ///
  static ExecutionEngine *create(Module *M,
                                 std::string *Err,
                                 JITMemoryManager *JMM,
                                 CodeGenOpt::Level OptLevel =
                                   CodeGenOpt::Default,
                                 bool GVsWithCode = true,
                                 Reloc::Model RM = Reloc::Default,
                                 CodeModel::Model CMM = CodeModel::JITDefault) {
    return ExecutionEngine::createJIT(M, Err, JMM, OptLevel, GVsWithCode,
                                      RM, CMM);
  }

  virtual void addModule(Module *M);

  /// removeModule - Remove a Module from the list of modules.  Returns true if
  /// M is found.
  virtual bool removeModule(Module *M);

  /// runFunction - Start execution with the specified function and arguments.
  ///
  virtual GenericValue runFunction(Function *F,
                                   const std::vector<GenericValue> &ArgValues);

  /// getPointerToNamedFunction - This method returns the address of the
  /// specified function by using the MemoryManager. As such it is only
  /// useful for resolving library symbols, not code generated symbols.
  ///
  /// If AbortOnFailure is false and no function with the given name is
  /// found, this function silently returns a null pointer. Otherwise,
  /// it prints a message to stderr and aborts.
  ///
  virtual void *getPointerToNamedFunction(const std::string &Name,
                                          bool AbortOnFailure = true);

  // CompilationCallback - Invoked the first time that a call site is found,
  // which causes lazy compilation of the target function.
  //
  static void CompilationCallback();

  /// getPointerToFunction - This returns the address of the specified function,
  /// compiling it if necessary.
  ///
  void *getPointerToFunction(Function *F);

  /// addPointerToBasicBlock - Adds address of the specific basic block.
  void addPointerToBasicBlock(const BasicBlock *BB, void *Addr);

  /// clearPointerToBasicBlock - Removes address of specific basic block.
  void clearPointerToBasicBlock(const BasicBlock *BB);

  /// getPointerToBasicBlock - This returns the address of the specified basic
  /// block, assuming function is compiled.
  void *getPointerToBasicBlock(BasicBlock *BB);

  /// getOrEmitGlobalVariable - Return the address of the specified global
  /// variable, possibly emitting it to memory if needed.  This is used by the
  /// Emitter.
  void *getOrEmitGlobalVariable(const GlobalVariable *GV);

  /// getPointerToFunctionOrStub - If the specified function has been
  /// code-gen'd, return a pointer to the function.  If not, compile it, or use
  /// a stub to implement lazy compilation if available.
  ///
  void *getPointerToFunctionOrStub(Function *F);

  /// recompileAndRelinkFunction - This method is used to force a function
  /// which has already been compiled, to be compiled again, possibly
  /// after it has been modified. Then the entry to the old copy is overwritten
  /// with a branch to the new copy. If there was no old copy, this acts
  /// just like JIT::getPointerToFunction().
  ///
  void *recompileAndRelinkFunction(Function *F);

  /// freeMachineCodeForFunction - deallocate memory used to code-generate this
  /// Function.
  ///
  void freeMachineCodeForFunction(Function *F);

  /// addPendingFunction - while jitting non-lazily, a called but non-codegen'd
  /// function was encountered.  Add it to a pending list to be processed after
  /// the current function.
  ///
  void addPendingFunction(Function *F);

  /// getCodeEmitter - Return the code emitter this JIT is emitting into.
  ///
  JITCodeEmitter *getCodeEmitter() const { return JCE; }

  static ExecutionEngine *createJIT(Module *M,
                                    std::string *ErrorStr,
                                    JITMemoryManager *JMM,
                                    bool GVsWithCode,
                                    TargetMachine *TM);

  // Run the JIT on F and return information about the generated code
  void runJITOnFunction(Function *F, MachineCodeInfo *MCI = 0);

  virtual void RegisterJITEventListener(JITEventListener *L);
  virtual void UnregisterJITEventListener(JITEventListener *L);
  /// These functions correspond to the methods on JITEventListener.  They
  /// iterate over the registered listeners and call the corresponding method on
  /// each.
  void NotifyFunctionEmitted(
      const Function &F, void *Code, size_t Size,
      const JITEvent_EmittedFunctionDetails &Details);
  void NotifyFreeingMachineCode(void *OldPtr);

  BasicBlockAddressMapTy &
  getBasicBlockAddressMap(const MutexGuard &) {
    return BasicBlockAddressMap;
  }


private:
  static JITCodeEmitter *createEmitter(JIT &J, JITMemoryManager *JMM,
                                       TargetMachine &tm);
  void runJITOnFunctionUnlocked(Function *F, const MutexGuard &locked);
  void updateFunctionStub(Function *F);
  void jitTheFunction(Function *F, const MutexGuard &locked);

protected:

  /// getMemoryforGV - Allocate memory for a global variable.
  virtual char* getMemoryForGV(const GlobalVariable* GV);

};

} // End llvm namespace

#endif
