//===-- JITEmitter.cpp - Write machine code to executable memory ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a MachineCodeEmitter object that is used by the JIT to
// write machine code to memory and remember where relocatable values are.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#include "JIT.h"
#include "JITDwarfEmitter.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/CodeGen/JITCodeEmitter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRelocation.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/JITMemoryManager.h"
#include "llvm/CodeGen/MachineCodeInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetJITInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Disassembler.h"
#include "llvm/System/Memory.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include <algorithm>
#ifndef NDEBUG
#include <iomanip>
#endif
using namespace llvm;

STATISTIC(NumBytes, "Number of bytes of machine code compiled");
STATISTIC(NumRelos, "Number of relocations applied");
STATISTIC(NumRetries, "Number of retries with more memory");
static JIT *TheJIT = 0;


//===----------------------------------------------------------------------===//
// JIT lazy compilation code.
//
namespace {
  class JITResolverState {
  public:
    typedef std::map<AssertingVH<Function>, void*> FunctionToStubMapTy;
    typedef std::map<void*, Function*> StubToFunctionMapTy;
    typedef std::map<AssertingVH<GlobalValue>, void*> GlobalToIndirectSymMapTy;
  private:
    /// FunctionToStubMap - Keep track of the stub created for a particular
    /// function so that we can reuse them if necessary.
    FunctionToStubMapTy FunctionToStubMap;

    /// StubToFunctionMap - Keep track of the function that each stub
    /// corresponds to.
    StubToFunctionMapTy StubToFunctionMap;

    /// GlobalToIndirectSymMap - Keep track of the indirect symbol created for a
    /// particular GlobalVariable so that we can reuse them if necessary.
    GlobalToIndirectSymMapTy GlobalToIndirectSymMap;

  public:
    FunctionToStubMapTy& getFunctionToStubMap(const MutexGuard& locked) {
      assert(locked.holds(TheJIT->lock));
      return FunctionToStubMap;
    }

    StubToFunctionMapTy& getStubToFunctionMap(const MutexGuard& locked) {
      assert(locked.holds(TheJIT->lock));
      return StubToFunctionMap;
    }

    GlobalToIndirectSymMapTy& getGlobalToIndirectSymMap(const MutexGuard& locked) {
      assert(locked.holds(TheJIT->lock));
      return GlobalToIndirectSymMap;
    }
  };

  /// JITResolver - Keep track of, and resolve, call sites for functions that
  /// have not yet been compiled.
  class JITResolver {
    typedef JITResolverState::FunctionToStubMapTy FunctionToStubMapTy;
    typedef JITResolverState::StubToFunctionMapTy StubToFunctionMapTy;
    typedef JITResolverState::GlobalToIndirectSymMapTy GlobalToIndirectSymMapTy;

    /// LazyResolverFn - The target lazy resolver function that we actually
    /// rewrite instructions to use.
    TargetJITInfo::LazyResolverFn LazyResolverFn;

    JITResolverState state;

    /// ExternalFnToStubMap - This is the equivalent of FunctionToStubMap for
    /// external functions.
    std::map<void*, void*> ExternalFnToStubMap;

    /// revGOTMap - map addresses to indexes in the GOT
    std::map<void*, unsigned> revGOTMap;
    unsigned nextGOTIndex;

    static JITResolver *TheJITResolver;
  public:
    explicit JITResolver(JIT &jit) : nextGOTIndex(0) {
      TheJIT = &jit;

      LazyResolverFn = jit.getJITInfo().getLazyResolverFunction(JITCompilerFn);
      assert(TheJITResolver == 0 && "Multiple JIT resolvers?");
      TheJITResolver = this;
    }
    
    ~JITResolver() {
      TheJITResolver = 0;
    }

    /// getFunctionStubIfAvailable - This returns a pointer to a function stub
    /// if it has already been created.
    void *getFunctionStubIfAvailable(Function *F);

    /// getFunctionStub - This returns a pointer to a function stub, creating
    /// one on demand as needed.  If empty is true, create a function stub
    /// pointing at address 0, to be filled in later.
    void *getFunctionStub(Function *F);

    /// getExternalFunctionStub - Return a stub for the function at the
    /// specified address, created lazily on demand.
    void *getExternalFunctionStub(void *FnAddr);

    /// getGlobalValueIndirectSym - Return an indirect symbol containing the
    /// specified GV address.
    void *getGlobalValueIndirectSym(GlobalValue *V, void *GVAddress);

    /// AddCallbackAtLocation - If the target is capable of rewriting an
    /// instruction without the use of a stub, record the location of the use so
    /// we know which function is being used at the location.
    void *AddCallbackAtLocation(Function *F, void *Location) {
      MutexGuard locked(TheJIT->lock);
      /// Get the target-specific JIT resolver function.
      state.getStubToFunctionMap(locked)[Location] = F;
      return (void*)(intptr_t)LazyResolverFn;
    }
    
    void getRelocatableGVs(SmallVectorImpl<GlobalValue*> &GVs,
                           SmallVectorImpl<void*> &Ptrs);
    
    GlobalValue *invalidateStub(void *Stub);

    /// getGOTIndexForAddress - Return a new or existing index in the GOT for
    /// an address.  This function only manages slots, it does not manage the
    /// contents of the slots or the memory associated with the GOT.
    unsigned getGOTIndexForAddr(void *addr);

    /// JITCompilerFn - This function is called to resolve a stub to a compiled
    /// address.  If the LLVM Function corresponding to the stub has not yet
    /// been compiled, this function compiles it first.
    static void *JITCompilerFn(void *Stub);
  };
}

JITResolver *JITResolver::TheJITResolver = 0;

/// getFunctionStubIfAvailable - This returns a pointer to a function stub
/// if it has already been created.
void *JITResolver::getFunctionStubIfAvailable(Function *F) {
  MutexGuard locked(TheJIT->lock);

  // If we already have a stub for this function, recycle it.
  void *&Stub = state.getFunctionToStubMap(locked)[F];
  return Stub;
}

/// getFunctionStub - This returns a pointer to a function stub, creating
/// one on demand as needed.
void *JITResolver::getFunctionStub(Function *F) {
  MutexGuard locked(TheJIT->lock);

  // If we already have a stub for this function, recycle it.
  void *&Stub = state.getFunctionToStubMap(locked)[F];
  if (Stub) return Stub;

  // Call the lazy resolver function unless we are JIT'ing non-lazily, in which
  // case we must resolve the symbol now.
  void *Actual =  TheJIT->isLazyCompilationDisabled() 
    ? (void *)0 : (void *)(intptr_t)LazyResolverFn;
   
  // If this is an external declaration, attempt to resolve the address now
  // to place in the stub.
  if (F->isDeclaration() && !F->hasNotBeenReadFromBitcode()) {
    Actual = TheJIT->getPointerToFunction(F);

    // If we resolved the symbol to a null address (eg. a weak external)
    // don't emit a stub. Return a null pointer to the application.  If dlsym
    // stubs are enabled, not being able to resolve the address is not
    // meaningful.
    if (!Actual && !TheJIT->areDlsymStubsEnabled()) return 0;
  }

  // Codegen a new stub, calling the lazy resolver or the actual address of the
  // external function, if it was resolved.
  Stub = TheJIT->getJITInfo().emitFunctionStub(F, Actual,
                                               *TheJIT->getCodeEmitter());

  if (Actual != (void*)(intptr_t)LazyResolverFn) {
    // If we are getting the stub for an external function, we really want the
    // address of the stub in the GlobalAddressMap for the JIT, not the address
    // of the external function.
    TheJIT->updateGlobalMapping(F, Stub);
  }

  DEBUG(errs() << "JIT: Stub emitted at [" << Stub << "] for function '"
        << F->getName() << "'\n");

  // Finally, keep track of the stub-to-Function mapping so that the
  // JITCompilerFn knows which function to compile!
  state.getStubToFunctionMap(locked)[Stub] = F;
  
  // If we are JIT'ing non-lazily but need to call a function that does not
  // exist yet, add it to the JIT's work list so that we can fill in the stub
  // address later.
  if (!Actual && TheJIT->isLazyCompilationDisabled())
    if (!F->isDeclaration() || F->hasNotBeenReadFromBitcode())
      TheJIT->addPendingFunction(F);
  
  return Stub;
}

/// getGlobalValueIndirectSym - Return a lazy pointer containing the specified
/// GV address.
void *JITResolver::getGlobalValueIndirectSym(GlobalValue *GV, void *GVAddress) {
  MutexGuard locked(TheJIT->lock);

  // If we already have a stub for this global variable, recycle it.
  void *&IndirectSym = state.getGlobalToIndirectSymMap(locked)[GV];
  if (IndirectSym) return IndirectSym;

  // Otherwise, codegen a new indirect symbol.
  IndirectSym = TheJIT->getJITInfo().emitGlobalValueIndirectSym(GV, GVAddress,
                                                     *TheJIT->getCodeEmitter());

  DEBUG(errs() << "JIT: Indirect symbol emitted at [" << IndirectSym 
        << "] for GV '" << GV->getName() << "'\n");

  return IndirectSym;
}

/// getExternalFunctionStub - Return a stub for the function at the
/// specified address, created lazily on demand.
void *JITResolver::getExternalFunctionStub(void *FnAddr) {
  // If we already have a stub for this function, recycle it.
  void *&Stub = ExternalFnToStubMap[FnAddr];
  if (Stub) return Stub;

  Stub = TheJIT->getJITInfo().emitFunctionStub(0, FnAddr,
                                               *TheJIT->getCodeEmitter());

  DOUT << "JIT: Stub emitted at [" << Stub
       << "] for external function at '" << FnAddr << "'\n";
  return Stub;
}

unsigned JITResolver::getGOTIndexForAddr(void* addr) {
  unsigned idx = revGOTMap[addr];
  if (!idx) {
    idx = ++nextGOTIndex;
    revGOTMap[addr] = idx;
    DOUT << "JIT: Adding GOT entry " << idx << " for addr [" << addr << "]\n";
  }
  return idx;
}

void JITResolver::getRelocatableGVs(SmallVectorImpl<GlobalValue*> &GVs,
                                    SmallVectorImpl<void*> &Ptrs) {
  MutexGuard locked(TheJIT->lock);
  
  FunctionToStubMapTy &FM = state.getFunctionToStubMap(locked);
  GlobalToIndirectSymMapTy &GM = state.getGlobalToIndirectSymMap(locked);
  
  for (FunctionToStubMapTy::iterator i = FM.begin(), e = FM.end(); i != e; ++i){
    Function *F = i->first;
    if (F->isDeclaration() && F->hasExternalLinkage()) {
      GVs.push_back(i->first);
      Ptrs.push_back(i->second);
    }
  }
  for (GlobalToIndirectSymMapTy::iterator i = GM.begin(), e = GM.end();
       i != e; ++i) {
    GVs.push_back(i->first);
    Ptrs.push_back(i->second);
  }
}

GlobalValue *JITResolver::invalidateStub(void *Stub) {
  MutexGuard locked(TheJIT->lock);
  
  FunctionToStubMapTy &FM = state.getFunctionToStubMap(locked);
  StubToFunctionMapTy &SM = state.getStubToFunctionMap(locked);
  GlobalToIndirectSymMapTy &GM = state.getGlobalToIndirectSymMap(locked);
  
  // Look up the cheap way first, to see if it's a function stub we are
  // invalidating.  If so, remove it from both the forward and reverse maps.
  if (SM.find(Stub) != SM.end()) {
    Function *F = SM[Stub];
    SM.erase(Stub);
    FM.erase(F);
    return F;
  }
  
  // Otherwise, it might be an indirect symbol stub.  Find it and remove it.
  for (GlobalToIndirectSymMapTy::iterator i = GM.begin(), e = GM.end();
       i != e; ++i) {
    if (i->second != Stub)
      continue;
    GlobalValue *GV = i->first;
    GM.erase(i);
    return GV;
  }
  
  // Lastly, check to see if it's in the ExternalFnToStubMap.
  for (std::map<void *, void *>::iterator i = ExternalFnToStubMap.begin(),
       e = ExternalFnToStubMap.end(); i != e; ++i) {
    if (i->second != Stub)
      continue;
    ExternalFnToStubMap.erase(i);
    break;
  }
  
  return 0;
}

/// JITCompilerFn - This function is called when a lazy compilation stub has
/// been entered.  It looks up which function this stub corresponds to, compiles
/// it if necessary, then returns the resultant function pointer.
void *JITResolver::JITCompilerFn(void *Stub) {
  JITResolver &JR = *TheJITResolver;
  
  Function* F = 0;
  void* ActualPtr = 0;

  {
    // Only lock for getting the Function. The call getPointerToFunction made
    // in this function might trigger function materializing, which requires
    // JIT lock to be unlocked.
    MutexGuard locked(TheJIT->lock);

    // The address given to us for the stub may not be exactly right, it might be
    // a little bit after the stub.  As such, use upper_bound to find it.
    StubToFunctionMapTy::iterator I =
      JR.state.getStubToFunctionMap(locked).upper_bound(Stub);
    assert(I != JR.state.getStubToFunctionMap(locked).begin() &&
           "This is not a known stub!");
    F = (--I)->second;
    ActualPtr = I->first;
  }

  // If we have already code generated the function, just return the address.
  void *Result = TheJIT->getPointerToGlobalIfAvailable(F);
  
  if (!Result) {
    // Otherwise we don't have it, do lazy compilation now.
    
    // If lazy compilation is disabled, emit a useful error message and abort.
    if (TheJIT->isLazyCompilationDisabled()) {
      llvm_report_error("LLVM JIT requested to do lazy compilation of function '"
                        + F->getName() + "' when lazy compiles are disabled!");
    }
  
    // We might like to remove the stub from the StubToFunction map.
    // We can't do that! Multiple threads could be stuck, waiting to acquire the
    // lock above. As soon as the 1st function finishes compiling the function,
    // the next one will be released, and needs to be able to find the function
    // it needs to call.
    //JR.state.getStubToFunctionMap(locked).erase(I);

    DEBUG(errs() << "JIT: Lazily resolving function '" << F->getName()
          << "' In stub ptr = " << Stub << " actual ptr = "
          << ActualPtr << "\n");

    Result = TheJIT->getPointerToFunction(F);
  }
  
  // Reacquire the lock to erase the stub in the map.
  MutexGuard locked(TheJIT->lock);

  // We don't need to reuse this stub in the future, as F is now compiled.
  JR.state.getFunctionToStubMap(locked).erase(F);

  // FIXME: We could rewrite all references to this stub if we knew them.

  // What we will do is set the compiled function address to map to the
  // same GOT entry as the stub so that later clients may update the GOT
  // if they see it still using the stub address.
  // Note: this is done so the Resolver doesn't have to manage GOT memory
  // Do this without allocating map space if the target isn't using a GOT
  if(JR.revGOTMap.find(Stub) != JR.revGOTMap.end())
    JR.revGOTMap[Result] = JR.revGOTMap[Stub];

  return Result;
}

//===----------------------------------------------------------------------===//
// JITEmitter code.
//
namespace {
  /// JITEmitter - The JIT implementation of the MachineCodeEmitter, which is
  /// used to output functions to memory for execution.
  class JITEmitter : public JITCodeEmitter {
    JITMemoryManager *MemMgr;

    // When outputting a function stub in the context of some other function, we
    // save BufferBegin/BufferEnd/CurBufferPtr here.
    uint8_t *SavedBufferBegin, *SavedBufferEnd, *SavedCurBufferPtr;

    // When reattempting to JIT a function after running out of space, we store
    // the estimated size of the function we're trying to JIT here, so we can
    // ask the memory manager for at least this much space.  When we
    // successfully emit the function, we reset this back to zero.
    uintptr_t SizeEstimate;

    /// Relocations - These are the relocations that the function needs, as
    /// emitted.
    std::vector<MachineRelocation> Relocations;
    
    /// MBBLocations - This vector is a mapping from MBB ID's to their address.
    /// It is filled in by the StartMachineBasicBlock callback and queried by
    /// the getMachineBasicBlockAddress callback.
    std::vector<uintptr_t> MBBLocations;

    /// ConstantPool - The constant pool for the current function.
    ///
    MachineConstantPool *ConstantPool;

    /// ConstantPoolBase - A pointer to the first entry in the constant pool.
    ///
    void *ConstantPoolBase;

    /// ConstPoolAddresses - Addresses of individual constant pool entries.
    ///
    SmallVector<uintptr_t, 8> ConstPoolAddresses;

    /// JumpTable - The jump tables for the current function.
    ///
    MachineJumpTableInfo *JumpTable;
    
    /// JumpTableBase - A pointer to the first entry in the jump table.
    ///
    void *JumpTableBase;

    /// Resolver - This contains info about the currently resolved functions.
    JITResolver Resolver;
    
    /// DE - The dwarf emitter for the jit.
    JITDwarfEmitter *DE;

    /// LabelLocations - This vector is a mapping from Label ID's to their 
    /// address.
    std::vector<uintptr_t> LabelLocations;

    /// MMI - Machine module info for exception informations
    MachineModuleInfo* MMI;

    // GVSet - a set to keep track of which globals have been seen
    SmallPtrSet<const GlobalVariable*, 8> GVSet;

    // CurFn - The llvm function being emitted.  Only valid during 
    // finishFunction().
    const Function *CurFn;

    /// Information about emitted code, which is passed to the
    /// JITEventListeners.  This is reset in startFunction and used in
    /// finishFunction.
    JITEvent_EmittedFunctionDetails EmissionDetails;

    // CurFnStubUses - For a given Function, a vector of stubs that it
    // references.  This facilitates the JIT detecting that a stub is no
    // longer used, so that it may be deallocated.
    DenseMap<const Function *, SmallVector<void*, 1> > CurFnStubUses;
    
    // StubFnRefs - For a given pointer to a stub, a set of Functions which
    // reference the stub.  When the count of a stub's references drops to zero,
    // the stub is unused.
    DenseMap<void *, SmallPtrSet<const Function*, 1> > StubFnRefs;
    
    // ExtFnStubs - A map of external function names to stubs which have entries
    // in the JITResolver's ExternalFnToStubMap.
    StringMap<void *> ExtFnStubs;

    DebugLocTuple PrevDLT;

  public:
    JITEmitter(JIT &jit, JITMemoryManager *JMM)
        : SizeEstimate(0), Resolver(jit), MMI(0), CurFn(0) {
      MemMgr = JMM ? JMM : JITMemoryManager::CreateDefaultMemManager();
      if (jit.getJITInfo().needsGOT()) {
        MemMgr->AllocateGOT();
        DOUT << "JIT is managing a GOT\n";
      }

      if (DwarfExceptionHandling) DE = new JITDwarfEmitter(jit);
    }
    ~JITEmitter() { 
      delete MemMgr;
      if (DwarfExceptionHandling) delete DE;
    }

    /// classof - Methods for support type inquiry through isa, cast, and
    /// dyn_cast:
    ///
    static inline bool classof(const JITEmitter*) { return true; }
    static inline bool classof(const MachineCodeEmitter*) { return true; }
    
    JITResolver &getJITResolver() { return Resolver; }

    virtual void startFunction(MachineFunction &F);
    virtual bool finishFunction(MachineFunction &F);
    
    void emitConstantPool(MachineConstantPool *MCP);
    void initJumpTableInfo(MachineJumpTableInfo *MJTI);
    void emitJumpTableInfo(MachineJumpTableInfo *MJTI);
    
    virtual void startGVStub(const GlobalValue* GV, unsigned StubSize,
                                   unsigned Alignment = 1);
    virtual void startGVStub(const GlobalValue* GV, void *Buffer,
                             unsigned StubSize);
    virtual void* finishGVStub(const GlobalValue *GV);

    /// allocateSpace - Reserves space in the current block if any, or
    /// allocate a new one of the given size.
    virtual void *allocateSpace(uintptr_t Size, unsigned Alignment);

    /// allocateGlobal - Allocate memory for a global.  Unlike allocateSpace,
    /// this method does not allocate memory in the current output buffer,
    /// because a global may live longer than the current function.
    virtual void *allocateGlobal(uintptr_t Size, unsigned Alignment);

    virtual void addRelocation(const MachineRelocation &MR) {
      Relocations.push_back(MR);
    }
    
    virtual void StartMachineBasicBlock(MachineBasicBlock *MBB) {
      if (MBBLocations.size() <= (unsigned)MBB->getNumber())
        MBBLocations.resize((MBB->getNumber()+1)*2);
      MBBLocations[MBB->getNumber()] = getCurrentPCValue();
      DOUT << "JIT: Emitting BB" << MBB->getNumber() << " at ["
           << (void*) getCurrentPCValue() << "]\n";
    }

    virtual uintptr_t getConstantPoolEntryAddress(unsigned Entry) const;
    virtual uintptr_t getJumpTableEntryAddress(unsigned Entry) const;

    virtual uintptr_t getMachineBasicBlockAddress(MachineBasicBlock *MBB) const {
      assert(MBBLocations.size() > (unsigned)MBB->getNumber() && 
             MBBLocations[MBB->getNumber()] && "MBB not emitted!");
      return MBBLocations[MBB->getNumber()];
    }

    /// retryWithMoreMemory - Log a retry and deallocate all memory for the
    /// given function.  Increase the minimum allocation size so that we get
    /// more memory next time.
    void retryWithMoreMemory(MachineFunction &F);

    /// deallocateMemForFunction - Deallocate all memory for the specified
    /// function body.
    void deallocateMemForFunction(const Function *F);

    /// AddStubToCurrentFunction - Mark the current function being JIT'd as
    /// using the stub at the specified address. Allows
    /// deallocateMemForFunction to also remove stubs no longer referenced.
    void AddStubToCurrentFunction(void *Stub);
    
    /// getExternalFnStubs - Accessor for the JIT to find stubs emitted for
    /// MachineRelocations that reference external functions by name.
    const StringMap<void*> &getExternalFnStubs() const { return ExtFnStubs; }
    
    virtual void processDebugLoc(DebugLoc DL);

    virtual void emitLabel(uint64_t LabelID) {
      if (LabelLocations.size() <= LabelID)
        LabelLocations.resize((LabelID+1)*2);
      LabelLocations[LabelID] = getCurrentPCValue();
    }

    virtual uintptr_t getLabelAddress(uint64_t LabelID) const {
      assert(LabelLocations.size() > (unsigned)LabelID && 
             LabelLocations[LabelID] && "Label not emitted!");
      return LabelLocations[LabelID];
    }
 
    virtual void setModuleInfo(MachineModuleInfo* Info) {
      MMI = Info;
      if (DwarfExceptionHandling) DE->setModuleInfo(Info);
    }

    void setMemoryExecutable() {
      MemMgr->setMemoryExecutable();
    }
    
    JITMemoryManager *getMemMgr() const { return MemMgr; }

  private:
    void *getPointerToGlobal(GlobalValue *GV, void *Reference, bool NoNeedStub);
    void *getPointerToGVIndirectSym(GlobalValue *V, void *Reference,
                                    bool NoNeedStub);
    unsigned addSizeOfGlobal(const GlobalVariable *GV, unsigned Size);
    unsigned addSizeOfGlobalsInConstantVal(const Constant *C, unsigned Size);
    unsigned addSizeOfGlobalsInInitializer(const Constant *Init, unsigned Size);
    unsigned GetSizeOfGlobalsInBytes(MachineFunction &MF);
  };
}

void *JITEmitter::getPointerToGlobal(GlobalValue *V, void *Reference,
                                     bool DoesntNeedStub) {
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V))
    return TheJIT->getOrEmitGlobalVariable(GV);

  if (GlobalAlias *GA = dyn_cast<GlobalAlias>(V))
    return TheJIT->getPointerToGlobal(GA->resolveAliasedGlobal(false));

  // If we have already compiled the function, return a pointer to its body.
  Function *F = cast<Function>(V);
  void *ResultPtr;
  if (!DoesntNeedStub && !TheJIT->isLazyCompilationDisabled()) {
    // Return the function stub if it's already created.
    ResultPtr = Resolver.getFunctionStubIfAvailable(F);
    if (ResultPtr)
      AddStubToCurrentFunction(ResultPtr);
  } else {
    ResultPtr = TheJIT->getPointerToGlobalIfAvailable(F);
  }
  if (ResultPtr) return ResultPtr;

  // If this is an external function pointer, we can force the JIT to
  // 'compile' it, which really just adds it to the map.  In dlsym mode, 
  // external functions are forced through a stub, regardless of reloc type.
  if (F->isDeclaration() && !F->hasNotBeenReadFromBitcode() &&
      DoesntNeedStub && !TheJIT->areDlsymStubsEnabled())
    return TheJIT->getPointerToFunction(F);

  // Okay, the function has not been compiled yet, if the target callback
  // mechanism is capable of rewriting the instruction directly, prefer to do
  // that instead of emitting a stub.  This uses the lazy resolver, so is not
  // legal if lazy compilation is disabled.
  if (DoesntNeedStub && !TheJIT->isLazyCompilationDisabled())
    return Resolver.AddCallbackAtLocation(F, Reference);

  // Otherwise, we have to emit a stub.
  void *StubAddr = Resolver.getFunctionStub(F);

  // Add the stub to the current function's list of referenced stubs, so we can
  // deallocate them if the current function is ever freed.  It's possible to
  // return null from getFunctionStub in the case of a weak extern that fails
  // to resolve.
  if (StubAddr)
    AddStubToCurrentFunction(StubAddr);

  return StubAddr;
}

void *JITEmitter::getPointerToGVIndirectSym(GlobalValue *V, void *Reference,
                                            bool NoNeedStub) {
  // Make sure GV is emitted first, and create a stub containing the fully
  // resolved address.
  void *GVAddress = getPointerToGlobal(V, Reference, true);
  void *StubAddr = Resolver.getGlobalValueIndirectSym(V, GVAddress);
  
  // Add the stub to the current function's list of referenced stubs, so we can
  // deallocate them if the current function is ever freed.
  AddStubToCurrentFunction(StubAddr);
  
  return StubAddr;
}

void JITEmitter::AddStubToCurrentFunction(void *StubAddr) {
  if (!TheJIT->areDlsymStubsEnabled())
    return;
  
  assert(CurFn && "Stub added to current function, but current function is 0!");
  
  SmallVectorImpl<void*> &StubsUsed = CurFnStubUses[CurFn];
  StubsUsed.push_back(StubAddr);

  SmallPtrSet<const Function *, 1> &FnRefs = StubFnRefs[StubAddr];
  FnRefs.insert(CurFn);
}

void JITEmitter::processDebugLoc(DebugLoc DL) {
  if (!DL.isUnknown()) {
    DebugLocTuple CurDLT = EmissionDetails.MF->getDebugLocTuple(DL);

    if (CurDLT.CompileUnit != 0 && PrevDLT != CurDLT) {
      JITEvent_EmittedFunctionDetails::LineStart NextLine;
      NextLine.Address = getCurrentPCValue();
      NextLine.Loc = DL;
      EmissionDetails.LineStarts.push_back(NextLine);
    }

    PrevDLT = CurDLT;
  }
}

static unsigned GetConstantPoolSizeInBytes(MachineConstantPool *MCP,
                                           const TargetData *TD) {
  const std::vector<MachineConstantPoolEntry> &Constants = MCP->getConstants();
  if (Constants.empty()) return 0;

  unsigned Size = 0;
  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    MachineConstantPoolEntry CPE = Constants[i];
    unsigned AlignMask = CPE.getAlignment() - 1;
    Size = (Size + AlignMask) & ~AlignMask;
    const Type *Ty = CPE.getType();
    Size += TD->getTypeAllocSize(Ty);
  }
  return Size;
}

static unsigned GetJumpTableSizeInBytes(MachineJumpTableInfo *MJTI) {
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  if (JT.empty()) return 0;
  
  unsigned NumEntries = 0;
  for (unsigned i = 0, e = JT.size(); i != e; ++i)
    NumEntries += JT[i].MBBs.size();

  unsigned EntrySize = MJTI->getEntrySize();

  return NumEntries * EntrySize;
}

static uintptr_t RoundUpToAlign(uintptr_t Size, unsigned Alignment) {
  if (Alignment == 0) Alignment = 1;
  // Since we do not know where the buffer will be allocated, be pessimistic. 
  return Size + Alignment;
}

/// addSizeOfGlobal - add the size of the global (plus any alignment padding)
/// into the running total Size.

unsigned JITEmitter::addSizeOfGlobal(const GlobalVariable *GV, unsigned Size) {
  const Type *ElTy = GV->getType()->getElementType();
  size_t GVSize = (size_t)TheJIT->getTargetData()->getTypeAllocSize(ElTy);
  size_t GVAlign = 
      (size_t)TheJIT->getTargetData()->getPreferredAlignment(GV);
  DOUT << "JIT: Adding in size " << GVSize << " alignment " << GVAlign;
  DEBUG(GV->dump());
  // Assume code section ends with worst possible alignment, so first
  // variable needs maximal padding.
  if (Size==0)
    Size = 1;
  Size = ((Size+GVAlign-1)/GVAlign)*GVAlign;
  Size += GVSize;
  return Size;
}

/// addSizeOfGlobalsInConstantVal - find any globals that we haven't seen yet
/// but are referenced from the constant; put them in GVSet and add their
/// size into the running total Size.

unsigned JITEmitter::addSizeOfGlobalsInConstantVal(const Constant *C, 
                                              unsigned Size) {
  // If its undefined, return the garbage.
  if (isa<UndefValue>(C))
    return Size;

  // If the value is a ConstantExpr
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    Constant *Op0 = CE->getOperand(0);
    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr:
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::BitCast: {
      Size = addSizeOfGlobalsInConstantVal(Op0, Size);
      break;
    }
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {
      Size = addSizeOfGlobalsInConstantVal(Op0, Size);
      Size = addSizeOfGlobalsInConstantVal(CE->getOperand(1), Size);
      break;
    }
    default: {
       std::string msg;
       raw_string_ostream Msg(msg);
       Msg << "ConstantExpr not handled: " << *CE;
       llvm_report_error(Msg.str());
    }
    }
  }

  if (C->getType()->getTypeID() == Type::PointerTyID)
    if (const GlobalVariable* GV = dyn_cast<GlobalVariable>(C))
      if (GVSet.insert(GV))
        Size = addSizeOfGlobal(GV, Size);

  return Size;
}

/// addSizeOfGLobalsInInitializer - handle any globals that we haven't seen yet
/// but are referenced from the given initializer.

unsigned JITEmitter::addSizeOfGlobalsInInitializer(const Constant *Init, 
                                              unsigned Size) {
  if (!isa<UndefValue>(Init) &&
      !isa<ConstantVector>(Init) &&
      !isa<ConstantAggregateZero>(Init) &&
      !isa<ConstantArray>(Init) &&
      !isa<ConstantStruct>(Init) &&
      Init->getType()->isFirstClassType())
    Size = addSizeOfGlobalsInConstantVal(Init, Size);
  return Size;
}

/// GetSizeOfGlobalsInBytes - walk the code for the function, looking for
/// globals; then walk the initializers of those globals looking for more.
/// If their size has not been considered yet, add it into the running total
/// Size.

unsigned JITEmitter::GetSizeOfGlobalsInBytes(MachineFunction &MF) {
  unsigned Size = 0;
  GVSet.clear();

  for (MachineFunction::iterator MBB = MF.begin(), E = MF.end(); 
       MBB != E; ++MBB) {
    for (MachineBasicBlock::const_iterator I = MBB->begin(), E = MBB->end();
         I != E; ++I) {
      const TargetInstrDesc &Desc = I->getDesc();
      const MachineInstr &MI = *I;
      unsigned NumOps = Desc.getNumOperands();
      for (unsigned CurOp = 0; CurOp < NumOps; CurOp++) {
        const MachineOperand &MO = MI.getOperand(CurOp);
        if (MO.isGlobal()) {
          GlobalValue* V = MO.getGlobal();
          const GlobalVariable *GV = dyn_cast<const GlobalVariable>(V);
          if (!GV)
            continue;
          // If seen in previous function, it will have an entry here.
          if (TheJIT->getPointerToGlobalIfAvailable(GV))
            continue;
          // If seen earlier in this function, it will have an entry here.
          // FIXME: it should be possible to combine these tables, by
          // assuming the addresses of the new globals in this module
          // start at 0 (or something) and adjusting them after codegen
          // complete.  Another possibility is to grab a marker bit in GV.
          if (GVSet.insert(GV))
            // A variable as yet unseen.  Add in its size.
            Size = addSizeOfGlobal(GV, Size);
        }
      }
    }
  }
  DOUT << "JIT: About to look through initializers\n";
  // Look for more globals that are referenced only from initializers.
  // GVSet.end is computed each time because the set can grow as we go.
  for (SmallPtrSet<const GlobalVariable *, 8>::iterator I = GVSet.begin(); 
       I != GVSet.end(); I++) {
    const GlobalVariable* GV = *I;
    if (GV->hasInitializer())
      Size = addSizeOfGlobalsInInitializer(GV->getInitializer(), Size);
  }

  return Size;
}

void JITEmitter::startFunction(MachineFunction &F) {
  DEBUG(errs() << "JIT: Starting CodeGen of Function "
        << F.getFunction()->getName() << "\n");

  uintptr_t ActualSize = 0;
  // Set the memory writable, if it's not already
  MemMgr->setMemoryWritable();
  if (MemMgr->NeedsExactSize()) {
    DOUT << "JIT: ExactSize\n";
    const TargetInstrInfo* TII = F.getTarget().getInstrInfo();
    MachineJumpTableInfo *MJTI = F.getJumpTableInfo();
    MachineConstantPool *MCP = F.getConstantPool();
    
    // Ensure the constant pool/jump table info is at least 4-byte aligned.
    ActualSize = RoundUpToAlign(ActualSize, 16);
    
    // Add the alignment of the constant pool
    ActualSize = RoundUpToAlign(ActualSize, MCP->getConstantPoolAlignment());

    // Add the constant pool size
    ActualSize += GetConstantPoolSizeInBytes(MCP, TheJIT->getTargetData());

    // Add the aligment of the jump table info
    ActualSize = RoundUpToAlign(ActualSize, MJTI->getAlignment());

    // Add the jump table size
    ActualSize += GetJumpTableSizeInBytes(MJTI);
    
    // Add the alignment for the function
    ActualSize = RoundUpToAlign(ActualSize,
                                std::max(F.getFunction()->getAlignment(), 8U));

    // Add the function size
    ActualSize += TII->GetFunctionSizeInBytes(F);

    DOUT << "JIT: ActualSize before globals " << ActualSize << "\n";
    // Add the size of the globals that will be allocated after this function.
    // These are all the ones referenced from this function that were not
    // previously allocated.
    ActualSize += GetSizeOfGlobalsInBytes(F);
    DOUT << "JIT: ActualSize after globals " << ActualSize << "\n";
  } else if (SizeEstimate > 0) {
    // SizeEstimate will be non-zero on reallocation attempts.
    ActualSize = SizeEstimate;
  }

  BufferBegin = CurBufferPtr = MemMgr->startFunctionBody(F.getFunction(),
                                                         ActualSize);
  BufferEnd = BufferBegin+ActualSize;
  
  // Ensure the constant pool/jump table info is at least 4-byte aligned.
  emitAlignment(16);

  emitConstantPool(F.getConstantPool());
  initJumpTableInfo(F.getJumpTableInfo());

  // About to start emitting the machine code for the function.
  emitAlignment(std::max(F.getFunction()->getAlignment(), 8U));
  TheJIT->updateGlobalMapping(F.getFunction(), CurBufferPtr);

  MBBLocations.clear();

  EmissionDetails.MF = &F;
  EmissionDetails.LineStarts.clear();
}

bool JITEmitter::finishFunction(MachineFunction &F) {
  if (CurBufferPtr == BufferEnd) {
    // We must call endFunctionBody before retrying, because
    // deallocateMemForFunction requires it.
    MemMgr->endFunctionBody(F.getFunction(), BufferBegin, CurBufferPtr);
    retryWithMoreMemory(F);
    return true;
  }

  emitJumpTableInfo(F.getJumpTableInfo());

  // FnStart is the start of the text, not the start of the constant pool and
  // other per-function data.
  uint8_t *FnStart =
    (uint8_t *)TheJIT->getPointerToGlobalIfAvailable(F.getFunction());

  // FnEnd is the end of the function's machine code.
  uint8_t *FnEnd = CurBufferPtr;

  if (!Relocations.empty()) {
    CurFn = F.getFunction();
    NumRelos += Relocations.size();

    // Resolve the relocations to concrete pointers.
    for (unsigned i = 0, e = Relocations.size(); i != e; ++i) {
      MachineRelocation &MR = Relocations[i];
      void *ResultPtr = 0;
      if (!MR.letTargetResolve()) {
        if (MR.isExternalSymbol()) {
          ResultPtr = TheJIT->getPointerToNamedFunction(MR.getExternalSymbol(),
                                                        false);
          DOUT << "JIT: Map \'" << MR.getExternalSymbol() << "\' to ["
               << ResultPtr << "]\n";  

          // If the target REALLY wants a stub for this function, emit it now.
          if (!MR.doesntNeedStub()) {
            if (!TheJIT->areDlsymStubsEnabled()) {
              ResultPtr = Resolver.getExternalFunctionStub(ResultPtr);
            } else {
              void *&Stub = ExtFnStubs[MR.getExternalSymbol()];
              if (!Stub) {
                Stub = Resolver.getExternalFunctionStub((void *)&Stub);
                AddStubToCurrentFunction(Stub);
              }
              ResultPtr = Stub;
            }
          }
        } else if (MR.isGlobalValue()) {
          ResultPtr = getPointerToGlobal(MR.getGlobalValue(),
                                         BufferBegin+MR.getMachineCodeOffset(),
                                         MR.doesntNeedStub());
        } else if (MR.isIndirectSymbol()) {
          ResultPtr = getPointerToGVIndirectSym(MR.getGlobalValue(),
                                          BufferBegin+MR.getMachineCodeOffset(),
                                          MR.doesntNeedStub());
        } else if (MR.isBasicBlock()) {
          ResultPtr = (void*)getMachineBasicBlockAddress(MR.getBasicBlock());
        } else if (MR.isConstantPoolIndex()) {
          ResultPtr = (void*)getConstantPoolEntryAddress(MR.getConstantPoolIndex());
        } else {
          assert(MR.isJumpTableIndex());
          ResultPtr=(void*)getJumpTableEntryAddress(MR.getJumpTableIndex());
        }

        MR.setResultPointer(ResultPtr);
      }

      // if we are managing the GOT and the relocation wants an index,
      // give it one
      if (MR.isGOTRelative() && MemMgr->isManagingGOT()) {
        unsigned idx = Resolver.getGOTIndexForAddr(ResultPtr);
        MR.setGOTIndex(idx);
        if (((void**)MemMgr->getGOTBase())[idx] != ResultPtr) {
          DOUT << "JIT: GOT was out of date for " << ResultPtr
               << " pointing at " << ((void**)MemMgr->getGOTBase())[idx]
               << "\n";
          ((void**)MemMgr->getGOTBase())[idx] = ResultPtr;
        }
      }
    }

    CurFn = 0;
    TheJIT->getJITInfo().relocate(BufferBegin, &Relocations[0],
                                  Relocations.size(), MemMgr->getGOTBase());
  }

  // Update the GOT entry for F to point to the new code.
  if (MemMgr->isManagingGOT()) {
    unsigned idx = Resolver.getGOTIndexForAddr((void*)BufferBegin);
    if (((void**)MemMgr->getGOTBase())[idx] != (void*)BufferBegin) {
      DOUT << "JIT: GOT was out of date for " << (void*)BufferBegin
           << " pointing at " << ((void**)MemMgr->getGOTBase())[idx] << "\n";
      ((void**)MemMgr->getGOTBase())[idx] = (void*)BufferBegin;
    }
  }

  // CurBufferPtr may have moved beyond FnEnd, due to memory allocation for
  // global variables that were referenced in the relocations.
  MemMgr->endFunctionBody(F.getFunction(), BufferBegin, CurBufferPtr);

  if (CurBufferPtr == BufferEnd) {
    retryWithMoreMemory(F);
    return true;
  } else {
    // Now that we've succeeded in emitting the function, reset the
    // SizeEstimate back down to zero.
    SizeEstimate = 0;
  }

  BufferBegin = CurBufferPtr = 0;
  NumBytes += FnEnd-FnStart;

  // Invalidate the icache if necessary.
  sys::Memory::InvalidateInstructionCache(FnStart, FnEnd-FnStart);

  TheJIT->NotifyFunctionEmitted(*F.getFunction(), FnStart, FnEnd-FnStart,
                                EmissionDetails);

  DEBUG(errs() << "JIT: Finished CodeGen of [" << (void*)FnStart
        << "] Function: " << F.getFunction()->getName()
        << ": " << (FnEnd-FnStart) << " bytes of text, "
        << Relocations.size() << " relocations\n");

  Relocations.clear();
  ConstPoolAddresses.clear();

  // Mark code region readable and executable if it's not so already.
  MemMgr->setMemoryExecutable();

#ifndef NDEBUG
  {
    if (sys::hasDisassembler()) {
      DOUT << "JIT: Disassembled code:\n";
      DOUT << sys::disassembleBuffer(FnStart, FnEnd-FnStart, (uintptr_t)FnStart);
    } else {
      DOUT << "JIT: Binary code:\n";
      DOUT << std::hex;
      uint8_t* q = FnStart;
      for (int i = 0; q < FnEnd; q += 4, ++i) {
        if (i == 4)
          i = 0;
        if (i == 0)
          DOUT << "JIT: " << std::setw(8) << std::setfill('0')
               << (long)(q - FnStart) << ": ";
        bool Done = false;
        for (int j = 3; j >= 0; --j) {
          if (q + j >= FnEnd)
            Done = true;
          else
            DOUT << std::setw(2) << std::setfill('0') << (unsigned short)q[j];
        }
        if (Done)
          break;
        DOUT << ' ';
        if (i == 3)
          DOUT << '\n';
      }
      DOUT << std::dec;
      DOUT<< '\n';
    }
  }
#endif
  if (DwarfExceptionHandling) {
    uintptr_t ActualSize = 0;
    SavedBufferBegin = BufferBegin;
    SavedBufferEnd = BufferEnd;
    SavedCurBufferPtr = CurBufferPtr;
    
    if (MemMgr->NeedsExactSize()) {
      ActualSize = DE->GetDwarfTableSizeInBytes(F, *this, FnStart, FnEnd);
    }

    BufferBegin = CurBufferPtr = MemMgr->startExceptionTable(F.getFunction(),
                                                             ActualSize);
    BufferEnd = BufferBegin+ActualSize;
    uint8_t* FrameRegister = DE->EmitDwarfTable(F, *this, FnStart, FnEnd);
    MemMgr->endExceptionTable(F.getFunction(), BufferBegin, CurBufferPtr,
                              FrameRegister);
    BufferBegin = SavedBufferBegin;
    BufferEnd = SavedBufferEnd;
    CurBufferPtr = SavedCurBufferPtr;

    TheJIT->RegisterTable(FrameRegister);
  }

  if (MMI)
    MMI->EndFunction();
 
  return false;
}

void JITEmitter::retryWithMoreMemory(MachineFunction &F) {
  DOUT << "JIT: Ran out of space for native code.  Reattempting.\n";
  Relocations.clear();  // Clear the old relocations or we'll reapply them.
  ConstPoolAddresses.clear();
  ++NumRetries;
  deallocateMemForFunction(F.getFunction());
  // Try again with at least twice as much free space.
  SizeEstimate = (uintptr_t)(2 * (BufferEnd - BufferBegin));
}

/// deallocateMemForFunction - Deallocate all memory for the specified
/// function body.  Also drop any references the function has to stubs.
void JITEmitter::deallocateMemForFunction(const Function *F) {
  MemMgr->deallocateMemForFunction(F);

  // If the function did not reference any stubs, return.
  if (CurFnStubUses.find(F) == CurFnStubUses.end())
    return;
  
  // For each referenced stub, erase the reference to this function, and then
  // erase the list of referenced stubs.
  SmallVectorImpl<void *> &StubList = CurFnStubUses[F];
  for (unsigned i = 0, e = StubList.size(); i != e; ++i) {
    void *Stub = StubList[i];
    
    // If we already invalidated this stub for this function, continue.
    if (StubFnRefs.count(Stub) == 0)
      continue;
      
    SmallPtrSet<const Function *, 1> &FnRefs = StubFnRefs[Stub];
    FnRefs.erase(F);
    
    // If this function was the last reference to the stub, invalidate the stub
    // in the JITResolver.  Were there a memory manager deallocateStub routine,
    // we could call that at this point too.
    if (FnRefs.empty()) {
      DOUT << "\nJIT: Invalidated Stub at [" << Stub << "]\n";
      StubFnRefs.erase(Stub);

      // Invalidate the stub.  If it is a GV stub, update the JIT's global
      // mapping for that GV to zero, otherwise, search the string map of
      // external function names to stubs and remove the entry for this stub.
      GlobalValue *GV = Resolver.invalidateStub(Stub);
      if (GV) {
        TheJIT->updateGlobalMapping(GV, 0);
      } else {
        for (StringMapIterator<void*> i = ExtFnStubs.begin(),
             e = ExtFnStubs.end(); i != e; ++i) {
          if (i->second == Stub) {
            ExtFnStubs.erase(i);
            break;
          }
        }
      }
    }
  }
  CurFnStubUses.erase(F);
}


void* JITEmitter::allocateSpace(uintptr_t Size, unsigned Alignment) {
  if (BufferBegin)
    return JITCodeEmitter::allocateSpace(Size, Alignment);

  // create a new memory block if there is no active one.
  // care must be taken so that BufferBegin is invalidated when a
  // block is trimmed
  BufferBegin = CurBufferPtr = MemMgr->allocateSpace(Size, Alignment);
  BufferEnd = BufferBegin+Size;
  return CurBufferPtr;
}

void* JITEmitter::allocateGlobal(uintptr_t Size, unsigned Alignment) {
  // Delegate this call through the memory manager.
  return MemMgr->allocateGlobal(Size, Alignment);
}

void JITEmitter::emitConstantPool(MachineConstantPool *MCP) {
  if (TheJIT->getJITInfo().hasCustomConstantPool())
    return;

  const std::vector<MachineConstantPoolEntry> &Constants = MCP->getConstants();
  if (Constants.empty()) return;

  unsigned Size = GetConstantPoolSizeInBytes(MCP, TheJIT->getTargetData());
  unsigned Align = MCP->getConstantPoolAlignment();
  ConstantPoolBase = allocateSpace(Size, Align);
  ConstantPool = MCP;

  if (ConstantPoolBase == 0) return;  // Buffer overflow.

  DOUT << "JIT: Emitted constant pool at [" << ConstantPoolBase
       << "] (size: " << Size << ", alignment: " << Align << ")\n";

  // Initialize the memory for all of the constant pool entries.
  unsigned Offset = 0;
  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    MachineConstantPoolEntry CPE = Constants[i];
    unsigned AlignMask = CPE.getAlignment() - 1;
    Offset = (Offset + AlignMask) & ~AlignMask;

    uintptr_t CAddr = (uintptr_t)ConstantPoolBase + Offset;
    ConstPoolAddresses.push_back(CAddr);
    if (CPE.isMachineConstantPoolEntry()) {
      // FIXME: add support to lower machine constant pool values into bytes!
      llvm_report_error("Initialize memory with machine specific constant pool"
                        "entry has not been implemented!");
    }
    TheJIT->InitializeMemory(CPE.Val.ConstVal, (void*)CAddr);
    DOUT << "JIT:   CP" << i << " at [0x"
         << std::hex << CAddr << std::dec << "]\n";

    const Type *Ty = CPE.Val.ConstVal->getType();
    Offset += TheJIT->getTargetData()->getTypeAllocSize(Ty);
  }
}

void JITEmitter::initJumpTableInfo(MachineJumpTableInfo *MJTI) {
  if (TheJIT->getJITInfo().hasCustomJumpTables())
    return;

  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  if (JT.empty()) return;
  
  unsigned NumEntries = 0;
  for (unsigned i = 0, e = JT.size(); i != e; ++i)
    NumEntries += JT[i].MBBs.size();

  unsigned EntrySize = MJTI->getEntrySize();

  // Just allocate space for all the jump tables now.  We will fix up the actual
  // MBB entries in the tables after we emit the code for each block, since then
  // we will know the final locations of the MBBs in memory.
  JumpTable = MJTI;
  JumpTableBase = allocateSpace(NumEntries * EntrySize, MJTI->getAlignment());
}

void JITEmitter::emitJumpTableInfo(MachineJumpTableInfo *MJTI) {
  if (TheJIT->getJITInfo().hasCustomJumpTables())
    return;

  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  if (JT.empty() || JumpTableBase == 0) return;
  
  if (TargetMachine::getRelocationModel() == Reloc::PIC_) {
    assert(MJTI->getEntrySize() == 4 && "Cross JIT'ing?");
    // For each jump table, place the offset from the beginning of the table
    // to the target address.
    int *SlotPtr = (int*)JumpTableBase;

    for (unsigned i = 0, e = JT.size(); i != e; ++i) {
      const std::vector<MachineBasicBlock*> &MBBs = JT[i].MBBs;
      // Store the offset of the basic block for this jump table slot in the
      // memory we allocated for the jump table in 'initJumpTableInfo'
      uintptr_t Base = (uintptr_t)SlotPtr;
      for (unsigned mi = 0, me = MBBs.size(); mi != me; ++mi) {
        uintptr_t MBBAddr = getMachineBasicBlockAddress(MBBs[mi]);
        *SlotPtr++ = TheJIT->getJITInfo().getPICJumpTableEntry(MBBAddr, Base);
      }
    }
  } else {
    assert(MJTI->getEntrySize() == sizeof(void*) && "Cross JIT'ing?");
    
    // For each jump table, map each target in the jump table to the address of 
    // an emitted MachineBasicBlock.
    intptr_t *SlotPtr = (intptr_t*)JumpTableBase;

    for (unsigned i = 0, e = JT.size(); i != e; ++i) {
      const std::vector<MachineBasicBlock*> &MBBs = JT[i].MBBs;
      // Store the address of the basic block for this jump table slot in the
      // memory we allocated for the jump table in 'initJumpTableInfo'
      for (unsigned mi = 0, me = MBBs.size(); mi != me; ++mi)
        *SlotPtr++ = getMachineBasicBlockAddress(MBBs[mi]);
    }
  }
}

void JITEmitter::startGVStub(const GlobalValue* GV, unsigned StubSize,
                             unsigned Alignment) {
  SavedBufferBegin = BufferBegin;
  SavedBufferEnd = BufferEnd;
  SavedCurBufferPtr = CurBufferPtr;
  
  BufferBegin = CurBufferPtr = MemMgr->allocateStub(GV, StubSize, Alignment);
  BufferEnd = BufferBegin+StubSize+1;
}

void JITEmitter::startGVStub(const GlobalValue* GV, void *Buffer,
                             unsigned StubSize) {
  SavedBufferBegin = BufferBegin;
  SavedBufferEnd = BufferEnd;
  SavedCurBufferPtr = CurBufferPtr;
  
  BufferBegin = CurBufferPtr = (uint8_t *)Buffer;
  BufferEnd = BufferBegin+StubSize+1;
}

void *JITEmitter::finishGVStub(const GlobalValue* GV) {
  NumBytes += getCurrentPCOffset();
  std::swap(SavedBufferBegin, BufferBegin);
  BufferEnd = SavedBufferEnd;
  CurBufferPtr = SavedCurBufferPtr;
  return SavedBufferBegin;
}

// getConstantPoolEntryAddress - Return the address of the 'ConstantNum' entry
// in the constant pool that was last emitted with the 'emitConstantPool'
// method.
//
uintptr_t JITEmitter::getConstantPoolEntryAddress(unsigned ConstantNum) const {
  assert(ConstantNum < ConstantPool->getConstants().size() &&
         "Invalid ConstantPoolIndex!");
  return ConstPoolAddresses[ConstantNum];
}

// getJumpTableEntryAddress - Return the address of the JumpTable with index
// 'Index' in the jumpp table that was last initialized with 'initJumpTableInfo'
//
uintptr_t JITEmitter::getJumpTableEntryAddress(unsigned Index) const {
  const std::vector<MachineJumpTableEntry> &JT = JumpTable->getJumpTables();
  assert(Index < JT.size() && "Invalid jump table index!");
  
  unsigned Offset = 0;
  unsigned EntrySize = JumpTable->getEntrySize();
  
  for (unsigned i = 0; i < Index; ++i)
    Offset += JT[i].MBBs.size();
  
   Offset *= EntrySize;
  
  return (uintptr_t)((char *)JumpTableBase + Offset);
}

//===----------------------------------------------------------------------===//
//  Public interface to this file
//===----------------------------------------------------------------------===//

JITCodeEmitter *JIT::createEmitter(JIT &jit, JITMemoryManager *JMM) {
  return new JITEmitter(jit, JMM);
}

// getPointerToNamedFunction - This function is used as a global wrapper to
// JIT::getPointerToNamedFunction for the purpose of resolving symbols when
// bugpoint is debugging the JIT. In that scenario, we are loading an .so and
// need to resolve function(s) that are being mis-codegenerated, so we need to
// resolve their addresses at runtime, and this is the way to do it.
extern "C" {
  void *getPointerToNamedFunction(const char *Name) {
    if (Function *F = TheJIT->FindFunctionNamed(Name))
      return TheJIT->getPointerToFunction(F);
    return TheJIT->getPointerToNamedFunction(Name);
  }
}

// getPointerToFunctionOrStub - If the specified function has been
// code-gen'd, return a pointer to the function.  If not, compile it, or use
// a stub to implement lazy compilation if available.
//
void *JIT::getPointerToFunctionOrStub(Function *F) {
  // If we have already code generated the function, just return the address.
  if (void *Addr = getPointerToGlobalIfAvailable(F))
    return Addr;
  
  // Get a stub if the target supports it.
  assert(isa<JITEmitter>(JCE) && "Unexpected MCE?");
  JITEmitter *JE = cast<JITEmitter>(getCodeEmitter());
  return JE->getJITResolver().getFunctionStub(F);
}

void JIT::updateFunctionStub(Function *F) {
  // Get the empty stub we generated earlier.
  assert(isa<JITEmitter>(JCE) && "Unexpected MCE?");
  JITEmitter *JE = cast<JITEmitter>(getCodeEmitter());
  void *Stub = JE->getJITResolver().getFunctionStub(F);

  // Tell the target jit info to rewrite the stub at the specified address,
  // rather than creating a new one.
  void *Addr = getPointerToGlobalIfAvailable(F);
  getJITInfo().emitFunctionStubAtAddr(F, Addr, Stub, *getCodeEmitter());
}

/// updateDlsymStubTable - Emit the data necessary to relocate the stubs
/// that were emitted during code generation.
///
void JIT::updateDlsymStubTable() {
  assert(isa<JITEmitter>(JCE) && "Unexpected MCE?");
  JITEmitter *JE = cast<JITEmitter>(getCodeEmitter());
  
  SmallVector<GlobalValue*, 8> GVs;
  SmallVector<void*, 8> Ptrs;
  const StringMap<void *> &ExtFns = JE->getExternalFnStubs();

  JE->getJITResolver().getRelocatableGVs(GVs, Ptrs);

  unsigned nStubs = GVs.size() + ExtFns.size();
  
  // If there are no relocatable stubs, return.
  if (nStubs == 0)
    return;

  // If there are no new relocatable stubs, return.
  void *CurTable = JE->getMemMgr()->getDlsymTable();
  if (CurTable && (*(unsigned *)CurTable == nStubs))
    return;
  
  // Calculate the size of the stub info
  unsigned offset = 4 + 4 * nStubs + sizeof(intptr_t) * nStubs;
  
  SmallVector<unsigned, 8> Offsets;
  for (unsigned i = 0; i != GVs.size(); ++i) {
    Offsets.push_back(offset);
    offset += GVs[i]->getName().size() + 1;
  }
  for (StringMapConstIterator<void*> i = ExtFns.begin(), e = ExtFns.end(); 
       i != e; ++i) {
    Offsets.push_back(offset);
    offset += strlen(i->first()) + 1;
  }
  
  // Allocate space for the new "stub", which contains the dlsym table.
  JE->startGVStub(0, offset, 4);
  
  // Emit the number of records
  JE->emitInt32(nStubs);
  
  // Emit the string offsets
  for (unsigned i = 0; i != nStubs; ++i)
    JE->emitInt32(Offsets[i]);
  
  // Emit the pointers.  Verify that they are at least 2-byte aligned, and set
  // the low bit to 0 == GV, 1 == Function, so that the client code doing the
  // relocation can write the relocated pointer at the appropriate place in
  // the stub.
  for (unsigned i = 0; i != GVs.size(); ++i) {
    intptr_t Ptr = (intptr_t)Ptrs[i];
    assert((Ptr & 1) == 0 && "Stub pointers must be at least 2-byte aligned!");
    
    if (isa<Function>(GVs[i]))
      Ptr |= (intptr_t)1;
           
    if (sizeof(Ptr) == 8)
      JE->emitInt64(Ptr);
    else
      JE->emitInt32(Ptr);
  }
  for (StringMapConstIterator<void*> i = ExtFns.begin(), e = ExtFns.end(); 
       i != e; ++i) {
    intptr_t Ptr = (intptr_t)i->second | 1;

    if (sizeof(Ptr) == 8)
      JE->emitInt64(Ptr);
    else
      JE->emitInt32(Ptr);
  }
  
  // Emit the strings.
  for (unsigned i = 0; i != GVs.size(); ++i)
    JE->emitString(GVs[i]->getName());
  for (StringMapConstIterator<void*> i = ExtFns.begin(), e = ExtFns.end(); 
       i != e; ++i)
    JE->emitString(i->first());
  
  // Tell the JIT memory manager where it is.  The JIT Memory Manager will
  // deallocate space for the old one, if one existed.
  JE->getMemMgr()->SetDlsymTable(JE->finishGVStub(0));
}

/// freeMachineCodeForFunction - release machine code memory for given Function.
///
void JIT::freeMachineCodeForFunction(Function *F) {

  // Delete translation for this from the ExecutionEngine, so it will get
  // retranslated next time it is used.
  void *OldPtr = updateGlobalMapping(F, 0);

  if (OldPtr)
    TheJIT->NotifyFreeingMachineCode(*F, OldPtr);

  // Free the actual memory for the function body and related stuff.
  assert(isa<JITEmitter>(JCE) && "Unexpected MCE?");
  cast<JITEmitter>(JCE)->deallocateMemForFunction(F);
}
