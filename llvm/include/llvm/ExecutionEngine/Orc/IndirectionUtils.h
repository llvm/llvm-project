//===-- IndirectionUtils.h - Utilities for adding indirections --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains utilities for adding indirections and breaking up modules.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_INDIRECTIONUTILS_H
#define LLVM_EXECUTIONENGINE_ORC_INDIRECTIONUTILS_H

#include "JITSymbol.h"
#include "LambdaResolver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Support/Process.h"
#include <sstream>

namespace llvm {
namespace orc {

/// @brief Target-independent base class for compile callback management.
class JITCompileCallbackManager {
public:

  typedef std::function<TargetAddress()> CompileFtor;

  /// @brief Handle to a newly created compile callback. Can be used to get an
  ///        IR constant representing the address of the trampoline, and to set
  ///        the compile action for the callback.
  class CompileCallbackInfo {
  public:
    CompileCallbackInfo(TargetAddress Addr, CompileFtor &Compile)
      : Addr(Addr), Compile(Compile) {}

    TargetAddress getAddress() const { return Addr; }
    void setCompileAction(CompileFtor Compile) {
      this->Compile = std::move(Compile);
    }
  private:
    TargetAddress Addr;
    CompileFtor &Compile;
  };

  /// @brief Construct a JITCompileCallbackManager.
  /// @param ErrorHandlerAddress The address of an error handler in the target
  ///                            process to be used if a compile callback fails.
  JITCompileCallbackManager(TargetAddress ErrorHandlerAddress)
    : ErrorHandlerAddress(ErrorHandlerAddress) {}

  virtual ~JITCompileCallbackManager() {}

  /// @brief Execute the callback for the given trampoline id. Called by the JIT
  ///        to compile functions on demand.
  TargetAddress executeCompileCallback(TargetAddress TrampolineAddr) {
    auto I = ActiveTrampolines.find(TrampolineAddr);
    // FIXME: Also raise an error in the Orc error-handler when we finally have
    //        one.
    if (I == ActiveTrampolines.end())
      return ErrorHandlerAddress;

    // Found a callback handler. Yank this trampoline out of the active list and
    // put it back in the available trampolines list, then try to run the
    // handler's compile and update actions.
    // Moving the trampoline ID back to the available list first means there's at
    // least one available trampoline if the compile action triggers a request for
    // a new one.
    auto Compile = std::move(I->second);
    ActiveTrampolines.erase(I);
    AvailableTrampolines.push_back(TrampolineAddr);

    if (auto Addr = Compile())
      return Addr;

    return ErrorHandlerAddress;
  }

  /// @brief Reserve a compile callback.
  CompileCallbackInfo getCompileCallback() {
    TargetAddress TrampolineAddr = getAvailableTrampolineAddr();
    auto &Compile = this->ActiveTrampolines[TrampolineAddr];
    return CompileCallbackInfo(TrampolineAddr, Compile);
  }

  /// @brief Get a CompileCallbackInfo for an existing callback.
  CompileCallbackInfo getCompileCallbackInfo(TargetAddress TrampolineAddr) {
    auto I = ActiveTrampolines.find(TrampolineAddr);
    assert(I != ActiveTrampolines.end() && "Not an active trampoline.");
    return CompileCallbackInfo(I->first, I->second);
  }

  /// @brief Release a compile callback.
  ///
  ///   Note: Callbacks are auto-released after they execute. This method should
  /// only be called to manually release a callback that is not going to
  /// execute.
  void releaseCompileCallback(TargetAddress TrampolineAddr) {
    auto I = ActiveTrampolines.find(TrampolineAddr);
    assert(I != ActiveTrampolines.end() && "Not an active trampoline.");
    ActiveTrampolines.erase(I);
    AvailableTrampolines.push_back(TrampolineAddr);
  }

protected:
  TargetAddress ErrorHandlerAddress;

  typedef std::map<TargetAddress, CompileFtor> TrampolineMapT;
  TrampolineMapT ActiveTrampolines;
  std::vector<TargetAddress> AvailableTrampolines;

private:

  TargetAddress getAvailableTrampolineAddr() {
    if (this->AvailableTrampolines.empty())
      grow();
    assert(!this->AvailableTrampolines.empty() &&
           "Failed to grow available trampolines.");
    TargetAddress TrampolineAddr = this->AvailableTrampolines.back();
    this->AvailableTrampolines.pop_back();
    return TrampolineAddr;
  }

  // Create new trampolines - to be implemented in subclasses.
  virtual void grow() = 0;

  virtual void anchor();
};

/// @brief Manage compile callbacks for in-process JITs.
template <typename TargetT>
class LocalJITCompileCallbackManager : public JITCompileCallbackManager {
public:

  /// @brief Construct a InProcessJITCompileCallbackManager.
  /// @param ErrorHandlerAddress The address of an error handler in the target
  ///                            process to be used if a compile callback fails.
  LocalJITCompileCallbackManager(TargetAddress ErrorHandlerAddress)
    : JITCompileCallbackManager(ErrorHandlerAddress) {

    /// Set up the resolver block.
    std::error_code EC;
    ResolverBlock =
      sys::OwningMemoryBlock(
        sys::Memory::allocateMappedMemory(TargetT::ResolverCodeSize, nullptr,
                                          sys::Memory::MF_READ |
                                          sys::Memory::MF_WRITE, EC));
    assert(!EC && "Failed to allocate resolver block");

    TargetT::writeResolverCode(static_cast<uint8_t *>(ResolverBlock.base()),
                               &reenter, this);

    EC = sys::Memory::protectMappedMemory(ResolverBlock.getMemoryBlock(),
                                          sys::Memory::MF_READ |
                                              sys::Memory::MF_EXEC);
    assert(!EC && "Failed to mprotect resolver block");
  }

private:

  static TargetAddress reenter(void *CCMgr, void *TrampolineId) {
    JITCompileCallbackManager *Mgr =
      static_cast<JITCompileCallbackManager*>(CCMgr);
    return Mgr->executeCompileCallback(
             static_cast<TargetAddress>(
               reinterpret_cast<uintptr_t>(TrampolineId)));
  }

  void grow() override {
    assert(this->AvailableTrampolines.empty() && "Growing prematurely?");

    std::error_code EC;
    auto TrampolineBlock =
      sys::OwningMemoryBlock(
        sys::Memory::allocateMappedMemory(sys::Process::getPageSize(), nullptr,
                                          sys::Memory::MF_READ |
                                          sys::Memory::MF_WRITE, EC));
    assert(!EC && "Failed to allocate trampoline block");


    unsigned NumTrampolines =
      (sys::Process::getPageSize() - TargetT::PointerSize) /
        TargetT::TrampolineSize;

    uint8_t *TrampolineMem = static_cast<uint8_t*>(TrampolineBlock.base());
    TargetT::writeTrampolines(TrampolineMem, ResolverBlock.base(),
                              NumTrampolines);

    for (unsigned I = 0; I < NumTrampolines; ++I)
      this->AvailableTrampolines.push_back(
          static_cast<TargetAddress>(reinterpret_cast<uintptr_t>(
              TrampolineMem + (I * TargetT::TrampolineSize))));

    EC = sys::Memory::protectMappedMemory(TrampolineBlock.getMemoryBlock(),
                                          sys::Memory::MF_READ |
                                              sys::Memory::MF_EXEC);
    assert(!EC && "Failed to mprotect trampoline block");

    TrampolineBlocks.push_back(std::move(TrampolineBlock));
  }

  sys::OwningMemoryBlock ResolverBlock;
  std::vector<sys::OwningMemoryBlock> TrampolineBlocks;
};

/// @brief Base class for managing collections of named indirect stubs.
class IndirectStubsManager {
public:

  /// @brief Map type for initializing the manager. See init.
  typedef StringMap<std::pair<TargetAddress, JITSymbolFlags>> StubInitsMap;

  virtual ~IndirectStubsManager() {}

  /// @brief Create a single stub with the given name, target address and flags.
  virtual std::error_code createStub(StringRef StubName, TargetAddress StubAddr,
                                     JITSymbolFlags StubFlags) = 0;

  /// @brief Create StubInits.size() stubs with the given names, target
  ///        addresses, and flags.
  virtual std::error_code createStubs(const StubInitsMap &StubInits) = 0;

  /// @brief Find the stub with the given name. If ExportedStubsOnly is true,
  ///        this will only return a result if the stub's flags indicate that it
  ///        is exported.
  virtual JITSymbol findStub(StringRef Name, bool ExportedStubsOnly) = 0;

  /// @brief Find the implementation-pointer for the stub.
  virtual JITSymbol findPointer(StringRef Name) = 0;

  /// @brief Change the value of the implementation pointer for the stub.
  virtual std::error_code updatePointer(StringRef Name, TargetAddress NewAddr) = 0;
private:
  virtual void anchor();
};

/// @brief IndirectStubsManager implementation for the host architecture, e.g.
///        OrcX86_64. (See OrcArchitectureSupport.h).
template <typename TargetT>
class LocalIndirectStubsManager : public IndirectStubsManager {
public:

  std::error_code createStub(StringRef StubName, TargetAddress StubAddr,
                             JITSymbolFlags StubFlags) override {
    if (auto EC = reserveStubs(1))
      return EC;

    createStubInternal(StubName, StubAddr, StubFlags);

    return std::error_code();
  }

  std::error_code createStubs(const StubInitsMap &StubInits) override {
    if (auto EC = reserveStubs(StubInits.size()))
      return EC;

    for (auto &Entry : StubInits)
      createStubInternal(Entry.first(), Entry.second.first,
                         Entry.second.second);

    return std::error_code();
  }

  JITSymbol findStub(StringRef Name, bool ExportedStubsOnly) override {
    auto I = StubIndexes.find(Name);
    if (I == StubIndexes.end())
      return nullptr;
    auto Key = I->second.first;
    void *StubAddr = IndirectStubsInfos[Key.first].getStub(Key.second);
    assert(StubAddr && "Missing stub address");
    auto StubTargetAddr =
      static_cast<TargetAddress>(reinterpret_cast<uintptr_t>(StubAddr));
    auto StubSymbol = JITSymbol(StubTargetAddr, I->second.second);
    if (ExportedStubsOnly && !StubSymbol.isExported())
      return nullptr;
    return StubSymbol;
  }

  JITSymbol findPointer(StringRef Name) override {
    auto I = StubIndexes.find(Name);
    if (I == StubIndexes.end())
      return nullptr;
    auto Key = I->second.first;
    void *PtrAddr = IndirectStubsInfos[Key.first].getPtr(Key.second);
    assert(PtrAddr && "Missing pointer address");
    auto PtrTargetAddr =
      static_cast<TargetAddress>(reinterpret_cast<uintptr_t>(PtrAddr));
    return JITSymbol(PtrTargetAddr, I->second.second);
  }

  std::error_code updatePointer(StringRef Name, TargetAddress NewAddr) override {
    auto I = StubIndexes.find(Name);
    assert(I != StubIndexes.end() && "No stub pointer for symbol");
    auto Key = I->second.first;
    *IndirectStubsInfos[Key.first].getPtr(Key.second) =
      reinterpret_cast<void*>(static_cast<uintptr_t>(NewAddr));
    return std::error_code();
  }

private:

  std::error_code reserveStubs(unsigned NumStubs) {
    if (NumStubs <= FreeStubs.size())
      return std::error_code();

    unsigned NewStubsRequired = NumStubs - FreeStubs.size();
    unsigned NewBlockId = IndirectStubsInfos.size();
    typename TargetT::IndirectStubsInfo ISI;
    if (auto EC = TargetT::emitIndirectStubsBlock(ISI, NewStubsRequired,
                                                  nullptr))
      return EC;
    for (unsigned I = 0; I < ISI.getNumStubs(); ++I)
      FreeStubs.push_back(std::make_pair(NewBlockId, I));
    IndirectStubsInfos.push_back(std::move(ISI));
    return std::error_code();
  }

  void createStubInternal(StringRef StubName, TargetAddress InitAddr,
                          JITSymbolFlags StubFlags) {
    auto Key = FreeStubs.back();
    FreeStubs.pop_back();
    *IndirectStubsInfos[Key.first].getPtr(Key.second) =
      reinterpret_cast<void*>(static_cast<uintptr_t>(InitAddr));
    StubIndexes[StubName] = std::make_pair(Key, StubFlags);
  }

  std::vector<typename TargetT::IndirectStubsInfo> IndirectStubsInfos;
  typedef std::pair<uint16_t, uint16_t> StubKey;
  std::vector<StubKey> FreeStubs;
  StringMap<std::pair<StubKey, JITSymbolFlags>> StubIndexes;
};

/// @brief Build a function pointer of FunctionType with the given constant
///        address.
///
///   Usage example: Turn a trampoline address into a function pointer constant
/// for use in a stub.
Constant* createIRTypedAddress(FunctionType &FT, TargetAddress Addr);

/// @brief Create a function pointer with the given type, name, and initializer
///        in the given Module.
GlobalVariable* createImplPointer(PointerType &PT, Module &M,
                                  const Twine &Name, Constant *Initializer);

/// @brief Turn a function declaration into a stub function that makes an
///        indirect call using the given function pointer.
void makeStub(Function &F, Value &ImplPointer);

/// @brief Raise linkage types and rename as necessary to ensure that all
///        symbols are accessible for other modules.
///
///   This should be called before partitioning a module to ensure that the
/// partitions retain access to each other's symbols.
void makeAllSymbolsExternallyAccessible(Module &M);

/// @brief Clone a function declaration into a new module.
///
///   This function can be used as the first step towards creating a callback
/// stub (see makeStub), or moving a function body (see moveFunctionBody).
///
///   If the VMap argument is non-null, a mapping will be added between F and
/// the new declaration, and between each of F's arguments and the new
/// declaration's arguments. This map can then be passed in to moveFunction to
/// move the function body if required. Note: When moving functions between
/// modules with these utilities, all decls should be cloned (and added to a
/// single VMap) before any bodies are moved. This will ensure that references
/// between functions all refer to the versions in the new module.
Function* cloneFunctionDecl(Module &Dst, const Function &F,
                            ValueToValueMapTy *VMap = nullptr);

/// @brief Move the body of function 'F' to a cloned function declaration in a
///        different module (See related cloneFunctionDecl).
///
///   If the target function declaration is not supplied via the NewF parameter
/// then it will be looked up via the VMap.
///
///   This will delete the body of function 'F' from its original parent module,
/// but leave its declaration.
void moveFunctionBody(Function &OrigF, ValueToValueMapTy &VMap,
                      ValueMaterializer *Materializer = nullptr,
                      Function *NewF = nullptr);

/// @brief Clone a global variable declaration into a new module.
GlobalVariable* cloneGlobalVariableDecl(Module &Dst, const GlobalVariable &GV,
                                        ValueToValueMapTy *VMap = nullptr);

/// @brief Move global variable GV from its parent module to cloned global
///        declaration in a different module.
///
///   If the target global declaration is not supplied via the NewGV parameter
/// then it will be looked up via the VMap.
///
///   This will delete the initializer of GV from its original parent module,
/// but leave its declaration.
void moveGlobalVariableInitializer(GlobalVariable &OrigGV,
                                   ValueToValueMapTy &VMap,
                                   ValueMaterializer *Materializer = nullptr,
                                   GlobalVariable *NewGV = nullptr);

/// @brief Clone
GlobalAlias* cloneGlobalAliasDecl(Module &Dst, const GlobalAlias &OrigA,
                                  ValueToValueMapTy &VMap);

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_INDIRECTIONUTILS_H
