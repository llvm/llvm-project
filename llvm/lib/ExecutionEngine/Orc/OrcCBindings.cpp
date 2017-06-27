//===----------- OrcCBindings.cpp - C bindings for the Orc APIs -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OrcCBindingsStack.h"
#include "llvm-c/OrcBindings.h"

using namespace llvm;

LLVMSharedModuleRef LLVMOrcMakeSharedModule(LLVMModuleRef Mod) {
  return wrap(new std::shared_ptr<Module>(unwrap(Mod)));
}

void LLVMOrcDisposeSharedModuleRef(LLVMSharedModuleRef SharedMod) {
  delete unwrap(SharedMod);
}

LLVMSharedObjectBufferRef
LLVMOrcMakeSharedObjectBuffer(LLVMMemoryBufferRef ObjBuffer) {
  return wrap(new std::shared_ptr<MemoryBuffer>(unwrap(ObjBuffer)));
}

void
LLVMOrcDisposeSharedObjectBufferRef(LLVMSharedObjectBufferRef SharedObjBuffer) {
  delete unwrap(SharedObjBuffer);
}

LLVMOrcJITStackRef LLVMOrcCreateInstance(LLVMTargetMachineRef TM) {
  TargetMachine *TM2(unwrap(TM));

  Triple T(TM2->getTargetTriple());

  auto CompileCallbackMgr = orc::createLocalCompileCallbackManager(T, 0);
  auto IndirectStubsMgrBuilder =
      orc::createLocalIndirectStubsManagerBuilder(T);

  OrcCBindingsStack *JITStack = new OrcCBindingsStack(
      *TM2, std::move(CompileCallbackMgr), IndirectStubsMgrBuilder);

  return wrap(JITStack);
}

const char *LLVMOrcGetErrorMsg(LLVMOrcJITStackRef JITStack) {
  OrcCBindingsStack &J = *unwrap(JITStack);
  return J.getErrorMessage().c_str();
}

void LLVMOrcGetMangledSymbol(LLVMOrcJITStackRef JITStack, char **MangledName,
                             const char *SymbolName) {
  OrcCBindingsStack &J = *unwrap(JITStack);
  std::string Mangled = J.mangle(SymbolName);
  *MangledName = new char[Mangled.size() + 1];
  strcpy(*MangledName, Mangled.c_str());
}

void LLVMOrcDisposeMangledSymbol(char *MangledName) { delete[] MangledName; }

LLVMOrcTargetAddress
LLVMOrcCreateLazyCompileCallback(LLVMOrcJITStackRef JITStack,
                                 LLVMOrcLazyCompileCallbackFn Callback,
                                 void *CallbackCtx) {
  OrcCBindingsStack &J = *unwrap(JITStack);
  return J.createLazyCompileCallback(Callback, CallbackCtx);
}

LLVMOrcErrorCode LLVMOrcCreateIndirectStub(LLVMOrcJITStackRef JITStack,
                                           const char *StubName,
                                           LLVMOrcTargetAddress InitAddr) {
  OrcCBindingsStack &J = *unwrap(JITStack);
  return J.createIndirectStub(StubName, InitAddr);
}

LLVMOrcErrorCode LLVMOrcSetIndirectStubPointer(LLVMOrcJITStackRef JITStack,
                                               const char *StubName,
                                               LLVMOrcTargetAddress NewAddr) {
  OrcCBindingsStack &J = *unwrap(JITStack);
  return J.setIndirectStubPointer(StubName, NewAddr);
}

LLVMOrcModuleHandle
LLVMOrcAddEagerlyCompiledIR(LLVMOrcJITStackRef JITStack,
                            LLVMSharedModuleRef Mod,
                            LLVMOrcSymbolResolverFn SymbolResolver,
                            void *SymbolResolverCtx) {
  OrcCBindingsStack &J = *unwrap(JITStack);
  std::shared_ptr<Module> *M(unwrap(Mod));
  return J.addIRModuleEager(*M, SymbolResolver, SymbolResolverCtx);
}

LLVMOrcModuleHandle
LLVMOrcAddLazilyCompiledIR(LLVMOrcJITStackRef JITStack,
                           LLVMSharedModuleRef Mod,
                           LLVMOrcSymbolResolverFn SymbolResolver,
                           void *SymbolResolverCtx) {
  OrcCBindingsStack &J = *unwrap(JITStack);
  std::shared_ptr<Module> *M(unwrap(Mod));
  return J.addIRModuleLazy(*M, SymbolResolver, SymbolResolverCtx);
}

void LLVMOrcRemoveModule(LLVMOrcJITStackRef JITStack, LLVMOrcModuleHandle H) {
  OrcCBindingsStack &J = *unwrap(JITStack);
  J.removeModule(H);
}

LLVMOrcTargetAddress LLVMOrcGetSymbolAddress(LLVMOrcJITStackRef JITStack,
                                             const char *SymbolName) {
  OrcCBindingsStack &J = *unwrap(JITStack);
  auto Sym = J.findSymbol(SymbolName, true);
  return Sym.getAddress();
}

void LLVMOrcDisposeInstance(LLVMOrcJITStackRef JITStack) {
  delete unwrap(JITStack);
}
