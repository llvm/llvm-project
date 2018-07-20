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
#include "llvm/ExecutionEngine/JITEventListener.h"

using namespace llvm;

LLVMOrcJITStackRef LLVMOrcCreateInstance(LLVMTargetMachineRef TM) {
  TargetMachine *TM2(unwrap(TM));

  Triple T(TM2->getTargetTriple());

  auto IndirectStubsMgrBuilder =
      orc::createLocalIndirectStubsManagerBuilder(T);

  OrcCBindingsStack *JITStack =
      new OrcCBindingsStack(*TM2, std::move(IndirectStubsMgrBuilder));

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

LLVMOrcErrorCode
LLVMOrcCreateLazyCompileCallback(LLVMOrcJITStackRef JITStack,
                                 LLVMOrcTargetAddress *RetAddr,
                                 LLVMOrcLazyCompileCallbackFn Callback,
                                 void *CallbackCtx) {
  OrcCBindingsStack &J = *unwrap(JITStack);
  return J.createLazyCompileCallback(*RetAddr, Callback, CallbackCtx);
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

LLVMOrcErrorCode
LLVMOrcAddEagerlyCompiledIR(LLVMOrcJITStackRef JITStack,
                            LLVMOrcModuleHandle *RetHandle, LLVMModuleRef Mod,
                            LLVMOrcSymbolResolverFn SymbolResolver,
                            void *SymbolResolverCtx) {
  OrcCBindingsStack &J = *unwrap(JITStack);
  std::unique_ptr<Module> M(unwrap(Mod));
  return J.addIRModuleEager(*RetHandle, std::move(M), SymbolResolver,
                            SymbolResolverCtx);
}

LLVMOrcErrorCode
LLVMOrcAddLazilyCompiledIR(LLVMOrcJITStackRef JITStack,
                           LLVMOrcModuleHandle *RetHandle, LLVMModuleRef Mod,
                           LLVMOrcSymbolResolverFn SymbolResolver,
                           void *SymbolResolverCtx) {
  OrcCBindingsStack &J = *unwrap(JITStack);
  std::unique_ptr<Module> M(unwrap(Mod));
  return J.addIRModuleLazy(*RetHandle, std::move(M), SymbolResolver,
                           SymbolResolverCtx);
}

LLVMOrcErrorCode
LLVMOrcAddObjectFile(LLVMOrcJITStackRef JITStack,
                     LLVMOrcModuleHandle *RetHandle,
                     LLVMMemoryBufferRef Obj,
                     LLVMOrcSymbolResolverFn SymbolResolver,
                     void *SymbolResolverCtx) {
  OrcCBindingsStack &J = *unwrap(JITStack);
  std::unique_ptr<MemoryBuffer> O(unwrap(Obj));
  return J.addObject(*RetHandle, std::move(O), SymbolResolver,
                     SymbolResolverCtx);
}

LLVMOrcErrorCode LLVMOrcRemoveModule(LLVMOrcJITStackRef JITStack,
                                     LLVMOrcModuleHandle H) {
  OrcCBindingsStack &J = *unwrap(JITStack);
  return J.removeModule(H);
}

LLVMOrcErrorCode LLVMOrcGetSymbolAddress(LLVMOrcJITStackRef JITStack,
                                         LLVMOrcTargetAddress *RetAddr,
                                         const char *SymbolName) {
  OrcCBindingsStack &J = *unwrap(JITStack);
  return J.findSymbolAddress(*RetAddr, SymbolName, true);
}

LLVMOrcErrorCode LLVMOrcGetSymbolAddressIn(LLVMOrcJITStackRef JITStack,
                                           LLVMOrcTargetAddress *RetAddr,
                                           LLVMOrcModuleHandle H,
                                           const char *SymbolName) {
  OrcCBindingsStack &J = *unwrap(JITStack);
  return J.findSymbolAddressIn(*RetAddr, H, SymbolName, true);
}

LLVMOrcErrorCode LLVMOrcDisposeInstance(LLVMOrcJITStackRef JITStack) {
  auto *J = unwrap(JITStack);
  auto Err = J->shutdown();
  delete J;
  return Err;
}

void LLVMOrcRegisterJITEventListener(LLVMOrcJITStackRef JITStack, LLVMJITEventListenerRef L)
{
  unwrap(JITStack)->RegisterJITEventListener(unwrap(L));
}

void LLVMOrcUnregisterJITEventListener(LLVMOrcJITStackRef JITStack, LLVMJITEventListenerRef L)
{
  unwrap(JITStack)->UnregisterJITEventListener(unwrap(L));
}
