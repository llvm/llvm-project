//===-- BitReader.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/BitReader.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/LLVMContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include <string>
#include <cstring>

using namespace llvm;

/* Builds a module from the bitcode in the specified memory buffer, returning a
   reference to the module via the OutModule parameter. Returns 0 on success.
   Optionally returns a human-readable error message via OutMessage. */
LLVMBool LLVMParseBitcode(LLVMMemoryBufferRef MemBuf,
                          LLVMModuleRef *OutModule, char **OutMessage) {
  return LLVMParseBitcodeInContext(wrap(&getGlobalContext()), MemBuf, OutModule,
                                   OutMessage);
}

LLVMBool LLVMParseBitcodeInContext(LLVMContextRef ContextRef,
                                   LLVMMemoryBufferRef MemBuf,
                                   LLVMModuleRef *OutModule,
                                   char **OutMessage) {
  std::string Message;
  
  *OutModule = wrap(ParseBitcodeFile(unwrap(MemBuf), *unwrap(ContextRef),
                                     &Message));
  if (!*OutModule) {
    if (OutMessage)
      *OutMessage = strdup(Message.c_str());
    return 1;
  }
  
  return 0;
}

/* Reads a module from the specified path, returning via the OutModule parameter
   a module provider which performs lazy deserialization. Returns 0 on success.
   Optionally returns a human-readable error message via OutMessage. */ 
LLVMBool LLVMGetBitcodeModuleInContext(LLVMContextRef ContextRef,
                                       LLVMMemoryBufferRef MemBuf,
                                       LLVMModuleRef *OutM,
                                       char **OutMessage) {
  std::string Message;
  
  *OutM = wrap(getLazyBitcodeModule(unwrap(MemBuf), *unwrap(ContextRef),
                                    &Message));
  if (!*OutM) {
    if (OutMessage)
      *OutMessage = strdup(Message.c_str());
    return 1;
  }
  
  return 0;

}

LLVMBool LLVMGetBitcodeModule(LLVMMemoryBufferRef MemBuf, LLVMModuleRef *OutM,
                              char **OutMessage) {
  return LLVMGetBitcodeModuleInContext(LLVMGetGlobalContext(), MemBuf, OutM,
                                       OutMessage);
}

/* Deprecated: Use LLVMGetBitcodeModuleInContext instead. */
LLVMBool LLVMGetBitcodeModuleProviderInContext(LLVMContextRef ContextRef,
                                               LLVMMemoryBufferRef MemBuf,
                                               LLVMModuleProviderRef *OutMP,
                                               char **OutMessage) {
  return LLVMGetBitcodeModuleInContext(ContextRef, MemBuf,
                                       reinterpret_cast<LLVMModuleRef*>(OutMP),
                                       OutMessage);
}

/* Deprecated: Use LLVMGetBitcodeModule instead. */
LLVMBool LLVMGetBitcodeModuleProvider(LLVMMemoryBufferRef MemBuf,
                                      LLVMModuleProviderRef *OutMP,
                                      char **OutMessage) {
  return LLVMGetBitcodeModuleProviderInContext(LLVMGetGlobalContext(), MemBuf,
                                               OutMP, OutMessage);
}
