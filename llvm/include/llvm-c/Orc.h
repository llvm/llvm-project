/*===---------------- llvm-c/Orc.h - OrcV2 C bindings -----------*- C++ -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to libLLVMOrcJIT.a, which implements  *|
|* JIT compilation of LLVM IR.                                                *|
|*                                                                            *|
|* Many exotic languages can interoperate with C code but have a harder time  *|
|* with C++ due to name mangling. So in addition to C, this interface enables *|
|* tools written in such languages.                                           *|
|*                                                                            *|
|* Note: This interface is experimental. It is *NOT* stable, and may be       *|
|*       changed without warning. Only C API usage documentation is           *|
|*       provided. See the C++ documentation for all higher level ORC API     *|
|*       details.                                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_ORC_H
#define LLVM_C_ORC_H

#include "llvm-c/Error.h"
#include "llvm-c/Types.h"

LLVM_C_EXTERN_C_BEGIN

typedef struct LLVMOrcOpaqueThreadSafeContext *LLVMOrcThreadSafeContextRef;
typedef struct LLVMOrcOpaqueThreadSafeModule *LLVMOrcThreadSafeModuleRef;
typedef struct LLVMOrcOpaqueLLJIT *LLVMOrcLLJITRef;
typedef uint64_t LLVMOrcJITTargetAddress;

/**
 * Create a ThreadSafeContext containing a new LLVMContext.
 *
 * Ownership of the underlying ThreadSafeContext data is shared: Clients
 * can and should dispose of their ThreadSafeContext as soon as they no longer
 * need to refer to it directly. Other references (e.g. from ThreadSafeModules
 * will keep the data alive as long as it is needed.
 */
LLVMOrcThreadSafeContextRef LLVMOrcCreateNewThreadSafeContext(void);

/**
 * Get a reference to the wrapped LLVMContext.
 */
LLVMContextRef
LLVMOrcThreadSafeContextGetContext(LLVMOrcThreadSafeContextRef TSCtx);

/**
 * Dispose of a ThreadSafeContext.
 */
void LLVMOrcDisposeThreadSafeContext(LLVMOrcThreadSafeContextRef TSCtx);

/**
 * Create a ThreadSafeModule wrapper around the given LLVM module. This takes
 * ownership of the M argument which should not be disposed of or referenced
 * after this function returns.
 *
 * Ownership of the ThreadSafeModule is unique: If it is transferred to the JIT
 * (e.g. by LLVMOrcLLJITAddLLVMIRModule), in which case the client is no longer
 * responsible for it. If it is not transferred to the JIT then the client
 * should call LLVMOrcDisposeThreadSafeModule to dispose of it.
 */
LLVMOrcThreadSafeModuleRef
LLVMOrcCreateNewThreadSafeModule(LLVMModuleRef M,
                                 LLVMOrcThreadSafeContextRef TSCtx);

/**
 * Dispose of a ThreadSafeModule.
 */
void LLVMOrcDisposeThreadSafeModule(LLVMOrcThreadSafeModuleRef TSM);

/**
 * Create an LLJIT instance using all default values.
 *
 * The LLJIT instance is uniquely owned by the client and automatically manages
 * the memory of all JIT'd code and all modules that are transferred to it
 * (e.g. via LLVMOrcLLJITAddLLVMIRModule). Disposing of the LLJIT instance will
 * free all memory managed by the JIT, including JIT'd code and not-yet
 * compiled modules.
 */
LLVMErrorRef LLVMOrcCreateDefaultLLJIT(LLVMOrcLLJITRef *Result);

/**
 * Dispose of an LLJIT instance.
 */
LLVMErrorRef LLVMOrcDisposeLLJIT(LLVMOrcLLJITRef J);

/**
 * Add an IR module to the main JITDylib of the given LLJIT instance. This
 * operation takes ownership of the TSM argument which should not be disposed
 * of or referenced once this function returns.
 */
LLVMErrorRef LLVMOrcLLJITAddLLVMIRModule(LLVMOrcLLJITRef J,
                                         LLVMOrcThreadSafeModuleRef TSM);
/**
 * Look up the given symbol in the main JITDylib of the given LLJIT instance.
 *
 * This operation does not take ownership of the Name argument.
 */
LLVMErrorRef LLVMOrcLLJITLookup(LLVMOrcLLJITRef J,
                                LLVMOrcJITTargetAddress *Result,
                                const char *Name);

LLVM_C_EXTERN_C_END

#endif /* LLVM_C_ORC_H */
