/*===-- llvm-c/ExecutionEngine.h - ExecutionEngine Lib C Iface --*- C++ -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to libLLVMExecutionEngine.o, which    *|
|* implements various analyses of the LLVM IR.                                *|
|*                                                                            *|
|* Many exotic languages can interoperate with C code but have a harder time  *|
|* with C++ due to name mangling. So in addition to C, this interface enables *|
|* tools written in such languages.                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_EXECUTIONENGINE_H
#define LLVM_C_EXECUTIONENGINE_H

#include "llvm-c/ExternC.h"
#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"
#include "llvm-c/Types.h"
#include "llvm-c/Visibility.h"

LLVM_C_EXTERN_C_BEGIN

/**
 * @defgroup LLVMCExecutionEngine Execution Engine
 * @ingroup LLVMC
 *
 * @{
 */

/**
 * Empty function used to force the linker to link MCJIT.
 * Has no effect when called on a pre-built library (dylib interface).
 */
LLVM_C_ABI void LLVMLinkInMCJIT(void);
/**
 * Empty function used to force the linker to link the LLVM interpreter.
 * Has no effect when called on a pre-built library (dylib interface).
 */
LLVM_C_ABI void LLVMLinkInInterpreter(void);

typedef struct LLVMOpaqueGenericValue *LLVMGenericValueRef;
typedef struct LLVMOpaqueExecutionEngine *LLVMExecutionEngineRef;
typedef struct LLVMOpaqueMCJITMemoryManager *LLVMMCJITMemoryManagerRef;

struct LLVMMCJITCompilerOptions {
  unsigned OptLevel;
  LLVMCodeModel CodeModel;
  LLVMBool NoFramePointerElim;
  LLVMBool EnableFastISel;
  LLVMMCJITMemoryManagerRef MCJMM;
};

/*===-- Operations on generic values --------------------------------------===*/

LLVM_C_ABI LLVMGenericValueRef LLVMCreateGenericValueOfInt(LLVMTypeRef Ty,
                                                           unsigned long long N,
                                                           LLVMBool IsSigned);

LLVM_C_ABI LLVMGenericValueRef LLVMCreateGenericValueOfPointer(void *P);

LLVM_C_ABI LLVMGenericValueRef LLVMCreateGenericValueOfFloat(LLVMTypeRef Ty,
                                                             double N);

LLVM_C_ABI unsigned LLVMGenericValueIntWidth(LLVMGenericValueRef GenValRef);

LLVM_C_ABI unsigned long long LLVMGenericValueToInt(LLVMGenericValueRef GenVal,
                                                    LLVMBool IsSigned);

LLVM_C_ABI void *LLVMGenericValueToPointer(LLVMGenericValueRef GenVal);

LLVM_C_ABI double LLVMGenericValueToFloat(LLVMTypeRef TyRef,
                                          LLVMGenericValueRef GenVal);

LLVM_C_ABI void LLVMDisposeGenericValue(LLVMGenericValueRef GenVal);

/*===-- Operations on execution engines -----------------------------------===*/

LLVM_C_ABI LLVMBool LLVMCreateExecutionEngineForModule(
    LLVMExecutionEngineRef *OutEE, LLVMModuleRef M, char **OutError);

LLVM_C_ABI LLVMBool LLVMCreateInterpreterForModule(
    LLVMExecutionEngineRef *OutInterp, LLVMModuleRef M, char **OutError);

LLVM_C_ABI LLVMBool
LLVMCreateJITCompilerForModule(LLVMExecutionEngineRef *OutJIT, LLVMModuleRef M,
                               unsigned OptLevel, char **OutError);

LLVM_C_ABI void
LLVMInitializeMCJITCompilerOptions(struct LLVMMCJITCompilerOptions *Options,
                                   size_t SizeOfOptions);

/**
 * Create an MCJIT execution engine for a module, with the given options. It is
 * the responsibility of the caller to ensure that all fields in Options up to
 * the given SizeOfOptions are initialized. It is correct to pass a smaller
 * value of SizeOfOptions that omits some fields. The canonical way of using
 * this is:
 *
 * LLVMMCJITCompilerOptions options;
 * LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
 * ... fill in those options you care about
 * LLVMCreateMCJITCompilerForModule(&jit, mod, &options, sizeof(options),
 *                                  &error);
 *
 * Note that this is also correct, though possibly suboptimal:
 *
 * LLVMCreateMCJITCompilerForModule(&jit, mod, 0, 0, &error);
 */
LLVM_C_ABI LLVMBool LLVMCreateMCJITCompilerForModule(
    LLVMExecutionEngineRef *OutJIT, LLVMModuleRef M,
    struct LLVMMCJITCompilerOptions *Options, size_t SizeOfOptions,
    char **OutError);

LLVM_C_ABI void LLVMDisposeExecutionEngine(LLVMExecutionEngineRef EE);

LLVM_C_ABI void LLVMRunStaticConstructors(LLVMExecutionEngineRef EE);

LLVM_C_ABI void LLVMRunStaticDestructors(LLVMExecutionEngineRef EE);

LLVM_C_ABI int LLVMRunFunctionAsMain(LLVMExecutionEngineRef EE, LLVMValueRef F,
                                     unsigned ArgC, const char *const *ArgV,
                                     const char *const *EnvP);

LLVM_C_ABI LLVMGenericValueRef LLVMRunFunction(LLVMExecutionEngineRef EE,
                                               LLVMValueRef F, unsigned NumArgs,
                                               LLVMGenericValueRef *Args);

LLVM_C_ABI void LLVMFreeMachineCodeForFunction(LLVMExecutionEngineRef EE,
                                               LLVMValueRef F);

LLVM_C_ABI void LLVMAddModule(LLVMExecutionEngineRef EE, LLVMModuleRef M);

LLVM_C_ABI LLVMBool LLVMRemoveModule(LLVMExecutionEngineRef EE, LLVMModuleRef M,
                                     LLVMModuleRef *OutMod, char **OutError);

LLVM_C_ABI LLVMBool LLVMFindFunction(LLVMExecutionEngineRef EE,
                                     const char *Name, LLVMValueRef *OutFn);

LLVM_C_ABI void *LLVMRecompileAndRelinkFunction(LLVMExecutionEngineRef EE,
                                                LLVMValueRef Fn);

LLVM_C_ABI LLVMTargetDataRef
LLVMGetExecutionEngineTargetData(LLVMExecutionEngineRef EE);
LLVM_C_ABI LLVMTargetMachineRef
LLVMGetExecutionEngineTargetMachine(LLVMExecutionEngineRef EE);

LLVM_C_ABI void LLVMAddGlobalMapping(LLVMExecutionEngineRef EE,
                                     LLVMValueRef Global, void *Addr);

LLVM_C_ABI void *LLVMGetPointerToGlobal(LLVMExecutionEngineRef EE,
                                        LLVMValueRef Global);

LLVM_C_ABI uint64_t LLVMGetGlobalValueAddress(LLVMExecutionEngineRef EE,
                                              const char *Name);

LLVM_C_ABI uint64_t LLVMGetFunctionAddress(LLVMExecutionEngineRef EE,
                                           const char *Name);

/// Returns true on error, false on success. If true is returned then the error
/// message is copied to OutStr and cleared in the ExecutionEngine instance.
LLVM_C_ABI LLVMBool LLVMExecutionEngineGetErrMsg(LLVMExecutionEngineRef EE,
                                                 char **OutError);

/*===-- Operations on memory managers -------------------------------------===*/

typedef uint8_t *(*LLVMMemoryManagerAllocateCodeSectionCallback)(
  void *Opaque, uintptr_t Size, unsigned Alignment, unsigned SectionID,
  const char *SectionName);
typedef uint8_t *(*LLVMMemoryManagerAllocateDataSectionCallback)(
  void *Opaque, uintptr_t Size, unsigned Alignment, unsigned SectionID,
  const char *SectionName, LLVMBool IsReadOnly);
typedef LLVMBool (*LLVMMemoryManagerFinalizeMemoryCallback)(
  void *Opaque, char **ErrMsg);
typedef void (*LLVMMemoryManagerDestroyCallback)(void *Opaque);

/**
 * Create a simple custom MCJIT memory manager. This memory manager can
 * intercept allocations in a module-oblivious way. This will return NULL
 * if any of the passed functions are NULL.
 *
 * @param Opaque An opaque client object to pass back to the callbacks.
 * @param AllocateCodeSection Allocate a block of memory for executable code.
 * @param AllocateDataSection Allocate a block of memory for data.
 * @param FinalizeMemory Set page permissions and flush cache. Return 0 on
 *   success, 1 on error.
 */
LLVM_C_ABI LLVMMCJITMemoryManagerRef LLVMCreateSimpleMCJITMemoryManager(
    void *Opaque,
    LLVMMemoryManagerAllocateCodeSectionCallback AllocateCodeSection,
    LLVMMemoryManagerAllocateDataSectionCallback AllocateDataSection,
    LLVMMemoryManagerFinalizeMemoryCallback FinalizeMemory,
    LLVMMemoryManagerDestroyCallback Destroy);

LLVM_C_ABI void LLVMDisposeMCJITMemoryManager(LLVMMCJITMemoryManagerRef MM);

/*===-- JIT Event Listener functions -------------------------------------===*/

LLVM_C_ABI LLVMJITEventListenerRef LLVMCreateGDBRegistrationListener(void);
LLVM_C_ABI LLVMJITEventListenerRef LLVMCreateIntelJITEventListener(void);
LLVM_C_ABI LLVMJITEventListenerRef LLVMCreateOProfileJITEventListener(void);
LLVM_C_ABI LLVMJITEventListenerRef LLVMCreatePerfJITEventListener(void);

/**
 * @}
 */

LLVM_C_EXTERN_C_END

#endif
