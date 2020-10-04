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
|* JIT compilation of LLVM IR. Minimal documentation of C API specific issues *|
|* (especially memory ownership rules) is provided. Core Orc concepts are     *|
|* documented in llvm/docs/ORCv2.rst and APIs are documented in the C++       *|
|* headers                                                                    *|
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
#include "llvm-c/TargetMachine.h"
#include "llvm-c/Types.h"

LLVM_C_EXTERN_C_BEGIN

/**
 * Represents an address in the target process.
 */
typedef uint64_t LLVMOrcJITTargetAddress;

/**
 * A reference to an orc::ExecutionSession instance.
 */
typedef struct LLVMOrcOpaqueExecutionSession *LLVMOrcExecutionSessionRef;

/**
 * A reference to an orc::SymbolStringPool table entry.
 */
typedef struct LLVMOrcQuaqueSymbolStringPoolEntryPtr
    *LLVMOrcSymbolStringPoolEntryRef;

/**
 * A reference to an orc::JITDylib instance.
 */
typedef struct LLVMOrcOpaqueJITDylib *LLVMOrcJITDylibRef;

/**
 * A reference to an orc::JITDylib::DefinitionGenerator.
 */
typedef struct LLVMOrcOpaqueJITDylibDefinitionGenerator
    *LLVMOrcJITDylibDefinitionGeneratorRef;

/**
 * Predicate function for SymbolStringPoolEntries.
 */
typedef int (*LLVMOrcSymbolPredicate)(LLVMOrcSymbolStringPoolEntryRef Sym,
                                      void *Ctx);

/**
 * A reference to an orc::ThreadSafeContext instance.
 */
typedef struct LLVMOrcOpaqueThreadSafeContext *LLVMOrcThreadSafeContextRef;

/**
 * A reference to an orc::ThreadSafeModule instance.
 */
typedef struct LLVMOrcOpaqueThreadSafeModule *LLVMOrcThreadSafeModuleRef;

/**
 * A reference to an orc::JITTargetMachineBuilder instance.
 */
typedef struct LLVMOrcOpaqueJITTargetMachineBuilder
    *LLVMOrcJITTargetMachineBuilderRef;

/**
 * A reference to an orc::LLJITBuilder instance.
 */
typedef struct LLVMOrcOpaqueLLJITBuilder *LLVMOrcLLJITBuilderRef;

/**
 * A reference to an orc::LLJIT instance.
 */
typedef struct LLVMOrcOpaqueLLJIT *LLVMOrcLLJITRef;

/**
 * Intern a string in the ExecutionSession's SymbolStringPool and return a
 * reference to it. This increments the ref-count of the pool entry, and the
 * returned value should be released once the client is done with it by
 * calling LLVMOrReleaseSymbolStringPoolEntry.
 *
 * Since strings are uniqued within the SymbolStringPool
 * LLVMOrcSymbolStringPoolEntryRefs can be compared by value to test string
 * equality.
 *
 * Note that this function does not perform linker-mangling on the string.
 */
LLVMOrcSymbolStringPoolEntryRef
LLVMOrcExecutionSessionIntern(LLVMOrcExecutionSessionRef ES, const char *Name);

/**
 * Reduces the ref-count for of a SymbolStringPool entry.
 */
void LLVMOrcReleaseSymbolStringPoolEntry(LLVMOrcSymbolStringPoolEntryRef S);

/**
 * Create a "bare" JITDylib.
 *
 * The client is responsible for ensuring that the JITDylib's name is unique,
 * e.g. by calling LLVMOrcExecutionSessionGetJTIDylibByName first.
 *
 * This call does not install any library code or symbols into the newly
 * created JITDylib. The client is responsible for all configuration.
 */
LLVMOrcJITDylibRef
LLVMOrcExecutionSessionCreateBareJITDylib(LLVMOrcExecutionSessionRef ES,
                                          const char *Name);

/**
 * Create a JITDylib.
 *
 * The client is responsible for ensuring that the JITDylib's name is unique,
 * e.g. by calling LLVMOrcExecutionSessionGetJTIDylibByName first.
 *
 * If a Platform is attached to the ExecutionSession then
 * Platform::setupJITDylib will be called to install standard platform symbols
 * (e.g. standard library interposes). If no Platform is installed then this
 * call is equivalent to LLVMExecutionSessionRefCreateBareJITDylib and will
 * always return success.
 */
LLVMErrorRef
LLVMOrcExecutionSessionCreateJITDylib(LLVMOrcExecutionSessionRef ES,
                                      LLVMOrcJITDylibRef *Result,
                                      const char *Name);

/**
 * Returns the JITDylib with the given name, or NULL if no such JITDylib
 * exists.
 */
LLVMOrcJITDylibRef LLVMOrcExecutionSessionGetJITDylibByName(const char *Name);

/**
 * Dispose of a JITDylib::DefinitionGenerator. This should only be called if
 * ownership has not been passed to a JITDylib (e.g. because some error
 * prevented the client from calling LLVMOrcJITDylibAddGenerator).
 */
void LLVMOrcDisposeJITDylibDefinitionGenerator(
    LLVMOrcJITDylibDefinitionGeneratorRef DG);

/**
 * Add a JITDylib::DefinitionGenerator to the given JITDylib.
 *
 * The JITDylib will take ownership of the given generator: The client is no
 * longer responsible for managing its memory.
 */
void LLVMOrcJITDylibAddGenerator(LLVMOrcJITDylibRef JD,
                                 LLVMOrcJITDylibDefinitionGeneratorRef DG);

/**
 * Get a DynamicLibrarySearchGenerator that will reflect process symbols into
 * the JITDylib. On success the resulting generator is owned by the client.
 * Ownership is typically transferred by adding the instance to a JITDylib
 * using LLVMOrcJITDylibAddGenerator,
 *
 * The GlobalPrefix argument specifies the character that appears on the front
 * of linker-mangled symbols for the target platform (e.g. '_' on MachO).
 * If non-null, this character will be stripped from the start of all symbol
 * strings before passing the remaining substring to dlsym.
 *
 * The optional Filter and Ctx arguments can be used to supply a symbol name
 * filter: Only symbols for which the filter returns true will be visible to
 * JIT'd code. If the Filter argument is null then all process symbols will
 * be visible to JIT'd code. Note that the symbol name passed to the Filter
 * function is the full mangled symbol: The client is responsible for stripping
 * the global prefix if present.
 */
LLVMErrorRef LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(
    LLVMOrcJITDylibDefinitionGeneratorRef *Result, char GlobalPrefx,
    LLVMOrcSymbolPredicate Filter, void *FilterCtx);

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
 * Dispose of a ThreadSafeModule. This should only be called if ownership has
 * not been passed to LLJIT (e.g. because some error prevented the client from
 * adding this to the JIT).
 */
void LLVMOrcDisposeThreadSafeModule(LLVMOrcThreadSafeModuleRef TSM);

/**
 * Create a JITTargetMachineBuilder by detecting the host.
 *
 * On success the client owns the resulting JITTargetMachineBuilder. It must be
 * passed to a consuming operation (e.g. LLVMOrcCreateLLJITBuilder) or disposed
 * of by calling LLVMOrcDisposeJITTargetMachineBuilder.
 */
LLVMErrorRef LLVMOrcJITTargetMachineBuilderDetectHost(
    LLVMOrcJITTargetMachineBuilderRef *Result);

/**
 * Create a JITTargetMachineBuilder from the given TargetMachine template.
 *
 * This operation takes ownership of the given TargetMachine and destroys it
 * before returing. The resulting JITTargetMachineBuilder is owned by the client
 * and must be passed to a consuming operation (e.g. LLVMOrcCreateLLJITBuilder)
 * or disposed of by calling LLVMOrcDisposeJITTargetMachineBuilder.
 */
LLVMOrcJITTargetMachineBuilderRef
LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(LLVMTargetMachineRef TM);

/**
 * Dispose of a JITTargetMachineBuilder.
 */
void LLVMOrcDisposeJITTargetMachineBuilder(
    LLVMOrcJITTargetMachineBuilderRef JTMB);

/**
 * Create an LLJITTargetMachineBuilder.
 *
 * The client owns the resulting LLJITBuilder and should dispose of it using
 * LLVMOrcDisposeLLJITBuilder once they are done with it.
 */
LLVMOrcLLJITBuilderRef LLVMOrcCreateLLJITBuilder(void);

/**
 * Dispose of an LLVMOrcLLJITBuilderRef. This should only be called if ownership
 * has not been passed to LLVMOrcCreateLLJIT (e.g. because some error prevented
 * that function from being called).
 */
void LLVMOrcDisposeLLJITBuilder(LLVMOrcLLJITBuilderRef Builder);

/**
 * Set the JITTargetMachineBuilder to be used when constructing the LLJIT
 * instance. Calling this function is optional: if it is not called then the
 * LLJITBuilder will use JITTargeTMachineBuilder::detectHost to construct a
 * JITTargetMachineBuilder.
 */
void LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(
    LLVMOrcLLJITBuilderRef Builder, LLVMOrcJITTargetMachineBuilderRef JTMB);

/**
 * Create an LLJIT instance from an LLJITBuilder.
 *
 * This operation takes ownership of the Builder argument: clients should not
 * dispose of the builder after calling this function (even if the function
 * returns an error). If a null Builder argument is provided then a
 * default-constructed LLJITBuilder will be used.
 *
 * On success the resulting LLJIT instance is uniquely owned by the client and
 * automatically manages the memory of all JIT'd code and all modules that are
 * transferred to it (e.g. via LLVMOrcLLJITAddLLVMIRModule). Disposing of the
 * LLJIT instance will free all memory managed by the JIT, including JIT'd code
 * and not-yet compiled modules.
 */
LLVMErrorRef LLVMOrcCreateLLJIT(LLVMOrcLLJITRef *Result,
                                LLVMOrcLLJITBuilderRef Builder);

/**
 * Dispose of an LLJIT instance.
 */
LLVMErrorRef LLVMOrcDisposeLLJIT(LLVMOrcLLJITRef J);

/**
 * Get a reference to the ExecutionSession for this LLJIT instance.
 *
 * The ExecutionSession is owned by the LLJIT instance. The client is not
 * responsible for managing its memory.
 */
LLVMOrcExecutionSessionRef LLVMOrcLLJITGetExecutionSession(LLVMOrcLLJITRef J);

/**
 * Return a reference to the Main JITDylib.
 *
 * The JITDylib is owned by the LLJIT instance. The client is not responsible
 * for managing its memory.
 */
LLVMOrcJITDylibRef LLVMOrcLLJITGetMainJITDylib(LLVMOrcLLJITRef J);

/**
 * Return the target triple for this LLJIT instance. This string is owned by
 * the LLJIT instance and should not be freed by the client.
 */
const char *LLVMOrcLLJITGetTripleString(LLVMOrcLLJITRef J);

/**
 * Returns the global prefix character according to the LLJIT's DataLayout.
 */
char LLVMOrcLLJITGetGlobalPrefix(LLVMOrcLLJITRef J);

/**
 * Mangles the given string according to the LLJIT instance's DataLayout, then
 * interns the result in the SymbolStringPool and returns a reference to the
 * pool entry. Clients should call LLVMOrcReleaseSymbolStringPoolEntry to
 * decrement the ref-count on the pool entry once they are finished with this
 * value.
 */
LLVMOrcSymbolStringPoolEntryRef
LLVMOrcLLJITMangleAndIntern(LLVMOrcLLJITRef J, const char *UnmangledName);

/**
 * Add a buffer representing an object file to the given JITDylib in the given
 * LLJIT instance. This operation transfers ownership of the buffer to the
 * LLJIT instance. The buffer should not be disposed of or referenced once this
 * function returns.
 */
LLVMErrorRef LLVMOrcLLJITAddObjectFile(LLVMOrcLLJITRef J, LLVMOrcJITDylibRef JD,
                                       LLVMMemoryBufferRef ObjBuffer);

/**
 * Add an IR module to the given JITDylib of the given LLJIT instance. This
 * operation transfers ownership of the TSM argument to the LLJIT instance.
 * The TSM argument should not be 3disposed of or referenced once this
 * function returns.
 */
LLVMErrorRef LLVMOrcLLJITAddLLVMIRModule(LLVMOrcLLJITRef J,
                                         LLVMOrcJITDylibRef JD,
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
