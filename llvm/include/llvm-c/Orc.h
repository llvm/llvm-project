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
 * Represents generic linkage flags for a symbol definition.
 */
typedef enum {
  LLVMJITSymbolGenericFlagsExported = 1U << 0,
  LLVMJITSymbolGenericFlagsWeak = 1U << 1
} LLVMJITSymbolGenericFlags;

/**
 * Represents target specific flags for a symbol definition.
 */
typedef uint8_t LLVMJITTargetSymbolFlags;

/**
 * Represents the linkage flags for a symbol definition.
 */
typedef struct {
  uint8_t GenericFlags;
  uint8_t TargetFlags;
} LLVMJITSymbolFlags;

/**
 * Represents an evaluated symbol address and flags.
 */
typedef struct {
  LLVMOrcJITTargetAddress Address;
  LLVMJITSymbolFlags Flags;
} LLVMJITEvaluatedSymbol;

/**
 * A reference to an orc::ExecutionSession instance.
 */
typedef struct LLVMOrcOpaqueExecutionSession *LLVMOrcExecutionSessionRef;

/**
 * Error reporter function.
 */
typedef void (*LLVMOrcErrorReporterFunction)(void *Ctx, LLVMErrorRef Err);

/**
 * A reference to an orc::SymbolStringPool.
 */
typedef struct LLVMOrcOpaqueSymbolStringPool *LLVMOrcSymbolStringPoolRef;

/**
 * A reference to an orc::SymbolStringPool table entry.
 */
typedef struct LLVMOrcOpaqueSymbolStringPoolEntry
    *LLVMOrcSymbolStringPoolEntryRef;

/**
 * Represents a pair of a symbol name and an evaluated symbol.
 */
typedef struct {
  LLVMOrcSymbolStringPoolEntryRef Name;
  LLVMJITEvaluatedSymbol Sym;
} LLVMJITCSymbolMapPair;

/**
 * Represents a list of (SymbolStringPtr, JITEvaluatedSymbol) pairs that can be
 * used to construct a SymbolMap.
 */
typedef LLVMJITCSymbolMapPair *LLVMOrcCSymbolMapPairs;

/**
 * Lookup kind. This can be used by definition generators when deciding whether
 * to produce a definition for a requested symbol.
 *
 * This enum should be kept in sync with llvm::orc::LookupKind.
 */
typedef enum {
  LLVMOrcLookupKindStatic,
  LLVMOrcLookupKindDLSym
} LLVMOrcLookupKind;

/**
 * JITDylib lookup flags. This can be used by definition generators when
 * deciding whether to produce a definition for a requested symbol.
 *
 * This enum should be kept in sync with llvm::orc::JITDylibLookupFlags.
 */
typedef enum {
  LLVMOrcJITDylibLookupFlagsMatchExportedSymbolsOnly,
  LLVMOrcJITDylibLookupFlagsMatchAllSymbols
} LLVMOrcJITDylibLookupFlags;

/**
 * Symbol lookup flags for lookup sets. This should be kept in sync with
 * llvm::orc::SymbolLookupFlags.
 */
typedef enum {
  LLVMOrcSymbolLookupFlagsRequiredSymbol,
  LLVMOrcSymbolLookupFlagsWeaklyReferencedSymbol
} LLVMOrcSymbolLookupFlags;

/**
 * An element type for a symbol lookup set.
 */
typedef struct {
  LLVMOrcSymbolStringPoolEntryRef Name;
  LLVMOrcSymbolLookupFlags LookupFlags;
} LLVMOrcCLookupSetElement;

/**
 * A set of symbols to look up / generate.
 *
 * The list is terminated with an element containing a null pointer for the
 * Name field.
 *
 * If a client creates an instance of this type then they are responsible for
 * freeing it, and for ensuring that all strings have been retained over the
 * course of its life. Clients receiving a copy from a callback are not
 * responsible for managing lifetime or retain counts.
 */
typedef LLVMOrcCLookupSetElement *LLVMOrcCLookupSet;

/**
 * A reference to an orc::MaterializationUnit.
 */
typedef struct LLVMOrcOpaqueMaterializationUnit *LLVMOrcMaterializationUnitRef;

/**
 * A reference to an orc::JITDylib instance.
 */
typedef struct LLVMOrcOpaqueJITDylib *LLVMOrcJITDylibRef;

/**
 * A reference to an orc::ResourceTracker instance.
 */
typedef struct LLVMOrcOpaqueResourceTracker *LLVMOrcResourceTrackerRef;

/**
 * A reference to an orc::DefinitionGenerator.
 */
typedef struct LLVMOrcOpaqueDefinitionGenerator
    *LLVMOrcDefinitionGeneratorRef;

/**
 * An opaque lookup state object. Instances of this type can be captured to
 * suspend a lookup while a custom generator function attempts to produce a
 * definition.
 *
 * If a client captures a lookup state object then they must eventually call
 * LLVMOrcLookupStateContinueLookup to restart the lookup. This is required
 * in order to release memory allocated for the lookup state, even if errors
 * have occurred while the lookup was suspended (if these errors have made the
 * lookup impossible to complete then it will issue its own error before
 * destruction).
 */
typedef struct LLVMOrcOpaqueLookupState *LLVMOrcLookupStateRef;

/**
 * A custom generator function. This can be used to create a custom generator
 * object using LLVMOrcCreateCustomCAPIDefinitionGenerator. The resulting
 * object can be attached to a JITDylib, via LLVMOrcJITDylibAddGenerator, to
 * receive callbacks when lookups fail to match existing definitions.
 *
 * GeneratorObj will contain the address of the custom generator object.
 *
 * Ctx will contain the context object passed to
 * LLVMOrcCreateCustomCAPIDefinitionGenerator.
 *
 * LookupState will contain a pointer to an LLVMOrcLookupStateRef object. This
 * can optionally be modified to make the definition generation process
 * asynchronous: If the LookupStateRef value is copied, and the original
 * LLVMOrcLookupStateRef set to null, the lookup will be suspended. Once the
 * asynchronous definition process has been completed clients must call
 * LLVMOrcLookupStateContinueLookup to continue the lookup (this should be
 * done unconditionally, even if errors have occurred in the mean time, to
 * free the lookup state memory and notify the query object of the failures. If
 * LookupState is captured this function must return LLVMErrorSuccess.
 *
 * The Kind argument can be inspected to determine the lookup kind (e.g.
 * as-if-during-static-link, or as-if-during-dlsym).
 *
 * The JD argument specifies which JITDylib the definitions should be generated
 * into.
 *
 * The JDLookupFlags argument can be inspected to determine whether the original
 * lookup included non-exported symobls.
 *
 * Finally, the LookupSet argument contains the set of symbols that could not
 * be found in JD already (the set of generation candidates).
 */
typedef LLVMErrorRef (*LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction)(
    LLVMOrcDefinitionGeneratorRef GeneratorObj, void *Ctx,
    LLVMOrcLookupStateRef *LookupState, LLVMOrcLookupKind Kind,
    LLVMOrcJITDylibRef JD, LLVMOrcJITDylibLookupFlags JDLookupFlags,
    LLVMOrcCLookupSet LookupSet, size_t LookupSetSize);

/**
 * Predicate function for SymbolStringPoolEntries.
 */
typedef int (*LLVMOrcSymbolPredicate)(void *Ctx,
                                      LLVMOrcSymbolStringPoolEntryRef Sym);

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
 * A reference to an orc::ObjectLayer instance.
 */
typedef struct LLVMOrcOpaqueObjectLayer *LLVMOrcObjectLayerRef;

/**
 * Attach a custom error reporter function to the ExecutionSession.
 *
 * The error reporter will be called to deliver failure notices that can not be
 * directly reported to a caller. For example, failure to resolve symbols in
 * the JIT linker is typically reported via the error reporter (callers
 * requesting definitions from the JIT will typically be delivered a
 * FailureToMaterialize error instead).
 */
void LLVMOrcExecutionSessionSetErrorReporter(
    LLVMOrcExecutionSessionRef ES, LLVMOrcErrorReporterFunction ReportError,
    void *Ctx);

/**
 * Return a reference to the SymbolStringPool for an ExecutionSession.
 *
 * Ownership of the pool remains with the ExecutionSession: The caller is
 * not required to free the pool.
 */
LLVMOrcSymbolStringPoolRef
LLVMOrcExecutionSessionGetSymbolStringPool(LLVMOrcExecutionSessionRef ES);

/**
 * Clear all unreferenced symbol string pool entries.
 *
 * This can be called at any time to release unused entries in the
 * ExecutionSession's string pool. Since it locks the pool (preventing
 * interning of any new strings) it is recommended that it only be called
 * infrequently, ideally when the caller has reason to believe that some
 * entries will have become unreferenced, e.g. after removing a module or
 * closing a JITDylib.
 */
void LLVMOrcSymbolStringPoolClearDeadEntries(LLVMOrcSymbolStringPoolRef SSP);

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
 * Increments the ref-count for a SymbolStringPool entry.
 */
void LLVMOrcRetainSymbolStringPoolEntry(LLVMOrcSymbolStringPoolEntryRef S);

/**
 * Reduces the ref-count for of a SymbolStringPool entry.
 */
void LLVMOrcReleaseSymbolStringPoolEntry(LLVMOrcSymbolStringPoolEntryRef S);

const char *LLVMOrcSymbolStringPoolEntryStr(LLVMOrcSymbolStringPoolEntryRef S);

/**
 * Reduces the ref-count of a ResourceTracker.
 */
void LLVMOrcReleaseResourceTracker(LLVMOrcResourceTrackerRef RT);

/**
 * Transfers tracking of all resources associated with resource tracker SrcRT
 * to resource tracker DstRT.
 */
void LLVMOrcResourceTrackerTransferTo(LLVMOrcResourceTrackerRef SrcRT,
                                      LLVMOrcResourceTrackerRef DstRT);

/**
 * Remove all resources associated with the given tracker. See
 * ResourceTracker::remove().
 */
LLVMErrorRef LLVMOrcResourceTrackerRemove(LLVMOrcResourceTrackerRef RT);

/**
 * Dispose of a JITDylib::DefinitionGenerator. This should only be called if
 * ownership has not been passed to a JITDylib (e.g. because some error
 * prevented the client from calling LLVMOrcJITDylibAddGenerator).
 */
void LLVMOrcDisposeDefinitionGenerator(
    LLVMOrcDefinitionGeneratorRef DG);

/**
 * Dispose of a MaterializationUnit.
 */
void LLVMOrcDisposeMaterializationUnit(LLVMOrcMaterializationUnitRef MU);

/**
 * Create a MaterializationUnit to define the given symbols as pointing to
 * the corresponding raw addresses.
 */
LLVMOrcMaterializationUnitRef
LLVMOrcAbsoluteSymbols(LLVMOrcCSymbolMapPairs Syms, size_t NumPairs);

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
 * Return a reference to a newly created resource tracker associated with JD.
 * The tracker is returned with an initial ref-count of 1, and must be released
 * with LLVMOrcReleaseResourceTracker when no longer needed.
 */
LLVMOrcResourceTrackerRef
LLVMOrcJITDylibCreateResourceTracker(LLVMOrcJITDylibRef JD);

/**
 * Return a reference to the default resource tracker for the given JITDylib.
 * This operation will increase the retain count of the tracker: Clients should
 * call LLVMOrcReleaseResourceTracker when the result is no longer needed.
 */
LLVMOrcResourceTrackerRef
LLVMOrcJITDylibGetDefaultResourceTracker(LLVMOrcJITDylibRef JD);

/**
 * Add the given MaterializationUnit to the given JITDylib.
 *
 * If this operation succeeds then JITDylib JD will take ownership of MU.
 * If the operation fails then ownership remains with the caller who should
 * call LLVMOrcDisposeMaterializationUnit to destroy it.
 */
LLVMErrorRef LLVMOrcJITDylibDefine(LLVMOrcJITDylibRef JD,
                                   LLVMOrcMaterializationUnitRef MU);

/**
 * Calls remove on all trackers associated with this JITDylib, see
 * JITDylib::clear().
 */
LLVMErrorRef LLVMOrcJITDylibClear(LLVMOrcJITDylibRef JD);

/**
 * Add a DefinitionGenerator to the given JITDylib.
 *
 * The JITDylib will take ownership of the given generator: The client is no
 * longer responsible for managing its memory.
 */
void LLVMOrcJITDylibAddGenerator(LLVMOrcJITDylibRef JD,
                                 LLVMOrcDefinitionGeneratorRef DG);

/**
 * Create a custom generator.
 */
LLVMOrcDefinitionGeneratorRef LLVMOrcCreateCustomCAPIDefinitionGenerator(
    LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction F, void *Ctx);

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
    LLVMOrcDefinitionGeneratorRef *Result, char GlobalPrefx,
    LLVMOrcSymbolPredicate Filter, void *FilterCtx);

/**
 * Create a ThreadSafeContext containing a new LLVMContext.
 *
 * Ownership of the underlying ThreadSafeContext data is shared: Clients
 * can and should dispose of their ThreadSafeContext as soon as they no longer
 * need to refer to it directly. Other references (e.g. from ThreadSafeModules)
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
 * (e.g. by LLVMOrcLLJITAddLLVMIRModule) then the client is no longer
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
 * Dispose of an ObjectLayer.
 */
void LLVMOrcDisposeObjectLayer(LLVMOrcObjectLayerRef ObjLayer);

LLVM_C_EXTERN_C_END

#endif /* LLVM_C_ORC_H */
