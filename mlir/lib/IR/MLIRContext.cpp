//===- MLIRContext.cpp - MLIR Type Classes --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "AffineExprDetail.h"
#include "AffineMapDetail.h"
#include "AttributeDetail.h"
#include "IntegerSetDetail.h"
#include "TypeDetail.h"
#include "mlir/IR/Action.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/RWMutex.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <optional>

#define DEBUG_TYPE "mlircontext"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// MLIRContext CommandLine Options
//===----------------------------------------------------------------------===//

namespace {
/// This struct contains command line options that can be used to initialize
/// various bits of an MLIRContext. This uses a struct wrapper to avoid the need
/// for global command line options.
struct MLIRContextOptions {
  llvm::cl::opt<bool> disableThreading{
      "mlir-disable-threading",
      llvm::cl::desc("Disable multi-threading within MLIR, overrides any "
                     "further call to MLIRContext::enableMultiThreading()")};

  llvm::cl::opt<bool> printOpOnDiagnostic{
      "mlir-print-op-on-diagnostic",
      llvm::cl::desc("When a diagnostic is emitted on an operation, also print "
                     "the operation as an attached note"),
      llvm::cl::init(true)};

  llvm::cl::opt<bool> printStackTraceOnDiagnostic{
      "mlir-print-stacktrace-on-diagnostic",
      llvm::cl::desc("When a diagnostic is emitted, also print the stack trace "
                     "as an attached note")};
};
} // namespace

static llvm::ManagedStatic<MLIRContextOptions> clOptions;

static bool isThreadingGloballyDisabled() {
#if LLVM_ENABLE_THREADS != 0
  return clOptions.isConstructed() && clOptions->disableThreading;
#else
  return true;
#endif
}

/// Register a set of useful command-line options that can be used to configure
/// various flags within the MLIRContext. These flags are used when constructing
/// an MLIR context for initialization.
void mlir::registerMLIRContextCLOptions() {
  // Make sure that the options struct has been initialized.
  *clOptions;
}

//===----------------------------------------------------------------------===//
// Locking Utilities
//===----------------------------------------------------------------------===//

namespace {
/// Utility writer lock that takes a runtime flag that specifies if we really
/// need to lock.
struct ScopedWriterLock {
  ScopedWriterLock(llvm::sys::SmartRWMutex<true> &mutexParam, bool shouldLock)
      : mutex(shouldLock ? &mutexParam : nullptr) {
    if (mutex)
      mutex->lock();
  }
  ~ScopedWriterLock() {
    if (mutex)
      mutex->unlock();
  }
  llvm::sys::SmartRWMutex<true> *mutex;
};
} // namespace

//===----------------------------------------------------------------------===//
// MLIRContextImpl
//===----------------------------------------------------------------------===//

namespace mlir {
/// This is the implementation of the MLIRContext class, using the pImpl idiom.
/// This class is completely private to this file, so everything is public.
class MLIRContextImpl {
public:
  //===--------------------------------------------------------------------===//
  // Debugging
  //===--------------------------------------------------------------------===//

  /// An action handler for handling actions that are dispatched through this
  /// context.
  std::function<void(function_ref<void()>, const tracing::Action &)>
      actionHandler;

  //===--------------------------------------------------------------------===//
  // Diagnostics
  //===--------------------------------------------------------------------===//
  DiagnosticEngine diagEngine;

  //===--------------------------------------------------------------------===//
  // Options
  //===--------------------------------------------------------------------===//

  /// In most cases, creating operation in unregistered dialect is not desired
  /// and indicate a misconfiguration of the compiler. This option enables to
  /// detect such use cases
  bool allowUnregisteredDialects = false;

  /// Enable support for multi-threading within MLIR.
  bool threadingIsEnabled = true;

  /// Track if we are currently executing in a threaded execution environment
  /// (like the pass-manager): this is only a debugging feature to help reducing
  /// the chances of data races one some context APIs.
#ifndef NDEBUG
  std::atomic<int> multiThreadedExecutionContext{0};
#endif

  /// If the operation should be attached to diagnostics printed via the
  /// Operation::emit methods.
  bool printOpOnDiagnostic = true;

  /// If the current stack trace should be attached when emitting diagnostics.
  bool printStackTraceOnDiagnostic = false;

  //===--------------------------------------------------------------------===//
  // Other
  //===--------------------------------------------------------------------===//

  /// This points to the ThreadPool used when processing MLIR tasks in parallel.
  /// It can't be nullptr when multi-threading is enabled. Otherwise if
  /// multi-threading is disabled, and the threadpool wasn't externally provided
  /// using `setThreadPool`, this will be nullptr.
  llvm::ThreadPool *threadPool = nullptr;

  /// In case where the thread pool is owned by the context, this ensures
  /// destruction with the context.
  std::unique_ptr<llvm::ThreadPool> ownedThreadPool;

  /// This is a list of dialects that are created referring to this context.
  /// The MLIRContext owns the objects.
  DenseMap<StringRef, std::unique_ptr<Dialect>> loadedDialects;
  DialectRegistry dialectsRegistry;

  /// An allocator used for AbstractAttribute and AbstractType objects.
  llvm::BumpPtrAllocator abstractDialectSymbolAllocator;

  /// This is a mapping from operation name to the operation info describing it.
  llvm::StringMap<std::unique_ptr<OperationName::Impl>> operations;

  /// A vector of operation info specifically for registered operations.
  llvm::StringMap<RegisteredOperationName> registeredOperations;

  /// This is a sorted container of registered operations for a deterministic
  /// and efficient `getRegisteredOperations` implementation.
  SmallVector<RegisteredOperationName, 0> sortedRegisteredOperations;

  /// A mutex used when accessing operation information.
  llvm::sys::SmartRWMutex<true> operationInfoMutex;

  //===--------------------------------------------------------------------===//
  // Affine uniquing
  //===--------------------------------------------------------------------===//

  // Affine expression, map and integer set uniquing.
  StorageUniquer affineUniquer;

  //===--------------------------------------------------------------------===//
  // Type uniquing
  //===--------------------------------------------------------------------===//

  DenseMap<TypeID, AbstractType *> registeredTypes;
  StorageUniquer typeUniquer;

  /// Cached Type Instances.
  Float8E5M2Type f8E5M2Ty;
  Float8E4M3FNType f8E4M3FNTy;
  Float8E5M2FNUZType f8E5M2FNUZTy;
  Float8E4M3FNUZType f8E4M3FNUZTy;
  Float8E4M3B11FNUZType f8E4M3B11FNUZTy;
  BFloat16Type bf16Ty;
  Float16Type f16Ty;
  Float32Type f32Ty;
  Float64Type f64Ty;
  Float80Type f80Ty;
  Float128Type f128Ty;
  IndexType indexTy;
  IntegerType int1Ty, int8Ty, int16Ty, int32Ty, int64Ty, int128Ty;
  NoneType noneType;

  //===--------------------------------------------------------------------===//
  // Attribute uniquing
  //===--------------------------------------------------------------------===//

  DenseMap<TypeID, AbstractAttribute *> registeredAttributes;
  StorageUniquer attributeUniquer;

  /// Cached Attribute Instances.
  BoolAttr falseAttr, trueAttr;
  UnitAttr unitAttr;
  UnknownLoc unknownLocAttr;
  DictionaryAttr emptyDictionaryAttr;
  StringAttr emptyStringAttr;

  /// Map of string attributes that may reference a dialect, that are awaiting
  /// that dialect to be loaded.
  llvm::sys::SmartMutex<true> dialectRefStrAttrMutex;
  DenseMap<StringRef, SmallVector<StringAttrStorage *>>
      dialectReferencingStrAttrs;

public:
  MLIRContextImpl(bool threadingIsEnabled)
      : threadingIsEnabled(threadingIsEnabled) {
    if (threadingIsEnabled) {
      ownedThreadPool = std::make_unique<llvm::ThreadPool>();
      threadPool = ownedThreadPool.get();
    }
  }
  ~MLIRContextImpl() {
    for (auto typeMapping : registeredTypes)
      typeMapping.second->~AbstractType();
    for (auto attrMapping : registeredAttributes)
      attrMapping.second->~AbstractAttribute();
  }
};
} // namespace mlir

MLIRContext::MLIRContext(Threading setting)
    : MLIRContext(DialectRegistry(), setting) {}

MLIRContext::MLIRContext(const DialectRegistry &registry, Threading setting)
    : impl(new MLIRContextImpl(setting == Threading::ENABLED &&
                               !isThreadingGloballyDisabled())) {
  // Initialize values based on the command line flags if they were provided.
  if (clOptions.isConstructed()) {
    printOpOnDiagnostic(clOptions->printOpOnDiagnostic);
    printStackTraceOnDiagnostic(clOptions->printStackTraceOnDiagnostic);
  }

  // Pre-populate the registry.
  registry.appendTo(impl->dialectsRegistry);

  // Ensure the builtin dialect is always pre-loaded.
  getOrLoadDialect<BuiltinDialect>();

  // Initialize several common attributes and types to avoid the need to lock
  // the context when accessing them.

  //// Types.
  /// Floating-point Types.
  impl->f8E5M2Ty = TypeUniquer::get<Float8E5M2Type>(this);
  impl->f8E4M3FNTy = TypeUniquer::get<Float8E4M3FNType>(this);
  impl->f8E5M2FNUZTy = TypeUniquer::get<Float8E5M2FNUZType>(this);
  impl->f8E4M3FNUZTy = TypeUniquer::get<Float8E4M3FNUZType>(this);
  impl->f8E4M3B11FNUZTy = TypeUniquer::get<Float8E4M3B11FNUZType>(this);
  impl->bf16Ty = TypeUniquer::get<BFloat16Type>(this);
  impl->f16Ty = TypeUniquer::get<Float16Type>(this);
  impl->f32Ty = TypeUniquer::get<Float32Type>(this);
  impl->f64Ty = TypeUniquer::get<Float64Type>(this);
  impl->f80Ty = TypeUniquer::get<Float80Type>(this);
  impl->f128Ty = TypeUniquer::get<Float128Type>(this);
  /// Index Type.
  impl->indexTy = TypeUniquer::get<IndexType>(this);
  /// Integer Types.
  impl->int1Ty = TypeUniquer::get<IntegerType>(this, 1, IntegerType::Signless);
  impl->int8Ty = TypeUniquer::get<IntegerType>(this, 8, IntegerType::Signless);
  impl->int16Ty =
      TypeUniquer::get<IntegerType>(this, 16, IntegerType::Signless);
  impl->int32Ty =
      TypeUniquer::get<IntegerType>(this, 32, IntegerType::Signless);
  impl->int64Ty =
      TypeUniquer::get<IntegerType>(this, 64, IntegerType::Signless);
  impl->int128Ty =
      TypeUniquer::get<IntegerType>(this, 128, IntegerType::Signless);
  /// None Type.
  impl->noneType = TypeUniquer::get<NoneType>(this);

  //// Attributes.
  //// Note: These must be registered after the types as they may generate one
  //// of the above types internally.
  /// Unknown Location Attribute.
  impl->unknownLocAttr = AttributeUniquer::get<UnknownLoc>(this);
  /// Bool Attributes.
  impl->falseAttr = IntegerAttr::getBoolAttrUnchecked(impl->int1Ty, false);
  impl->trueAttr = IntegerAttr::getBoolAttrUnchecked(impl->int1Ty, true);
  /// Unit Attribute.
  impl->unitAttr = AttributeUniquer::get<UnitAttr>(this);
  /// The empty dictionary attribute.
  impl->emptyDictionaryAttr = DictionaryAttr::getEmptyUnchecked(this);
  /// The empty string attribute.
  impl->emptyStringAttr = StringAttr::getEmptyStringAttrUnchecked(this);

  // Register the affine storage objects with the uniquer.
  impl->affineUniquer
      .registerParametricStorageType<AffineBinaryOpExprStorage>();
  impl->affineUniquer
      .registerParametricStorageType<AffineConstantExprStorage>();
  impl->affineUniquer.registerParametricStorageType<AffineDimExprStorage>();
  impl->affineUniquer.registerParametricStorageType<AffineMapStorage>();
  impl->affineUniquer.registerParametricStorageType<IntegerSetStorage>();
}

MLIRContext::~MLIRContext() = default;

/// Copy the specified array of elements into memory managed by the provided
/// bump pointer allocator.  This assumes the elements are all PODs.
template <typename T>
static ArrayRef<T> copyArrayRefInto(llvm::BumpPtrAllocator &allocator,
                                    ArrayRef<T> elements) {
  auto result = allocator.Allocate<T>(elements.size());
  std::uninitialized_copy(elements.begin(), elements.end(), result);
  return ArrayRef<T>(result, elements.size());
}

//===----------------------------------------------------------------------===//
// Action Handling
//===----------------------------------------------------------------------===//

void MLIRContext::registerActionHandler(HandlerTy handler) {
  getImpl().actionHandler = std::move(handler);
}

/// Dispatch the provided action to the handler if any, or just execute it.
void MLIRContext::executeActionInternal(function_ref<void()> actionFn,
                                        const tracing::Action &action) {
  assert(getImpl().actionHandler);
  getImpl().actionHandler(actionFn, action);
}

bool MLIRContext::hasActionHandler() { return (bool)getImpl().actionHandler; }

//===----------------------------------------------------------------------===//
// Diagnostic Handlers
//===----------------------------------------------------------------------===//

/// Returns the diagnostic engine for this context.
DiagnosticEngine &MLIRContext::getDiagEngine() { return getImpl().diagEngine; }

//===----------------------------------------------------------------------===//
// Dialect and Operation Registration
//===----------------------------------------------------------------------===//

void MLIRContext::appendDialectRegistry(const DialectRegistry &registry) {
  if (registry.isSubsetOf(impl->dialectsRegistry))
    return;

  assert(impl->multiThreadedExecutionContext == 0 &&
         "appending to the MLIRContext dialect registry while in a "
         "multi-threaded execution context");
  registry.appendTo(impl->dialectsRegistry);

  // For the already loaded dialects, apply any possible extensions immediately.
  registry.applyExtensions(this);
}

const DialectRegistry &MLIRContext::getDialectRegistry() {
  return impl->dialectsRegistry;
}

/// Return information about all registered IR dialects.
std::vector<Dialect *> MLIRContext::getLoadedDialects() {
  std::vector<Dialect *> result;
  result.reserve(impl->loadedDialects.size());
  for (auto &dialect : impl->loadedDialects)
    result.push_back(dialect.second.get());
  llvm::array_pod_sort(result.begin(), result.end(),
                       [](Dialect *const *lhs, Dialect *const *rhs) -> int {
                         return (*lhs)->getNamespace() < (*rhs)->getNamespace();
                       });
  return result;
}
std::vector<StringRef> MLIRContext::getAvailableDialects() {
  std::vector<StringRef> result;
  for (auto dialect : impl->dialectsRegistry.getDialectNames())
    result.push_back(dialect);
  return result;
}

/// Get a registered IR dialect with the given namespace. If none is found,
/// then return nullptr.
Dialect *MLIRContext::getLoadedDialect(StringRef name) {
  // Dialects are sorted by name, so we can use binary search for lookup.
  auto it = impl->loadedDialects.find(name);
  return (it != impl->loadedDialects.end()) ? it->second.get() : nullptr;
}

Dialect *MLIRContext::getOrLoadDialect(StringRef name) {
  Dialect *dialect = getLoadedDialect(name);
  if (dialect)
    return dialect;
  DialectAllocatorFunctionRef allocator =
      impl->dialectsRegistry.getDialectAllocator(name);
  return allocator ? allocator(this) : nullptr;
}

/// Get a dialect for the provided namespace and TypeID: abort the program if a
/// dialect exist for this namespace with different TypeID. Returns a pointer to
/// the dialect owned by the context.
Dialect *
MLIRContext::getOrLoadDialect(StringRef dialectNamespace, TypeID dialectID,
                              function_ref<std::unique_ptr<Dialect>()> ctor) {
  auto &impl = getImpl();
  // Get the correct insertion position sorted by namespace.
  auto dialectIt = impl.loadedDialects.try_emplace(dialectNamespace, nullptr);

  if (dialectIt.second) {
    LLVM_DEBUG(llvm::dbgs()
               << "Load new dialect in Context " << dialectNamespace << "\n");
#ifndef NDEBUG
    if (impl.multiThreadedExecutionContext != 0)
      llvm::report_fatal_error(
          "Loading a dialect (" + dialectNamespace +
          ") while in a multi-threaded execution context (maybe "
          "the PassManager): this can indicate a "
          "missing `dependentDialects` in a pass for example.");
#endif // NDEBUG
    // loadedDialects entry is initialized to nullptr, indicating that the
    // dialect is currently being loaded. Re-lookup the address in
    // loadedDialects because the table might have been rehashed by recursive
    // dialect loading in ctor().
    std::unique_ptr<Dialect> &dialect = impl.loadedDialects[dialectNamespace] =
        ctor();
    assert(dialect && "dialect ctor failed");

    // Refresh all the identifiers dialect field, this catches cases where a
    // dialect may be loaded after identifier prefixed with this dialect name
    // were already created.
    auto stringAttrsIt = impl.dialectReferencingStrAttrs.find(dialectNamespace);
    if (stringAttrsIt != impl.dialectReferencingStrAttrs.end()) {
      for (StringAttrStorage *storage : stringAttrsIt->second)
        storage->referencedDialect = dialect.get();
      impl.dialectReferencingStrAttrs.erase(stringAttrsIt);
    }

    // Apply any extensions to this newly loaded dialect.
    impl.dialectsRegistry.applyExtensions(dialect.get());
    return dialect.get();
  }

#ifndef NDEBUG
  if (dialectIt.first->second == nullptr)
    llvm::report_fatal_error(
        "Loading (and getting) a dialect (" + dialectNamespace +
        ") while the same dialect is still loading: use loadDialect instead "
        "of getOrLoadDialect.");
#endif // NDEBUG

  // Abort if dialect with namespace has already been registered.
  std::unique_ptr<Dialect> &dialect = dialectIt.first->second;
  if (dialect->getTypeID() != dialectID)
    llvm::report_fatal_error("a dialect with namespace '" + dialectNamespace +
                             "' has already been registered");

  return dialect.get();
}

bool MLIRContext::isDialectLoading(StringRef dialectNamespace) {
  auto it = getImpl().loadedDialects.find(dialectNamespace);
  // nullptr indicates that the dialect is currently being loaded.
  return it != getImpl().loadedDialects.end() && it->second == nullptr;
}

DynamicDialect *MLIRContext::getOrLoadDynamicDialect(
    StringRef dialectNamespace, function_ref<void(DynamicDialect *)> ctor) {
  auto &impl = getImpl();
  // Get the correct insertion position sorted by namespace.
  auto dialectIt = impl.loadedDialects.find(dialectNamespace);

  if (dialectIt != impl.loadedDialects.end()) {
    if (auto *dynDialect = dyn_cast<DynamicDialect>(dialectIt->second.get()))
      return dynDialect;
    llvm::report_fatal_error("a dialect with namespace '" + dialectNamespace +
                             "' has already been registered");
  }

  LLVM_DEBUG(llvm::dbgs() << "Load new dynamic dialect in Context "
                          << dialectNamespace << "\n");
#ifndef NDEBUG
  if (impl.multiThreadedExecutionContext != 0)
    llvm::report_fatal_error(
        "Loading a dynamic dialect (" + dialectNamespace +
        ") while in a multi-threaded execution context (maybe "
        "the PassManager): this can indicate a "
        "missing `dependentDialects` in a pass for example.");
#endif

  auto name = StringAttr::get(this, dialectNamespace);
  auto *dialect = new DynamicDialect(name, this);
  (void)getOrLoadDialect(name, dialect->getTypeID(), [dialect, ctor]() {
    ctor(dialect);
    return std::unique_ptr<DynamicDialect>(dialect);
  });
  // This is the same result as `getOrLoadDialect` (if it didn't failed),
  // since it has the same TypeID, and TypeIDs are unique.
  return dialect;
}

void MLIRContext::loadAllAvailableDialects() {
  for (StringRef name : getAvailableDialects())
    getOrLoadDialect(name);
}

llvm::hash_code MLIRContext::getRegistryHash() {
  llvm::hash_code hash(0);
  // Factor in number of loaded dialects, attributes, operations, types.
  hash = llvm::hash_combine(hash, impl->loadedDialects.size());
  hash = llvm::hash_combine(hash, impl->registeredAttributes.size());
  hash = llvm::hash_combine(hash, impl->registeredOperations.size());
  hash = llvm::hash_combine(hash, impl->registeredTypes.size());
  return hash;
}

bool MLIRContext::allowsUnregisteredDialects() {
  return impl->allowUnregisteredDialects;
}

void MLIRContext::allowUnregisteredDialects(bool allowing) {
  assert(impl->multiThreadedExecutionContext == 0 &&
         "changing MLIRContext `allow-unregistered-dialects` configuration "
         "while in a multi-threaded execution context");
  impl->allowUnregisteredDialects = allowing;
}

/// Return true if multi-threading is enabled by the context.
bool MLIRContext::isMultithreadingEnabled() {
  return impl->threadingIsEnabled && llvm::llvm_is_multithreaded();
}

/// Set the flag specifying if multi-threading is disabled by the context.
void MLIRContext::disableMultithreading(bool disable) {
  // This API can be overridden by the global debugging flag
  // --mlir-disable-threading
  if (isThreadingGloballyDisabled())
    return;
  assert(impl->multiThreadedExecutionContext == 0 &&
         "changing MLIRContext `disable-threading` configuration while "
         "in a multi-threaded execution context");

  impl->threadingIsEnabled = !disable;

  // Update the threading mode for each of the uniquers.
  impl->affineUniquer.disableMultithreading(disable);
  impl->attributeUniquer.disableMultithreading(disable);
  impl->typeUniquer.disableMultithreading(disable);

  // Destroy thread pool (stop all threads) if it is no longer needed, or create
  // a new one if multithreading was re-enabled.
  if (disable) {
    // If the thread pool is owned, explicitly set it to nullptr to avoid
    // keeping a dangling pointer around. If the thread pool is externally
    // owned, we don't do anything.
    if (impl->ownedThreadPool) {
      assert(impl->threadPool);
      impl->threadPool = nullptr;
      impl->ownedThreadPool.reset();
    }
  } else if (!impl->threadPool) {
    // The thread pool isn't externally provided.
    assert(!impl->ownedThreadPool);
    impl->ownedThreadPool = std::make_unique<llvm::ThreadPool>();
    impl->threadPool = impl->ownedThreadPool.get();
  }
}

void MLIRContext::setThreadPool(llvm::ThreadPool &pool) {
  assert(!isMultithreadingEnabled() &&
         "expected multi-threading to be disabled when setting a ThreadPool");
  impl->threadPool = &pool;
  impl->ownedThreadPool.reset();
  enableMultithreading();
}

unsigned MLIRContext::getNumThreads() {
  if (isMultithreadingEnabled()) {
    assert(impl->threadPool &&
           "multi-threading is enabled but threadpool not set");
    return impl->threadPool->getThreadCount();
  }
  // No multithreading or active thread pool. Return 1 thread.
  return 1;
}

llvm::ThreadPool &MLIRContext::getThreadPool() {
  assert(isMultithreadingEnabled() &&
         "expected multi-threading to be enabled within the context");
  assert(impl->threadPool &&
         "multi-threading is enabled but threadpool not set");
  return *impl->threadPool;
}

void MLIRContext::enterMultiThreadedExecution() {
#ifndef NDEBUG
  ++impl->multiThreadedExecutionContext;
#endif
}
void MLIRContext::exitMultiThreadedExecution() {
#ifndef NDEBUG
  --impl->multiThreadedExecutionContext;
#endif
}

/// Return true if we should attach the operation to diagnostics emitted via
/// Operation::emit.
bool MLIRContext::shouldPrintOpOnDiagnostic() {
  return impl->printOpOnDiagnostic;
}

/// Set the flag specifying if we should attach the operation to diagnostics
/// emitted via Operation::emit.
void MLIRContext::printOpOnDiagnostic(bool enable) {
  assert(impl->multiThreadedExecutionContext == 0 &&
         "changing MLIRContext `print-op-on-diagnostic` configuration while in "
         "a multi-threaded execution context");
  impl->printOpOnDiagnostic = enable;
}

/// Return true if we should attach the current stacktrace to diagnostics when
/// emitted.
bool MLIRContext::shouldPrintStackTraceOnDiagnostic() {
  return impl->printStackTraceOnDiagnostic;
}

/// Set the flag specifying if we should attach the current stacktrace when
/// emitting diagnostics.
void MLIRContext::printStackTraceOnDiagnostic(bool enable) {
  assert(impl->multiThreadedExecutionContext == 0 &&
         "changing MLIRContext `print-stacktrace-on-diagnostic` configuration "
         "while in a multi-threaded execution context");
  impl->printStackTraceOnDiagnostic = enable;
}

/// Return information about all registered operations.
ArrayRef<RegisteredOperationName> MLIRContext::getRegisteredOperations() {
  return impl->sortedRegisteredOperations;
}

bool MLIRContext::isOperationRegistered(StringRef name) {
  return RegisteredOperationName::lookup(name, this).has_value();
}

void Dialect::addType(TypeID typeID, AbstractType &&typeInfo) {
  auto &impl = context->getImpl();
  assert(impl.multiThreadedExecutionContext == 0 &&
         "Registering a new type kind while in a multi-threaded execution "
         "context");
  auto *newInfo =
      new (impl.abstractDialectSymbolAllocator.Allocate<AbstractType>())
          AbstractType(std::move(typeInfo));
  if (!impl.registeredTypes.insert({typeID, newInfo}).second)
    llvm::report_fatal_error("Dialect Type already registered.");
}

void Dialect::addAttribute(TypeID typeID, AbstractAttribute &&attrInfo) {
  auto &impl = context->getImpl();
  assert(impl.multiThreadedExecutionContext == 0 &&
         "Registering a new attribute kind while in a multi-threaded execution "
         "context");
  auto *newInfo =
      new (impl.abstractDialectSymbolAllocator.Allocate<AbstractAttribute>())
          AbstractAttribute(std::move(attrInfo));
  if (!impl.registeredAttributes.insert({typeID, newInfo}).second)
    llvm::report_fatal_error("Dialect Attribute already registered.");
}

//===----------------------------------------------------------------------===//
// AbstractAttribute
//===----------------------------------------------------------------------===//

/// Get the dialect that registered the attribute with the provided typeid.
const AbstractAttribute &AbstractAttribute::lookup(TypeID typeID,
                                                   MLIRContext *context) {
  const AbstractAttribute *abstract = lookupMutable(typeID, context);
  if (!abstract)
    llvm::report_fatal_error("Trying to create an Attribute that was not "
                             "registered in this MLIRContext.");
  return *abstract;
}

AbstractAttribute *AbstractAttribute::lookupMutable(TypeID typeID,
                                                    MLIRContext *context) {
  auto &impl = context->getImpl();
  auto it = impl.registeredAttributes.find(typeID);
  if (it == impl.registeredAttributes.end())
    return nullptr;
  return it->second;
}

//===----------------------------------------------------------------------===//
// OperationName
//===----------------------------------------------------------------------===//

OperationName::Impl::Impl(StringRef name, Dialect *dialect, TypeID typeID,
                          detail::InterfaceMap interfaceMap)
    : Impl(StringAttr::get(dialect->getContext(), name), dialect, typeID,
           std::move(interfaceMap)) {}

OperationName::OperationName(StringRef name, MLIRContext *context) {
  MLIRContextImpl &ctxImpl = context->getImpl();

  // Check for an existing name in read-only mode.
  bool isMultithreadingEnabled = context->isMultithreadingEnabled();
  if (isMultithreadingEnabled) {
    // Check the registered info map first. In the overwhelmingly common case,
    // the entry will be in here and it also removes the need to acquire any
    // locks.
    auto registeredIt = ctxImpl.registeredOperations.find(name);
    if (LLVM_LIKELY(registeredIt != ctxImpl.registeredOperations.end())) {
      impl = registeredIt->second.impl;
      return;
    }

    llvm::sys::SmartScopedReader<true> contextLock(ctxImpl.operationInfoMutex);
    auto it = ctxImpl.operations.find(name);
    if (it != ctxImpl.operations.end()) {
      impl = it->second.get();
      return;
    }
  }

  // Acquire a writer-lock so that we can safely create the new instance.
  ScopedWriterLock lock(ctxImpl.operationInfoMutex, isMultithreadingEnabled);

  auto it = ctxImpl.operations.insert({name, nullptr});
  if (it.second) {
    auto nameAttr = StringAttr::get(context, name);
    it.first->second = std::make_unique<UnregisteredOpModel>(
        nameAttr, nameAttr.getReferencedDialect(), TypeID::get<void>(),
        detail::InterfaceMap());
  }
  impl = it.first->second.get();
}

StringRef OperationName::getDialectNamespace() const {
  if (Dialect *dialect = getDialect())
    return dialect->getNamespace();
  return getStringRef().split('.').first;
}

LogicalResult
OperationName::UnregisteredOpModel::foldHook(Operation *, ArrayRef<Attribute>,
                                             SmallVectorImpl<OpFoldResult> &) {
  return failure();
}
void OperationName::UnregisteredOpModel::getCanonicalizationPatterns(
    RewritePatternSet &, MLIRContext *) {}
bool OperationName::UnregisteredOpModel::hasTrait(TypeID) { return false; }

OperationName::ParseAssemblyFn
OperationName::UnregisteredOpModel::getParseAssemblyFn() {
  llvm::report_fatal_error("getParseAssemblyFn hook called on unregistered op");
}
void OperationName::UnregisteredOpModel::populateDefaultAttrs(
    const OperationName &, NamedAttrList &) {}
void OperationName::UnregisteredOpModel::printAssembly(
    Operation *op, OpAsmPrinter &p, StringRef defaultDialect) {
  p.printGenericOp(op);
}
LogicalResult
OperationName::UnregisteredOpModel::verifyInvariants(Operation *) {
  return success();
}
LogicalResult
OperationName::UnregisteredOpModel::verifyRegionInvariants(Operation *) {
  return success();
}

//===----------------------------------------------------------------------===//
// RegisteredOperationName
//===----------------------------------------------------------------------===//

std::optional<RegisteredOperationName>
RegisteredOperationName::lookup(StringRef name, MLIRContext *ctx) {
  auto &impl = ctx->getImpl();
  auto it = impl.registeredOperations.find(name);
  if (it != impl.registeredOperations.end())
    return it->getValue();
  return std::nullopt;
}

void RegisteredOperationName::insert(
    std::unique_ptr<RegisteredOperationName::Impl> ownedImpl,
    ArrayRef<StringRef> attrNames) {
  RegisteredOperationName::Impl *impl = ownedImpl.get();
  MLIRContext *ctx = impl->getDialect()->getContext();
  auto &ctxImpl = ctx->getImpl();
  assert(ctxImpl.multiThreadedExecutionContext == 0 &&
         "registering a new operation kind while in a multi-threaded execution "
         "context");

  // Register the attribute names of this operation.
  MutableArrayRef<StringAttr> cachedAttrNames;
  if (!attrNames.empty()) {
    cachedAttrNames = MutableArrayRef<StringAttr>(
        ctxImpl.abstractDialectSymbolAllocator.Allocate<StringAttr>(
            attrNames.size()),
        attrNames.size());
    for (unsigned i : llvm::seq<unsigned>(0, attrNames.size()))
      new (&cachedAttrNames[i]) StringAttr(StringAttr::get(ctx, attrNames[i]));
    impl->attributeNames = cachedAttrNames;
  }
  StringRef name = impl->getName().strref();
  // Insert the operation info if it doesn't exist yet.
  auto it = ctxImpl.operations.insert({name, nullptr});
  it.first->second = std::move(ownedImpl);

  // Update the registered info for this operation.
  auto emplaced = ctxImpl.registeredOperations.try_emplace(
      name, RegisteredOperationName(impl));
  assert(emplaced.second && "operation name registration must be successful");

  // Add emplaced operation name to the sorted operations container.
  RegisteredOperationName &value = emplaced.first->getValue();
  ctxImpl.sortedRegisteredOperations.insert(
      llvm::upper_bound(ctxImpl.sortedRegisteredOperations, value,
                        [](auto &lhs, auto &rhs) {
                          return lhs.getIdentifier().compare(
                              rhs.getIdentifier());
                        }),
      value);
}

//===----------------------------------------------------------------------===//
// AbstractType
//===----------------------------------------------------------------------===//

const AbstractType &AbstractType::lookup(TypeID typeID, MLIRContext *context) {
  const AbstractType *type = lookupMutable(typeID, context);
  if (!type)
    llvm::report_fatal_error(
        "Trying to create a Type that was not registered in this MLIRContext.");
  return *type;
}

AbstractType *AbstractType::lookupMutable(TypeID typeID, MLIRContext *context) {
  auto &impl = context->getImpl();
  auto it = impl.registeredTypes.find(typeID);
  if (it == impl.registeredTypes.end())
    return nullptr;
  return it->second;
}

//===----------------------------------------------------------------------===//
// Type uniquing
//===----------------------------------------------------------------------===//

/// Returns the storage uniquer used for constructing type storage instances.
/// This should not be used directly.
StorageUniquer &MLIRContext::getTypeUniquer() { return getImpl().typeUniquer; }

Float8E5M2Type Float8E5M2Type::get(MLIRContext *context) {
  return context->getImpl().f8E5M2Ty;
}
Float8E4M3FNType Float8E4M3FNType::get(MLIRContext *context) {
  return context->getImpl().f8E4M3FNTy;
}
Float8E5M2FNUZType Float8E5M2FNUZType::get(MLIRContext *context) {
  return context->getImpl().f8E5M2FNUZTy;
}
Float8E4M3FNUZType Float8E4M3FNUZType::get(MLIRContext *context) {
  return context->getImpl().f8E4M3FNUZTy;
}
Float8E4M3B11FNUZType Float8E4M3B11FNUZType::get(MLIRContext *context) {
  return context->getImpl().f8E4M3B11FNUZTy;
}
BFloat16Type BFloat16Type::get(MLIRContext *context) {
  return context->getImpl().bf16Ty;
}
Float16Type Float16Type::get(MLIRContext *context) {
  return context->getImpl().f16Ty;
}
Float32Type Float32Type::get(MLIRContext *context) {
  return context->getImpl().f32Ty;
}
Float64Type Float64Type::get(MLIRContext *context) {
  return context->getImpl().f64Ty;
}
Float80Type Float80Type::get(MLIRContext *context) {
  return context->getImpl().f80Ty;
}
Float128Type Float128Type::get(MLIRContext *context) {
  return context->getImpl().f128Ty;
}

/// Get an instance of the IndexType.
IndexType IndexType::get(MLIRContext *context) {
  return context->getImpl().indexTy;
}

/// Return an existing integer type instance if one is cached within the
/// context.
static IntegerType
getCachedIntegerType(unsigned width,
                     IntegerType::SignednessSemantics signedness,
                     MLIRContext *context) {
  if (signedness != IntegerType::Signless)
    return IntegerType();

  switch (width) {
  case 1:
    return context->getImpl().int1Ty;
  case 8:
    return context->getImpl().int8Ty;
  case 16:
    return context->getImpl().int16Ty;
  case 32:
    return context->getImpl().int32Ty;
  case 64:
    return context->getImpl().int64Ty;
  case 128:
    return context->getImpl().int128Ty;
  default:
    return IntegerType();
  }
}

IntegerType IntegerType::get(MLIRContext *context, unsigned width,
                             IntegerType::SignednessSemantics signedness) {
  if (auto cached = getCachedIntegerType(width, signedness, context))
    return cached;
  return Base::get(context, width, signedness);
}

IntegerType
IntegerType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                        MLIRContext *context, unsigned width,
                        SignednessSemantics signedness) {
  if (auto cached = getCachedIntegerType(width, signedness, context))
    return cached;
  return Base::getChecked(emitError, context, width, signedness);
}

/// Get an instance of the NoneType.
NoneType NoneType::get(MLIRContext *context) {
  if (NoneType cachedInst = context->getImpl().noneType)
    return cachedInst;
  // Note: May happen when initializing the singleton attributes of the builtin
  // dialect.
  return Base::get(context);
}

//===----------------------------------------------------------------------===//
// Attribute uniquing
//===----------------------------------------------------------------------===//

/// Returns the storage uniquer used for constructing attribute storage
/// instances. This should not be used directly.
StorageUniquer &MLIRContext::getAttributeUniquer() {
  return getImpl().attributeUniquer;
}

/// Initialize the given attribute storage instance.
void AttributeUniquer::initializeAttributeStorage(AttributeStorage *storage,
                                                  MLIRContext *ctx,
                                                  TypeID attrID) {
  storage->initializeAbstractAttribute(AbstractAttribute::lookup(attrID, ctx));
}

BoolAttr BoolAttr::get(MLIRContext *context, bool value) {
  return value ? context->getImpl().trueAttr : context->getImpl().falseAttr;
}

UnitAttr UnitAttr::get(MLIRContext *context) {
  return context->getImpl().unitAttr;
}

UnknownLoc UnknownLoc::get(MLIRContext *context) {
  return context->getImpl().unknownLocAttr;
}

/// Return empty dictionary.
DictionaryAttr DictionaryAttr::getEmpty(MLIRContext *context) {
  return context->getImpl().emptyDictionaryAttr;
}

void StringAttrStorage::initialize(MLIRContext *context) {
  // Check for a dialect namespace prefix, if there isn't one we don't need to
  // do any additional initialization.
  auto dialectNamePair = value.split('.');
  if (dialectNamePair.first.empty() || dialectNamePair.second.empty())
    return;

  // If one exists, we check to see if this dialect is loaded. If it is, we set
  // the dialect now, if it isn't we record this storage for initialization
  // later if the dialect ever gets loaded.
  if ((referencedDialect = context->getLoadedDialect(dialectNamePair.first)))
    return;

  MLIRContextImpl &impl = context->getImpl();
  llvm::sys::SmartScopedLock<true> lock(impl.dialectRefStrAttrMutex);
  impl.dialectReferencingStrAttrs[dialectNamePair.first].push_back(this);
}

/// Return an empty string.
StringAttr StringAttr::get(MLIRContext *context) {
  return context->getImpl().emptyStringAttr;
}

//===----------------------------------------------------------------------===//
// AffineMap uniquing
//===----------------------------------------------------------------------===//

StorageUniquer &MLIRContext::getAffineUniquer() {
  return getImpl().affineUniquer;
}

AffineMap AffineMap::getImpl(unsigned dimCount, unsigned symbolCount,
                             ArrayRef<AffineExpr> results,
                             MLIRContext *context) {
  auto &impl = context->getImpl();
  auto *storage = impl.affineUniquer.get<AffineMapStorage>(
      [&](AffineMapStorage *storage) { storage->context = context; }, dimCount,
      symbolCount, results);
  return AffineMap(storage);
}

/// Check whether the arguments passed to the AffineMap::get() are consistent.
/// This method checks whether the highest index of dimensional identifier
/// present in result expressions is less than `dimCount` and the highest index
/// of symbolic identifier present in result expressions is less than
/// `symbolCount`.
LLVM_ATTRIBUTE_UNUSED static bool
willBeValidAffineMap(unsigned dimCount, unsigned symbolCount,
                     ArrayRef<AffineExpr> results) {
  int64_t maxDimPosition = -1;
  int64_t maxSymbolPosition = -1;
  getMaxDimAndSymbol(ArrayRef<ArrayRef<AffineExpr>>(results), maxDimPosition,
                     maxSymbolPosition);
  if ((maxDimPosition >= dimCount) || (maxSymbolPosition >= symbolCount)) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "maximum dimensional identifier position in result expression must "
           "be less than `dimCount` and maximum symbolic identifier position "
           "in result expression must be less than `symbolCount`\n");
    return false;
  }
  return true;
}

AffineMap AffineMap::get(MLIRContext *context) {
  return getImpl(/*dimCount=*/0, /*symbolCount=*/0, /*results=*/{}, context);
}

AffineMap AffineMap::get(unsigned dimCount, unsigned symbolCount,
                         MLIRContext *context) {
  return getImpl(dimCount, symbolCount, /*results=*/{}, context);
}

AffineMap AffineMap::get(unsigned dimCount, unsigned symbolCount,
                         AffineExpr result) {
  assert(willBeValidAffineMap(dimCount, symbolCount, {result}));
  return getImpl(dimCount, symbolCount, {result}, result.getContext());
}

AffineMap AffineMap::get(unsigned dimCount, unsigned symbolCount,
                         ArrayRef<AffineExpr> results, MLIRContext *context) {
  assert(willBeValidAffineMap(dimCount, symbolCount, results));
  return getImpl(dimCount, symbolCount, results, context);
}

//===----------------------------------------------------------------------===//
// Integer Sets: these are allocated into the bump pointer, and are immutable.
// Unlike AffineMap's, these are uniqued only if they are small.
//===----------------------------------------------------------------------===//

IntegerSet IntegerSet::get(unsigned dimCount, unsigned symbolCount,
                           ArrayRef<AffineExpr> constraints,
                           ArrayRef<bool> eqFlags) {
  // The number of constraints can't be zero.
  assert(!constraints.empty());
  assert(constraints.size() == eqFlags.size());

  auto &impl = constraints[0].getContext()->getImpl();
  auto *storage = impl.affineUniquer.get<IntegerSetStorage>(
      [](IntegerSetStorage *) {}, dimCount, symbolCount, constraints, eqFlags);
  return IntegerSet(storage);
}

//===----------------------------------------------------------------------===//
// StorageUniquerSupport
//===----------------------------------------------------------------------===//

/// Utility method to generate a callback that can be used to generate a
/// diagnostic when checking the construction invariants of a storage object.
/// This is defined out-of-line to avoid the need to include Location.h.
llvm::unique_function<InFlightDiagnostic()>
mlir::detail::getDefaultDiagnosticEmitFn(MLIRContext *ctx) {
  return [ctx] { return emitError(UnknownLoc::get(ctx)); };
}
llvm::unique_function<InFlightDiagnostic()>
mlir::detail::getDefaultDiagnosticEmitFn(const Location &loc) {
  return [=] { return emitError(loc); };
}
