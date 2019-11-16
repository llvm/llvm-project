//===- llvm/LLVMContext.h - Class for managing "global" state ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares LLVMContext, a container of "global" state in LLVM, such
// as the global type and constant uniquing tables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_LLVMCONTEXT_H
#define LLVM_IR_LLVMCONTEXT_H

#include "llvm-c/Types.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/Support/CBindingWrapping.h"
#include <cstdint>
#include <memory>
#include <string>

namespace llvm {

class DiagnosticInfo;
enum DiagnosticSeverity : char;
class Function;
class Instruction;
class LLVMContextImpl;
class Module;
class OptPassGate;
template <typename T> class SmallVectorImpl;
class SMDiagnostic;
class StringRef;
class Twine;
class RemarkStreamer;
class raw_ostream;

namespace SyncScope {

typedef uint8_t ID;

/// Known synchronization scope IDs, which always have the same value.  All
/// synchronization scope IDs that LLVM has special knowledge of are listed
/// here.  Additionally, this scheme allows LLVM to efficiently check for
/// specific synchronization scope ID without comparing strings.
enum {
  /// Synchronized with respect to signal handlers executing in the same thread.
  SingleThread = 0,

  /// Synchronized with respect to all concurrently executing threads.
  System = 1
};

} // end namespace SyncScope

/// This is an important class for using LLVM in a threaded context.  It
/// (opaquely) owns and manages the core "global" data of LLVM's core
/// infrastructure, including the type and constant uniquing tables.
/// LLVMContext itself provides no locking guarantees, so you should be careful
/// to have one context per thread.
class LLVMContext {
public:
  LLVMContextImpl *const pImpl;
  LLVMContext();
  LLVMContext(LLVMContext &) = delete;
  LLVMContext &operator=(const LLVMContext &) = delete;
  ~LLVMContext();

  // Pinned metadata names, which always have the same value.  This is a
  // compile-time performance optimization, not a correctness optimization.
  enum : unsigned {
#define LLVM_FIXED_MD_KIND(EnumID, Name, Value) EnumID = Value,
#include "llvm/IR/FixedMetadataKinds.def"
#undef LLVM_FIXED_MD_KIND
  };

  /// Known operand bundle tag IDs, which always have the same value.  All
  /// operand bundle tags that LLVM has special knowledge of are listed here.
  /// Additionally, this scheme allows LLVM to efficiently check for specific
  /// operand bundle tags without comparing strings.
  enum : unsigned {
    OB_deopt = 0,         // "deopt"
    OB_funclet = 1,       // "funclet"
    OB_gc_transition = 2, // "gc-transition"
    OB_cfguardtarget = 3, // "cfguardtarget"
  };

  /// getMDKindID - Return a unique non-zero ID for the specified metadata kind.
  /// This ID is uniqued across modules in the current LLVMContext.
  unsigned getMDKindID(StringRef Name) const;

  /// getMDKindNames - Populate client supplied SmallVector with the name for
  /// custom metadata IDs registered in this LLVMContext.
  void getMDKindNames(SmallVectorImpl<StringRef> &Result) const;

  /// getOperandBundleTags - Populate client supplied SmallVector with the
  /// bundle tags registered in this LLVMContext.  The bundle tags are ordered
  /// by increasing bundle IDs.
  /// \see LLVMContext::getOperandBundleTagID
  void getOperandBundleTags(SmallVectorImpl<StringRef> &Result) const;

  /// getOperandBundleTagID - Maps a bundle tag to an integer ID.  Every bundle
  /// tag registered with an LLVMContext has an unique ID.
  uint32_t getOperandBundleTagID(StringRef Tag) const;

  /// getOrInsertSyncScopeID - Maps synchronization scope name to
  /// synchronization scope ID.  Every synchronization scope registered with
  /// LLVMContext has unique ID except pre-defined ones.
  SyncScope::ID getOrInsertSyncScopeID(StringRef SSN);

  /// getSyncScopeNames - Populates client supplied SmallVector with
  /// synchronization scope names registered with LLVMContext.  Synchronization
  /// scope names are ordered by increasing synchronization scope IDs.
  void getSyncScopeNames(SmallVectorImpl<StringRef> &SSNs) const;

  /// Define the GC for a function
  void setGC(const Function &Fn, std::string GCName);

  /// Return the GC for a function
  const std::string &getGC(const Function &Fn);

  /// Remove the GC for a function
  void deleteGC(const Function &Fn);

  /// Return true if the Context runtime configuration is set to discard all
  /// value names. When true, only GlobalValue names will be available in the
  /// IR.
  bool shouldDiscardValueNames() const;

  /// Set the Context runtime configuration to discard all value name (but
  /// GlobalValue). Clients can use this flag to save memory and runtime,
  /// especially in release mode.
  void setDiscardValueNames(bool Discard);

  /// Whether there is a string map for uniquing debug info
  /// identifiers across the context.  Off by default.
  bool isODRUniquingDebugTypes() const;
  void enableDebugTypeODRUniquing();
  void disableDebugTypeODRUniquing();

  using InlineAsmDiagHandlerTy = void (*)(const SMDiagnostic&, void *Context,
                                          unsigned LocCookie);

  /// Defines the type of a yield callback.
  /// \see LLVMContext::setYieldCallback.
  using YieldCallbackTy = void (*)(LLVMContext *Context, void *OpaqueHandle);

  /// setInlineAsmDiagnosticHandler - This method sets a handler that is invoked
  /// when problems with inline asm are detected by the backend.  The first
  /// argument is a function pointer and the second is a context pointer that
  /// gets passed into the DiagHandler.
  ///
  /// LLVMContext doesn't take ownership or interpret either of these
  /// pointers.
  void setInlineAsmDiagnosticHandler(InlineAsmDiagHandlerTy DiagHandler,
                                     void *DiagContext = nullptr);

  /// getInlineAsmDiagnosticHandler - Return the diagnostic handler set by
  /// setInlineAsmDiagnosticHandler.
  InlineAsmDiagHandlerTy getInlineAsmDiagnosticHandler() const;

  /// getInlineAsmDiagnosticContext - Return the diagnostic context set by
  /// setInlineAsmDiagnosticHandler.
  void *getInlineAsmDiagnosticContext() const;

  /// setDiagnosticHandlerCallBack - This method sets a handler call back
  /// that is invoked when the backend needs to report anything to the user.
  /// The first argument is a function pointer and the second is a context pointer
  /// that gets passed into the DiagHandler.  The third argument should be set to
  /// true if the handler only expects enabled diagnostics.
  ///
  /// LLVMContext doesn't take ownership or interpret either of these
  /// pointers.
  void setDiagnosticHandlerCallBack(
      DiagnosticHandler::DiagnosticHandlerTy DiagHandler,
      void *DiagContext = nullptr, bool RespectFilters = false);

  /// setDiagnosticHandler - This method sets unique_ptr to object of DiagnosticHandler
  /// to provide custom diagnostic handling. The first argument is unique_ptr of object
  /// of type DiagnosticHandler or a derived of that.   The third argument should be
  /// set to true if the handler only expects enabled diagnostics.
  ///
  /// Ownership of this pointer is moved to LLVMContextImpl.
  void setDiagnosticHandler(std::unique_ptr<DiagnosticHandler> &&DH,
                            bool RespectFilters = false);

  /// getDiagnosticHandlerCallBack - Return the diagnostic handler call back set by
  /// setDiagnosticHandlerCallBack.
  DiagnosticHandler::DiagnosticHandlerTy getDiagnosticHandlerCallBack() const;

  /// getDiagnosticContext - Return the diagnostic context set by
  /// setDiagnosticContext.
  void *getDiagnosticContext() const;

  /// getDiagHandlerPtr - Returns const raw pointer of DiagnosticHandler set by
  /// setDiagnosticHandler.
  const DiagnosticHandler *getDiagHandlerPtr() const;

  /// getDiagnosticHandler - transfers owenership of DiagnosticHandler unique_ptr
  /// to caller.
  std::unique_ptr<DiagnosticHandler> getDiagnosticHandler();

  /// Return if a code hotness metric should be included in optimization
  /// diagnostics.
  bool getDiagnosticsHotnessRequested() const;
  /// Set if a code hotness metric should be included in optimization
  /// diagnostics.
  void setDiagnosticsHotnessRequested(bool Requested);

  /// Return the minimum hotness value a diagnostic would need in order
  /// to be included in optimization diagnostics. If there is no minimum, this
  /// returns None.
  uint64_t getDiagnosticsHotnessThreshold() const;

  /// Set the minimum hotness value a diagnostic needs in order to be
  /// included in optimization diagnostics.
  void setDiagnosticsHotnessThreshold(uint64_t Threshold);

  /// Return the streamer used by the backend to save remark diagnostics. If it
  /// does not exist, diagnostics are not saved in a file but only emitted via
  /// the diagnostic handler.
  RemarkStreamer *getRemarkStreamer();
  const RemarkStreamer *getRemarkStreamer() const;

  /// Set the diagnostics output used for optimization diagnostics.
  /// This filename may be embedded in a section for tools to find the
  /// diagnostics whenever they're needed.
  ///
  /// If a remark streamer is already set, it will be replaced with
  /// \p RemarkStreamer.
  ///
  /// By default, diagnostics are not saved in a file but only emitted via the
  /// diagnostic handler.  Even if an output file is set, the handler is invoked
  /// for each diagnostic message.
  void setRemarkStreamer(std::unique_ptr<RemarkStreamer> RemarkStreamer);

  /// Get the prefix that should be printed in front of a diagnostic of
  ///        the given \p Severity
  static const char *getDiagnosticMessagePrefix(DiagnosticSeverity Severity);

  /// Report a message to the currently installed diagnostic handler.
  ///
  /// This function returns, in particular in the case of error reporting
  /// (DI.Severity == \a DS_Error), so the caller should leave the compilation
  /// process in a self-consistent state, even though the generated code
  /// need not be correct.
  ///
  /// The diagnostic message will be implicitly prefixed with a severity keyword
  /// according to \p DI.getSeverity(), i.e., "error: " for \a DS_Error,
  /// "warning: " for \a DS_Warning, and "note: " for \a DS_Note.
  void diagnose(const DiagnosticInfo &DI);

  /// Registers a yield callback with the given context.
  ///
  /// The yield callback function may be called by LLVM to transfer control back
  /// to the client that invoked the LLVM compilation. This can be used to yield
  /// control of the thread, or perform periodic work needed by the client.
  /// There is no guaranteed frequency at which callbacks must occur; in fact,
  /// the client is not guaranteed to ever receive this callback. It is at the
  /// sole discretion of LLVM to do so and only if it can guarantee that
  /// suspending the thread won't block any forward progress in other LLVM
  /// contexts in the same process.
  ///
  /// At a suspend point, the state of the current LLVM context is intentionally
  /// undefined. No assumptions about it can or should be made. Only LLVM
  /// context API calls that explicitly state that they can be used during a
  /// yield callback are allowed to be used. Any other API calls into the
  /// context are not supported until the yield callback function returns
  /// control to LLVM. Other LLVM contexts are unaffected by this restriction.
  void setYieldCallback(YieldCallbackTy Callback, void *OpaqueHandle);

  /// Calls the yield callback (if applicable).
  ///
  /// This transfers control of the current thread back to the client, which may
  /// suspend the current thread. Only call this method when LLVM doesn't hold
  /// any global mutex or cannot block the execution in another LLVM context.
  void yield();

  /// emitError - Emit an error message to the currently installed error handler
  /// with optional location information.  This function returns, so code should
  /// be prepared to drop the erroneous construct on the floor and "not crash".
  /// The generated code need not be correct.  The error message will be
  /// implicitly prefixed with "error: " and should not end with a ".".
  void emitError(unsigned LocCookie, const Twine &ErrorStr);
  void emitError(const Instruction *I, const Twine &ErrorStr);
  void emitError(const Twine &ErrorStr);

  /// Access the object which can disable optional passes and individual
  /// optimizations at compile time.
  OptPassGate &getOptPassGate() const;

  /// Set the object which can disable optional passes and individual
  /// optimizations at compile time.
  ///
  /// The lifetime of the object must be guaranteed to extend as long as the
  /// LLVMContext is used by compilation.
  void setOptPassGate(OptPassGate&);

private:
  // Module needs access to the add/removeModule methods.
  friend class Module;

  /// addModule - Register a module as being instantiated in this context.  If
  /// the context is deleted, the module will be deleted as well.
  void addModule(Module*);

  /// removeModule - Unregister a module from this context.
  void removeModule(Module*);
};

// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(LLVMContext, LLVMContextRef)

/* Specialized opaque context conversions.
 */
inline LLVMContext **unwrap(LLVMContextRef* Tys) {
  return reinterpret_cast<LLVMContext**>(Tys);
}

inline LLVMContextRef *wrap(const LLVMContext **Tys) {
  return reinterpret_cast<LLVMContextRef*>(const_cast<LLVMContext**>(Tys));
}

} // end namespace llvm

#endif // LLVM_IR_LLVMCONTEXT_H
