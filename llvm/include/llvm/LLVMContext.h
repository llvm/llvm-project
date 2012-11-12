//===-- llvm/LLVMContext.h - Class for managing "global" state --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares LLVMContext, a container of "global" state in LLVM, such
// as the global type and constant uniquing tables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LLVMCONTEXT_H
#define LLVM_LLVMCONTEXT_H

#include "llvm/Support/Compiler.h"

namespace llvm {

class LLVMContextImpl;
class StringRef;
class Twine;
class Instruction;
class Module;
class SMDiagnostic;
template <typename T> class SmallVectorImpl;

/// This is an important class for using LLVM in a threaded context.  It
/// (opaquely) owns and manages the core "global" data of LLVM's core 
/// infrastructure, including the type and constant uniquing tables.
/// LLVMContext itself provides no locking guarantees, so you should be careful
/// to have one context per thread.
class LLVMContext {
public:
  LLVMContextImpl *const pImpl;
  LLVMContext();
  ~LLVMContext();
  
  // Pinned metadata names, which always have the same value.  This is a
  // compile-time performance optimization, not a correctness optimization.
  enum {
    MD_dbg = 0,  // "dbg"
    MD_tbaa = 1, // "tbaa"
    MD_prof = 2,  // "prof"
    MD_fpmath = 3,  // "fpmath"
    MD_range = 4, // "range"
    MD_tbaa_struct = 5 // "tbaa.struct"
  };
  
  /// getMDKindID - Return a unique non-zero ID for the specified metadata kind.
  /// This ID is uniqued across modules in the current LLVMContext.
  unsigned getMDKindID(StringRef Name) const;
  
  /// getMDKindNames - Populate client supplied SmallVector with the name for
  /// custom metadata IDs registered in this LLVMContext.
  void getMDKindNames(SmallVectorImpl<StringRef> &Result) const;
  
  
  typedef void (*InlineAsmDiagHandlerTy)(const SMDiagnostic&, void *Context,
                                         unsigned LocCookie);
  
  /// setInlineAsmDiagnosticHandler - This method sets a handler that is invoked
  /// when problems with inline asm are detected by the backend.  The first
  /// argument is a function pointer and the second is a context pointer that
  /// gets passed into the DiagHandler.
  ///
  /// LLVMContext doesn't take ownership or interpret either of these
  /// pointers.
  void setInlineAsmDiagnosticHandler(InlineAsmDiagHandlerTy DiagHandler,
                                     void *DiagContext = 0);

  /// getInlineAsmDiagnosticHandler - Return the diagnostic handler set by
  /// setInlineAsmDiagnosticHandler.
  InlineAsmDiagHandlerTy getInlineAsmDiagnosticHandler() const;

  /// getInlineAsmDiagnosticContext - Return the diagnostic context set by
  /// setInlineAsmDiagnosticHandler.
  void *getInlineAsmDiagnosticContext() const;
  
  
  /// emitError - Emit an error message to the currently installed error handler
  /// with optional location information.  This function returns, so code should
  /// be prepared to drop the erroneous construct on the floor and "not crash".
  /// The generated code need not be correct.  The error message will be
  /// implicitly prefixed with "error: " and should not end with a ".".
  void emitError(unsigned LocCookie, const Twine &ErrorStr);
  void emitError(const Instruction *I, const Twine &ErrorStr);
  void emitError(const Twine &ErrorStr);

private:
  LLVMContext(LLVMContext&) LLVM_DELETED_FUNCTION;
  void operator=(LLVMContext&) LLVM_DELETED_FUNCTION;

  /// addModule - Register a module as being instantiated in this context.  If
  /// the context is deleted, the module will be deleted as well.
  void addModule(Module*);
  
  /// removeModule - Unregister a module from this context.
  void removeModule(Module*);
  
  // Module needs access to the add/removeModule methods.
  friend class Module;
};

/// getGlobalContext - Returns a global context.  This is for LLVM clients that
/// only care about operating on a single thread.
extern LLVMContext &getGlobalContext();

}

#endif
