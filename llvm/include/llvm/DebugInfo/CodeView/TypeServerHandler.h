//===- TypeServerHandler.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_TYPESERVERHANDLER_H
#define LLVM_DEBUGINFO_CODEVIEW_TYPESERVERHANDLER_H

#include "llvm/Support/Error.h"

namespace llvm {
namespace codeview {

class TypeServer2Record;
class TypeVisitorCallbacks;

class TypeServerHandler {
public:
  virtual ~TypeServerHandler() = default;

  /// Handle a TypeServer record.  If the implementation returns true
  /// the record will not be processed by the top-level visitor.  If
  /// it returns false, it will be processed.  If it returns an Error,
  /// then the top-level visitor will fail.
  virtual Expected<bool> handle(TypeServer2Record &TS,
                                TypeVisitorCallbacks &Callbacks) {
    return false;
  }
};

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_TYPESERVERHANDLER_H
