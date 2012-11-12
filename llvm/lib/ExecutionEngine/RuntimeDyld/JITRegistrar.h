//===-- JITRegistrar.h - Registers objects with a debugger ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTION_ENGINE_JIT_REGISTRAR_H
#define LLVM_EXECUTION_ENGINE_JIT_REGISTRAR_H

#include "llvm/ExecutionEngine/ObjectBuffer.h"

namespace llvm {

/// Global access point for the JIT debugging interface.
class JITRegistrar {
public:
  /// Instantiates the JIT service.
  JITRegistrar() {}

  /// Unregisters each object that was previously registered and releases all
  /// internal resources.
  virtual ~JITRegistrar() {}

  /// Creates an entry in the JIT registry for the buffer @p Object,
  /// which must contain an object file in executable memory with any
  /// debug information for the debugger.
  virtual void registerObject(const ObjectBuffer &Object) = 0;

  /// Removes the internal registration of @p Object, and
  /// frees associated resources.
  /// Returns true if @p Object was previously registered.
  virtual bool deregisterObject(const ObjectBuffer &Object) = 0;

  /// Returns a reference to a GDB JIT registrar singleton
  static JITRegistrar& getGDBRegistrar();
};

} // end namespace llvm

#endif // LLVM_EXECUTION_ENGINE_JIT_REGISTRAR_H
