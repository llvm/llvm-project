//=- MachineFunctionInitializer.h - machine function initializer --*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares an interface that allows custom machine function
// initialization.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEFUNCTIONINITIALIZER_H
#define LLVM_CODEGEN_MACHINEFUNCTIONINITIALIZER_H

namespace llvm {

class MachineFunction;

/// This interface provides a way to initialize machine functions after they are
/// created by the machine function analysis pass.
class MachineFunctionInitializer {
  virtual void anchor();

public:
  virtual ~MachineFunctionInitializer() = default;

  /// Initialize the machine function.
  ///
  /// Return true if error occurred.
  virtual bool initializeMachineFunction(MachineFunction &MF) = 0;
};

} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINEFUNCTIONINITIALIZER_H
