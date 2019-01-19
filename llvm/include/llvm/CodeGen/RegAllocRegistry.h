//===- llvm/CodeGen/RegAllocRegistry.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation for register allocator function
// pass registry (RegisterRegAlloc).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGALLOCREGISTRY_H
#define LLVM_CODEGEN_REGALLOCREGISTRY_H

#include "llvm/CodeGen/MachinePassRegistry.h"

namespace llvm {

class FunctionPass;

//===----------------------------------------------------------------------===//
///
/// RegisterRegAlloc class - Track the registration of register allocators.
///
//===----------------------------------------------------------------------===//
class RegisterRegAlloc : public MachinePassRegistryNode<FunctionPass *(*)()> {
public:
  using FunctionPassCtor = FunctionPass *(*)();

  static MachinePassRegistry<FunctionPassCtor> Registry;

  RegisterRegAlloc(const char *N, const char *D, FunctionPassCtor C)
      : MachinePassRegistryNode(N, D, C) {
    Registry.Add(this);
  }

  ~RegisterRegAlloc() { Registry.Remove(this); }

  // Accessors.
  RegisterRegAlloc *getNext() const {
    return (RegisterRegAlloc *)MachinePassRegistryNode::getNext();
  }

  static RegisterRegAlloc *getList() {
    return (RegisterRegAlloc *)Registry.getList();
  }

  static FunctionPassCtor getDefault() { return Registry.getDefault(); }

  static void setDefault(FunctionPassCtor C) { Registry.setDefault(C); }

  static void setListener(MachinePassRegistryListener<FunctionPassCtor> *L) {
    Registry.setListener(L);
  }
};

} // end namespace llvm

#endif // LLVM_CODEGEN_REGALLOCREGISTRY_H
