//===-- ModuleProvider.cpp - Base implementation for module providers -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Minimal implementation of the abstract interface for providing a module.
//
//===----------------------------------------------------------------------===//

#include "llvm/ModuleProvider.h"
#include "llvm/Module.h"
using namespace llvm;

/// ctor - always have a valid Module
///
ModuleProvider::ModuleProvider() : TheModule(0) { }

/// dtor - when we leave, we take our Module with us
///
ModuleProvider::~ModuleProvider() {
  delete TheModule;
}
