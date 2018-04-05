//===-- llvm/Support/Compiler.h - Compiler abstraction support --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Due to layering constraints (Support depends on Demangler) this is a thin
// wrapper around the implementation that lives in llvm-c, though most clients
// can/should think of this as being provided by Support for simplicity (not
// many clients are aware of their dependency on Demangler/it's a weird place to
// own this - but didn't seem to justify splitting Support into "lower support"
// and "upper support").
//
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/Compiler.h"
