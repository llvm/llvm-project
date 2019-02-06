//===- llvm/Support/Options.cpp - Debug options support ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the helper objects for defining debug options using the
// new API built on cl::opt, but not requiring the use of static globals.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Options.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;

OptionRegistry::~OptionRegistry() {
  for (auto IT = Options.begin(); IT != Options.end(); ++IT)
    delete IT->second;
}

void OptionRegistry::addOption(void *Key, cl::Option *O) {
  assert(Options.find(Key) == Options.end() &&
         "Argument with this key already registerd");
  Options.insert(std::make_pair(Key, O));
}

static ManagedStatic<OptionRegistry> OR;

OptionRegistry &OptionRegistry::instance() { return *OR; }
