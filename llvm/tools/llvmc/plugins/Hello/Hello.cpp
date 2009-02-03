//===- Hello.cpp - Example code from "Writing an LLVM Pass" ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Test plugin for LLVMC. Shows how to write plugins without using TableGen.
//
//===----------------------------------------------------------------------===//

#include "llvm/CompilerDriver/CompilationGraph.h"
#include "llvm/CompilerDriver/Plugin.h"

#include <iostream>

namespace {
struct MyPlugin : public llvmc::BasePlugin {
  void PopulateLanguageMap(llvmc::LanguageMap&) const
  { std::cout << "Hello!\n"; }

  void PopulateCompilationGraph(llvmc::CompilationGraph&) const
  {}
};

static llvmc::RegisterPlugin<MyPlugin> RP("Hello", "Hello World plugin");

}
