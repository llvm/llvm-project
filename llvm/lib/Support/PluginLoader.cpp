//===-- PluginLoader.cpp - Implement -load command line option ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the -load <plugin> command line option handler.
//
//===----------------------------------------------------------------------===//

#define DONT_GET_PLUGIN_LOADER_OPTION
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/Streams.h"
#include "llvm/System/DynamicLibrary.h"
#include <ostream>
#include <vector>
using namespace llvm;

static ManagedStatic<std::vector<std::string> > Plugins;

void PluginLoader::operator=(const std::string &Filename) {
  std::string Error;
  if (sys::DynamicLibrary::LoadLibraryPermanently(Filename.c_str(), &Error)) {
    cerr << "Error opening '" << Filename << "': " << Error
         << "\n  -load request ignored.\n";
  } else {
    Plugins->push_back(Filename);
  }
}

unsigned PluginLoader::getNumPlugins() {
  return Plugins.isConstructed() ? Plugins->size() : 0;
}

std::string &PluginLoader::getPlugin(unsigned num) {
  assert(Plugins.isConstructed() && num < Plugins->size() &&
         "Asking for an out of bounds plugin");
  return (*Plugins)[num];
}
