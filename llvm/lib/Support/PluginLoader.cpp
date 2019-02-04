//===-- PluginLoader.cpp - Implement -load command line option ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the -load <plugin> command line option handler.
//
//===----------------------------------------------------------------------===//

#define DONT_GET_PLUGIN_LOADER_OPTION
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>
using namespace llvm;

static ManagedStatic<std::vector<std::string> > Plugins;
static ManagedStatic<sys::SmartMutex<true> > PluginsLock;

void PluginLoader::operator=(const std::string &Filename) {
  sys::SmartScopedLock<true> Lock(*PluginsLock);
  std::string Error;
  if (sys::DynamicLibrary::LoadLibraryPermanently(Filename.c_str(), &Error)) {
    errs() << "Error opening '" << Filename << "': " << Error
           << "\n  -load request ignored.\n";
  } else {
    Plugins->push_back(Filename);
  }
}

unsigned PluginLoader::getNumPlugins() {
  sys::SmartScopedLock<true> Lock(*PluginsLock);
  return Plugins.isConstructed() ? Plugins->size() : 0;
}

std::string &PluginLoader::getPlugin(unsigned num) {
  sys::SmartScopedLock<true> Lock(*PluginsLock);
  assert(Plugins.isConstructed() && num < Plugins->size() &&
         "Asking for an out of bounds plugin");
  return (*Plugins)[num];
}
