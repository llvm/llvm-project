//===--- Plugin.h - The LLVM Compiler Driver --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Plugin support for llvmc2.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMC2_PLUGIN_H
#define LLVM_TOOLS_LLVMC2_PLUGIN_H

#include "llvm/Support/Registry.h"

namespace llvmc {

  class LanguageMap;
  class CompilationGraph;

  /// BasePlugin - An abstract base class for all LLVMC plugins.
  struct BasePlugin {

    /// Priority - Plugin priority, useful for handling dependencies
    /// between plugins. Plugins with lower priorities are loaded
    /// first.
    virtual int Priority() const = 0;

    /// PopulateLanguageMap - The auto-generated function that fills in
    /// the language map (map from file extensions to language names).
    virtual void PopulateLanguageMap(LanguageMap&) const = 0;

    /// PopulateCompilationGraph - The auto-generated function that
    /// populates the compilation graph with nodes and edges.
    virtual void PopulateCompilationGraph(CompilationGraph&) const = 0;

    /// Needed to avoid a compiler warning.
    virtual ~BasePlugin() {}
  };

  typedef llvm::Registry<BasePlugin> PluginRegistry;

  template <class P>
  struct RegisterPlugin
    : public PluginRegistry::Add<P> {
    typedef PluginRegistry::Add<P> Base;

    RegisterPlugin(const char* Name = "Nameless",
                   const char* Desc = "Auto-generated plugin")
      : Base(Name, Desc) {}
  };


  /// PluginLoader - Helper class used by the main program for
  /// lifetime management.
  struct PluginLoader {
    PluginLoader();
    ~PluginLoader();

    /// PopulateLanguageMap - Fills in the language map by calling
    /// PopulateLanguageMap methods of all plugins.
    void PopulateLanguageMap(LanguageMap& langMap);

    /// PopulateCompilationGraph - Populates the compilation graph by
    /// calling PopulateCompilationGraph methods of all plugins.
    void PopulateCompilationGraph(CompilationGraph& tools);

  private:
    // noncopyable
    PluginLoader(const PluginLoader& other);
    const PluginLoader& operator=(const PluginLoader& other);
  };

}

#endif // LLVM_TOOLS_LLVMC2_PLUGIN_H
