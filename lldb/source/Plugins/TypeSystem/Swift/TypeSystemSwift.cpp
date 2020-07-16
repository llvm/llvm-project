//===-- TypeSystemSwift.cpp ==---------------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "Plugins/TypeSystem/Swift/TypeSystemSwift.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Target/SwiftLanguageRuntime.h"

LLDB_PLUGIN_DEFINE(TypeSystemSwift)

using namespace lldb_private;

static lldb::TypeSystemSP CreateTypeSystemInstance(lldb::LanguageType language,
                                                   Module *module,
                                                   Target *target,
                                                   const char *extra_options) {
  // This should be called with either a target or a module.
  if (module) {
    assert(!target);
    assert(StringRef(extra_options).empty());
    return SwiftASTContext::CreateInstance(language, *module);
  } else if (target) {
    assert(!module);
    return SwiftASTContext::CreateInstance(language, *target, extra_options);
  }
  llvm_unreachable("Neither type nor module given to CreateTypeSystemInstance");
}

void TypeSystemSwift::Initialize() {
  LanguageSet swift;
  SwiftLanguageRuntime::Initialize();
  swift.Insert(lldb::eLanguageTypeSwift);
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "Swift type system and AST context plug-in",
                                CreateTypeSystemInstance, swift, swift);
}

void TypeSystemSwift::Terminate() {
  PluginManager::UnregisterPlugin(CreateTypeSystemInstance);
  SwiftLanguageRuntime::Terminate();
}

ConstString TypeSystemSwift::GetPluginNameStatic() {
  return ConstString("swift");
}

ConstString TypeSystemSwift::GetPluginName() {
  return TypeSystemSwift::GetPluginNameStatic();
}

uint32_t TypeSystemSwift::GetPluginVersion() { return 1; }
