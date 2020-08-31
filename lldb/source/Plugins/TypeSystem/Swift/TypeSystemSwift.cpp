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
#include <lldb/lldb-enumerations.h>
#include <llvm/ADT/StringRef.h>

LLDB_PLUGIN_DEFINE(TypeSystemSwift)

using namespace lldb;
using namespace lldb_private;
using llvm::StringRef;

/// TypeSystem Plugin functionality.
/// \{
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

LanguageSet TypeSystemSwift::GetSupportedLanguagesForTypes() {
  LanguageSet swift;
  swift.Insert(lldb::eLanguageTypeSwift);
  return swift;
}

void TypeSystemSwift::Initialize() {
  SwiftLanguageRuntime::Initialize();
  LanguageSet swift = GetSupportedLanguagesForTypes();
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

/// \}


bool TypeSystemSwift::IsFloatingPointType(opaque_compiler_type_t type,
                                          uint32_t &count, bool &is_complex) {
  count = 0;
  is_complex = false;
  if (GetTypeInfo(type, nullptr) & eTypeIsFloat) {
    count = 1;
    return true;
  }
  return false;
}

bool TypeSystemSwift::IsIntegerType(opaque_compiler_type_t type,
                                    bool &is_signed) {
  return (GetTypeInfo(type, nullptr) & eTypeIsInteger);
}

bool TypeSystemSwift::IsScalarType(opaque_compiler_type_t type) {
  if (!type)
    return false;

  return (GetTypeInfo(type, nullptr) & eTypeIsScalar) != 0;
}

bool TypeSystemSwift::ShouldTreatScalarValueAsAddress(
    opaque_compiler_type_t type) {
  return Flags(GetTypeInfo(type, nullptr))
      .AnySet(eTypeInstanceIsPointer | eTypeIsReference);
}
