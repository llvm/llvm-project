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

#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "lldb/Core/PluginManager.h"
#include <lldb/lldb-enumerations.h>
#include <llvm/ADT/StringRef.h>

LLDB_PLUGIN_DEFINE(TypeSystemSwift)

using namespace lldb;
using namespace lldb_private;
using llvm::StringRef;

TypeSystemSwift::TypeSystemSwift() : TypeSystem() {}

/// TypeSystem Plugin functionality.
/// \{
static lldb::TypeSystemSP CreateTypeSystemInstance(lldb::LanguageType language,
                                                   Module *module,
                                                   Target *target,
                                                   const char *extra_options) {
  if (language != eLanguageTypeSwift)
    return {};

  // This should be called with either a target or a module.
  if (module) {
    assert(!target);
    assert(StringRef(extra_options).empty());
    return std::shared_ptr<TypeSystemSwiftTypeRef>(
        new TypeSystemSwiftTypeRef(*module));
  } else if (target) {
    assert(!module);
    return std::shared_ptr<TypeSystemSwiftTypeRefForExpressions>(
        new TypeSystemSwiftTypeRefForExpressions(language, *target,
                                                 extra_options));
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

/// \}

void TypeSystemSwift::DumpValue(
    lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx, Stream *s,
    lldb::Format format, const DataExtractor &data, lldb::offset_t data_offset,
    size_t data_byte_size, uint32_t bitfield_bit_size,
    uint32_t bitfield_bit_offset, bool show_types, bool show_summary,
    bool verbose, uint32_t depth) {}

void TypeSystemSwift::Dump(llvm::raw_ostream &output) {
  // TODO: What to dump?
}

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

uint32_t TypeSystemSwift::GetIndexOfChildWithName(
    opaque_compiler_type_t type, const char *name, ExecutionContext *exe_ctx,
    bool omit_empty_base_classes) {
  std::vector<uint32_t> child_indexes;
  size_t num_child_indexes = GetIndexOfChildMemberWithName(
      type, name, exe_ctx, omit_empty_base_classes, child_indexes);
  return num_child_indexes == 1 ? child_indexes.front() : UINT32_MAX;
}

lldb::Format TypeSystemSwift::GetFormat(opaque_compiler_type_t type) {
  auto swift_flags = GetTypeInfo(type, nullptr);

  if (swift_flags & eTypeIsInteger)
    return eFormatDecimal;

  if (swift_flags & eTypeIsFloat)
    return eFormatFloat;

  if (swift_flags & eTypeIsPointer || swift_flags & eTypeIsClass)
    return eFormatAddressInfo;

  if (swift_flags & eTypeIsClass)
    return eFormatHex;

  if (swift_flags & eTypeIsGenericTypeParam)
    return eFormatUnsigned;

  if (swift_flags & eTypeIsFuncPrototype || swift_flags & eTypeIsBlock)
    return eFormatAddressInfo;

  return eFormatBytes;
}

namespace llvm {
llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           TypeSystemSwift::NonTriviallyManagedReferenceKind k) {
  switch (k) {
  case TypeSystemSwift::NonTriviallyManagedReferenceKind::eWeak:
    os << "eWeak";
    break;
  case TypeSystemSwift::NonTriviallyManagedReferenceKind:: eUnowned:
    os << "eUnowned";
    break;
  case TypeSystemSwift::NonTriviallyManagedReferenceKind::eUnmanaged:
    os << "eUnmanaged";
    break;
  }
  return os;
}
} // namespace llvm
