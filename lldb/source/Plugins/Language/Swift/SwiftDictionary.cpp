//===-- SwiftDictionary.cpp -------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "SwiftDictionary.h"

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "Plugins/TypeSystem/Swift/SwiftASTContext.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SwiftLanguageRuntime.h"

#include "swift/AST/ASTContext.h"
#include "llvm/ADT/StringRef.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace lldb_private::formatters::swift;

const DictionaryConfig &
DictionaryConfig::Get() {
  static DictionaryConfig g_config{};
  return g_config;
}

DictionaryConfig::DictionaryConfig()
  : HashedCollectionConfig() {
  m_summaryProviderName =
    ConstString("Swift.Dictionary summary provider");
  m_syntheticChildrenName =
    ConstString("Swift.Dictionary synthetic children");
  m_collection_demangledRegex =
    ConstString("^Swift\\.Dictionary<.+,.+>$");

  // Note: We need have use the old _Tt names here because those are
  // still used to name classes in the ObjC runtime.

  m_nativeStorageRoot_mangled =
    ConstString("_TtCs22__RawDictionaryStorage");
  m_nativeStorageRoot_demangled =
    ConstString("Swift.__RawDictionaryStorage");

  // Native storage class
  m_nativeStorage_mangledRegex_ObjC =
    ConstString("^_TtGCs18_DictionaryStorage.*");
  m_nativeStorage_demangledPrefix =
    ConstString("Swift._DictionaryStorage<");
  m_nativeStorage_demangledRegex =
    ConstString("^Swift\\._DictionaryStorage<.+,.+>$");

  // Type-punned empty dictionary
  m_emptyStorage_mangled_ObjC =
    ConstString("_TtCs26__EmptyDictionarySingleton");
  m_emptyStorage_demangled
    = ConstString("Swift.__EmptyDictionarySingleton");

  // Deferred non-verbatim bridged dictionary
  m_deferredBridgedStorage_mangledRegex_ObjC
    = ConstString("^_TtGCs26_SwiftDeferredNSDictionary.*");
  m_deferredBridgedStorage_demangledPrefix
    = ConstString("Swift._SwiftDeferredNSDictionary<");
  m_deferredBridgedStorage_demangledRegex
    = ConstString("^Swift\\._SwiftDeferredNSDictionary<.+,.+>$");
}

bool
DictionaryConfig::SummaryProvider(
  ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  auto handler = DictionaryConfig::Get().CreateHandler(valobj);

  if (!handler)
    return false;

  auto count = handler->GetCount();

  stream.Printf("%zu key/value pair%s", count, (count == 1 ? "" : "s"));

  return true;
};

SyntheticChildrenFrontEnd *
DictionaryConfig::SyntheticChildrenCreator(
  CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;
  return new HashedSyntheticChildrenFrontEnd(
    DictionaryConfig::Get(), valobj_sp);
}
