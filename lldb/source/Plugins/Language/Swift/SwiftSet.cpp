//===-- SwiftSet.cpp --------------------------------------------*- C++ -*-===//
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

#include "SwiftSet.h"

#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/SwiftASTContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SwiftLanguageRuntime.h"

#include "swift/AST/ASTContext.h"
#include "llvm/ADT/StringRef.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace lldb_private::formatters::swift;


const SetConfig &
SetConfig::Get() {
  static SetConfig g_config{};
  return g_config;
}

SetConfig::SetConfig()
  : HashedCollectionConfig() {
  m_summaryProviderName =
    ConstString("Swift.Set summary provider");
  m_syntheticChildrenName =
    ConstString("Swift.Set synthetic children");
  m_collection_demangledRegex =
    ConstString("^Swift\\.Set<.+>$");

  // Note: We need to have the old _Tt names here because those are
  // still used to name classes in the ObjC runtime.

  m_nativeStorageRoot_mangled =
    ConstString("_TtCs15__RawSetStorage");
  m_nativeStorageRoot_demangled =
    ConstString("Swift.__RawSetStorage");

    // Native storage class
  m_nativeStorage_mangledRegex_ObjC =
    ConstString("^_TtGCs11_SetStorage.*");
  m_nativeStorage_demangledPrefix =
    ConstString("Swift._SetStorage<");
  m_nativeStorage_demangledRegex =
    ConstString("^Swift\\._SetStorage<.+>$");

  // Type-punned empty set
  m_emptyStorage_mangled_ObjC = ConstString("_TtCs19__EmptySetSingleton");
  m_emptyStorage_demangled = ConstString("Swift.__EmptySetSingleton");

  // Deferred non-verbatim bridged set
  m_deferredBridgedStorage_mangledRegex_ObjC
    = ConstString("^_TtGCs19_SwiftDeferredNSSet.*");
  m_deferredBridgedStorage_demangledPrefix
    = ConstString("Swift._SwiftDeferredNSSet<");
  m_deferredBridgedStorage_demangledRegex
    = ConstString("^Swift\\._SwiftDeferredNSSet<.+>$");
}

bool
SetConfig::SummaryProvider(
  ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  auto handler = SetConfig::Get().CreateHandler(valobj);

  if (!handler)
    return false;

  auto count = handler->GetCount();

  stream.Printf("%zu value%s", count, (count == 1 ? "" : "s"));

  return true;
};

SyntheticChildrenFrontEnd *
SetConfig::SyntheticChildrenCreator(
  CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;
  return new HashedSyntheticChildrenFrontEnd(SetConfig::Get(), valobj_sp);
}
