//===-- SwiftSet.cpp --------------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See http://swift.org/LICENSE.txt for license information
// See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "SwiftSet.h"

#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/SwiftASTContext.h"
#include "lldb/Target/Process.h"

#include "swift/AST/ASTContext.h"
#include "llvm/ADT/StringRef.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace lldb_private::formatters::swift;

namespace lldb_private {
namespace formatters {
namespace swift {
class SwiftSetNativeBufferHandler
    : public SwiftHashedContainerNativeBufferHandler {
public:
  SwiftHashedContainerBufferHandler::Kind GetKind() { return Kind::eSet; }

  static ConstString GetMangledStorageTypeName();

  static ConstString GetDemangledStorageTypeName();

  virtual lldb::ValueObjectSP GetElementAtIndex(size_t);

  SwiftSetNativeBufferHandler(ValueObjectSP nativeStorage_sp,
                              CompilerType key_type)
      : SwiftHashedContainerNativeBufferHandler(nativeStorage_sp, key_type,
                                                CompilerType()) {}
  friend class SwiftHashedContainerBufferHandler;

private:
};

class SwiftSetSyntheticFrontEndBufferHandler
    : public SwiftHashedContainerSyntheticFrontEndBufferHandler {
public:
  SwiftHashedContainerBufferHandler::Kind GetKind() { return Kind::eSet; }

  virtual ~SwiftSetSyntheticFrontEndBufferHandler() {}

  SwiftSetSyntheticFrontEndBufferHandler(lldb::ValueObjectSP valobj_sp)
      : SwiftHashedContainerSyntheticFrontEndBufferHandler(valobj_sp) {}
  friend class SwiftHashedContainerBufferHandler;

private:
};

class SetSyntheticFrontEnd : public HashedContainerSyntheticFrontEnd {
public:
  SetSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
      : HashedContainerSyntheticFrontEnd(valobj_sp) {}

  virtual bool Update();

  virtual ~SetSyntheticFrontEnd() = default;
};
}
}
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::swift::SetSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return NULL;
  return (new SetSyntheticFrontEnd(valobj_sp));
}

bool lldb_private::formatters::swift::SetSyntheticFrontEnd::Update() {
  m_buffer = SwiftHashedContainerBufferHandler::CreateBufferHandler(
      m_backend,
      [](ValueObjectSP a, CompilerType b,
         CompilerType c) -> SwiftHashedContainerBufferHandler * {
        return new SwiftSetNativeBufferHandler(a, b);
      },
      [](ValueObjectSP a) -> SwiftHashedContainerBufferHandler * {
        return new SwiftSetSyntheticFrontEndBufferHandler(a);
      },
      SwiftSetNativeBufferHandler::GetMangledStorageTypeName(),
      SwiftSetNativeBufferHandler::GetDemangledStorageTypeName());
  return false;
}

bool lldb_private::formatters::swift::Set_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  auto handler = SwiftHashedContainerBufferHandler::CreateBufferHandler(
      valobj,
      [](ValueObjectSP a, CompilerType b,
         CompilerType c) -> SwiftHashedContainerBufferHandler * {
        return new SwiftSetNativeBufferHandler(a, b);
      },
      [](ValueObjectSP a) -> SwiftHashedContainerBufferHandler * {
        return new SwiftSetSyntheticFrontEndBufferHandler(a);
      },
      SwiftSetNativeBufferHandler::GetMangledStorageTypeName(),
      SwiftSetNativeBufferHandler::GetDemangledStorageTypeName());

  if (!handler)
    return false;

  auto count = handler->GetCount();

  stream.Printf("%zu value%s", count, (count == 1 ? "" : "s"));

  return true;
};

lldb::ValueObjectSP SwiftSetNativeBufferHandler::GetElementAtIndex(size_t idx) {
  ValueObjectSP parent_element(
      this->SwiftHashedContainerNativeBufferHandler::GetElementAtIndex(idx));
  if (!parent_element)
    return parent_element;
  static ConstString g_key("key");
  ValueObjectSP key_child(parent_element->GetChildMemberWithName(g_key, true));
  return key_child ? (key_child->SetName(parent_element->GetName()), key_child)
                   : parent_element;
}

ConstString SwiftSetNativeBufferHandler::GetMangledStorageTypeName() {
  static ConstString g_name("_TtCs22_NativeSetStorageOwner");
  return g_name;
}

ConstString SwiftSetNativeBufferHandler::GetDemangledStorageTypeName() {
  static ConstString g_name(
      "Swift._NativeSetStorageOwner with unmangled suffix");
  return g_name;
}
