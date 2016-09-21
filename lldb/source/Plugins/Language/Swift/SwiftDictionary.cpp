//===-- SwiftDictionary.cpp -------------------------------------*- C++ -*-===//
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

#include "SwiftDictionary.h"

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
class SwiftDictionaryNativeBufferHandler
    : public SwiftHashedContainerNativeBufferHandler {
public:
  SwiftHashedContainerBufferHandler::Kind GetKind() {
    return Kind::eDictionary;
  }

  static ConstString GetMangledStorageTypeName();

  static ConstString GetDemangledStorageTypeName();

  SwiftDictionaryNativeBufferHandler(ValueObjectSP nativeStorage_sp,
                                     CompilerType key_type,
                                     CompilerType value_type)
      : SwiftHashedContainerNativeBufferHandler(nativeStorage_sp, key_type,
                                                value_type) {}
  friend class SwiftHashedContainerBufferHandler;

private:
};

class SwiftDictionarySyntheticFrontEndBufferHandler
    : public SwiftHashedContainerSyntheticFrontEndBufferHandler {
public:
  SwiftHashedContainerBufferHandler::Kind GetKind() {
    return Kind::eDictionary;
  }

  virtual ~SwiftDictionarySyntheticFrontEndBufferHandler() {}

  SwiftDictionarySyntheticFrontEndBufferHandler(lldb::ValueObjectSP valobj_sp)
      : SwiftHashedContainerSyntheticFrontEndBufferHandler(valobj_sp) {}
  friend class SwiftHashedContainerBufferHandler;

private:
};

class DictionarySyntheticFrontEnd : public HashedContainerSyntheticFrontEnd {
public:
  DictionarySyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
      : HashedContainerSyntheticFrontEnd(valobj_sp) {}

  virtual bool Update();

  virtual ~DictionarySyntheticFrontEnd() = default;
};
}
}
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::swift::DictionarySyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return NULL;
  return (new DictionarySyntheticFrontEnd(valobj_sp));
}

bool lldb_private::formatters::swift::DictionarySyntheticFrontEnd::Update() {
  m_buffer = SwiftHashedContainerBufferHandler::CreateBufferHandler(
      m_backend,
      [](ValueObjectSP a, CompilerType b,
         CompilerType c) -> SwiftHashedContainerBufferHandler * {
        return new SwiftDictionaryNativeBufferHandler(a, b, c);
      },
      [](ValueObjectSP a) -> SwiftHashedContainerBufferHandler * {
        return new SwiftDictionarySyntheticFrontEndBufferHandler(a);
      },
      SwiftDictionaryNativeBufferHandler::GetMangledStorageTypeName(),
      SwiftDictionaryNativeBufferHandler::GetDemangledStorageTypeName());
  return false;
}

bool lldb_private::formatters::swift::Dictionary_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  auto handler = SwiftHashedContainerBufferHandler::CreateBufferHandler(
      valobj,
      [](ValueObjectSP a, CompilerType b,
         CompilerType c) -> SwiftHashedContainerBufferHandler * {
        return new SwiftDictionaryNativeBufferHandler(a, b, c);
      },
      [](ValueObjectSP a) -> SwiftHashedContainerBufferHandler * {
        return new SwiftDictionarySyntheticFrontEndBufferHandler(a);
      },
      SwiftDictionaryNativeBufferHandler::GetMangledStorageTypeName(),
      SwiftDictionaryNativeBufferHandler::GetDemangledStorageTypeName());

  if (!handler)
    return false;

  auto count = handler->GetCount();

  stream.Printf("%zu key/value pair%s", count, (count == 1 ? "" : "s"));

  return true;
};

ConstString SwiftDictionaryNativeBufferHandler::GetMangledStorageTypeName() {
  static ConstString g_name("_TtCs29_NativeDictionaryStorageOwner");
  return g_name;
}

ConstString SwiftDictionaryNativeBufferHandler::GetDemangledStorageTypeName() {
  static ConstString g_name(
      "Swift._NativeDictionaryStorageOwner");
  return g_name;
}
