//===-- SwiftHashedContainer.h ----------------------------------*- C++ -*-===//
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

#ifndef liblldb_SwiftHashedContainer_h_
#define liblldb_SwiftHashedContainer_h_

#include "lldb/lldb-forward.h"

#include "lldb/Utility/ConstString.h"
#include "lldb/DataFormatters/FormatClasses.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Target/Target.h"

#include <functional>

namespace lldb_private {
namespace formatters {
namespace swift {

class HashedStorageHandler;
typedef std::unique_ptr<HashedStorageHandler> HashedStorageHandlerUP;

class HashedCollectionConfig {
public:
  void RegisterSummaryProviders(
    lldb::TypeCategoryImplSP swift_category_sp,
    TypeSummaryImpl::Flags flags) const;
  void RegisterSyntheticChildrenCreators(
    lldb::TypeCategoryImplSP swift_category_sp,
    SyntheticChildren::Flags flags) const;

  bool IsNativeStorageName(ConstString name) const;

  bool IsEmptyStorageName(ConstString name) const;

  bool IsDeferredBridgedStorageName(ConstString name) const;

  HashedStorageHandlerUP
  CreateHandler(ValueObject &valobj) const;

  virtual ~HashedCollectionConfig() = default;

private:
  HashedStorageHandlerUP
  CreateEmptyHandler(CompilerType elem_type = CompilerType()) const;

  HashedStorageHandlerUP
  _CreateNativeHandler(
    lldb::ValueObjectSP storage_sp,
    CompilerType key_type,
    CompilerType value_type) const;

  HashedStorageHandlerUP
  CreateNativeHandler(
    lldb::ValueObjectSP value_sp,
    lldb::ValueObjectSP storage_sp) const;

  HashedStorageHandlerUP
  CreateCocoaHandler(lldb::ValueObjectSP storage_sp) const;

  lldb::ValueObjectSP
  StorageObjectAtAddress(
    const ExecutionContext &exe_ctx,
    lldb::addr_t address) const;

  lldb::ValueObjectSP
  CocoaObjectAtAddress(
    const ExecutionContext &exe_ctx,
    lldb::addr_t address) const;

protected:
  HashedCollectionConfig() {}

  virtual CXXFunctionSummaryFormat::Callback
  GetSummaryProvider() const = 0;

  virtual CXXSyntheticChildren::CreateFrontEndCallback
  GetSyntheticChildrenCreator() const = 0;

  virtual CXXSyntheticChildren::CreateFrontEndCallback
  GetCocoaSyntheticChildrenCreator() const = 0;

  ConstString m_summaryProviderName;
  ConstString m_syntheticChildrenName;

  ConstString m_collection_demangledRegex;
  
  ConstString m_nativeStorageRoot_mangled;
  ConstString m_nativeStorageRoot_demangled;

  ConstString m_nativeStorage_mangledRegex_ObjC;
  ConstString m_nativeStorage_demangledPrefix;
  ConstString m_nativeStorage_demangledRegex;

  ConstString m_emptyStorage_mangled_ObjC;
  ConstString m_emptyStorage_demangled;

  ConstString m_deferredBridgedStorage_mangledRegex_ObjC;
  ConstString m_deferredBridgedStorage_demangledPrefix;
  ConstString m_deferredBridgedStorage_demangledRegex;
};

// Some part of the buffer handling logic needs to be shared between summary and
// synthetic children
// If I was only making synthetic children, this would be best modelled as
// different FrontEnds
class HashedStorageHandler {
public:
  virtual size_t GetCount() = 0;

  virtual CompilerType GetElementType() = 0;

  virtual lldb::ValueObjectSP GetElementAtIndex(size_t) = 0;

  virtual bool IsValid() = 0;

  virtual ~HashedStorageHandler() {}
};

class HashedSyntheticChildrenFrontEnd : public SyntheticChildrenFrontEnd {
public:
  HashedSyntheticChildrenFrontEnd(
    const HashedCollectionConfig &config,
    lldb::ValueObjectSP valobj_sp);

  virtual size_t CalculateNumChildren();

  virtual lldb::ValueObjectSP GetChildAtIndex(size_t idx);

  virtual bool Update();

  virtual bool MightHaveChildren();

  virtual size_t GetIndexOfChildWithName(ConstString name);

  virtual ~HashedSyntheticChildrenFrontEnd() = default;

private:
  const HashedCollectionConfig &m_config;
  HashedStorageHandlerUP m_buffer;
};
}
}
}

#endif // liblldb_SwiftHashedContainer_h_
