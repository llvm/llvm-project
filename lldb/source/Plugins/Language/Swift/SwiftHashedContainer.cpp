//===-- SwiftHashedContainer.cpp --------------------------------*- C++ -*-===//
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

#include "SwiftHashedContainer.h"

#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/SwiftASTContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SwiftLanguageRuntime.h"
#include "lldb/Utility/DataBufferHeap.h"

#include "Plugins/Language/ObjC/NSDictionary.h"

#include "swift/AST/ASTContext.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace lldb_private::formatters::swift;

namespace lldb_private {
namespace formatters {
namespace swift {

class EmptyHashedStorageHandler: public HashedStorageHandler {
public:
  EmptyHashedStorageHandler(CompilerType elem_type)
    : m_elem_type(elem_type) {}

  virtual size_t GetCount() override { return 0; }

  virtual CompilerType GetElementType() override { return m_elem_type; }

  virtual ValueObjectSP GetElementAtIndex(size_t) override {
    return ValueObjectSP();
  }

  virtual bool IsValid() override { return true; }

  virtual ~EmptyHashedStorageHandler() {}

private:
  CompilerType m_elem_type;
};

class NativeHashedStorageHandler: public HashedStorageHandler {
public:
  NativeHashedStorageHandler(ValueObjectSP storage_sp,
                             CompilerType key_type,
                             CompilerType value_type);

  virtual size_t GetCount() override { return m_count; }

  virtual CompilerType GetElementType() override { return m_element_type; }


  virtual ValueObjectSP GetElementAtIndex(size_t) override;

  virtual bool IsValid() override;

  virtual ~NativeHashedStorageHandler() override {}

protected:
  typedef uint64_t Index;
  typedef uint64_t Bucket;

  bool UpdateBuckets();
  bool FailBuckets();

  size_t GetBucketCount() { return 1 << m_scale; }
  size_t GetWordWidth() { return m_ptr_size * 8; }
  size_t GetWordCount() { return std::max(1ul, GetBucketCount() / GetWordWidth()); }

  uint64_t GetMetadataWord(int index, Status &error);

  lldb::addr_t GetLocationOfKeyInBucket(Bucket b) {
    return m_keys_ptr + (b * m_key_stride);
  }

  lldb::addr_t GetLocationOfValueInBucket(Bucket b) {
    return m_value_stride
      ? m_values_ptr + (b * m_value_stride)
      : LLDB_INVALID_ADDRESS;
  }

  // these are sharp tools that assume that the Bucket contains valid
  // data and the destination buffer has enough room to store the data
  // to - use with caution
  bool GetDataForKeyInBucket(Bucket b, void *data_ptr) {
    if (!data_ptr)
      return false;

    lldb::addr_t addr = GetLocationOfKeyInBucket(b);
    Status error;
    m_process->ReadMemory(addr, data_ptr, m_key_stride, error);
    if (error.Fail())
      return false;

    return true;
  }

  bool GetDataForValueInBucket(Bucket b, void *data_ptr) {
    if (!data_ptr || !m_value_stride)
      return false;

    lldb::addr_t addr = GetLocationOfValueInBucket(b);
    Status error;
    m_process->ReadMemory(addr, data_ptr, m_value_stride, error);
    if (error.Fail())
      return false;

    return true;
  }

private:
  ValueObject *m_storage;
  Process *m_process;
  uint32_t m_ptr_size;
  uint64_t m_count;
  uint64_t m_scale;
  lldb::addr_t m_metadata_ptr;
  lldb::addr_t m_keys_ptr;
  lldb::addr_t m_values_ptr;
  CompilerType m_element_type;
  uint64_t m_key_stride;
  uint64_t m_value_stride;
  uint64_t m_key_stride_padded;
  // Cached mapping from index to occupied bucket.
  std::vector<Bucket> m_occupiedBuckets;
  bool m_failedToGetBuckets;
};

// Storage handler for Swift 4.2 storage class instances
class LegacyNativeHashedStorageHandler: public HashedStorageHandler {
public:
  LegacyNativeHashedStorageHandler(ValueObjectSP storage_sp,
                                   CompilerType key_type,
                                   CompilerType value_type);

  virtual size_t GetCount() override { return m_count; }

  virtual CompilerType GetElementType() override { return m_element_type; }

  virtual ValueObjectSP GetElementAtIndex(size_t) override;

  virtual bool IsValid() override;

  virtual ~LegacyNativeHashedStorageHandler() override {}

protected:
  typedef uint64_t Index;
  typedef uint64_t Cell;

  bool ReadBitmaskAtIndex(Index, Status &error);

  lldb::addr_t GetLocationOfKeyAtCell(Cell i) {
    return m_keys_ptr + (i * m_key_stride);
  }

  lldb::addr_t GetLocationOfValueAtCell(Cell i) {
    return m_value_stride
      ? m_values_ptr + (i * m_value_stride)
      : LLDB_INVALID_ADDRESS;
  }

  // these are sharp tools that assume that the Cell contains valid data and the
  // destination buffer
  // has enough room to store the data to - use with caution
  bool GetDataForKeyAtCell(Cell i, void *data_ptr) {
    if (!data_ptr)
      return false;

    lldb::addr_t addr = GetLocationOfKeyAtCell(i);
    Status error;
    m_process->ReadMemory(addr, data_ptr, m_key_stride, error);
    if (error.Fail())
      return false;

    return true;
  }

  bool GetDataForValueAtCell(Cell i, void *data_ptr) {
    if (!data_ptr || !m_value_stride)
      return false;

    lldb::addr_t addr = GetLocationOfValueAtCell(i);
    Status error;
    m_process->ReadMemory(addr, data_ptr, m_value_stride, error);
    if (error.Fail())
      return false;

    return true;
  }

private:
  ValueObject *m_nativeStorage;
  Process *m_process;
  uint32_t m_ptr_size;
  uint64_t m_count;
  uint64_t m_bucketCount;
  lldb::addr_t m_bitmask_ptr;
  lldb::addr_t m_keys_ptr;
  lldb::addr_t m_values_ptr;
  CompilerType m_element_type;
  uint64_t m_key_stride;
  uint64_t m_value_stride;
  uint64_t m_key_stride_padded;
  std::map<lldb::addr_t, uint64_t> m_bitmask_cache;
};

class CocoaHashedStorageHandler: public HashedStorageHandler {
public:
  CocoaHashedStorageHandler(
    ValueObjectSP cocoaObject_sp,
    SyntheticChildrenFrontEnd *frontend)
    : m_cocoaObject_sp(cocoaObject_sp), m_frontend(frontend) {}

  virtual size_t GetCount() override {
    return m_frontend->CalculateNumChildren();
  }

  virtual CompilerType GetElementType() override {
    // this doesn't make sense here - the synthetic children know best
    return CompilerType();
  }

  virtual ValueObjectSP GetElementAtIndex(size_t idx) override {
    return m_frontend->GetChildAtIndex(idx);
  }

  virtual bool IsValid() override {
    return m_frontend.get() != nullptr;
  }

  virtual ~CocoaHashedStorageHandler() {}

private:
  // reader beware: this entails you must only pass self-rooted
  // valueobjects to this class
  ValueObjectSP m_cocoaObject_sp; 
  std::unique_ptr<SyntheticChildrenFrontEnd> m_frontend;
};

}
}
}

void
HashedCollectionConfig::RegisterSummaryProviders(
  lldb::TypeCategoryImplSP swift_category_sp,
  TypeSummaryImpl::Flags flags
) const {
#ifndef LLDB_DISABLE_PYTHON
  using lldb_private::formatters::AddCXXSummary;

  auto summaryProvider = GetSummaryProvider();
  AddCXXSummary(swift_category_sp, summaryProvider,
                m_summaryProviderName.AsCString(),
                m_collection_demangledRegex, flags, true);

  AddCXXSummary(swift_category_sp, summaryProvider,
                m_summaryProviderName.AsCString(),
                m_nativeStorage_demangledRegex, flags, true);
  AddCXXSummary(swift_category_sp, summaryProvider,
                m_summaryProviderName.AsCString(),
                m_emptyStorage_demangled, flags, false);
  AddCXXSummary(swift_category_sp, summaryProvider,
                m_summaryProviderName.AsCString(),
                m_deferredBridgedStorage_demangledRegex, flags, true);
  AddCXXSummary(swift_category_sp, summaryProvider,
                m_summaryProviderName.AsCString(),
                m_legacy_nativeStorage_demangledRegex, flags, true);
  AddCXXSummary(swift_category_sp, summaryProvider,
                m_summaryProviderName.AsCString(),
                m_legacy_emptyStorage_demangled, flags, false);

  flags.SetSkipPointers(false);
  AddCXXSummary(swift_category_sp, summaryProvider,
                m_summaryProviderName.AsCString(),
                m_nativeStorage_mangledRegex_ObjC, flags, true);
  AddCXXSummary(swift_category_sp, summaryProvider,
                m_summaryProviderName.AsCString(),
                m_emptyStorage_mangled_ObjC, flags, false);
  AddCXXSummary(swift_category_sp, summaryProvider,
                m_summaryProviderName.AsCString(),
                m_deferredBridgedStorage_mangledRegex_ObjC, flags, true);
  AddCXXSummary(swift_category_sp, summaryProvider,
                m_summaryProviderName.AsCString(),
                m_legacy_nativeStorage_mangledRegex_ObjC, flags, true);
  AddCXXSummary(swift_category_sp, summaryProvider,
                m_summaryProviderName.AsCString(),
                m_legacy_emptyStorage_mangled_ObjC, flags, false);

#endif // LLDB_DISABLE_PYTHON
}

void
HashedCollectionConfig::RegisterSyntheticChildrenCreators(
  lldb::TypeCategoryImplSP swift_category_sp,
  SyntheticChildren::Flags flags
) const {
#ifndef LLDB_DISABLE_PYTHON
  using lldb_private::formatters::AddCXXSynthetic;

  auto creator = GetSyntheticChildrenCreator();
  AddCXXSynthetic(swift_category_sp, creator,
                  m_syntheticChildrenName.AsCString(),
                  m_collection_demangledRegex, flags, true);

  AddCXXSynthetic(swift_category_sp, creator,
                  m_syntheticChildrenName.AsCString(),
                  m_nativeStorage_demangledRegex, flags, true);
  AddCXXSynthetic(swift_category_sp, creator,
                  m_syntheticChildrenName.AsCString(),
                  m_emptyStorage_demangled, flags, false);
  AddCXXSynthetic(swift_category_sp, creator,
                  m_syntheticChildrenName.AsCString(),
                  m_deferredBridgedStorage_demangledRegex, flags, true);
  AddCXXSynthetic(swift_category_sp, creator,
                  m_syntheticChildrenName.AsCString(),
                  m_legacy_nativeStorage_demangledRegex, flags, true);
  AddCXXSynthetic(swift_category_sp, creator,
                  m_syntheticChildrenName.AsCString(),
                  m_legacy_emptyStorage_demangled, flags, false);

  flags.SetSkipPointers(false);
  AddCXXSynthetic(swift_category_sp, creator,
                  m_syntheticChildrenName.AsCString(),
                  m_nativeStorage_mangledRegex_ObjC, flags, true);
  AddCXXSynthetic(swift_category_sp, creator,
                  m_syntheticChildrenName.AsCString(),
                  m_emptyStorage_mangled_ObjC, flags, false);
  AddCXXSynthetic(swift_category_sp, creator,
                  m_syntheticChildrenName.AsCString(),
                  m_deferredBridgedStorage_mangledRegex_ObjC, flags, true);
  AddCXXSynthetic(swift_category_sp, creator,
                  m_syntheticChildrenName.AsCString(),
                  m_legacy_nativeStorage_mangledRegex_ObjC, flags, true);
  AddCXXSynthetic(swift_category_sp, creator,
                  m_syntheticChildrenName.AsCString(),
                  m_legacy_emptyStorage_mangled_ObjC, flags, false);
#endif // LLDB_DISABLE_PYTHON
}

bool
HashedCollectionConfig::IsNativeStorageName(ConstString name) const {
  assert(m_nativeStorage_demangledPrefix);
  auto n = name.GetStringRef();
  if (n.startswith(m_nativeStorage_demangledPrefix.GetStringRef()))
    return true;
  if (n.startswith(m_legacy_nativeStorage_demangledPrefix.GetStringRef()))
    return true;
  return false;     
}

bool
HashedCollectionConfig::IsEmptyStorageName(ConstString name) const {
  assert(m_emptyStorage_demangled);
  return name == m_emptyStorage_demangled || name == m_legacy_emptyStorage_demangled;
}

bool
HashedCollectionConfig::IsDeferredBridgedStorageName(ConstString name) const {
  assert(m_deferredBridgedStorage_demangledPrefix);
  auto n = name.GetStringRef();
  return n.startswith(m_deferredBridgedStorage_demangledPrefix.GetStringRef());
}

HashedStorageHandlerUP
HashedCollectionConfig::CreateEmptyHandler(CompilerType elem_type) const {
  return HashedStorageHandlerUP(new EmptyHashedStorageHandler(elem_type));
}

ValueObjectSP
HashedCollectionConfig::SwiftObjectAtAddress(
    const ExecutionContext &exe_ctx,
    lldb::addr_t address) const {

  if (address == LLDB_INVALID_ADDRESS)
    return nullptr;

  ProcessSP process_sp = exe_ctx.GetProcessSP();
  if (!process_sp)
    return nullptr;
  
  // Create a ValueObject with a Swift AnyObject type referencing the
  // same address.
  Status error;
  ExecutionContextScope *exe_scope = exe_ctx.GetBestExecutionContextScope();
  auto reader =
    process_sp->GetTarget().GetScratchSwiftASTContext(error, *exe_scope);
  SwiftASTContext *ast_ctx = reader.get();
  if (!ast_ctx)
    return nullptr;
  if (error.Fail())
    return nullptr;

  CompilerType anyObject_type = ast_ctx->GetAnyObjectType();
  if (!anyObject_type)
    return nullptr;

  lldb::DataBufferSP buffer(
    new lldb_private::DataBufferHeap(&address, sizeof(lldb::addr_t)));
  return ValueObjectConstResult::Create(
    exe_scope, anyObject_type, ConstString("swift"),
    buffer, exe_ctx.GetByteOrder(), exe_ctx.GetAddressByteSize());
}

ValueObjectSP
HashedCollectionConfig::CocoaObjectAtAddress(
  const ExecutionContext &exe_ctx,
  lldb::addr_t address) const {

  if (address == LLDB_INVALID_ADDRESS)
    return nullptr;
  ProcessSP process_sp = exe_ctx.GetProcessSP();
  if (!process_sp)
    return nullptr;
  CompilerType id = exe_ctx.GetTargetSP()
    ->GetScratchClangASTContext()
    ->GetBasicType(lldb::eBasicTypeObjCID);
  InferiorSizedWord isw(address, *process_sp);
  return ValueObject::CreateValueObjectFromData(
    "cocoa", isw.GetAsData(process_sp->GetByteOrder()), exe_ctx, id);
}

HashedStorageHandlerUP
HashedCollectionConfig::_CreateNativeHandler(
  lldb::ValueObjectSP storage_sp,
  CompilerType key_type,
  CompilerType value_type) const {
  auto handler = HashedStorageHandlerUP(
    new NativeHashedStorageHandler(storage_sp, key_type, value_type));
  if (!handler->IsValid())
    return nullptr;
  return handler;  
}

HashedStorageHandlerUP
HashedCollectionConfig::_CreateLegacyNativeHandler(
  lldb::ValueObjectSP storage_sp,
  CompilerType key_type,
  CompilerType value_type) const {
  auto handler = HashedStorageHandlerUP(
    new LegacyNativeHashedStorageHandler(storage_sp, key_type, value_type));
  if (!handler->IsValid())
    return nullptr;
  return handler;  
}

HashedStorageHandlerUP
HashedCollectionConfig::CreateNativeHandler(
  ValueObjectSP value_sp,
  ValueObjectSP storage_sp) const {
  if (!storage_sp)
    return nullptr;

  // To prevent reading uninitialized data, first try to get the
  // runtime class of storage_sp and verify that it's of a known type.
  // If thissuccessful, get the correct key_type and value_type directly
  // from its generic arguments instead of using value_sp.
  storage_sp = storage_sp->GetQualifiedRepresentationIfAvailable(
    lldb::eDynamicCanRunTarget, false);

  auto type = storage_sp->GetCompilerType();

  auto typeName = type.GetTypeName().GetStringRef();
  if (typeName == m_emptyStorage_demangled.GetStringRef()) {
    return CreateEmptyHandler();
  }
  
  if (typeName.startswith(m_nativeStorage_demangledPrefix.GetStringRef())) {
    auto key_type = type.GetGenericArgumentType(0);
    auto value_type = type.GetGenericArgumentType(1);
    if (key_type.IsValid()) {
      return _CreateNativeHandler(storage_sp, key_type, value_type);
    }
  }

  if (typeName.startswith(m_legacy_nativeStorage_demangledPrefix.GetStringRef())) {
    auto key_type = type.GetGenericArgumentType(0);
    auto value_type = type.GetGenericArgumentType(1);
    if (key_type.IsValid()) {
      return _CreateLegacyNativeHandler(storage_sp, key_type, value_type);
    }
  }    
  
  // Fallback: If we couldn't get the dynamic type, assume storage_sp
  // is some valid storage class instance, and attempt to get
  // key/value types from value_sp.
  type = value_sp->GetCompilerType();
  CompilerType key_type = type.GetGenericArgumentType(0);
  CompilerType value_type = type.GetGenericArgumentType(1);
  if (typeName == m_legacy_emptyStorage_demangled.GetStringRef()) {
    // In Swift 4.2, the empty storage singleton had the storage root
    // class as its dynamic type.
    return _CreateLegacyNativeHandler(storage_sp, key_type, value_type);
  }
  if (typeName == m_nativeStorageRoot_demangled.GetStringRef()) {
    return _CreateNativeHandler(storage_sp, key_type, value_type);
  }
  return nullptr;
}

HashedStorageHandlerUP
HashedCollectionConfig::CreateCocoaHandler(ValueObjectSP storage_sp) const {
  auto cocoaChildrenCreator = GetCocoaSyntheticChildrenCreator();
  auto frontend = cocoaChildrenCreator(nullptr, storage_sp);
  if (!frontend) {
    return nullptr;
  }
  // Cocoa frontends must be updated before use
  frontend->Update();
  auto handler = HashedStorageHandlerUP(
    new CocoaHashedStorageHandler(storage_sp, frontend));
  if (!handler->IsValid())
    return nullptr;
  return handler;
}

//===----------------------------------------------------------------------===//

NativeHashedStorageHandler::NativeHashedStorageHandler(
  ValueObjectSP nativeStorage_sp,
  CompilerType key_type,
  CompilerType value_type
) : m_storage(nativeStorage_sp.get()),
    m_process(nullptr),
    m_ptr_size(0),
    m_count(0),
    m_scale(0),
    m_metadata_ptr(LLDB_INVALID_ADDRESS),
    m_keys_ptr(LLDB_INVALID_ADDRESS),
    m_values_ptr(LLDB_INVALID_ADDRESS),
    m_element_type(),
    m_key_stride(key_type.GetByteStride()),
    m_value_stride(0),
    m_key_stride_padded(m_key_stride),
    m_occupiedBuckets(),
    m_failedToGetBuckets(false) {
  static ConstString g__count("_count");
  static ConstString g__scale("_scale");
  static ConstString g__rawElements("_rawElements");
  static ConstString g__rawKeys("_rawKeys");
  static ConstString g__rawValues("_rawValues");

  static ConstString g__value("_value");
  static ConstString g__rawValue("_rawValue");

  static ConstString g_key("key");
  static ConstString g_value("value");

  if (!m_storage)
    return;
  if (!key_type)
    return;

  if (value_type) {
    m_value_stride = value_type.GetByteStride();
    if (SwiftASTContext *swift_ast =
            llvm::dyn_cast_or_null<SwiftASTContext>(key_type.GetTypeSystem())) {
      std::vector<SwiftASTContext::TupleElement> tuple_elements{
          {g_key, key_type}, {g_value, value_type}};
      m_element_type = swift_ast->CreateTupleType(tuple_elements);
      m_key_stride_padded = m_element_type.GetByteStride() - m_value_stride;
    }
  } else {
    m_element_type = key_type;
  }

  if (!m_element_type)
    return;

  m_process = m_storage->GetProcessSP().get();
  if (!m_process)
    return;

  m_ptr_size = m_process->GetAddressByteSize();

  auto count_sp = m_storage->GetChildAtNamePath({g__count, g__value});
  if (!count_sp)
    return;
  m_count = count_sp->GetValueAsUnsigned(0);

  auto scale_sp = m_storage->GetChildAtNamePath({g__scale, g__value});
  if (!scale_sp)
    return;
  auto scale = scale_sp->GetValueAsUnsigned(0);
  if (scale > m_ptr_size)
    return;
  m_scale = scale;

  auto keys_ivar = value_type ? g__rawKeys : g__rawElements;
  auto keys_sp = m_storage->GetChildAtNamePath({keys_ivar, g__rawValue});
  if (!keys_sp)
    return;
  m_keys_ptr = keys_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);

  auto last_field_ptr = keys_sp->GetAddressOf();

  if (value_type) {
    auto values_sp = m_storage->GetChildAtNamePath({g__rawValues, g__rawValue});
    if (!values_sp)
      return;
    m_values_ptr = values_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
    last_field_ptr = values_sp->GetAddressOf();
  }

  m_metadata_ptr = last_field_ptr + m_ptr_size;

  // Make sure we can read the first and last word of the bitmap.  
  if (IsValid()) {
    Status error;
    GetMetadataWord(0, error);
    GetMetadataWord(GetWordCount() - 1, error);
    if (error.Fail()) {
      m_metadata_ptr = LLDB_INVALID_ADDRESS;
    }
  }
}

bool NativeHashedStorageHandler::IsValid() {
  return (m_storage != nullptr)
    && (m_process != nullptr)
    && (m_ptr_size > 0)
    && (m_element_type.IsValid())
    && (m_metadata_ptr != LLDB_INVALID_ADDRESS)
    && (m_keys_ptr != LLDB_INVALID_ADDRESS)
    && (m_value_stride == 0 || m_values_ptr != LLDB_INVALID_ADDRESS)
    // Check counts.
    && (m_count <= GetBucketCount())
    // Buffers are tail-allocated in this order: metadata, keys, values
    && (m_metadata_ptr < m_keys_ptr)
    && (m_value_stride == 0 || m_keys_ptr < m_values_ptr);
}

uint64_t
NativeHashedStorageHandler::GetMetadataWord(int index, Status &error) {
  if (index >= GetWordCount()) {
    error.SetErrorToGenericError();
    return 0;
  }
  const lldb::addr_t effective_ptr = m_metadata_ptr + (index * m_ptr_size);
  uint64_t data = m_process->ReadUnsignedIntegerFromMemory(
    effective_ptr, m_ptr_size,
    0, error);
  return data;
}

bool
NativeHashedStorageHandler::FailBuckets() {
  m_failedToGetBuckets = true;
  std::vector<Bucket>().swap(m_occupiedBuckets);
  return false;
}

bool
NativeHashedStorageHandler::UpdateBuckets() {
  if (m_failedToGetBuckets)
    return false;
  if (!m_occupiedBuckets.empty())
    return true;
  // Scan bitmap for occupied buckets.
  m_occupiedBuckets.reserve(m_count);
  size_t bucketCount = GetBucketCount();
  size_t wordWidth = GetWordWidth();
  size_t wordCount = GetWordCount();
  for (size_t wordIndex = 0; wordIndex < wordCount; wordIndex++) {
    Status error;
    auto word = GetMetadataWord(wordIndex, error);
    if (error.Fail()) {
      return FailBuckets();
    }
    if (wordCount == 1) {
      // Mask off out-of-bounds bits from first partial word.
      word &= (1 << bucketCount) - 1;
    }
    for (size_t bit = 0; bit < wordWidth; bit++) {
      if ((word & (1ULL << bit)) != 0) {
        if (m_occupiedBuckets.size() == m_count) {
          return FailBuckets();
        }
        m_occupiedBuckets.push_back(wordIndex * wordWidth + bit);
      }
    }
  }
  if (m_occupiedBuckets.size() != m_count) {
    return FailBuckets();
  }
  return true;
}

ValueObjectSP
NativeHashedStorageHandler::GetElementAtIndex(size_t idx) {
  if (!UpdateBuckets())
    return nullptr;
  if (!IsValid())
    return nullptr;
  if (idx >= m_occupiedBuckets.size())
    return nullptr;
  Bucket bucket = m_occupiedBuckets[idx];
  DataBufferSP full_buffer_sp(
    new DataBufferHeap(m_key_stride_padded + m_value_stride, 0));
  uint8_t *key_buffer_ptr = full_buffer_sp->GetBytes();
  uint8_t *value_buffer_ptr =
    m_value_stride ? (key_buffer_ptr + m_key_stride_padded) : nullptr;
  if (!GetDataForKeyInBucket(bucket, key_buffer_ptr))
    return nullptr;
  if (value_buffer_ptr != nullptr &&
      !GetDataForValueInBucket(bucket, value_buffer_ptr))
    return nullptr;
  DataExtractor full_data;
  full_data.SetData(full_buffer_sp);
  StreamString name;
  name.Printf("[%zu]", idx);
  return ValueObjectConstResult::Create(
    m_process, m_element_type, ConstString(name.GetData()), full_data);
}

//===----------------------------------------------------------------------===//

ValueObjectSP
LegacyNativeHashedStorageHandler::GetElementAtIndex(size_t idx) {
  ValueObjectSP null_valobj_sp;
  if (idx >= m_count)
    return null_valobj_sp;
  if (!IsValid())
    return null_valobj_sp;
  int64_t found_idx = -1;
  Status error;
  for (Cell cell_idx = 0; cell_idx < m_bucketCount; cell_idx++) {
    const bool used = ReadBitmaskAtIndex(cell_idx, error);
    if (error.Fail()) {
      Status bitmask_error;
      bitmask_error.SetErrorStringWithFormat(
              "Failed to read bit-mask index from Dictionary: %s",
              error.AsCString());
      return ValueObjectConstResult::Create(m_process, bitmask_error);
    }
    if (!used)
      continue;
    if (++found_idx == idx) {
      // you found it!!!
      DataBufferSP full_buffer_sp(
          new DataBufferHeap(m_key_stride_padded + m_value_stride, 0));
      uint8_t *key_buffer_ptr = full_buffer_sp->GetBytes();
      uint8_t *value_buffer_ptr =
          m_value_stride ? (key_buffer_ptr + m_key_stride_padded) : nullptr;
      if (GetDataForKeyAtCell(cell_idx, key_buffer_ptr) &&
          (value_buffer_ptr == nullptr ||
           GetDataForValueAtCell(cell_idx, value_buffer_ptr))) {
        DataExtractor full_data;
        full_data.SetData(full_buffer_sp);
        StreamString name;
        name.Printf("[%zu]", idx);
        return ValueObjectConstResult::Create(
            m_process, m_element_type, ConstString(name.GetData()), full_data);
      }
    }
  }
  return null_valobj_sp;
}

bool LegacyNativeHashedStorageHandler::ReadBitmaskAtIndex(Index i, Status &error) {
  if (i >= m_bucketCount)
    return false;
  const size_t word = i / (8 * m_ptr_size);
  const size_t offset = i % (8 * m_ptr_size);
  const lldb::addr_t effective_ptr = m_bitmask_ptr + (word * m_ptr_size);
  uint64_t data = 0;

  auto cached = m_bitmask_cache.find(effective_ptr);
  if (cached != m_bitmask_cache.end()) {
    data = cached->second;
  } else {
    data = m_process->ReadUnsignedIntegerFromMemory(effective_ptr, m_ptr_size,
                                                    0, error);
    if (error.Fail())
      return false;
    m_bitmask_cache[effective_ptr] = data;
  }

  const uint64_t mask = static_cast<uint64_t>(1UL << offset);
  const uint64_t value = (data & mask);
  return (0 != value);
}

LegacyNativeHashedStorageHandler::LegacyNativeHashedStorageHandler(
  ValueObjectSP nativeStorage_sp,
  CompilerType key_type,
  CompilerType value_type
) : m_nativeStorage(nativeStorage_sp.get()), m_process(nullptr),
    m_ptr_size(0), m_count(0), m_bucketCount(0),
    m_bitmask_ptr(LLDB_INVALID_ADDRESS), m_keys_ptr(LLDB_INVALID_ADDRESS),
    m_values_ptr(LLDB_INVALID_ADDRESS), m_element_type(),
    m_key_stride(key_type.GetByteStride()), m_value_stride(0),
    m_key_stride_padded(m_key_stride), m_bitmask_cache() {
  static ConstString g_initializedEntries("initializedEntries");
  static ConstString g_values("values");
  static ConstString g__rawValue("_rawValue");
  static ConstString g_keys("keys");

  static ConstString g_key("key");
  static ConstString g_value("value");
  static ConstString g__value("_value");

  static ConstString g_capacity("capacity");
  static ConstString g_bucketCount("bucketCount");
  static ConstString g_count("count");

  if (!m_nativeStorage)
    return;
  if (!key_type)
    return;

  if (value_type) {
    m_value_stride = value_type.GetByteStride();
    if (SwiftASTContext *swift_ast =
            llvm::dyn_cast_or_null<SwiftASTContext>(key_type.GetTypeSystem())) {
      std::vector<SwiftASTContext::TupleElement> tuple_elements{
          {g_key, key_type}, {g_value, value_type}};
      m_element_type = swift_ast->CreateTupleType(tuple_elements);
      m_key_stride_padded = m_element_type.GetByteStride() - m_value_stride;
    }
  } else {
    m_element_type = key_type;
  }

  if (!m_element_type)
    return;

  m_process = m_nativeStorage->GetProcessSP().get();
  if (!m_process)
    return;

  m_ptr_size = m_process->GetAddressByteSize();

  auto bucketCount_sp =
    m_nativeStorage->GetChildAtNamePath({g_bucketCount, g__value});
  if (!bucketCount_sp) // <4.1: bucketCount was called capacity.
    bucketCount_sp = m_nativeStorage->GetChildAtNamePath({g_capacity, g__value});
  if (!bucketCount_sp)
    return;
  m_bucketCount = bucketCount_sp->GetValueAsUnsigned(0);
  auto count_sp = m_nativeStorage->GetChildAtNamePath({g_count, g__value});
  if (!count_sp)
    return;
  m_count = count_sp->GetValueAsUnsigned(0);

  m_nativeStorage = nativeStorage_sp.get();
  m_bitmask_ptr =
      m_nativeStorage
          ->GetChildAtNamePath({g_initializedEntries, g_values, g__rawValue})
          ->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);

  if (ValueObjectSP value_child_sp =
          m_nativeStorage->GetChildAtNamePath({g_values, g__rawValue})) {
    // it is fine not to pass a value_type, but if the value child exists, then
    // you have to pass one
    if (!value_type)
      return;
    m_values_ptr = value_child_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
  }
  m_keys_ptr = m_nativeStorage->GetChildAtNamePath({g_keys, g__rawValue})
                   ->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
  // Make sure we can read the bitmask at the ount index.  
  // and this will keep us from trying
  // to reconstruct many bajillions of invalid children.
  // Don't bother if the native buffer handler is invalid already, however. 
  if (IsValid())
  {
    Status error;
    ReadBitmaskAtIndex(m_bucketCount - 1, error);
    if (error.Fail())
    {
      m_bitmask_ptr = LLDB_INVALID_ADDRESS;
    }
  }
}

bool LegacyNativeHashedStorageHandler::IsValid() {
  return (m_nativeStorage != nullptr) && (m_process != nullptr) &&
         m_element_type.IsValid() && m_bitmask_ptr != LLDB_INVALID_ADDRESS &&
         m_keys_ptr != LLDB_INVALID_ADDRESS &&
         /*m_values_ptr != LLDB_INVALID_ADDRESS && you can't check values
            because some containers have only keys*/
         // The bucket count must be a power of two.
         m_bucketCount >= 1 && (m_bucketCount & (m_bucketCount - 1)) == 0 &&
         m_bucketCount >= m_count;
}

//===----------------------------------------------------------------------===//

HashedStorageHandlerUP
HashedCollectionConfig::CreateHandler(ValueObject &valobj) const {
  static ConstString g__variant("_variant"); // Swift 5
  static ConstString g__variantBuffer("_variantBuffer"); // Swift 4
  static ConstString g_native("native");
  static ConstString g_cocoa("cocoa");
  static ConstString g_nativeBuffer("nativeBuffer"); // Swift 4
  static ConstString g__storage("_storage");

  Status error;

  ValueObjectSP valobj_sp = valobj.GetSP();
  if (valobj_sp->GetObjectRuntimeLanguage() != eLanguageTypeSwift &&
      valobj_sp->IsPointerType()) {
    if (auto swiftval_sp = SwiftObjectAtAddress(
          valobj_sp->GetExecutionContextRef(),
          valobj_sp->GetPointerValue()))
      valobj_sp = swiftval_sp;
  }
  valobj_sp = valobj_sp->GetQualifiedRepresentationIfAvailable(
    lldb::eDynamicCanRunTarget, false);
  ConstString type_name_cs(valobj_sp->GetTypeName());

  if (IsNativeStorageName(type_name_cs)) {
    return CreateNativeHandler(valobj_sp, valobj_sp);
  }
  if (IsEmptyStorageName(type_name_cs)) {
    return CreateEmptyHandler();
  }
  if (IsDeferredBridgedStorageName(type_name_cs)) {
    auto storage_sp = valobj_sp->GetChildAtNamePath({g_native, g__storage});
    if (!storage_sp) // try Swift 4 name
      storage_sp = valobj_sp->GetChildAtNamePath({g_nativeBuffer, g__storage});
    return CreateNativeHandler(valobj_sp, storage_sp);
  }

  ValueObjectSP variant_sp =
    valobj_sp->GetChildMemberWithName(g__variant, true);
  if (!variant_sp) // try Swift 4 name
    variant_sp = valobj_sp->GetChildMemberWithName(g__variantBuffer, true);
  if (!variant_sp)
    return nullptr;

  ConstString variant_cs(variant_sp->GetValueAsCString());
  if (!variant_cs)
    return nullptr;

  if (g_cocoa == variant_cs) {
    // it's an NSDictionary/NSSet in disguise
    static ConstString g_object("object"); // Swift 5
    static ConstString g_cocoaDictionary("cocoaDictionary"); // Swift 4
    static ConstString g_cocoaSet("cocoaSet"); // Swift 4
    
    ValueObjectSP child_sp =
      variant_sp->GetChildAtNamePath({g_cocoa, g_object});
    if (!child_sp) // try Swift 4 name for dictionaries
      child_sp = variant_sp->GetChildAtNamePath({g_cocoa, g_cocoaDictionary});
    if (!child_sp) // try Swift 4 name for sets
      child_sp = variant_sp->GetChildAtNamePath({g_cocoa, g_cocoaSet});
    if (!child_sp)
      return nullptr;
    // child_sp is the _NSDictionary/_NSSet reference.
    ValueObjectSP ref_sp = child_sp->GetChildAtIndex(0, true); // instance
    if (!ref_sp)
      return nullptr;

    uint64_t cocoa_ptr = ref_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
    if (cocoa_ptr == LLDB_INVALID_ADDRESS)
      return nullptr;
    // FIXME: for some reason I need to zero out the MSB; figure out why
    cocoa_ptr &= 0x00FFFFFFFFFFFFFF;

    auto cocoa_sp = CocoaObjectAtAddress(valobj_sp->GetExecutionContextRef(),
                                         cocoa_ptr);
    if (!cocoa_sp)
      return nullptr;
    return CreateCocoaHandler(cocoa_sp);
  }
  if (g_native == variant_cs) {
    auto storage_sp = variant_sp->GetChildAtNamePath({g_native, g__storage});
    return CreateNativeHandler(valobj_sp, storage_sp);
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//

HashedSyntheticChildrenFrontEnd::HashedSyntheticChildrenFrontEnd(
  const HashedCollectionConfig &config,
  ValueObjectSP valobj_sp
) : SyntheticChildrenFrontEnd(*valobj_sp.get()),
    m_config(config),
    m_buffer()
{}

size_t
HashedSyntheticChildrenFrontEnd::CalculateNumChildren() {
  return m_buffer ? m_buffer->GetCount() : 0;
}

ValueObjectSP
HashedSyntheticChildrenFrontEnd::GetChildAtIndex(size_t idx) {
  if (!m_buffer)
    return ValueObjectSP();

  ValueObjectSP child_sp = m_buffer->GetElementAtIndex(idx);

  if (child_sp)
    child_sp->SetSyntheticChildrenGenerated(true);

  return child_sp;
}

bool
HashedSyntheticChildrenFrontEnd::Update() {
  m_buffer = m_config.CreateHandler(m_backend);
  return false;
}

bool
HashedSyntheticChildrenFrontEnd::MightHaveChildren() {
  return true;
}

size_t
HashedSyntheticChildrenFrontEnd::GetIndexOfChildWithName(
  const ConstString &name
) {
  if (!m_buffer)
    return UINT32_MAX;
  const char *item_name = name.GetCString();
  uint32_t idx = ExtractIndexFromString(item_name);
  if (idx < UINT32_MAX && idx >= CalculateNumChildren())
    return UINT32_MAX;
  return idx;
}
