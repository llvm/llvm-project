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
  uint64_t m_capacity;
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
#endif // LLDB_DISABLE_PYTHON
}

bool
HashedCollectionConfig::IsNativeStorageName(ConstString name) const {
  assert(m_nativeStorage_demangledPrefix);
  auto n = name.GetStringRef();
  return n.startswith(m_nativeStorage_demangledPrefix.GetStringRef());
}

bool
HashedCollectionConfig::IsEmptyStorageName(ConstString name) const {
  assert(m_emptyStorage_demangled);
  return name == m_emptyStorage_demangled;
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

HashedStorageHandlerUP
HashedCollectionConfig::CreateNativeHandler(
  lldb::ValueObjectSP storage_sp,
  CompilerType key_type,
  CompilerType value_type
) const {
  return HashedStorageHandlerUP(
    new NativeHashedStorageHandler(storage_sp, key_type, value_type));
}

HashedStorageHandlerUP
HashedCollectionConfig::CreateCocoaHandler(
  lldb::ValueObjectSP cocoaStorage_sp
) const {
  auto cocoaChildrenCreator = GetCocoaSyntheticChildrenCreator();
  auto frontend = cocoaChildrenCreator(nullptr, cocoaStorage_sp);
  if (!frontend) {
    return nullptr;
  }
  // Cocoa frontends must be updated before use
  frontend->Update();
  return HashedStorageHandlerUP(
    new CocoaHashedStorageHandler(cocoaStorage_sp, frontend));
}

ValueObjectSP
NativeHashedStorageHandler::GetElementAtIndex(size_t idx) {
  ValueObjectSP null_valobj_sp;
  if (idx >= m_count)
    return null_valobj_sp;
  if (!IsValid())
    return null_valobj_sp;
  int64_t found_idx = -1;
  Status error;
  for (Cell cell_idx = 0; cell_idx < m_capacity; cell_idx++) {
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

bool NativeHashedStorageHandler::ReadBitmaskAtIndex(Index i, Status &error) {
  if (i >= m_capacity)
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

NativeHashedStorageHandler::NativeHashedStorageHandler(
  ValueObjectSP nativeStorage_sp,
  CompilerType key_type,
  CompilerType value_type
) : m_nativeStorage(nativeStorage_sp.get()), m_process(nullptr),
    m_ptr_size(0), m_count(0), m_capacity(0),
    m_bitmask_ptr(LLDB_INVALID_ADDRESS), m_keys_ptr(LLDB_INVALID_ADDRESS),
    m_values_ptr(LLDB_INVALID_ADDRESS), m_element_type(),
    m_key_stride(key_type.GetByteStride()), m_value_stride(0),
    m_key_stride_padded(m_key_stride), m_bitmask_cache() {
  static ConstString g_initializedEntries("initializedEntries");
  static ConstString g_values("values");
  static ConstString g__rawValue("_rawValue");
  static ConstString g_keys("keys");
  static ConstString g_buffer("buffer");

  static ConstString g_key("key");
  static ConstString g_value("value");
  static ConstString g__value("_value");

  static ConstString g_capacity("bucketCount");
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
  } else
    m_element_type = key_type;

  if (!m_element_type)
    return;

  m_process = m_nativeStorage->GetProcessSP().get();
  if (!m_process)
    return;

  m_ptr_size = m_process->GetAddressByteSize();

  auto buffer_sp = m_nativeStorage->GetChildAtNamePath({g_buffer});
  if (buffer_sp) {
    auto buffer_ptr = buffer_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
    if (buffer_ptr == 0 || buffer_ptr == LLDB_INVALID_ADDRESS)
      return;

    Status error;
    m_capacity =
        m_process->ReadPointerFromMemory(buffer_ptr + 2 * m_ptr_size, error);
    if (error.Fail())
      return;
    m_count =
        m_process->ReadPointerFromMemory(buffer_ptr + 3 * m_ptr_size, error);
    if (error.Fail())
      return;
  } else {
    auto capacity_sp =
        m_nativeStorage->GetChildAtNamePath({g_capacity, g__value});
    if (!capacity_sp)
      return;
    m_capacity = capacity_sp->GetValueAsUnsigned(0);
    auto count_sp = m_nativeStorage->GetChildAtNamePath({g_count, g__value});
    if (!count_sp)
      return;
    m_count = count_sp->GetValueAsUnsigned(0);
  }

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
    ReadBitmaskAtIndex(m_capacity - 1, error);
    if (error.Fail())
    {
      m_bitmask_ptr = LLDB_INVALID_ADDRESS;
    }
  }
}

bool NativeHashedStorageHandler::IsValid() {
  return (m_nativeStorage != nullptr) && (m_process != nullptr) &&
         m_element_type.IsValid() && m_bitmask_ptr != LLDB_INVALID_ADDRESS &&
         m_keys_ptr != LLDB_INVALID_ADDRESS &&
         /*m_values_ptr != LLDB_INVALID_ADDRESS && you can't check values
            because some containers have only keys*/
         m_capacity >= m_count;
}

HashedStorageHandlerUP
HashedCollectionConfig::CreateNativeStorageHandler(
  ValueObject &valobj,
  lldb::addr_t storage_ptr,
  bool fail_on_no_children
) const {
  CompilerType valobj_type(valobj.GetCompilerType());
  CompilerType key_type = valobj_type.GetGenericArgumentType(0);
  CompilerType value_type = valobj_type.GetGenericArgumentType(1);

  static ConstString g_Native("native");
  static ConstString g_nativeStorage("nativeStorage");
  static ConstString g_buffer("buffer");

  Status error;

  ProcessSP process_sp(valobj.GetProcessSP());
  if (!process_sp)
    return nullptr;

  ValueObjectSP native_sp(valobj.GetChildAtNamePath({g_nativeStorage}));
  ValueObjectSP native_buffer_sp(
      valobj.GetChildAtNamePath({g_nativeStorage, g_buffer}));
  if (!native_sp || !native_buffer_sp) {
    if (fail_on_no_children)
      return nullptr;
    else {
      lldb::addr_t native_storage_ptr =
          storage_ptr + (3 * process_sp->GetAddressByteSize());
      native_storage_ptr =
          process_sp->ReadPointerFromMemory(native_storage_ptr, error);
      // (AnyObject,AnyObject)?
      ExecutionContext exe_ctx(process_sp);
      ExecutionContextScope *exe_scope = exe_ctx.GetBestExecutionContextScope();
      auto reader =
          process_sp->GetTarget().GetScratchSwiftASTContext(error, *exe_scope);
      SwiftASTContext *swift_ast_ctx = reader.get();
      if (swift_ast_ctx) {
        CompilerType element_type(swift_ast_ctx->GetTypeFromMangledTypename(
            SwiftLanguageRuntime::GetCurrentMangledName(
                "_TtGSqTPs9AnyObject_PS____")
                .c_str(),
            error));
        auto handler = CreateNativeHandler(native_sp, key_type, value_type);
        if (handler && handler->IsValid())
          return handler;
      }
      return nullptr;
    }
    }

    CompilerType child_type(native_sp->GetCompilerType());
    CompilerType element_type(child_type.GetGenericArgumentType(1));
    if (element_type.IsValid() == false ||
        child_type.GetGenericArgumentKind(1) != lldb::eBoundGenericKindType)
      return nullptr;
    lldb::addr_t native_storage_ptr = process_sp->ReadPointerFromMemory(storage_ptr + 2*process_sp->GetAddressByteSize(), error);
    if (error.Fail() || native_storage_ptr == LLDB_INVALID_ADDRESS)
        return nullptr;
    auto handler = CreateNativeHandler(native_sp, key_type, value_type);
    if (handler && handler->IsValid())
        return handler;

  return nullptr;
}

HashedStorageHandlerUP
HashedCollectionConfig::CreateHandler(ValueObject &valobj) const {
  static ConstString g__variantStorage("_variantStorage");
  static ConstString g__variantBuffer("_variantBuffer");
  static ConstString g_Native("native");
  static ConstString g_Cocoa("cocoa");
  static ConstString g_nativeStorage("nativeStorage");
  static ConstString g_nativeBuffer("nativeBuffer");
  static ConstString g_buffer("buffer");
  static ConstString g_storage("storage");
  static ConstString g__storage("_storage");
  static ConstString g_Some("some");

  Status error;

  ProcessSP process_sp(valobj.GetProcessSP());
  if (!process_sp)
    return nullptr;

  ConstString type_name_cs(valobj.GetTypeName());

  if (IsNativeStorageName(type_name_cs)) {
    return CreateNativeStorageHandler(valobj, valobj.GetPointerValue(), false);
  }
  if (IsEmptyStorageName(type_name_cs)) {
    return CreateEmptyHandler();
  }

  ValueObjectSP valobj_sp =
      valobj.GetSP()->GetQualifiedRepresentationIfAvailable(
          lldb::eDynamicCanRunTarget, false);

  ValueObjectSP _variantStorageSP(
      valobj_sp->GetChildMemberWithName(g__variantStorage, true));

  if (!_variantStorageSP)
    _variantStorageSP =
        valobj_sp->GetChildMemberWithName(g__variantBuffer, true);

  if (!_variantStorageSP) {
    if (IsDeferredBridgedStorageName(type_name_cs)) {
      ValueObjectSP storage_sp(
          valobj_sp->GetChildAtNamePath({g_nativeBuffer, g__storage}));
      if (storage_sp) {
        CompilerType child_type(valobj_sp->GetCompilerType());
        CompilerType key_type(child_type.GetGenericArgumentType(0));
        CompilerType value_type(child_type.GetGenericArgumentType(1));

        auto handler = CreateNativeHandler(storage_sp, key_type, value_type);
        if (handler && handler->IsValid())
          return handler;
      }
    }
    return nullptr;
  }

  ConstString storage_kind(_variantStorageSP->GetValueAsCString());

  if (!storage_kind)
    return nullptr;

  if (g_Cocoa == storage_kind) {
    ValueObjectSP child_sp(
        _variantStorageSP->GetChildMemberWithName(g_Native, true));
    if (!child_sp)
      return nullptr;
    // it's an NSDictionary in disguise
    uint64_t cocoa_storage_ptr =
        child_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
    if (cocoa_storage_ptr == LLDB_INVALID_ADDRESS || error.Fail())
      return nullptr;
    cocoa_storage_ptr &= 0x00FFFFFFFFFFFFFF; // for some reason I need to zero
                                             // out the MSB; figure out why
                                             // later
    CompilerType id =
        process_sp->GetTarget().GetScratchClangASTContext()->GetBasicType(
            lldb::eBasicTypeObjCID);
    InferiorSizedWord isw(cocoa_storage_ptr, *process_sp);
    ValueObjectSP cocoarr_sp = ValueObject::CreateValueObjectFromData(
        "cocoarr", isw.GetAsData(process_sp->GetByteOrder()),
        valobj.GetExecutionContextRef(), id);
    if (!cocoarr_sp)
      return nullptr;
    auto objc_runtime = process_sp->GetObjCLanguageRuntime();
    auto descriptor_sp = objc_runtime->GetClassDescriptor(*cocoarr_sp);
    if (!descriptor_sp)
      return nullptr;
    ConstString classname(descriptor_sp->GetClassName());
    if (IsNativeStorageName(classname)) {
      return CreateNativeStorageHandler(
        *_variantStorageSP, cocoa_storage_ptr, true);
    } else if (IsEmptyStorageName(classname)) {
      return CreateEmptyHandler();
    } else {
      auto handler = CreateCocoaHandler(cocoarr_sp);
      if (handler && handler->IsValid())
        return handler;
      return nullptr;
    }
  }
  if (g_Native == storage_kind) {
    ValueObjectSP native_sp(_variantStorageSP->GetChildAtNamePath({g_Native}));
    ValueObjectSP nativeStorage_sp(
        _variantStorageSP->GetChildAtNamePath({g_Native, g_nativeStorage}));
    if (!native_sp)
      return nullptr;
    if (!nativeStorage_sp)
      nativeStorage_sp =
          _variantStorageSP->GetChildAtNamePath({g_Native, g__storage});
    if (!nativeStorage_sp)
      return nullptr;

    CompilerType child_type(valobj.GetCompilerType());
    CompilerType key_type(child_type.GetGenericArgumentType(0));
    CompilerType value_type(child_type.GetGenericArgumentType(1));

    auto handler = CreateNativeHandler(nativeStorage_sp, key_type, value_type);
    if (handler && handler->IsValid())
      return handler;
    return nullptr;
  }

  return nullptr;
}

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
