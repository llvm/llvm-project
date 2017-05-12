//===-- NSSet.cpp -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "NSSet.h"

#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/Language.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

std::map<ConstString, CXXFunctionSummaryFormat::Callback> &
NSSet_Additionals::GetAdditionalSummaries() {
  static std::map<ConstString, CXXFunctionSummaryFormat::Callback> g_map;
  return g_map;
}

std::map<ConstString, CXXSyntheticChildren::CreateFrontEndCallback> &
NSSet_Additionals::GetAdditionalSynthetics() {
  static std::map<ConstString, CXXSyntheticChildren::CreateFrontEndCallback>
      g_map;
  return g_map;
}

namespace lldb_private {
namespace formatters {
class NSSetISyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  NSSetISyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  ~NSSetISyntheticFrontEnd() override;

  size_t CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(size_t idx) override;

  bool Update() override;

  bool MightHaveChildren() override;

  size_t GetIndexOfChildWithName(const ConstString &name) override;

private:
  struct DataDescriptor_32 {
    uint32_t _used : 26;
    uint32_t _szidx : 6;
  };

  struct DataDescriptor_64 {
    uint64_t _used : 58;
    uint32_t _szidx : 6;
  };

  struct SetItemDescriptor {
    lldb::addr_t item_ptr;
    lldb::ValueObjectSP valobj_sp;
  };

  ExecutionContextRef m_exe_ctx_ref;
  uint8_t m_ptr_size;
  DataDescriptor_32 *m_data_32;
  DataDescriptor_64 *m_data_64;
  lldb::addr_t m_data_ptr;
  std::vector<SetItemDescriptor> m_children;
};

class NSSetMSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  NSSetMSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  ~NSSetMSyntheticFrontEnd() override;

  size_t CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(size_t idx) override;

  bool Update() override;

  bool MightHaveChildren() override;

  size_t GetIndexOfChildWithName(const ConstString &name) override;

private:
  struct DataDescriptor_32 {
    uint32_t _used : 26;
    uint32_t _size;
    uint32_t _mutations;
    uint32_t _objs_addr;
  };

  struct DataDescriptor_64 {
    uint64_t _used : 58;
    uint64_t _size;
    uint64_t _mutations;
    uint64_t _objs_addr;
  };

  struct SetItemDescriptor {
    lldb::addr_t item_ptr;
    lldb::ValueObjectSP valobj_sp;
  };

  ExecutionContextRef m_exe_ctx_ref;
  uint8_t m_ptr_size;
  DataDescriptor_32 *m_data_32;
  DataDescriptor_64 *m_data_64;
  std::vector<SetItemDescriptor> m_children;
};

class NSSetCodeRunningSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  NSSetCodeRunningSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  ~NSSetCodeRunningSyntheticFrontEnd() override;

  size_t CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(size_t idx) override;

  bool Update() override;

  bool MightHaveChildren() override;

  size_t GetIndexOfChildWithName(const ConstString &name) override;
};
} // namespace formatters
} // namespace lldb_private

template <bool cf_style>
bool lldb_private::formatters::NSSetSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  static ConstString g_TypeHint("NSSet");

  ProcessSP process_sp = valobj.GetProcessSP();
  if (!process_sp)
    return false;

  ObjCLanguageRuntime *runtime =
      (ObjCLanguageRuntime *)process_sp->GetLanguageRuntime(
          lldb::eLanguageTypeObjC);

  if (!runtime)
    return false;

  ObjCLanguageRuntime::ClassDescriptorSP descriptor(
      runtime->GetClassDescriptor(valobj));

  if (!descriptor || !descriptor->IsValid())
    return false;

  uint32_t ptr_size = process_sp->GetAddressByteSize();
  bool is_64bit = (ptr_size == 8);

  lldb::addr_t valobj_addr = valobj.GetValueAsUnsigned(0);

  if (!valobj_addr)
    return false;

  uint64_t value = 0;

  ConstString class_name_cs = descriptor->GetClassName();
  const char *class_name = class_name_cs.GetCString();

  if (!class_name || !*class_name)
    return false;

  if (!strcmp(class_name, "__NSSetI")) {
    Status error;
    value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + ptr_size,
                                                      ptr_size, 0, error);
    if (error.Fail())
      return false;
    value &= (is_64bit ? ~0xFC00000000000000UL : ~0xFC000000U);
  } else if (!strcmp(class_name, "__NSSetM")) {
    Status error;
    value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + ptr_size,
                                                      ptr_size, 0, error);
    if (error.Fail())
      return false;
    value &= (is_64bit ? ~0xFC00000000000000UL : ~0xFC000000U);
  }
  /*else if (!strcmp(class_name,"__NSCFSet"))
   {
   Status error;
   value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + (is_64bit ?
   20 : 12), 4, 0, error);
   if (error.Fail())
   return false;
   if (is_64bit)
   value &= ~0x1fff000000000000UL;
   }
   else if (!strcmp(class_name,"NSCountedSet"))
   {
   Status error;
   value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + ptr_size,
   ptr_size, 0, error);
   if (error.Fail())
   return false;
   value = process_sp->ReadUnsignedIntegerFromMemory(value + (is_64bit ? 20 :
   12), 4, 0, error);
   if (error.Fail())
   return false;
   if (is_64bit)
   value &= ~0x1fff000000000000UL;
   }*/
  else {
    auto &map(NSSet_Additionals::GetAdditionalSummaries());
    auto iter = map.find(class_name_cs), end = map.end();
    if (iter != end)
      return iter->second(valobj, stream, options);
    else
      return false;
  }

  std::string prefix, suffix;
  if (Language *language = Language::FindPlugin(options.GetLanguage())) {
    if (!language->GetFormatterPrefixSuffix(valobj, g_TypeHint, prefix,
                                            suffix)) {
      prefix.clear();
      suffix.clear();
    }
  }

  stream.Printf("%s%" PRIu64 " %s%s%s", prefix.c_str(), value, "element",
                value == 1 ? "" : "s", suffix.c_str());
  return true;
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::NSSetSyntheticFrontEndCreator(
    CXXSyntheticChildren *synth, lldb::ValueObjectSP valobj_sp) {
  lldb::ProcessSP process_sp(valobj_sp->GetProcessSP());
  if (!process_sp)
    return nullptr;
  ObjCLanguageRuntime *runtime =
      (ObjCLanguageRuntime *)process_sp->GetLanguageRuntime(
          lldb::eLanguageTypeObjC);
  if (!runtime)
    return nullptr;

  CompilerType valobj_type(valobj_sp->GetCompilerType());
  Flags flags(valobj_type.GetTypeInfo());

  if (flags.IsClear(eTypeIsPointer)) {
    Status error;
    valobj_sp = valobj_sp->AddressOf(error);
    if (error.Fail() || !valobj_sp)
      return nullptr;
  }

  ObjCLanguageRuntime::ClassDescriptorSP descriptor(
      runtime->GetClassDescriptor(*valobj_sp));

  if (!descriptor || !descriptor->IsValid())
    return nullptr;

  ConstString class_name_cs = descriptor->GetClassName();
  const char *class_name = class_name_cs.GetCString();

  if (!class_name || !*class_name)
    return nullptr;

  if (!strcmp(class_name, "__NSSetI")) {
    return (new NSSetISyntheticFrontEnd(valobj_sp));
  } else if (!strcmp(class_name, "__NSSetM")) {
    return (new NSSetMSyntheticFrontEnd(valobj_sp));
  } else {
    auto &map(NSSet_Additionals::GetAdditionalSynthetics());
    auto iter = map.find(class_name_cs), end = map.end();
    if (iter != end)
      return iter->second(synth, valobj_sp);
    return nullptr;
  }
}

lldb_private::formatters::NSSetISyntheticFrontEnd::NSSetISyntheticFrontEnd(
    lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp), m_exe_ctx_ref(), m_ptr_size(8),
      m_data_32(nullptr), m_data_64(nullptr) {
  if (valobj_sp)
    Update();
}

lldb_private::formatters::NSSetISyntheticFrontEnd::~NSSetISyntheticFrontEnd() {
  delete m_data_32;
  m_data_32 = nullptr;
  delete m_data_64;
  m_data_64 = nullptr;
}

size_t
lldb_private::formatters::NSSetISyntheticFrontEnd::GetIndexOfChildWithName(
    const ConstString &name) {
  const char *item_name = name.GetCString();
  uint32_t idx = ExtractIndexFromString(item_name);
  if (idx < UINT32_MAX && idx >= CalculateNumChildren())
    return UINT32_MAX;
  return idx;
}

size_t
lldb_private::formatters::NSSetISyntheticFrontEnd::CalculateNumChildren() {
  if (!m_data_32 && !m_data_64)
    return 0;
  return (m_data_32 ? m_data_32->_used : m_data_64->_used);
}

bool lldb_private::formatters::NSSetISyntheticFrontEnd::Update() {
  m_children.clear();
  delete m_data_32;
  m_data_32 = nullptr;
  delete m_data_64;
  m_data_64 = nullptr;
  m_ptr_size = 0;
  ValueObjectSP valobj_sp = m_backend.GetSP();
  if (!valobj_sp)
    return false;
  if (!valobj_sp)
    return false;
  m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
  Status error;
  if (valobj_sp->IsPointerType()) {
    valobj_sp = valobj_sp->Dereference(error);
    if (error.Fail() || !valobj_sp)
      return false;
  }
  error.Clear();
  lldb::ProcessSP process_sp(valobj_sp->GetProcessSP());
  if (!process_sp)
    return false;
  m_ptr_size = process_sp->GetAddressByteSize();
  uint64_t data_location = valobj_sp->GetAddressOf() + m_ptr_size;
  if (m_ptr_size == 4) {
    m_data_32 = new DataDescriptor_32();
    process_sp->ReadMemory(data_location, m_data_32, sizeof(DataDescriptor_32),
                           error);
  } else {
    m_data_64 = new DataDescriptor_64();
    process_sp->ReadMemory(data_location, m_data_64, sizeof(DataDescriptor_64),
                           error);
  }
  if (error.Fail())
    return false;
  m_data_ptr = data_location + m_ptr_size;
  return false;
}

bool lldb_private::formatters::NSSetISyntheticFrontEnd::MightHaveChildren() {
  return true;
}

lldb::ValueObjectSP
lldb_private::formatters::NSSetISyntheticFrontEnd::GetChildAtIndex(size_t idx) {
  uint32_t num_children = CalculateNumChildren();

  if (idx >= num_children)
    return lldb::ValueObjectSP();

  ProcessSP process_sp = m_exe_ctx_ref.GetProcessSP();
  if (!process_sp)
    return lldb::ValueObjectSP();

  if (m_children.empty()) {
    // do the scan phase
    lldb::addr_t obj_at_idx = 0;

    uint32_t tries = 0;
    uint32_t test_idx = 0;

    while (tries < num_children) {
      obj_at_idx = m_data_ptr + (test_idx * m_ptr_size);
      if (!process_sp)
        return lldb::ValueObjectSP();
      Status error;
      obj_at_idx = process_sp->ReadPointerFromMemory(obj_at_idx, error);
      if (error.Fail())
        return lldb::ValueObjectSP();

      test_idx++;

      if (!obj_at_idx)
        continue;
      tries++;

      SetItemDescriptor descriptor = {obj_at_idx, lldb::ValueObjectSP()};

      m_children.push_back(descriptor);
    }
  }

  if (idx >= m_children.size()) // should never happen
    return lldb::ValueObjectSP();

  SetItemDescriptor &set_item = m_children[idx];
  if (!set_item.valobj_sp) {
    auto ptr_size = process_sp->GetAddressByteSize();
    DataBufferHeap buffer(ptr_size, 0);
    switch (ptr_size) {
    case 0: // architecture has no clue?? - fail
      return lldb::ValueObjectSP();
    case 4:
      *((uint32_t *)buffer.GetBytes()) = (uint32_t)set_item.item_ptr;
      break;
    case 8:
      *((uint64_t *)buffer.GetBytes()) = (uint64_t)set_item.item_ptr;
      break;
    default:
      assert(false && "pointer size is not 4 nor 8 - get out of here ASAP");
    }
    StreamString idx_name;
    idx_name.Printf("[%" PRIu64 "]", (uint64_t)idx);

    DataExtractor data(buffer.GetBytes(), buffer.GetByteSize(),
                       process_sp->GetByteOrder(),
                       process_sp->GetAddressByteSize());

    set_item.valobj_sp = CreateValueObjectFromData(
        idx_name.GetString(), data, m_exe_ctx_ref,
        m_backend.GetCompilerType().GetBasicTypeFromAST(
            lldb::eBasicTypeObjCID));
  }
  return set_item.valobj_sp;
}

lldb_private::formatters::NSSetMSyntheticFrontEnd::NSSetMSyntheticFrontEnd(
    lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp), m_exe_ctx_ref(), m_ptr_size(8),
      m_data_32(nullptr), m_data_64(nullptr) {
  if (valobj_sp)
    Update();
}

lldb_private::formatters::NSSetMSyntheticFrontEnd::~NSSetMSyntheticFrontEnd() {
  delete m_data_32;
  m_data_32 = nullptr;
  delete m_data_64;
  m_data_64 = nullptr;
}

size_t
lldb_private::formatters::NSSetMSyntheticFrontEnd::GetIndexOfChildWithName(
    const ConstString &name) {
  const char *item_name = name.GetCString();
  uint32_t idx = ExtractIndexFromString(item_name);
  if (idx < UINT32_MAX && idx >= CalculateNumChildren())
    return UINT32_MAX;
  return idx;
}

size_t
lldb_private::formatters::NSSetMSyntheticFrontEnd::CalculateNumChildren() {
  if (!m_data_32 && !m_data_64)
    return 0;
  return (m_data_32 ? m_data_32->_used : m_data_64->_used);
}

bool lldb_private::formatters::NSSetMSyntheticFrontEnd::Update() {
  m_children.clear();
  ValueObjectSP valobj_sp = m_backend.GetSP();
  m_ptr_size = 0;
  delete m_data_32;
  m_data_32 = nullptr;
  delete m_data_64;
  m_data_64 = nullptr;
  if (!valobj_sp)
    return false;
  if (!valobj_sp)
    return false;
  m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
  Status error;
  if (valobj_sp->IsPointerType()) {
    valobj_sp = valobj_sp->Dereference(error);
    if (error.Fail() || !valobj_sp)
      return false;
  }
  error.Clear();
  lldb::ProcessSP process_sp(valobj_sp->GetProcessSP());
  if (!process_sp)
    return false;
  m_ptr_size = process_sp->GetAddressByteSize();
  uint64_t data_location = valobj_sp->GetAddressOf() + m_ptr_size;
  if (m_ptr_size == 4) {
    m_data_32 = new DataDescriptor_32();
    process_sp->ReadMemory(data_location, m_data_32, sizeof(DataDescriptor_32),
                           error);
  } else {
    m_data_64 = new DataDescriptor_64();
    process_sp->ReadMemory(data_location, m_data_64, sizeof(DataDescriptor_64),
                           error);
  }
  if (error.Fail())
    return false;
  return false;
}

bool lldb_private::formatters::NSSetMSyntheticFrontEnd::MightHaveChildren() {
  return true;
}

lldb::ValueObjectSP
lldb_private::formatters::NSSetMSyntheticFrontEnd::GetChildAtIndex(size_t idx) {
  lldb::addr_t m_objs_addr =
      (m_data_32 ? m_data_32->_objs_addr : m_data_64->_objs_addr);

  uint32_t num_children = CalculateNumChildren();

  if (idx >= num_children)
    return lldb::ValueObjectSP();

  ProcessSP process_sp = m_exe_ctx_ref.GetProcessSP();
  if (!process_sp)
    return lldb::ValueObjectSP();

  if (m_children.empty()) {
    // do the scan phase
    lldb::addr_t obj_at_idx = 0;

    uint32_t tries = 0;
    uint32_t test_idx = 0;

    while (tries < num_children) {
      obj_at_idx = m_objs_addr + (test_idx * m_ptr_size);
      if (!process_sp)
        return lldb::ValueObjectSP();
      Status error;
      obj_at_idx = process_sp->ReadPointerFromMemory(obj_at_idx, error);
      if (error.Fail())
        return lldb::ValueObjectSP();

      test_idx++;

      if (!obj_at_idx)
        continue;
      tries++;

      SetItemDescriptor descriptor = {obj_at_idx, lldb::ValueObjectSP()};

      m_children.push_back(descriptor);
    }
  }

  if (idx >= m_children.size()) // should never happen
    return lldb::ValueObjectSP();

  SetItemDescriptor &set_item = m_children[idx];
  if (!set_item.valobj_sp) {
    auto ptr_size = process_sp->GetAddressByteSize();
    DataBufferHeap buffer(ptr_size, 0);
    switch (ptr_size) {
    case 0: // architecture has no clue?? - fail
      return lldb::ValueObjectSP();
    case 4:
      *((uint32_t *)buffer.GetBytes()) = (uint32_t)set_item.item_ptr;
      break;
    case 8:
      *((uint64_t *)buffer.GetBytes()) = (uint64_t)set_item.item_ptr;
      break;
    default:
      assert(false && "pointer size is not 4 nor 8 - get out of here ASAP");
    }
    StreamString idx_name;
    idx_name.Printf("[%" PRIu64 "]", (uint64_t)idx);

    DataExtractor data(buffer.GetBytes(), buffer.GetByteSize(),
                       process_sp->GetByteOrder(),
                       process_sp->GetAddressByteSize());

    set_item.valobj_sp = CreateValueObjectFromData(
        idx_name.GetString(), data, m_exe_ctx_ref,
        m_backend.GetCompilerType().GetBasicTypeFromAST(
            lldb::eBasicTypeObjCID));
  }
  return set_item.valobj_sp;
}

template bool lldb_private::formatters::NSSetSummaryProvider<true>(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options);

template bool lldb_private::formatters::NSSetSummaryProvider<false>(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options);
