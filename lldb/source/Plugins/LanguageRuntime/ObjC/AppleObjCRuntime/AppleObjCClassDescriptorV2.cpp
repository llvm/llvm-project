//===-- AppleObjCClassDescriptorV2.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AppleObjCClassDescriptorV2.h"

#include "lldb/Expression/FunctionCaller.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/Language.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/ErrorExtras.h"

using namespace lldb;
using namespace lldb_private;

static lldb::addr_t GetClassDataMask(Process *process) {
  switch (process->GetAddressByteSize()) {
  case 4:
    return 0xfffffffcUL;
  case 8:
    return 0x00007ffffffffff8UL;
  default:
    break;
  }

  return LLDB_INVALID_ADDRESS;
}

llvm::Expected<ClassDescriptorV2::objc_class_t>
ClassDescriptorV2::objc_class_t::Read(Process *process, lldb::addr_t addr) {
  size_t ptr_size = process->GetAddressByteSize();

  size_t objc_class_size = ptr_size    // uintptr_t isa;
                           + ptr_size  // Class superclass;
                           + ptr_size  // void *cache;
                           + ptr_size  // IMP *vtable;
                           + ptr_size; // uintptr_t data_NEVER_USE;

  DataBufferHeap objc_class_buf(objc_class_size, '\0');
  Status error;

  process->ReadMemory(addr, objc_class_buf.GetBytes(), objc_class_size, error);
  if (error.Fail())
    return error.takeError();

  DataExtractor extractor(objc_class_buf.GetBytes(), objc_class_size,
                          process->GetByteOrder(),
                          process->GetAddressByteSize());

  lldb::offset_t cursor = 0;

  ObjCLanguageRuntime::ObjCISA isa =
      extractor.GetAddress_unchecked(&cursor); // uintptr_t isa;
  ObjCLanguageRuntime::ObjCISA superclass =
      extractor.GetAddress_unchecked(&cursor); // Class superclass;
  lldb::addr_t cache_ptr =
      extractor.GetAddress_unchecked(&cursor); // void *cache;
  lldb::addr_t vtable_ptr =
      extractor.GetAddress_unchecked(&cursor); // IMP *vtable;
  lldb::addr_t data_NEVER_USE =
      extractor.GetAddress_unchecked(&cursor); // uintptr_t data_NEVER_USE;

  uint8_t flags = (uint8_t)(data_NEVER_USE & (lldb::addr_t)3);
  lldb::addr_t data_ptr = data_NEVER_USE & GetClassDataMask(process);

  if (ABISP abi_sp = process->GetABI()) {
    isa = abi_sp->FixCodeAddress(isa);
    superclass = abi_sp->FixCodeAddress(superclass);
    data_ptr = abi_sp->FixCodeAddress(data_ptr);
  }
  return objc_class_t{isa, superclass, cache_ptr, vtable_ptr, data_ptr, flags};
}

llvm::Expected<ClassDescriptorV2::class_rw_t>
ClassDescriptorV2::class_rw_t::Read(Process *process, lldb::addr_t addr) {
  size_t ptr_size = process->GetAddressByteSize();

  size_t size = sizeof(uint32_t)   // uint32_t flags;
                + sizeof(uint32_t) // uint32_t version;
                + ptr_size         // const class_ro_t *ro;
                + ptr_size         // union { method_list_t **method_lists;
                                   // method_list_t *method_list; };
                + ptr_size         // struct chained_property_list *properties;
                + ptr_size         // const protocol_list_t **protocols;
                + ptr_size         // Class firstSubclass;
                + ptr_size;        // Class nextSiblingClass;

  DataBufferHeap buffer(size, '\0');
  Status error;

  process->ReadMemory(addr, buffer.GetBytes(), size, error);
  if (error.Fail())
    return error.takeError();

  DataExtractor extractor(buffer.GetBytes(), size, process->GetByteOrder(),
                          process->GetAddressByteSize());

  class_rw_t result{};
  lldb::offset_t cursor = 0;
  result.m_flags = extractor.GetU32_unchecked(&cursor);
  result.m_version = extractor.GetU32_unchecked(&cursor);
  result.m_ro_ptr = extractor.GetAddress_unchecked(&cursor);
  if (ABISP abi_sp = process->GetABI())
    result.m_ro_ptr = abi_sp->FixCodeAddress(result.m_ro_ptr);
  result.m_method_list_ptr = extractor.GetAddress_unchecked(&cursor);
  result.m_properties_ptr = extractor.GetAddress_unchecked(&cursor);

  if (result.m_ro_ptr & 1) {
    DataBufferHeap buffer(ptr_size, '\0');
    process->ReadMemory(result.m_ro_ptr ^ 1, buffer.GetBytes(), ptr_size,
                        error);
    if (error.Fail())
      return error.takeError();
    DataExtractor extractor(buffer.GetBytes(), ptr_size,
                            process->GetByteOrder(),
                            process->GetAddressByteSize());
    lldb::offset_t cursor = 0;
    result.m_ro_ptr = extractor.GetAddress_unchecked(&cursor);
    if (ABISP abi_sp = process->GetABI())
      result.m_ro_ptr = abi_sp->FixCodeAddress(result.m_ro_ptr);
  }

  return result;
}

llvm::Expected<ClassDescriptorV2::class_ro_t>
ClassDescriptorV2::class_ro_t::Read(Process *process, lldb::addr_t addr) {
  size_t ptr_size = process->GetAddressByteSize();

  size_t size = sizeof(uint32_t)   // uint32_t flags;
                + sizeof(uint32_t) // uint32_t instanceStart;
                + sizeof(uint32_t) // uint32_t instanceSize;
                + (ptr_size == 8 ? sizeof(uint32_t)
                                 : 0) // uint32_t reserved; // __LP64__ only
                + ptr_size            // const uint8_t *ivarLayout;
                + ptr_size            // const char *name;
                + ptr_size            // const method_list_t *baseMethods;
                + ptr_size            // const protocol_list_t *baseProtocols;
                + ptr_size            // const ivar_list_t *ivars;
                + ptr_size            // const uint8_t *weakIvarLayout;
                + ptr_size;           // const property_list_t *baseProperties;

  DataBufferHeap buffer(size, '\0');
  Status error;

  process->ReadMemory(addr, buffer.GetBytes(), size, error);
  if (error.Fail())
    return error.takeError();

  DataExtractor extractor(buffer.GetBytes(), size, process->GetByteOrder(),
                          process->GetAddressByteSize());

  class_ro_t result{};
  lldb::offset_t cursor = 0;

  result.m_flags = extractor.GetU32_unchecked(&cursor);
  result.m_instanceStart = extractor.GetU32_unchecked(&cursor);
  result.m_instanceSize = extractor.GetU32_unchecked(&cursor);
  if (ptr_size == 8)
    result.m_reserved = extractor.GetU32_unchecked(&cursor);
  else
    result.m_reserved = 0;
  result.m_ivarLayout_ptr = extractor.GetAddress_unchecked(&cursor);
  result.m_name_ptr = extractor.GetAddress_unchecked(&cursor);
  result.m_baseMethods_ptr = extractor.GetAddress_unchecked(&cursor);
  result.m_baseProtocols_ptr = extractor.GetAddress_unchecked(&cursor);
  result.m_ivars_ptr = extractor.GetAddress_unchecked(&cursor);
  result.m_weakIvarLayout_ptr = extractor.GetAddress_unchecked(&cursor);
  result.m_baseProperties_ptr = extractor.GetAddress_unchecked(&cursor);

  DataBufferHeap name_buf(1024, '\0');

  process->ReadCStringFromMemory(result.m_name_ptr, (char *)name_buf.GetBytes(),
                                 name_buf.GetByteSize(), error);

  if (error.Fail())
    return error.takeError();

  result.m_name.assign((char *)name_buf.GetBytes());

  return result;
}

llvm::Expected<ClassDescriptorV2::class_ro_t>
ClassDescriptorV2::Read_class_row(Process *process,
                                  const objc_class_t &objc_class) {
  Status error;
  uint32_t class_row_t_flags = process->ReadUnsignedIntegerFromMemory(
      objc_class.m_data_ptr, sizeof(uint32_t), 0, error);
  if (!error.Success())
    return error.takeError();

  if (class_row_t_flags & RW_REALIZED) {
    // Only class_rw->m_ro_ptr is used, the rw class doesn't need to exist.
    auto class_rw = class_rw_t::Read(process, objc_class.m_data_ptr);
    if (!class_rw)
      return class_rw.takeError();
    return class_ro_t::Read(process, class_rw->m_ro_ptr);
  }
  return class_ro_t::Read(process, objc_class.m_data_ptr);
}

llvm::Expected<ClassDescriptorV2::method_list_t>
ClassDescriptorV2::method_list_t::Read(Process *process, lldb::addr_t addr) {
  size_t size = sizeof(uint32_t)    // uint32_t entsize_NEVER_USE;
                + sizeof(uint32_t); // uint32_t count;

  DataBufferHeap buffer(size, '\0');
  Status error;

  if (ABISP abi_sp = process->GetABI())
    addr = abi_sp->FixCodeAddress(addr);
  process->ReadMemory(addr, buffer.GetBytes(), size, error);
  if (error.Fail())
    return error.takeError();

  DataExtractor extractor(buffer.GetBytes(), size, process->GetByteOrder(),
                          process->GetAddressByteSize());

  lldb::offset_t cursor = 0;

  uint32_t entsize_raw = extractor.GetU32_unchecked(&cursor);
  bool is_small = (entsize_raw & 0x80000000) != 0;
  bool has_direct_selector = (entsize_raw & 0x40000000) != 0;
  bool has_relative_types = (entsize_raw & 0x20000000) != 0;
  uint16_t entsize = entsize_raw & 0xfffc;
  uint32_t count = extractor.GetU32_unchecked(&cursor);
  addr_t first_ptr = addr + cursor;

  return method_list_t{
      entsize, is_small, has_direct_selector, has_relative_types,
      count,   first_ptr};
}

void ClassDescriptorV2::method_t::ReadNames(
    llvm::MutableArrayRef<method_t> methods, Process &process) {
  std::vector<lldb::addr_t> str_addresses;
  str_addresses.reserve(2 * methods.size());
  for (auto &method : methods)
    str_addresses.push_back(method.m_name_ptr);
  for (auto &method : methods)
    str_addresses.push_back(method.m_types_ptr);

  llvm::SmallVector<std::optional<std::string>> read_result =
      process.ReadCStringsFromMemory(str_addresses);
  auto names = llvm::MutableArrayRef(read_result).take_front(methods.size());
  auto types = llvm::MutableArrayRef(read_result).take_back(methods.size());

  for (auto [name_str, type_str, method] : llvm::zip(names, types, methods)) {
    if (name_str)
      method.m_name = std::move(*name_str);
    if (type_str)
      method.m_types = std::move(*type_str);
  }
}

llvm::SmallVector<ClassDescriptorV2::method_t, 0>
ClassDescriptorV2::ReadMethods(llvm::ArrayRef<lldb::addr_t> addresses,
                               lldb::addr_t relative_string_base_addr,
                               bool is_small, bool has_direct_sel,
                               bool has_relative_types) const {
  lldb_private::Process *process = m_runtime.GetProcess();
  if (!process)
    return {};

  const size_t size = method_t::GetSize(process, is_small);
  const size_t num_methods = addresses.size();

  llvm::SmallVector<uint8_t, 0> buffer(num_methods * size, 0);

  llvm::SmallVector<Range<addr_t, size_t>> mem_ranges =
      llvm::to_vector(llvm::map_range(llvm::seq(num_methods), [&](size_t idx) {
        return Range<addr_t, size_t>(addresses[idx], size);
      }));

  llvm::SmallVector<llvm::MutableArrayRef<uint8_t>> read_results =
      process->ReadMemoryRanges(mem_ranges, buffer);

  llvm::SmallVector<method_t, 0> methods;
  methods.reserve(num_methods);
  for (auto [addr, memory] : llvm::zip(addresses, read_results)) {
    // Ignore partial reads.
    if (memory.size() != size)
      continue;

    DataExtractor extractor(memory.data(), size, process->GetByteOrder(),
                            process->GetAddressByteSize());
    methods.push_back(method_t());
    methods.back().Read(extractor, process, addr, relative_string_base_addr,
                        is_small, has_direct_sel, has_relative_types);
  }

  method_t::ReadNames(methods, *process);
  return methods;
}

bool ClassDescriptorV2::method_t::Read(DataExtractor &extractor,
                                       Process *process, lldb::addr_t addr,
                                       lldb::addr_t relative_string_base_addr,
                                       bool is_small, bool has_direct_sel,
                                       bool has_relative_types) {
  lldb::offset_t cursor = 0;

  if (is_small) {
    uint32_t nameref_offset = extractor.GetU32_unchecked(&cursor);
    uint32_t types_offset = extractor.GetU32_unchecked(&cursor);
    uint32_t imp_offset = extractor.GetU32_unchecked(&cursor);

    m_name_ptr = addr + nameref_offset;

    Status error;
    if (!has_direct_sel) {
      // The SEL offset points to a SELRef. We need to dereference twice.
      m_name_ptr = process->ReadPointerFromMemory(m_name_ptr, error);
      if (error.Fail())
        return false;
    } else if (relative_string_base_addr != LLDB_INVALID_ADDRESS) {
      m_name_ptr = relative_string_base_addr + nameref_offset;
    }
    if (has_relative_types)
      m_types_ptr = relative_string_base_addr + types_offset;
    else
      m_types_ptr = addr + 4 + types_offset;
    m_imp_ptr = addr + 8 + imp_offset;
  } else {
    m_name_ptr = extractor.GetAddress_unchecked(&cursor);
    m_types_ptr = extractor.GetAddress_unchecked(&cursor);
    m_imp_ptr = extractor.GetAddress_unchecked(&cursor);
  }

  return true;
}

llvm::Expected<ClassDescriptorV2::ivar_list_t>
ClassDescriptorV2::ivar_list_t::Read(Process *process, lldb::addr_t addr) {
  size_t size = sizeof(uint32_t)    // uint32_t entsize;
                + sizeof(uint32_t); // uint32_t count;

  DataBufferHeap buffer(size, '\0');
  Status error;

  process->ReadMemory(addr, buffer.GetBytes(), size, error);
  if (error.Fail())
    return error.takeError();

  DataExtractor extractor(buffer.GetBytes(), size, process->GetByteOrder(),
                          process->GetAddressByteSize());

  lldb::offset_t cursor = 0;
  uint32_t entsize = extractor.GetU32_unchecked(&cursor);
  uint32_t count = extractor.GetU32_unchecked(&cursor);
  lldb::addr_t first_ptr = addr + cursor;
  return ivar_list_t{entsize, count, first_ptr};
}

llvm::Expected<ClassDescriptorV2::ivar_t>
ClassDescriptorV2::ivar_t::Read(Process *process, lldb::addr_t addr) {
  size_t size = GetSize(process);

  DataBufferHeap buffer(size, '\0');
  Status error;

  process->ReadMemory(addr, buffer.GetBytes(), size, error);
  if (error.Fail())
    return error.takeError();

  DataExtractor extractor(buffer.GetBytes(), size, process->GetByteOrder(),
                          process->GetAddressByteSize());

  ivar_t result{};
  lldb::offset_t cursor = 0;

  result.m_offset_ptr = extractor.GetAddress_unchecked(&cursor);
  result.m_name_ptr = extractor.GetAddress_unchecked(&cursor);
  result.m_type_ptr = extractor.GetAddress_unchecked(&cursor);
  result.m_alignment = extractor.GetU32_unchecked(&cursor);
  result.m_size = extractor.GetU32_unchecked(&cursor);

  llvm::SmallVector<std::optional<std::string>> strs =
      process->ReadCStringsFromMemory({result.m_name_ptr, result.m_type_ptr});

  if (!strs[0])
    return llvm::createStringErrorV(
        "Failed to read ivar_t::m_name_str at address {0:x}",
        result.m_name_ptr);
  if (!strs[1])
    return llvm::createStringErrorV(
        "Failed to read ivar_t::m_type_str at address {0:x}",
        result.m_type_ptr);

  result.m_name = std::move(*strs[0]);
  result.m_type = std::move(*strs[1]);
  return result;
}

llvm::Expected<ClassDescriptorV2::relative_list_entry_t>
ClassDescriptorV2::relative_list_entry_t::Read(Process *process,
                                               lldb::addr_t addr) {
  size_t size = sizeof(uint64_t); // m_image_index : 16
                                  // m_list_offset : 48

  DataBufferHeap buffer(size, '\0');
  Status error;

  process->ReadMemory(addr, buffer.GetBytes(), size, error);
  if (error.Fail())
    return llvm::joinErrors(
        error.takeError(),
        llvm::createStringErrorV(
            "Failed to read relative_list_entry_t at address {0:x}", addr));

  DataExtractor extractor(buffer.GetBytes(), size, process->GetByteOrder(),
                          process->GetAddressByteSize());
  lldb::offset_t cursor = 0;
  uint64_t raw_entry = extractor.GetU64_unchecked(&cursor);
  uint16_t image_index = raw_entry & 0xFFFF;
  int64_t list_offset = llvm::SignExtend64<48>(raw_entry >> 16);
  return relative_list_entry_t{image_index, list_offset};
}

llvm::Expected<ClassDescriptorV2::relative_list_list_t>
ClassDescriptorV2::relative_list_list_t::Read(Process *process,
                                              lldb::addr_t addr) {
  size_t size = sizeof(uint32_t)    // m_entsize
                + sizeof(uint32_t); // m_count

  DataBufferHeap buffer(size, '\0');
  Status error;

  process->ReadMemory(addr, buffer.GetBytes(), size, error);
  if (error.Fail())
    return llvm::joinErrors(
        error.takeError(),
        llvm::createStringErrorV(
            "Failed to read relative_list_list_t at address {0:x}", addr));

  DataExtractor extractor(buffer.GetBytes(), size, process->GetByteOrder(),
                          process->GetAddressByteSize());
  lldb::offset_t cursor = 0;
  uint32_t entsize = extractor.GetU32_unchecked(&cursor);
  uint32_t count = extractor.GetU32_unchecked(&cursor);
  lldb::addr_t first_ptr = addr + cursor;
  return relative_list_list_t{entsize, count, first_ptr};
}

llvm::Expected<ClassDescriptorV2::method_list_t>
ClassDescriptorV2::GetMethodList(Process *process,
                                 lldb::addr_t method_list_ptr) {
  auto method_list =
      ClassDescriptorV2::method_list_t::Read(process, method_list_ptr);
  if (!method_list)
    return method_list.takeError();

  const size_t method_size =
      method_t::GetSize(process, method_list->m_is_small);
  if (method_list->m_entsize != method_size)
    return llvm::createStringErrorV(
        "method_list_t at address {0:x} has an entsize of {1:x}"
        " but method size should be {2:x}",
        method_list_ptr, method_list->m_entsize, method_size);

  return *method_list;
}

void ClassDescriptorV2::ProcessMethodList(
    std::function<bool(const char *, const char *)> const &instance_method_func,
    ClassDescriptorV2::method_list_t &method_list) const {
  auto idx_to_method_addr = [&](uint32_t idx) {
    return method_list.m_first_ptr + (idx * method_list.m_entsize);
  };
  llvm::SmallVector<addr_t> addresses = llvm::to_vector(llvm::map_range(
      llvm::seq<uint32_t>(method_list.m_count), idx_to_method_addr));

  llvm::SmallVector<method_t, 0> methods =
      ReadMethods(addresses, m_runtime.GetRelativeSelectorBaseAddr(),
                  method_list.m_is_small, method_list.m_has_direct_selector,
                  method_list.m_has_relative_types);

  for (const auto &method : methods)
    if (instance_method_func(method.m_name.c_str(), method.m_types.c_str()))
      break;
}

// The relevant data structures:
//  - relative_list_list_t
//    - uint32_t count
//    - uint32_t entsize
//    - Followed by <count> number of relative_list_entry_t of size <entsize>
//
//  - relative_list_entry_t
//    - uint64_t image_index : 16
//    - int64_t list_offset : 48
//    - Note: The above 2 fit into 8 bytes always
//
//    image_index corresponds to an image in the shared cache
//    list_offset is used to calculate the address of the method_list_t we want
llvm::Error ClassDescriptorV2::ProcessRelativeMethodLists(
    std::function<bool(const char *, const char *)> const &instance_method_func,
    lldb::addr_t relative_method_list_ptr) const {
  lldb_private::Process *process = m_runtime.GetProcess();

  // 1. Process the count and entsize of the relative_list_list_t
  auto relative_method_lists =
      relative_list_list_t::Read(process, relative_method_list_ptr);
  if (!relative_method_lists)
    return relative_method_lists.takeError();

  for (uint32_t i = 0; i < relative_method_lists->m_count; i++) {
    // 2. Extract the image index and the list offset from the
    // relative_list_entry_t
    const lldb::addr_t entry_addr = relative_method_lists->m_first_ptr +
                                    (i * relative_method_lists->m_entsize);
    auto entry = relative_list_entry_t::Read(process, entry_addr);
    if (!entry)
      return entry.takeError();

    // 3. Calculate the pointer to the method_list_t from the
    // relative_list_entry_t
    const lldb::addr_t method_list_addr = entry_addr + entry->m_list_offset;

    // 4. Get the method_list_t from the pointer
    llvm::Expected<method_list_t> method_list =
        GetMethodList(process, method_list_addr);
    if (!method_list)
      return method_list.takeError();

    // 5. Cache the result so we don't need to reconstruct it later.
    m_image_to_method_lists[entry->m_image_index].emplace_back(*method_list);

    // 6. If the relevant image is loaded, add the methods to the Decl
    if (!m_runtime.IsSharedCacheImageLoaded(entry->m_image_index))
      continue;

    ProcessMethodList(instance_method_func, *method_list);
  }

  // We need to keep track of the last time we updated so we can re-update the
  // type information in the future
  m_last_version_updated = m_runtime.GetSharedCacheImageHeaderVersion();

  return llvm::Error::success();
}

bool ClassDescriptorV2::Describe(
    std::function<void(ObjCLanguageRuntime::ObjCISA)> const &superclass_func,
    std::function<bool(const char *, const char *)> const &instance_method_func,
    std::function<bool(const char *, const char *)> const &class_method_func,
    std::function<bool(const char *, const char *, lldb::addr_t,
                       uint64_t)> const &ivar_func) const {
  lldb_private::Process *process = m_runtime.GetProcess();

  auto objc_class = objc_class_t::Read(process, m_objc_class_ptr);
  if (!objc_class) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Types), objc_class.takeError(), "{0}");
    return false;
  }
  auto class_ro = Read_class_row(process, *objc_class);
  if (!class_ro) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Types), class_ro.takeError(), "{0}");
    return false;
  }

  static ConstString NSObject_name("NSObject");

  if (m_name != NSObject_name && superclass_func)
    superclass_func(objc_class->m_superclass);

  if (instance_method_func) {
    // This is a relative list of lists
    if (class_ro->m_baseMethods_ptr & 1) {
      if (llvm::Error err = ProcessRelativeMethodLists(
              instance_method_func, class_ro->m_baseMethods_ptr ^ 1)) {
        LLDB_LOG_ERROR(GetLog(LLDBLog::Types), std::move(err), "{0}");
        return false;
      }
    } else {
      llvm::Expected<method_list_t> base_method_list =
          GetMethodList(process, class_ro->m_baseMethods_ptr);
      if (base_method_list)
        ProcessMethodList(instance_method_func, *base_method_list);
      else
        LLDB_LOG_ERROR(GetLog(LLDBLog::Types), base_method_list.takeError(),
                       "{0}");
    }
  }

  if (class_method_func) {
    AppleObjCRuntime::ClassDescriptorSP metaclass(GetMetaclass());

    // We don't care about the metaclass's superclass, or its class methods.
    // Its instance methods are our class methods.

    if (metaclass) {
      metaclass->Describe(
          std::function<void(ObjCLanguageRuntime::ObjCISA)>(nullptr),
          class_method_func,
          std::function<bool(const char *, const char *)>(nullptr),
          std::function<bool(const char *, const char *, lldb::addr_t,
                             uint64_t)>(nullptr));
    }
  }

  if (ivar_func) {
    if (class_ro->m_ivars_ptr != 0) {
      auto ivar_list = ivar_list_t::Read(process, class_ro->m_ivars_ptr);
      if (!ivar_list) {
        LLDB_LOG_ERROR(GetLog(LLDBLog::Types), ivar_list.takeError(), "{0}");
        return false;
      }

      if (ivar_list->m_entsize != ivar_t::GetSize(process))
        return false;

      for (uint32_t i = 0, e = ivar_list->m_count; i < e; ++i) {
        auto ivar = ivar_t::Read(process, ivar_list->m_first_ptr +
                                              (i * ivar_list->m_entsize));
        if (!ivar) {
          LLDB_LOG_ERROR(GetLog(LLDBLog::Types), ivar.takeError(), "{0}");
          continue;
        }

        if (ivar_func(ivar->m_name.c_str(), ivar->m_type.c_str(),
                      ivar->m_offset_ptr, ivar->m_size))
          break;
      }
    }
  }

  return true;
}

ConstString ClassDescriptorV2::GetClassName() {
  if (!m_name) {
    lldb_private::Process *process = m_runtime.GetProcess();

    if (process) {
      auto objc_class = objc_class_t::Read(process, m_objc_class_ptr);
      if (!objc_class) {
        LLDB_LOG_ERROR(GetLog(LLDBLog::Types), objc_class.takeError(), "{0}");
        return m_name;
      }
      auto class_ro = Read_class_row(process, *objc_class);
      if (!class_ro) {
        LLDB_LOG_ERROR(GetLog(LLDBLog::Types), class_ro.takeError(), "{0}");
        return m_name;
      }

      m_name = ConstString(class_ro->m_name);
    }
  }
  return m_name;
}

ObjCLanguageRuntime::ClassDescriptorSP ClassDescriptorV2::GetSuperclass() {
  lldb_private::Process *process = m_runtime.GetProcess();

  if (!process)
    return ObjCLanguageRuntime::ClassDescriptorSP();

  auto objc_class = objc_class_t::Read(process, m_objc_class_ptr);
  if (!objc_class) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Types), objc_class.takeError(), "{0}");
    return ObjCLanguageRuntime::ClassDescriptorSP();
  }

  return m_runtime.ObjCLanguageRuntime::GetClassDescriptorFromISA(
      objc_class->m_superclass);
}

ObjCLanguageRuntime::ClassDescriptorSP ClassDescriptorV2::GetMetaclass() const {
  lldb_private::Process *process = m_runtime.GetProcess();

  if (!process)
    return ObjCLanguageRuntime::ClassDescriptorSP();

  auto objc_class = objc_class_t::Read(process, m_objc_class_ptr);
  if (!objc_class) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Types), objc_class.takeError(), "{0}");
    return ObjCLanguageRuntime::ClassDescriptorSP();
  }

  lldb::addr_t candidate_isa = m_runtime.GetPointerISA(objc_class->m_isa);

  return ObjCLanguageRuntime::ClassDescriptorSP(
      new ClassDescriptorV2(m_runtime, candidate_isa, nullptr));
}

uint64_t ClassDescriptorV2::GetInstanceSize() {
  lldb_private::Process *process = m_runtime.GetProcess();

  if (process) {
    auto objc_class = objc_class_t::Read(process, m_objc_class_ptr);
    if (!objc_class) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Types), objc_class.takeError(), "{0}");
      return 0;
    }
    auto class_ro = Read_class_row(process, *objc_class);
    if (!class_ro) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Types), class_ro.takeError(), "{0}");
      return 0;
    }

    return class_ro->m_instanceSize;
  }

  return 0;
}

// From the ObjC runtime.
static uint8_t IS_SWIFT_STABLE = 1U << 1;

LanguageType ClassDescriptorV2::GetImplementationLanguage() const {
  if (auto *process = m_runtime.GetProcess()) {
    auto objc_class = objc_class_t::Read(process, m_objc_class_ptr);
    if (objc_class) {
      if (objc_class->m_flags & IS_SWIFT_STABLE)
        return lldb::eLanguageTypeSwift;
    } else {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Types), objc_class.takeError(), "{0}");
    }
  }
  return lldb::eLanguageTypeObjC;
}

ClassDescriptorV2::iVarsStorage::iVarsStorage() : m_ivars(), m_mutex() {}

size_t ClassDescriptorV2::iVarsStorage::size() { return m_ivars.size(); }

ClassDescriptorV2::iVarDescriptor &ClassDescriptorV2::iVarsStorage::
operator[](size_t idx) {
  return m_ivars[idx];
}

void ClassDescriptorV2::iVarsStorage::fill(AppleObjCRuntimeV2 &runtime,
                                           ClassDescriptorV2 &descriptor) {
  if (m_filled)
    return;
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  Log *log = GetLog(LLDBLog::Types);
  LLDB_LOG_VERBOSE(log, "class_name = {0}", descriptor.GetClassName());
  m_filled = true;
  ObjCLanguageRuntime::EncodingToTypeSP encoding_to_type_sp(
      runtime.GetEncodingToType());
  Process *process(runtime.GetProcess());
  if (!encoding_to_type_sp)
    return;
  descriptor.Describe(nullptr, nullptr, nullptr, [this, process,
                                                  encoding_to_type_sp,
                                                  log](const char *name,
                                                       const char *type,
                                                       lldb::addr_t offset_ptr,
                                                       uint64_t size) -> bool {
    const bool for_expression = false;
    const bool stop_loop = false;
    LLDB_LOG_VERBOSE(
        log, "name = {0}, encoding = {1}, offset_ptr = {2:x}, size = {3}", name,
        type, offset_ptr, size);
    CompilerType ivar_type =
        encoding_to_type_sp->RealizeType(type, for_expression);
    if (ivar_type) {
      LLDB_LOG_VERBOSE(
          log,
          "name = {0}, encoding = {1}, offset_ptr = {2:x}, size = "
          "{3}, type_size = {4}",
          name, type, offset_ptr, size,
          expectedToOptional(ivar_type.GetByteSize(nullptr)).value_or(0));
      Scalar offset_scalar;
      Status error;
      const int offset_ptr_size = 4;
      const bool is_signed = false;
      size_t read = process->ReadScalarIntegerFromMemory(
          offset_ptr, offset_ptr_size, is_signed, offset_scalar, error);
      if (error.Success() && 4 == read) {
        LLDB_LOG_VERBOSE(log, "offset_ptr = {0:x} --> {1}", offset_ptr,
                         offset_scalar.SInt());
        m_ivars.push_back(
            {ConstString(name), ivar_type, size, offset_scalar.SInt()});
      } else
        LLDB_LOG_VERBOSE(log, "offset_ptr = {0:x} --> read fail, read = %{1}",
                         offset_ptr, read);
    }
    return stop_loop;
  });
}

void ClassDescriptorV2::GetIVarInformation() {
  m_ivars_storage.fill(m_runtime, *this);
}
