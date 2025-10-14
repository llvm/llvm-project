//===-- LLDBMemoryReader.cpp ----------------------------------------------===//
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

#include "LLDBMemoryReader.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/Section.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include "swift/Demangling/Demangle.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace lldb;
using namespace lldb_private;

namespace lldb_private {
bool LLDBMemoryReader::queryDataLayout(DataLayoutQueryType type, void *inBuffer,
                                       void *outBuffer) {
  switch (type) {
  case DLQ_GetPtrAuthMask: {
    lldb::addr_t ptrauth_mask = m_process.GetDataAddressMask();
    if (ptrauth_mask == LLDB_INVALID_ADDRESS_MASK)
      return false;
    // The mask returned by the process masks out the non-addressable bits.
    uint64_t mask_pattern = ~ptrauth_mask;
    memcpy(outBuffer, &mask_pattern, m_process.GetAddressByteSize());
    return true;
  }
  case DLQ_GetObjCReservedLowBits: {
    auto *result = static_cast<uint8_t *>(outBuffer);
    auto &triple = m_process.GetTarget().GetArchitecture().GetTriple();
    if (triple.isMacOSX() && triple.getArch() == llvm::Triple::x86_64) {
      // Obj-C reserves low bit on 64-bit Intel macOS only.
      // Other Apple platforms don't reserve this bit (even when
      // running on x86_64-based simulators).
      *result = 1;
    } else {
      *result = 0;
    }
    return true;
  }
  case DLQ_GetPointerSize: {
    auto *result = static_cast<uint8_t *>(outBuffer);
    *result = m_process.GetAddressByteSize();
    return true;
  }
  case DLQ_GetSizeSize: {
    auto *result = static_cast<uint8_t *>(outBuffer);
    *result = m_process.GetAddressByteSize(); // FIXME: sizeof(size_t)
    return true;
  }
  case DLQ_GetLeastValidPointerValue: {
    auto *result = (uint64_t *)outBuffer;
    auto &triple = m_process.GetTarget().GetArchitecture().GetTriple();
    if (triple.isOSDarwin() && triple.isArch64Bit())
      *result = 0x100000000;
    else
      *result = 0x1000;
    return true;
  }
  case DLQ_GetObjCInteropIsEnabled: {
    auto *result = (bool *)outBuffer;
    *result = SwiftLanguageRuntime::GetObjCRuntime(m_process) != nullptr;
    return true;
  }
  }
}

swift::remote::RemoteAddress
LLDBMemoryReader::getSymbolAddress(const std::string &name) {
  lldbassert(!name.empty());
  if (name.empty())
    return swift::remote::RemoteAddress();

  Log *log = GetLog(LLDBLog::Types);

  LLDB_LOGV(log, "[MemoryReader] asked to retrieve the address of symbol {0}",
            name);

  ConstString name_cs(name.c_str(), name.size());
  SymbolContextList sc_list;
  m_process.GetTarget().GetImages().FindSymbolsWithNameAndType(
      name_cs, lldb::eSymbolTypeAny, sc_list);
  if (!sc_list.GetSize()) {
    LLDB_LOGV(log, "[MemoryReader] symbol resolution failed {0}", name);
    return swift::remote::RemoteAddress();
  }

  SymbolContext sym_ctx;
  // Remove undefined symbols from the list.
  size_t num_sc_matches = sc_list.GetSize();
  if (num_sc_matches > 1) {
    SymbolContextList tmp_sc_list(sc_list);
    sc_list.Clear();
    for (size_t idx = 0; idx < num_sc_matches; idx++) {
      tmp_sc_list.GetContextAtIndex(idx, sym_ctx);
      if (sym_ctx.symbol &&
          sym_ctx.symbol->GetType() != lldb::eSymbolTypeUndefined) {
        sc_list.Append(sym_ctx);
      }
    }
  }
  if (sc_list.GetSize() == 1 && sc_list.GetContextAtIndex(0, sym_ctx)) {
    if (sym_ctx.symbol) {
      auto load_addr = sym_ctx.symbol->GetLoadAddress(&m_process.GetTarget());
      LLDB_LOGV(log, "[MemoryReader] symbol resolved to {0:x}", load_addr);
      return swift::remote::RemoteAddress(
          load_addr, swift::remote::RemoteAddress::DefaultAddressSpace);
    }
  }

  // Empty list, resolution failed.
  if (sc_list.GetSize() == 0) {
    LLDB_LOGV(log, "[MemoryReader] symbol resolution failed {0}", name);
    return swift::remote::RemoteAddress();
  }

  // If there's a single symbol, then we're golden. If there's more than
  // a symbol, then just make sure all of them agree on the value.
  Status error;
  auto load_addr = sym_ctx.symbol->GetLoadAddress(&m_process.GetTarget());
  uint64_t sym_value = m_process.GetTarget().ReadUnsignedIntegerFromMemory(
      load_addr, m_process.GetAddressByteSize(), 0, error, true);
  for (unsigned i = 1; i < sc_list.GetSize(); ++i) {
    uint64_t other_sym_value =
        m_process.GetTarget().ReadUnsignedIntegerFromMemory(
            load_addr, m_process.GetAddressByteSize(), 0, error, true);
    if (sym_value != other_sym_value) {
      LLDB_LOGV(log, "[MemoryReader] symbol resolution failed {0}", name);
      return swift::remote::RemoteAddress();
    }
  }
  LLDB_LOGV(log, "[MemoryReader] symbol resolved to {0}", load_addr);
  return swift::remote::RemoteAddress(
      load_addr, swift::remote::RemoteAddress::DefaultAddressSpace);
}

std::unique_ptr<swift::SwiftObjectFileFormat>
GetSwiftObjectFileFormat(llvm::Triple::ObjectFormatType obj_format_type) {
  std::unique_ptr<swift::SwiftObjectFileFormat> obj_file_format;
  switch (obj_format_type) {
  case llvm::Triple::MachO:
    obj_file_format = std::make_unique<swift::SwiftObjectFileFormatMachO>();
    break;
  case llvm::Triple::ELF:
    obj_file_format = std::make_unique<swift::SwiftObjectFileFormatELF>();
    break;
  case llvm::Triple::COFF:
    obj_file_format = std::make_unique<swift::SwiftObjectFileFormatCOFF>();
    break;
  case llvm::Triple::Wasm:
    obj_file_format = std::make_unique<swift::SwiftObjectFileFormatWasm>();
    break;
  default:
    LLDB_LOG(GetLog(LLDBLog::Types), "Could not determine swift reflection "
                                     "section names for object format type");
    break;
  }
  return obj_file_format;
}

std::optional<swift::remote::RemoteAbsolutePointer>
LLDBMemoryReader::resolvePointerAsSymbol(swift::remote::RemoteAddress address) {
  // If an address has a symbol, that symbol provides additional useful data to
  // MetadataReader. Without the symbol, MetadataReader can derive the symbol
  // by loading other parts of reflection metadata, but that work has a cost.
  // For lldb, that data loading can be a significant performance hit. Providing
  // a symbol greatly reduces memory read traffic to the process.
  auto &target = m_process.GetTarget();
  if (!target.GetSwiftUseReflectionSymbols())
    return {};

  std::optional<Address> maybeAddr = remoteAddressToLLDBAddress(address);
  // This is not an assert, but should never happen.
  if (!maybeAddr)
    return {};

  Address addr;
  if (maybeAddr->IsSectionOffset()) {
    // `address` was tagged, and then successfully mapped (resolved).
    addr = *maybeAddr;
  } else {
    // `address` is a real load address.
    if (!target.ResolveLoadAddress(address.getRawAddress(), addr))
      return {};
  }

  if (auto section_sp = addr.GetSection()) {
    if (auto *obj_file = section_sp->GetObjectFile()) {
      auto obj_file_format_type =
          obj_file->GetArchitecture().GetTriple().getObjectFormat();
      if (auto swift_obj_file_format =
              GetSwiftObjectFileFormat(obj_file_format_type)) {
        if (!swift_obj_file_format->sectionContainsReflectionData(
                section_sp->GetName().GetStringRef()))
          return {};
      }
    }
  }

  if (auto *symbol = addr.CalculateSymbolContextSymbol()) {
    auto mangledName = symbol->GetMangled().GetMangledName().GetStringRef();
    // MemoryReader requires this to be a Swift symbol. LLDB can also be
    // aware of local symbols, so avoid returning those.
    using namespace swift::Demangle;
    if (isSwiftSymbol(mangledName) && !isOldFunctionTypeMangling(mangledName))
      return swift::remote::RemoteAbsolutePointer{mangledName, 0, address};
  }

  return {};
}

swift::remote::RemoteAbsolutePointer
LLDBMemoryReader::resolvePointer(swift::remote::RemoteAddress address,
                                 uint64_t readValue) {
  Log *log = GetLog(LLDBLog::Types);

  // We may have gotten a pointer to a process address, try to map it back
  // to a tagged address so further memory reads originating from it benefit
  // from the file-cache optimization.
  swift::remote::RemoteAbsolutePointer process_pointer{
      swift::remote::RemoteAddress{
          readValue, swift::remote::RemoteAddress::DefaultAddressSpace}};

  if (!readMetadataFromFileCacheEnabled()) {
    assert(address.getAddressSpace() ==
               swift::remote::RemoteAddress::DefaultAddressSpace &&
           "Unexpected address space!");
    return process_pointer;
  }

  // Try to strip the pointer before checking if we have it mapped.
  auto strippedPointer = signedPointerStripper(process_pointer);
  if (auto resolved = strippedPointer.getResolvedAddress())
    readValue = resolved.getRawAddress();

  auto &target = m_process.GetTarget();
  Address addr;
  if (!target.ResolveLoadAddress(readValue, addr)) {
    LLDB_LOGV(log,
              "[MemoryReader] Could not resolve load address of pointer {0:x} "
              "read from {1:x}.",
              readValue, address.getRawAddress());
    return process_pointer;
  }

  auto module_containing_pointer = addr.GetSection()->GetModule();

  // Check if the module containing the pointer is registered with
  // LLDBMemoryReader.
  auto pair_iterator = std::find_if(
      m_range_module_map.begin(), m_range_module_map.end(), [&](auto pair) {
        return std::get<ModuleSP>(pair) == module_containing_pointer;
      });

  // We haven't registered the image that contains the pointer.
  if (pair_iterator == m_range_module_map.end()) {
    LLDB_LOG(log,
             "[MemoryReader] Could not resolve find module containing pointer "
             "{0:x} read from {1:x}.",
             readValue, address.getRawAddress());
    return process_pointer;
  }

  // If the containing image is the first registered one, the image's tagged
  // start address for it is zero. Otherwise, the previous pair's address is the
  // start of the new address.
  uint64_t start_tagged_address = pair_iterator == m_range_module_map.begin()
                                      ? 0
                                      : std::prev(pair_iterator)->first;

  auto *section_list = module_containing_pointer->GetSectionList();
  if (section_list->GetSize() == 0) {
    LLDB_LOG(log, "[MemoryReader] Module with empty section list.");
    return {};
  }

  uint64_t tagged_address =
      start_tagged_address + addr.GetFileAddress() -
      section_list->GetSectionAtIndex(0)->GetFileAddress();

  if (tagged_address >= std::get<uint64_t>(*pair_iterator)) {
    // If the tagged address invades the next image's tagged address space,
    // something went wrong. Log it and just return the process address.
    LLDB_LOG(log,
             "[MemoryReader] Pointer {0:x} read from {1:x} resolved to tagged "
             "address {2:x}, which is outside its image address space.",
             readValue, address.getRawAddress(), tagged_address);
    return process_pointer;
  }

  swift::remote::RemoteAbsolutePointer tagged_pointer{
      swift::remote::RemoteAddress{tagged_address, LLDBAddressSpace}};

  if (tagged_address != (uint64_t)signedPointerStripper(tagged_pointer)
                            .getResolvedAddress()
                            .getRawAddress()) {
    lldbassert(false &&
               "Tagged pointer runs into pointer authentication mask!");
    return process_pointer;
  }

  LLDB_LOGV(log,
            "[MemoryReader] Successfully resolved pointer {0:x} read from "
            "{1:x} to tagged address {2:x}.",
            readValue, address.getRawAddress(), tagged_address);
  return tagged_pointer;
}

bool LLDBMemoryReader::readBytes(swift::remote::RemoteAddress address,
                                 uint8_t *dest, uint64_t size) {
  auto read_bytes_result = readBytesImpl(address, dest, size);
  return read_bytes_result != ReadBytesResult::fail;
}

LLDBMemoryReader::ReadBytesResult
LLDBMemoryReader::readBytesImpl(swift::remote::RemoteAddress address,
                                uint8_t *dest, uint64_t size) {
  Log *log = GetLog(LLDBLog::Types);
  if (m_local_buffer) {
    bool overflow = false;
    auto addr = address.getRawAddress();
    auto end = llvm::SaturatingAdd(addr, size, &overflow);
    if (overflow) {
      LLDB_LOGV(log, "[MemoryReader] address {0:x} + size {1} overflows", addr,
                size);
      return ReadBytesResult::fail;
    }
    if (addr >= *m_local_buffer &&
        end <= *m_local_buffer + m_local_buffer_size) {
      // If this crashes, the assumptions stated in
      // GetDynamicTypeAndAddress_Protocol() most likely no longer
      // hold.
      memcpy(dest, (void *)addr, size);
      return ReadBytesResult::success_from_file;
    }
  }

  LLDB_LOGV(log, "[MemoryReader] asked to read {0} bytes at address {1:x}",
            size, address.getRawAddress());
  std::optional<Address> maybeAddr =
      resolveRemoteAddressFromSymbolObjectFile(address);

  if (!maybeAddr)
    maybeAddr = remoteAddressToLLDBAddress(address);

  if (!maybeAddr) {
    LLDB_LOGV(log, "[MemoryReader] could not resolve address {0:x}",
              address.getRawAddress());
    return ReadBytesResult::fail;
  }
  auto addr = *maybeAddr;
  if (addr.IsSectionOffset()) {
    auto section = addr.GetSection();
    auto *object_file = section->GetObjectFile();
    if (object_file->GetType() == ObjectFile::Type::eTypeDebugInfo) {
      LLDB_LOGV(log, "[MemoryReader] Reading memory from symbol rich binary");

      if (object_file->ReadSectionData(section.get(), addr.GetOffset(), dest,
                                       size))
        return ReadBytesResult::success_from_file;
      return ReadBytesResult::fail;
    }
  }

  if (size > m_max_read_amount) {
    LLDB_LOGV(log, "[MemoryReader] memory read exceeds maximum allowed size");
    return ReadBytesResult::fail;
  }
  Target &target(m_process.GetTarget());
  Status error;
  // We only want to allow the file-cache optimization if we resolved the
  // address to section + offset.
  const bool force_live_memory =
      !readMetadataFromFileCacheEnabled() || !addr.IsSectionOffset();
  bool did_read_live_memory = false;
  if (size > target.ReadMemory(addr, dest, size, error, force_live_memory,
                               /*load_addr_ptr=*/nullptr,
                               &did_read_live_memory)) {
    LLDB_LOGV(log,
              "[MemoryReader] memory read returned fewer bytes than asked for");
    return ReadBytesResult::fail;
  }
  if (error.Fail()) {
    LLDB_LOGV(log, "[MemoryReader] memory read returned error: {0}",
              error.AsCString());
    return ReadBytesResult::fail;
  }

  auto format_data = [](auto dest, auto size) {
    StreamString stream;
    for (uint64_t i = 0; i < size; i++) {
      stream.PutHex8(dest[i]);
      stream.PutChar(' ');
    }
    return std::string(stream.GetData());
  };
  LLDB_LOGV(log, "[MemoryReader] memory read returned data: {0}",
            format_data(dest, size));

  return did_read_live_memory ? ReadBytesResult::success_from_memory
                              : ReadBytesResult::success_from_file;
}

bool LLDBMemoryReader::readString(swift::remote::RemoteAddress address,
                                  std::string &dest) {
  Log *log = GetLog(LLDBLog::Types);

  auto format_string = [](const std::string &dest) {
    std::string result;
    llvm::raw_string_ostream s(result);
    llvm::printEscapedString(dest, s);
    return result;
  };
  /// Very quickly (this is a high-firing event), copy a few
  /// human-readable bytes usable as detail in the progress update.
  auto short_string = [](const std::string &dest) {
    std::string result;
    result.reserve(24);
    size_t end = std::min<size_t>(dest.size(), 24);
    for (size_t i = 0; i < end; ++i) {
      signed char c = dest[i];
      result.append(1, c > 32 ? c : '.');
    }
    if (end < dest.size()) {
      result[21] = '.';
      result[22] = '.';
      result[23] = '.';
    }
    return result;
  };
  LLDB_LOGV(log, "[MemoryReader] asked to read string data at address {0:x}",
            address.getRawAddress());

  std::optional<Address> maybeAddr =
      resolveRemoteAddressFromSymbolObjectFile(address);

  if (!maybeAddr)
    maybeAddr = remoteAddressToLLDBAddress(address);

  if (!maybeAddr) {
    LLDB_LOGV(log, "[MemoryReader] could not resolve address {0:x}",
              address.getRawAddress());
    return false;
  }
  auto addr = *maybeAddr;
  if (addr.IsSectionOffset()) {
    auto section = addr.GetSection();
    auto *object_file = section->GetObjectFile();
    if (object_file->GetType() == ObjectFile::Type::eTypeDebugInfo) {
      LLDB_LOGV(log, "[MemoryReader] Reading memory from symbol rich binary");

      dest = object_file->GetCStrFromSection(section.get(), addr.GetOffset());
      LLDB_LOGV(log, "[MemoryReader] memory read returned string: \"{0}\"",
                format_string(dest));
      return true;
    }
  }

  Target &target(m_process.GetTarget());
  Status error;
  // We only want to allow the file-cache optimization if we resolved the
  // address to section + offset.
  const bool force_live_memory =
      !readMetadataFromFileCacheEnabled() || !addr.IsSectionOffset();
  target.ReadCStringFromMemory(addr, dest, error, force_live_memory);
  if (error.Success()) {
    if (m_progress_callback)
      m_progress_callback(short_string(dest));
    LLDB_LOGV(log, "[MemoryReader] memory read returned string: \"{0}\"",
              format_string(dest));
    return true;
  }
  LLDB_LOGV(log, "[MemoryReader] memory read returned error: {0}",
            error.AsCString());
  return false;
}

MemoryReaderLocalBufferHolder::~MemoryReaderLocalBufferHolder() {
  if (m_memory_reader)
    m_memory_reader->popLocalBuffer();
}

MemoryReaderLocalBufferHolder
LLDBMemoryReader::pushLocalBuffer(uint64_t local_buffer,
                                  uint64_t local_buffer_size) {
  lldbassert(!m_local_buffer);
  m_local_buffer = local_buffer;
  m_local_buffer_size = local_buffer_size;
  return MemoryReaderLocalBufferHolder(this);
}

void LLDBMemoryReader::popLocalBuffer() {
  lldbassert(m_local_buffer);
  m_local_buffer.reset();
  m_local_buffer_size = 0;
}

std::optional<std::pair<uint64_t, uint64_t>>
LLDBMemoryReader::addModuleToAddressMap(ModuleSP module,
                                        bool register_symbol_obj_file) {
  if (!readMetadataFromFileCacheEnabled())
    return {};

  assert(register_symbol_obj_file <=
             m_process.GetTarget().GetSwiftReadMetadataFromDSYM() &&
         "Trying to register symbol object file, but reading from it is "
         "disabled!");

  uint64_t module_start_address = 0;
  if (!m_range_module_map.empty())
    // We map the images contiguously one after the other.
    // The address that maps the last module is exactly the address the new
    // module should start at.
    module_start_address = m_range_module_map.back().first;

#ifndef NDEBUG
  static std::initializer_list<uint64_t> objc_bits = {
      SWIFT_ABI_ARM_IS_OBJC_BIT, SWIFT_ABI_X86_64_IS_OBJC_BIT,
      SWIFT_ABI_ARM64_IS_OBJC_BIT};

  for (auto objc_bit : objc_bits)
    assert((module_start_address & objc_bit) != objc_bit &&
           "LLDB file address bit clashes with an obj-c bit!");
#endif

  ObjectFile *object_file;
  if (register_symbol_obj_file) {
    auto *symbol_file = module->GetSymbolFile();
    if (!symbol_file)
      return {};
    object_file = symbol_file->GetObjectFile();
  } else {
    object_file = module->GetObjectFile();
  }

  if (!object_file)
    return {};

  SectionList *section_list = object_file->GetSectionList();

  auto section_list_size = section_list->GetSize();
  if (section_list_size == 0)
    return {};

  auto first_section = section_list->GetSectionAtIndex(0);
  auto last_section =
      section_list->GetSectionAtIndex(section_list->GetSize() - 1);

  // The total size is the last section's file address plus size, subtracting
  // the first section's file address.
  auto start_file_address = first_section->GetFileAddress();
  uint64_t end_file_address =
      last_section->GetFileAddress() + last_section->GetByteSize();
  auto size = end_file_address - start_file_address;
  auto module_end_address = module_start_address + size;

  // The address for the next image is the next pointer aligned address
  // available after the end of the current image.
  uint64_t next_module_start_address = llvm::alignTo(module_end_address, 8);
  m_range_module_map.emplace_back(next_module_start_address, module);

  if (register_symbol_obj_file)
    m_modules_with_metadata_in_symbol_obj_file.insert(module);

  return {{module_start_address, module_end_address}};
}

std::optional<std::pair<uint64_t, lldb::ModuleSP>>
LLDBMemoryReader::getFileAddressAndModuleForTaggedAddress(
    swift::remote::RemoteAddress tagged_address) const {
  Log *log(GetLog(LLDBLog::Types));

  if (!readMetadataFromFileCacheEnabled())
    return {};

  if (tagged_address.getAddressSpace() != LLDBAddressSpace)
    return {};

  // Dummy pair with the address we're looking for.
  auto comparison_pair =
      std::make_pair(tagged_address.getRawAddress(), ModuleSP());

  // Explicitly compare only the addresses, never the modules in the pairs.
  auto pair_iterator = std::lower_bound(
      m_range_module_map.begin(), m_range_module_map.end(), comparison_pair,
      [](auto &a, auto &b) { return a.first < b.first; });

  // If the address is larger than anything we have mapped the address is out
  if (pair_iterator == m_range_module_map.end()) {
    LLDB_LOG(log,
             "[MemoryReader] Address {0:x} is larger than the upper bound "
             "address of the mapped in modules",
             tagged_address.getRawAddress());
    return {};
  }

  ModuleSP module = pair_iterator->second;
  auto *section_list = module->GetSectionList();
  if (section_list->GetSize() == 0) {
    LLDB_LOG(log, "[MemoryReader] Module with empty section list.");
    return {};
  }
  uint64_t file_address;
  if (pair_iterator == m_range_module_map.begin())
    file_address = tagged_address.getRawAddress();
  else
    // The end of the previous section is the start of the current one.
    // We also need to add the first section's file address since we remove it
    // when constructing the range to module map.
    file_address =
        (tagged_address - std::prev(pair_iterator)->first).getRawAddress();

  // We also need to add the module's file address, since we subtract it when
  // building the range to module map.
  file_address += section_list->GetSectionAtIndex(0)->GetFileAddress();
  return {{file_address, module}};
}

bool LLDBMemoryReader::readRemoteAddressImpl(
    swift::remote::RemoteAddress address, swift::remote::RemoteAddress &out,
    std::size_t size) {
  assert((size == 4 || size == 8) &&
         "Only 32 or 64 bit architectures are supported!");
  if (size != 4 && size != 8)
    return false;

  uint64_t buf;
  auto read_bytes_result =
      readBytesImpl(address, reinterpret_cast<uint8_t *>(&buf), size);

  uint8_t addressSpace;
  switch (read_bytes_result) {
  case ReadBytesResult::success_from_file:
    addressSpace = LLDBAddressSpace;
    break;
  case ReadBytesResult::success_from_memory:
    addressSpace = swift::remote::RemoteAddress::DefaultAddressSpace;
    break;
  case ReadBytesResult::fail:
    return false;
  }
  ByteOrder byte_order = m_process.GetTarget().GetArchitecture().GetByteOrder();
  uint32_t byte_size =
      m_process.GetTarget().GetArchitecture().GetAddressByteSize();
  DataExtractor extractor((const void *)&buf, size, byte_order, byte_size);
  lldb::offset_t offset = 0;
  auto data = extractor.GetMaxU64(&offset, size);
  out = swift::remote::RemoteAddress(data, addressSpace);
  return true;
}

std::optional<swift::reflection::RemoteAddress>
LLDBMemoryReader::resolveRemoteAddress(
    swift::reflection::RemoteAddress address) const {
  std::optional<Address> lldb_address =
      LLDBMemoryReader::remoteAddressToLLDBAddress(address);
  if (!lldb_address)
    return {};
  lldb::addr_t addr = lldb_address->GetLoadAddress(&m_process.GetTarget());
  if (addr != LLDB_INVALID_ADDRESS)
    return swift::reflection::RemoteAddress(
        addr, swift::reflection::RemoteAddress::DefaultAddressSpace);
  return {};
}

std::optional<Address> LLDBMemoryReader::remoteAddressToLLDBAddress(
    swift::remote::RemoteAddress address) const {
  Log *log(GetLog(LLDBLog::Types));
  auto maybe_pair = getFileAddressAndModuleForTaggedAddress(address);
  if (!maybe_pair)
    return Address(address.getRawAddress());

  uint64_t file_address = maybe_pair->first;
  ModuleSP module = maybe_pair->second;

  if (m_modules_with_metadata_in_symbol_obj_file.count(module))
    return Address(address.getRawAddress());

  auto *object_file = module->GetObjectFile();
  if (!object_file)
    return {};

  Address resolved(file_address, object_file->GetSectionList());

  // If the address doesn't have a section it means we couldn't find a section
  // that contains that file address, and the "resolved" instance is wrong.
  // Calculate the virtual address by finding out the slide of the associated
  // module, and adding that to the file address.
  if (resolved.GetSection()) {
    LLDB_LOGV(log,
              "[MemoryReader] Successfully resolved mapped address {0:x} into "
              "file address {1:x}",
              address.getRawAddress(), resolved.GetFileAddress());
    return resolved;
  }
  auto *sec_list = module->GetSectionList();
  if (sec_list->GetSize() == 0) {
    LLDB_LOG(log,
             "[MemoryReader] Could not calculate virtual address from file "
             "address {0:x}, no sections in {1}",
             file_address, object_file->GetFileSpec().GetFilename());
    return {};
  }
  SectionSP sec = sec_list->GetSectionAtIndex(0);
  addr_t sec_file_address = sec->GetFileAddress();
  addr_t sec_load_address = sec->GetLoadBaseAddress(&m_process.GetTarget());

  if (sec_load_address < sec_file_address) {
    LLDB_LOG(log,
             "[MemoryReader] section load address {0:x} is smaller than "
             "section file address {1:x}",
             sec_load_address, sec_file_address);
    return {};
  }

  addr_t slide = sec_load_address - sec_file_address;

  bool overflow = false;
  addr_t virtual_address = llvm::SaturatingAdd(file_address, slide, &overflow);
  if (overflow) {
    LLDB_LOG(log, "[MemoryReader] file address {0:x} + slide {1:x} overflows",
             sec_load_address, sec_file_address);
    return {};
  }

  resolved = Address(virtual_address);
  LLDB_LOGV(log,
            "[MemoryReader] Could not find section with file address {0:x} "
            "and file {1}, resolved it into virtual address {2:x}",
            file_address, object_file->GetFileSpec().GetFilename(),
            virtual_address);
  return resolved;
}

std::optional<Address>
LLDBMemoryReader::resolveRemoteAddressFromSymbolObjectFile(
    swift::remote::RemoteAddress address) const {
  Log *log(GetLog(LLDBLog::Types));

  if (!m_process.GetTarget().GetSwiftReadMetadataFromDSYM())
    return {};

  auto maybe_pair = getFileAddressAndModuleForTaggedAddress(address);
  if (!maybe_pair)
    return {};

  uint64_t file_address = maybe_pair->first;
  ModuleSP module = maybe_pair->second;

  if (!m_modules_with_metadata_in_symbol_obj_file.count(module))
    return {};

  auto *symbol_file = module->GetSymbolFile();
  if (!symbol_file)
    return {};

  auto *object_file = symbol_file->GetObjectFile();
  if (!object_file)
    return {};

  Address resolved(file_address, object_file->GetSectionList());
  if (!resolved.IsSectionOffset()) {
    LLDB_LOG(log,
             "[MemoryReader] Could not make a real address out of file address "
             "{0:x} and object file {1}",
             file_address, object_file->GetFileSpec().GetFilename());
    return {};
  }

  if (!resolved.GetSection()
           ->GetParent()
           ->GetName()
           .GetStringRef()
           .contains_insensitive("DWARF")) {
    auto *main_object_file = module->GetObjectFile();
    resolved = Address(file_address, main_object_file->GetSectionList());
  }
  LLDB_LOGV(log,
            "[MemoryReader] Successfully resolved mapped address {0:x} into "
            "file address {1:x} from symbol object file.",
            address.getRawAddress(), file_address);
  return resolved;
}

bool LLDBMemoryReader::readMetadataFromFileCacheEnabled() const {
  return m_process.GetTarget().GetSwiftReadMetadataFromFileCache();
}
} // namespace lldb_private
