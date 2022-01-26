#include "LLDBMemoryReader.h"
#include "lldb/Core/Section.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Logging.h"

#include "llvm/Support/MathExtras.h"

using namespace lldb;
using namespace lldb_private;

namespace lldb_private {
bool LLDBMemoryReader::queryDataLayout(DataLayoutQueryType type, void *inBuffer,
                                       void *outBuffer) {
  switch (type) {
  // FIXME: add support for case DLQ_GetPtrAuthMask rdar://70729149
  case DLQ_GetPtrAuthMask:
    return false;
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
  }
}

swift::remote::RemoteAddress
LLDBMemoryReader::getSymbolAddress(const std::string &name) {
  lldbassert(!name.empty());
  if (name.empty())
    return swift::remote::RemoteAddress(nullptr);

  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES));

  LLDB_LOGV(log, "[MemoryReader] asked to retrieve the address of symbol {0}",
            name);

  ConstString name_cs(name.c_str(), name.size());
  SymbolContextList sc_list;
  m_process.GetTarget().GetImages().FindSymbolsWithNameAndType(
      name_cs, lldb::eSymbolTypeAny, sc_list);
  if (!sc_list.GetSize()) {
    LLDB_LOGV(log, "[MemoryReader] symbol resolution failed {0}", name);
    return swift::remote::RemoteAddress(nullptr);
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
      return swift::remote::RemoteAddress(load_addr);
    }
  }

  // Empty list, resolution failed.
  if (sc_list.GetSize() == 0) {
    LLDB_LOGV(log, "[MemoryReader] symbol resolution failed {0}", name);
    return swift::remote::RemoteAddress(nullptr);
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
      return swift::remote::RemoteAddress(nullptr);
    }
  }
  LLDB_LOGV(log, "[MemoryReader] symbol resolved to {0}", load_addr);
  return swift::remote::RemoteAddress(load_addr);
}

bool LLDBMemoryReader::readBytes(swift::remote::RemoteAddress address,
                                 uint8_t *dest, uint64_t size) {
  if (m_local_buffer) {
    auto addr = address.getAddressData();
    if (addr >= *m_local_buffer &&
        addr + size <= *m_local_buffer + m_local_buffer_size) {
      // If this crashes, the assumptions stated in
      // GetDynamicTypeAndAddress_Protocol() most likely no longer
      // hold.
      memcpy(dest, (void *)addr, size);
      return true;
    }
  }

  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES));

  LLDB_LOGV(log, "[MemoryReader] asked to read {0} bytes at address {1:x}",
            size, address.getAddressData());

  llvm::Optional<Address> maybeAddr =
      resolveRemoteAddress(address.getAddressData());
  if (!maybeAddr) {
    LLDB_LOGV(log, "[MemoryReader] could not resolve address {1:x}",
              address.getAddressData());
    return false;
  }
  auto addr = *maybeAddr;

  if (size > m_max_read_amount) {
    LLDB_LOGV(log, "[MemoryReader] memory read exceeds maximum allowed size");
    return false;
  }
  Target &target(m_process.GetTarget());
  Status error;
  // We only want to allow the file-cache optimization if we resolved the 
  // address to section + offset.
  const bool force_live_memory =
      !readMetadataFromFileCacheEnabled() || !addr.IsSectionOffset();
  if (size > target.ReadMemory(addr, dest, size, error, force_live_memory)) {
    LLDB_LOGV(log,
              "[MemoryReader] memory read returned fewer bytes than asked for");
    return false;
  }
  if (error.Fail()) {
    LLDB_LOGV(log, "[MemoryReader] memory read returned error: {0}",
              error.AsCString());
    return false;
  }

  auto format_data = [](auto dest, auto size) {
    StreamString stream;
    for (uint64_t i = 0; i < size; i++) {
      stream.PutHex8(dest[i]);
      stream.PutChar(' ');
    }
    return stream.GetData();
  };
  LLDB_LOGV(log, "[MemoryReader] memory read returned data: {0}",
            format_data(dest, size));

  return true;
}

bool LLDBMemoryReader::readString(swift::remote::RemoteAddress address,
                                  std::string &dest) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES));

  LLDB_LOGV(log, "[MemoryReader] asked to read string data at address {0x}",
            address.getAddressData());

  llvm::Optional<Address> maybeAddr =
      resolveRemoteAddress(address.getAddressData());
  if (!maybeAddr) {
    LLDB_LOGV(log, "[MemoryReader] could not resolve address {1:x}",
              address.getAddressData());
    return false;
  }
  auto addr = *maybeAddr;

  Target &target(m_process.GetTarget());
  Status error;
  // We only want to allow the file-cache optimization if we resolved the 
  // address to section + offset.
  const bool force_live_memory =
      !readMetadataFromFileCacheEnabled() || !addr.IsSectionOffset();
  target.ReadCStringFromMemory(addr, dest, error, force_live_memory);
  if (error.Success()) {
    auto format_string = [](const std::string &dest) {
      StreamString stream;
      for (auto c : dest) {
        if (c >= 32 && c <= 127) {
          stream << c;
        } else {
          stream << "\\0";
          stream.PutHex8(c);
        }
      }
      return stream.GetData();
    };
    LLDB_LOGV(log, "[MemoryReader] memory read returned data: \"{0}\"",
              format_string(dest));
    return true;
  }
  LLDB_LOGV(log, "[MemoryReader] memory read returned error: {0}",
            error.AsCString());
  return false;
}

void LLDBMemoryReader::pushLocalBuffer(uint64_t local_buffer,
                                       uint64_t local_buffer_size) {
  lldbassert(!m_local_buffer);
  m_local_buffer = local_buffer;
  m_local_buffer_size = local_buffer_size;
}

void LLDBMemoryReader::popLocalBuffer() {
  lldbassert(m_local_buffer);
  m_local_buffer.reset();
  m_local_buffer_size = 0;
}

llvm::Optional<std::pair<uint64_t, uint64_t>>
LLDBMemoryReader::addModuleToAddressMap(ModuleSP module) {
  if (!readMetadataFromFileCacheEnabled())
    return {};

  // The first available address is the mask, since subsequent images are mapped
  // in ascending order, all of them will contain this mask.
  uint64_t module_start_address = LLDB_FILE_ADDRESS_BIT;
  if (!m_range_module_map.empty())
    // We map the images contiguously one after the other, all with the tag bit
    // set.
    // The address that maps the last module is exactly the address the new
    // module should start at.
    module_start_address = m_range_module_map.back().first;

#ifndef NDEBUG
  static std::initializer_list<uint64_t> objc_bits = {
      SWIFT_ABI_ARM_IS_OBJC_BIT,
      SWIFT_ABI_X86_64_IS_OBJC_BIT,
      SWIFT_ABI_ARM64_IS_OBJC_BIT};

  for (auto objc_bit : objc_bits) 
    assert((module_start_address & objc_bit) != objc_bit &&
           "LLDB file address bit clashes with an obj-c bit!");
#endif

  SectionList *section_list = module->GetObjectFile()->GetSectionList();

  auto section_list_size = section_list->GetSize();
  if (section_list_size == 0)
    return {};

  auto last_section =
      section_list->GetSectionAtIndex(section_list->GetSize() - 1);
  // The virtual file address + the size of last section gives us the total size
  // of this image in memory.
  uint64_t size = last_section->GetFileAddress() + last_section->GetByteSize();
  auto module_end_address = module_start_address + size;

  // The address for the next image is the next pointer aligned address
  // available after the end of the current image.
  uint64_t next_module_start_address = llvm::alignTo(module_end_address, 8);
  m_range_module_map.emplace_back(next_module_start_address, module);
  return {{module_start_address, module_end_address}};
}

llvm::Optional<Address>
LLDBMemoryReader::resolveRemoteAddress(uint64_t address) const {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES));

  if (!m_process.GetTarget().GetSwiftReadMetadataFromFileCache())
    return Address(address);

  // If the address contains our mask, this is an image we registered.
  if (!(address & LLDB_FILE_ADDRESS_BIT))
    return Address(address);

  // Dummy pair with the address we're looking for.
  auto comparison_pair = std::make_pair(address, ModuleSP());

  // Explicitly compare only the addresses, never the modules in the pairs.
  auto pair_iterator = std::lower_bound(
      m_range_module_map.begin(), m_range_module_map.end(), comparison_pair,
      [](auto &a, auto &b) { return a.first < b.first; });

  // If the address is larger than anything we have mapped the address is out
  if (pair_iterator == m_range_module_map.end()) {
    LLDB_LOG(log,
              "[MemoryReader] Address {1:x} is larger than the upper bound "
              "address of the mapped in modules",
              address);
    return {};
  }

  ModuleSP module = pair_iterator->second;
  uint64_t file_address;
  if (pair_iterator == m_range_module_map.begin())
    // Since this is the first registered module,
    // clearing the tag bit will give the virtual file address.
    file_address = address & ~LLDB_FILE_ADDRESS_BIT;
  else
    // The end of the previous section is the start of the current one.
    file_address = address - std::prev(pair_iterator)->first;

  LLDB_LOGV(log,
            "[MemoryReader] Successfully resolved mapped address {1:x} "
            "into file address {1:x}",
            address, file_address);
  auto *object_file = module->GetObjectFile();
  if (!object_file)
    return {};

  Address resolved(file_address, object_file->GetSectionList());
  if (!resolved.IsValid()) {
    LLDB_LOG(log,
             "[MemoryReader] Could not make a real address out of file "
             "address {1:x} and object file {}",
             file_address, object_file->GetFileSpec().GetFilename());
    return {};
  }

  LLDB_LOGV(log,
            "[MemoryReader] Unsuccessfully resolved mapped address {1:x} "
            "into file address {1:x}",
            address, address);
  return resolved;
}

bool LLDBMemoryReader::readMetadataFromFileCacheEnabled() const {
  auto &triple = m_process.GetTarget().GetArchitecture().GetTriple();

  // 32 doesn't have a flag bit we can reliably use, so reading from filecache
  // is disabled on it.
  return m_process.GetTarget().GetSwiftReadMetadataFromFileCache() &&
         triple.isArch64Bit();
}
} // namespace lldb_private
