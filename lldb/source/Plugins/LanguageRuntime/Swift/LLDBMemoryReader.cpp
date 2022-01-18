#include "LLDBMemoryReader.h"
#include "lldb/Core/Section.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Logging.h"

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

  if (size > m_max_read_amount) {
    LLDB_LOGV(log, "[MemoryReader] memory read exceeds maximum allowed size");
    return false;
  }

  Target &target(m_process.GetTarget());
  Address addr(address.getAddressData());
  Status error;
  if (size > target.ReadMemory(addr, dest, size, error, true)) {
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

  LLDB_LOGV(log, "[MemoryReader] asked to read string data at address {0:x}",
            address.getAddressData());

  Target &target(m_process.GetTarget());
  Address addr(address.getAddressData());
  Status error;
  target.ReadCStringFromMemory(addr, dest, error);
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
  } else {
    LLDB_LOGV(log, "[MemoryReader] memory read returned error: {0}",
              error.AsCString());
    return false;
  }
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

} // namespace lldb_private
