//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_PROCESS_GDB_REMOTE_REGISTERPACKETLAYOUT_H
#define LLDB_PLUGINS_PROCESS_GDB_REMOTE_REGISTERPACKETLAYOUT_H

#include "lldb/Register/RegisterDescription.h"

namespace lldb_private::process_gdb_remote {

/// RSP placement for one description generation, without copies of its names,
/// numbering, types or relationships. Native storage offsets are not used.
class RegisterPacketLayout {
public:
  struct Entry {
    RegisterID id;
    uint32_t byte_offset;
    uint32_t byte_size;
    bool primary;
  };
  static llvm::Expected<RegisterPacketLayout>
  Create(std::shared_ptr<const RegisterDescription> description,
         llvm::ArrayRef<RegisterID> visible_order, lldb::ByteOrder byte_order);
  llvm::ArrayRef<Entry> GetEntries() const { return m_entries; }
  const RegisterDescription &GetDescription() const { return *m_description; }
  std::optional<uint32_t> GetWireNumber(RegisterID id) const;
  std::vector<uint32_t> GetWireNumbers(llvm::ArrayRef<RegisterID> ids) const;
  uint32_t GetByteSize() const { return m_byte_size; }

private:
  std::shared_ptr<const RegisterDescription> m_description;
  std::vector<Entry> m_entries;
  std::map<RegisterID, uint32_t> m_indices;
  uint32_t m_byte_size = 0;
};
} // namespace lldb_private::process_gdb_remote
#endif
