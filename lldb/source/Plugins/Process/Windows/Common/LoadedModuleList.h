//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_WINDOWS_COMMON_LOADEDMODULELIST_H
#define LLDB_SOURCE_PLUGINS_PROCESS_WINDOWS_COMMON_LOADEDMODULELIST_H

#include "lldb/Utility/FileSpec.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <map>
#include <optional>

namespace lldb_private {

/// The images mapped into the inferior. The same file can be mapped more than
/// once, and an UNLOAD_DLL_DEBUG_EVENT only carries a base address, so every
/// mapping has to be resolvable back to its file.
class LoadedModuleList {
  using Container = std::map<FileSpec, llvm::SmallVector<lldb::addr_t, 1>>;

public:
  void Add(const FileSpec &file_spec, lldb::addr_t base_addr) {
    m_modules[file_spec].push_back(base_addr);
  }

  /// Drop the mapping at \p base_addr. \return the file if that was its last
  /// mapping, an empty FileSpec otherwise.
  FileSpec Remove(lldb::addr_t base_addr) {
    for (auto it = m_modules.begin(); it != m_modules.end(); ++it) {
      auto &base_addrs = it->second;
      auto addr_it = llvm::find(base_addrs, base_addr);
      if (addr_it == base_addrs.end())
        continue;

      base_addrs.erase(addr_it);
      if (!base_addrs.empty())
        return {};

      FileSpec file_spec = it->first;
      m_modules.erase(it);
      return file_spec;
    }
    return {};
  }

  /// The address \p file_spec was first mapped at. That is the mapping the
  /// loader bound the image's imports against, so it is the one to report.
  std::optional<lldb::addr_t> GetBaseAddress(const FileSpec &file_spec) const {
    auto it = m_modules.find(file_spec);
    if (it == m_modules.end())
      return std::nullopt;
    return it->second.front();
  }

  /// \return the stored spelling of \p file_spec, or nullptr if it is not
  /// mapped.
  const FileSpec *FindFile(const FileSpec &file_spec) const {
    auto it = m_modules.find(file_spec);
    return it == m_modules.end() ? nullptr : &it->first;
  }

  bool IsEmpty() const { return m_modules.empty(); }
  size_t GetSize() const { return m_modules.size(); }

  Container::const_iterator begin() const { return m_modules.begin(); }
  Container::const_iterator end() const { return m_modules.end(); }

private:
  Container m_modules;
};

} // namespace lldb_private

#endif
