//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_OBJECTCONTAINER_CLANG_OFFLOAD_BUNDLE_OBJECTCONTAINERCLANGOFFLOADBUNDLE_H
#define LLDB_SOURCE_PLUGINS_OBJECTCONTAINER_CLANG_OFFLOAD_BUNDLE_OBJECTCONTAINERCLANGOFFLOADBUNDLE_H

#include "lldb/Symbol/ObjectContainer.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/FileSpec.h"

#include <vector>

class ObjectContainerClangOffloadBundle : public lldb_private::ObjectContainer {
public:
  ObjectContainerClangOffloadBundle(const lldb::ModuleSP &module_sp,
                                    lldb::DataBufferSP &data_sp,
                                    lldb::offset_t data_offset,
                                    const lldb_private::FileSpec *file,
                                    lldb::offset_t file_offset,
                                    lldb::offset_t length);

  ~ObjectContainerClangOffloadBundle() override;

  static void Initialize();
  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() {
    return "clang-offload-bundle";
  }

  static llvm::StringRef GetPluginDescriptionStatic() {
    return "Clang offload bundle object container reader for ELF files.";
  }

  static lldb_private::ObjectContainer *
  CreateInstance(const lldb::ModuleSP &module_sp, lldb::DataBufferSP &data_sp,
                 lldb::offset_t data_offset, const lldb_private::FileSpec *file,
                 lldb::offset_t file_offset, lldb::offset_t length);

  static lldb_private::ModuleSpecList
  GetModuleSpecifications(const lldb_private::FileSpec &file,
                          lldb::DataExtractorSP &extractor_sp,
                          lldb::offset_t file_offset, lldb::offset_t file_size);

  static bool MagicBytesMatch(const lldb_private::DataExtractor &data);

  bool ParseHeader() override;

  size_t GetNumArchitectures() const override;

  bool GetArchitectureAtIndex(uint32_t idx,
                              lldb_private::ArchSpec &arch) const override;

  lldb::ObjectFileSP GetObjectFile(const lldb_private::FileSpec *file) override;

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

protected:
  struct Entry {
    lldb_private::ArchSpec arch;
    uint64_t offset = 0;
    uint64_t size = 0;
  };

  std::vector<Entry> m_entries;

  static bool FindBundleEntries(const lldb_private::FileSpec &file,
                                lldb::DataExtractorSP extractor_sp,
                                lldb::offset_t file_offset,
                                lldb::offset_t file_size,
                                std::vector<Entry> &entries);
};

#endif // LLDB_SOURCE_PLUGINS_OBJECTCONTAINER_CLANG_OFFLOAD_BUNDLE_OBJECTCONTAINERCLANGOFFLOADBUNDLE_H
