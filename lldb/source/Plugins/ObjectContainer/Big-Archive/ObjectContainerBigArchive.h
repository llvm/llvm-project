//===-- ObjectContainerBigArchive.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_OBJECTCONTAINER_BIG_ARCHIVE_OBJECTCONTAINERBIGARCHIVE_H
#define LLDB_SOURCE_PLUGINS_OBJECTCONTAINER_BIG_ARCHIVE_OBJECTCONTAINERBIGARCHIVE_H

#include "lldb/Symbol/ObjectContainer.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/FileSpec.h"

// This file represents an AIX Big Archive and combines several files into one.
// It is the default library archive format for the AIX operating system.
// Ref: https://www.ibm.com/docs/en/aix/7.3.0?topic=formats-ar-file-format-big

class ObjectContainerBigArchive : public lldb_private::ObjectContainer {
public:
  ObjectContainerBigArchive(const lldb::ModuleSP &module_sp,
                            lldb::DataBufferSP &data_sp,
                            lldb::offset_t data_offset,
                            const lldb_private::FileSpec *file,
                            lldb::offset_t offset, lldb::offset_t length);

  ~ObjectContainerBigArchive() override;

  // Static Functions
  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "big-archive"; }

  static llvm::StringRef GetPluginDescriptionStatic() {
    return "Big Archive object container reader.";
  }

  static lldb_private::ObjectContainer *
  CreateInstance(const lldb::ModuleSP &module_sp, lldb::DataBufferSP &data_sp,
                 lldb::offset_t data_offset, const lldb_private::FileSpec *file,
                 lldb::offset_t offset, lldb::offset_t length);

  static size_t GetModuleSpecifications(const lldb_private::FileSpec &file,
                                        lldb::DataBufferSP &data_sp,
                                        lldb::offset_t data_offset,
                                        lldb::offset_t file_offset,
                                        lldb::offset_t length,
                                        lldb_private::ModuleSpecList &specs);

  // Member Functions
  bool ParseHeader() override;

  size_t GetNumObjects() const override {
    if (m_archive_sp)
      return m_archive_sp->GetNumObjects();
    return 0;
  }

  lldb::ObjectFileSP GetObjectFile(const lldb_private::FileSpec *file) override;

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

protected:
  struct Object {
    Object();

    void Clear();

    lldb::offset_t Extract(const lldb_private::DataExtractor &data,
                           lldb::offset_t offset);
    /// Object name in the archive.
    lldb_private::ConstString ar_name;

    /// Object modification time in the archive.
    uint32_t modification_time = 0;

    /// Object user id in the archive.
    uint16_t uid = 0;

    /// Object group id in the archive.
    uint16_t gid = 0;

    /// Object octal file permissions in the archive.
    uint16_t mode = 0;

    /// Object size in bytes in the archive.
    uint32_t size = 0;

    /// File offset in bytes from the beginning of the file of the object data.
    lldb::offset_t file_offset = 0;

    /// Length of the object data in bytes.
    lldb::offset_t file_size = 0;

    void Dump(lldb_private::Stream *s) const;
  };

  class Archive {
  public:
    typedef std::shared_ptr<Archive> shared_ptr;
    typedef std::multimap<lldb_private::FileSpec, shared_ptr> Map;

    Archive(const lldb_private::ArchSpec &arch,
            const llvm::sys::TimePoint<> &mod_time, lldb::offset_t file_offset,
            lldb::DataExtractorSP extractor_sp);

    ~Archive();

    size_t GetNumObjects() const { return m_objects.size(); }

    lldb::offset_t GetFileOffset() const { return m_file_offset; }

    const lldb_private::ArchSpec &GetArchitecture() const { return m_arch; }

    void SetArchitecture(const lldb_private::ArchSpec &arch) { m_arch = arch; }

    lldb_private::DataExtractor &GetData() { return *m_extractor_sp.get(); }
    lldb::DataExtractorSP &GetDataSP() { return m_extractor_sp; }

  protected:
    // Member Variables
    lldb_private::ArchSpec m_arch;
    llvm::sys::TimePoint<> m_modification_time;
    lldb::offset_t m_file_offset;
    std::vector<Object> m_objects;
    ///< The data extractor for this object container
    /// so we don't lose data if the .a files
    /// gets modified
    lldb::DataExtractorSP m_extractor_sp;
  };

  void SetArchive(Archive::shared_ptr &archive_sp);

  Archive::shared_ptr m_archive_sp;
};

#endif // LLDB_SOURCE_PLUGINS_OBJECTCONTAINER_BIG_ARCHIVE_OBJECTCONTAINERBIGARCHIVE_H
