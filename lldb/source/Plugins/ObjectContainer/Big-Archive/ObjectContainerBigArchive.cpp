//===-- ObjectContainerBigArchive.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ObjectContainerBigArchive.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/ArchSpec.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ObjectContainerBigArchive)

ObjectContainerBigArchive::Archive::Archive(const lldb_private::ArchSpec &arch,
                                            const llvm::sys::TimePoint<> &time,
                                            lldb::offset_t file_offset,
                                            lldb::DataExtractorSP extractor_sp)
    : m_arch(arch), m_modification_time(time), m_file_offset(file_offset),
      m_objects(), m_extractor_sp(extractor_sp) {}

ObjectContainerBigArchive::Archive::~Archive() = default;

void ObjectContainerBigArchive::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                GetModuleSpecifications);
}

void ObjectContainerBigArchive::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

ObjectContainer *ObjectContainerBigArchive::CreateInstance(
    const lldb::ModuleSP &module_sp, DataBufferSP &data_sp,
    lldb::offset_t data_offset, const FileSpec *file,
    lldb::offset_t file_offset, lldb::offset_t length) {
  return nullptr;
}

size_t ObjectContainerBigArchive::GetModuleSpecifications(
    const lldb_private::FileSpec &file, lldb::DataBufferSP &data_sp,
    lldb::offset_t data_offset, lldb::offset_t file_offset,
    lldb::offset_t file_size, lldb_private::ModuleSpecList &specs) {
  return 0;
}

ObjectContainerBigArchive::ObjectContainerBigArchive(
    const lldb::ModuleSP &module_sp, DataBufferSP &data_sp,
    lldb::offset_t data_offset, const lldb_private::FileSpec *file,
    lldb::offset_t file_offset, lldb::offset_t size)
    : ObjectContainer(module_sp, file, file_offset, size, data_sp, data_offset),
      m_archive_sp() {}

void ObjectContainerBigArchive::SetArchive(Archive::shared_ptr &archive_sp) {
  m_archive_sp = archive_sp;
}

ObjectContainerBigArchive::~ObjectContainerBigArchive() = default;

bool ObjectContainerBigArchive::ParseHeader() { return false; }

ObjectFileSP ObjectContainerBigArchive::GetObjectFile(const FileSpec *file) {
  return ObjectFileSP();
}
