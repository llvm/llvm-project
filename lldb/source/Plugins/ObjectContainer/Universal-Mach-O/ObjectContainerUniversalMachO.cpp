//===-- ObjectContainerUniversalMachO.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ObjectContainerUniversalMachO.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBuffer.h"
#include "lldb/Utility/Stream.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm::MachO;

void ObjectContainerUniversalMachO::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                GetModuleSpecifications);
}

void ObjectContainerUniversalMachO::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::ConstString ObjectContainerUniversalMachO::GetPluginNameStatic() {
  static ConstString g_name("mach-o");
  return g_name;
}

const char *ObjectContainerUniversalMachO::GetPluginDescriptionStatic() {
  return "Universal mach-o object container reader.";
}

ObjectContainer *ObjectContainerUniversalMachO::CreateInstance(
    const lldb::ModuleSP &module_sp, DataBufferSP &data_sp,
    lldb::offset_t data_offset, const FileSpec *file,
    lldb::offset_t file_offset, lldb::offset_t length) {
  // We get data when we aren't trying to look for cached container
  // information, so only try and look for an architecture slice if we get data
  if (data_sp) {
    DataExtractor data;
    data.SetData(data_sp, data_offset, length);
    if (ObjectContainerUniversalMachO::MagicBytesMatch(data)) {
      std::unique_ptr<ObjectContainerUniversalMachO> container_up(
          new ObjectContainerUniversalMachO(module_sp, data_sp, data_offset,
                                            file, file_offset, length));
      if (container_up->ParseHeader()) {
        return container_up.release();
      }
    }
  }
  return nullptr;
}

bool ObjectContainerUniversalMachO::MagicBytesMatch(const DataExtractor &data) {
  lldb::offset_t offset = 0;
  uint32_t magic = data.GetU32(&offset);
  return magic == FAT_MAGIC || magic == FAT_CIGAM;
}

ObjectContainerUniversalMachO::ObjectContainerUniversalMachO(
    const lldb::ModuleSP &module_sp, DataBufferSP &data_sp,
    lldb::offset_t data_offset, const FileSpec *file,
    lldb::offset_t file_offset, lldb::offset_t length)
    : ObjectContainer(module_sp, file, file_offset, length, data_sp,
                      data_offset),
      m_header(), m_fat_archs() {
  memset(&m_header, 0, sizeof(m_header));
}

ObjectContainerUniversalMachO::~ObjectContainerUniversalMachO() {}

bool ObjectContainerUniversalMachO::ParseHeader() {
  bool success = ParseHeader(m_data, m_header, m_fat_archs);
  // We no longer need any data, we parsed all we needed to parse and cached it
  // in m_header and m_fat_archs
  m_data.Clear();
  return success;
}

bool ObjectContainerUniversalMachO::ParseHeader(
    lldb_private::DataExtractor &data, llvm::MachO::fat_header &header,
    std::vector<llvm::MachO::fat_arch> &fat_archs) {
  bool success = false;
  // Store the file offset for this universal file as we could have a universal
  // .o file in a BSD archive, or be contained in another kind of object.
  // Universal mach-o files always have their headers in big endian.
  lldb::offset_t offset = 0;
  data.SetByteOrder(eByteOrderBig);
  header.magic = data.GetU32(&offset);
  fat_archs.clear();

  if (header.magic == FAT_MAGIC) {

    data.SetAddressByteSize(4);

    header.nfat_arch = data.GetU32(&offset);

    // Now we should have enough data for all of the fat headers, so lets index
    // them so we know how many architectures that this universal binary
    // contains.
    uint32_t arch_idx = 0;
    for (arch_idx = 0; arch_idx < header.nfat_arch; ++arch_idx) {
      if (data.ValidOffsetForDataOfSize(offset, sizeof(fat_arch))) {
        fat_arch arch;
        if (data.GetU32(&offset, &arch, sizeof(fat_arch) / sizeof(uint32_t)))
          fat_archs.push_back(arch);
      }
    }
    success = true;
  } else {
    memset(&header, 0, sizeof(header));
  }
  return success;
}

void ObjectContainerUniversalMachO::Dump(Stream *s) const {
  s->Printf("%p: ", static_cast<const void *>(this));
  s->Indent();
  const size_t num_archs = GetNumArchitectures();
  const size_t num_objects = GetNumObjects();
  s->Printf("ObjectContainerUniversalMachO, num_archs = %zu, num_objects = %zu",
            num_archs, num_objects);
  uint32_t i;
  ArchSpec arch;
  s->IndentMore();
  for (i = 0; i < num_archs; i++) {
    s->Indent();
    GetArchitectureAtIndex(i, arch);
    s->Printf("arch[%u] = %s\n", i, arch.GetArchitectureName());
  }
  for (i = 0; i < num_objects; i++) {
    s->Indent();
    s->Printf("object[%u] = %s\n", i, GetObjectNameAtIndex(i));
  }
  s->IndentLess();
  s->EOL();
}

size_t ObjectContainerUniversalMachO::GetNumArchitectures() const {
  return m_header.nfat_arch;
}

bool ObjectContainerUniversalMachO::GetArchitectureAtIndex(
    uint32_t idx, ArchSpec &arch) const {
  if (idx < m_header.nfat_arch) {
    arch.SetArchitecture(eArchTypeMachO, m_fat_archs[idx].cputype,
                         m_fat_archs[idx].cpusubtype);
    return true;
  }
  return false;
}

ObjectFileSP
ObjectContainerUniversalMachO::GetObjectFile(const FileSpec *file) {
  uint32_t arch_idx = 0;
  ArchSpec arch;
  // If the module hasn't specified an architecture yet, set it to the default
  // architecture:
  ModuleSP module_sp(GetModule());
  if (module_sp) {
    if (!module_sp->GetArchitecture().IsValid()) {
      arch = Target::GetDefaultArchitecture();
      if (!arch.IsValid())
        arch.SetTriple(LLDB_ARCH_DEFAULT);
    } else
      arch = module_sp->GetArchitecture();

    ArchSpec curr_arch;
    // First, try to find an exact match for the Arch of the Target.
    for (arch_idx = 0; arch_idx < m_header.nfat_arch; ++arch_idx) {
      if (GetArchitectureAtIndex(arch_idx, curr_arch) &&
          arch.IsExactMatch(curr_arch))
        break;
    }

    // Failing an exact match, try to find a compatible Arch of the Target.
    if (arch_idx >= m_header.nfat_arch) {
      for (arch_idx = 0; arch_idx < m_header.nfat_arch; ++arch_idx) {
        if (GetArchitectureAtIndex(arch_idx, curr_arch) &&
            arch.IsCompatibleMatch(curr_arch))
          break;
      }
    }

    if (arch_idx < m_header.nfat_arch) {
      DataBufferSP data_sp;
      lldb::offset_t data_offset = 0;
      return ObjectFile::FindPlugin(
          module_sp, file, m_offset + m_fat_archs[arch_idx].offset,
          m_fat_archs[arch_idx].size, data_sp, data_offset);
    }
  }
  return ObjectFileSP();
}

// PluginInterface protocol
lldb_private::ConstString ObjectContainerUniversalMachO::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t ObjectContainerUniversalMachO::GetPluginVersion() { return 1; }

size_t ObjectContainerUniversalMachO::GetModuleSpecifications(
    const lldb_private::FileSpec &file, lldb::DataBufferSP &data_sp,
    lldb::offset_t data_offset, lldb::offset_t file_offset,
    lldb::offset_t file_size, lldb_private::ModuleSpecList &specs) {
  const size_t initial_count = specs.GetSize();

  DataExtractor data;
  data.SetData(data_sp, data_offset, data_sp->GetByteSize());

  if (ObjectContainerUniversalMachO::MagicBytesMatch(data)) {
    llvm::MachO::fat_header header;
    std::vector<llvm::MachO::fat_arch> fat_archs;
    if (ParseHeader(data, header, fat_archs)) {
      for (const llvm::MachO::fat_arch &fat_arch : fat_archs) {
        const lldb::offset_t slice_file_offset = fat_arch.offset + file_offset;
        if (fat_arch.offset < file_size && file_size > slice_file_offset) {
          ObjectFile::GetModuleSpecifications(
              file, slice_file_offset, file_size - slice_file_offset, specs);
        }
      }
    }
  }
  return specs.GetSize() - initial_count;
}
