//===-- ObjectFileMinidump.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ObjectFileMinidump.h"

#include "MinidumpFileBuilder.h"

#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Target/Process.h"

#include "llvm/Support/FileSystem.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ObjectFileMinidump)

void ObjectFileMinidump::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), GetPluginDescriptionStatic(), CreateInstance,
      CreateMemoryInstance, GetModuleSpecifications, SaveCore);
}

void ObjectFileMinidump::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

ObjectFile *ObjectFileMinidump::CreateInstance(
    const lldb::ModuleSP &module_sp, lldb::DataBufferSP data_sp,
    lldb::offset_t data_offset, const lldb_private::FileSpec *file,
    lldb::offset_t offset, lldb::offset_t length) {
  return nullptr;
}

ObjectFile *ObjectFileMinidump::CreateMemoryInstance(
    const lldb::ModuleSP &module_sp, WritableDataBufferSP data_sp,
    const ProcessSP &process_sp, lldb::addr_t header_addr) {
  return nullptr;
}

size_t ObjectFileMinidump::GetModuleSpecifications(
    const lldb_private::FileSpec &file, lldb::DataBufferSP &data_sp,
    lldb::offset_t data_offset, lldb::offset_t file_offset,
    lldb::offset_t length, lldb_private::ModuleSpecList &specs) {
  specs.Clear();
  return 0;
}

bool ObjectFileMinidump::SaveCore(const lldb::ProcessSP &process_sp,
                                  const lldb_private::FileSpec &outfile,
                                  lldb::SaveCoreStyle &core_style,
                                  lldb_private::Status &error) {
  // Set default core style if it isn't set.
  if (core_style == SaveCoreStyle::eSaveCoreUnspecified)
    core_style = SaveCoreStyle::eSaveCoreStackOnly;

  if (!process_sp)
    return false;

  MinidumpFileBuilder builder;

  Target &target = process_sp->GetTarget();

  error = builder.AddSystemInfo(target.GetArchitecture().GetTriple());
  if (error.Fail())
    return false;

  error = builder.AddModuleList(target);
  if (error.Fail())
    return false;

  builder.AddMiscInfo(process_sp);

  error = builder.AddThreadList(process_sp);
  if (error.Fail())
    return false;

  // Add any exceptions but only if there are any in any threads.
  builder.AddExceptions(process_sp);

  error = builder.AddMemoryList(process_sp, core_style);
  if (error.Fail())
    return false;

  if (target.GetArchitecture().GetTriple().getOS() ==
      llvm::Triple::OSType::Linux) {
    builder.AddLinuxFileStreams(process_sp);
  }

  llvm::Expected<lldb::FileUP> maybe_core_file = FileSystem::Instance().Open(
      outfile, File::eOpenOptionWriteOnly | File::eOpenOptionCanCreate);
  if (!maybe_core_file) {
    error = maybe_core_file.takeError();
    return false;
  }
  lldb::FileUP core_file = std::move(maybe_core_file.get());

  error = builder.Dump(core_file);
  if (error.Fail())
    return false;

  return true;
}
