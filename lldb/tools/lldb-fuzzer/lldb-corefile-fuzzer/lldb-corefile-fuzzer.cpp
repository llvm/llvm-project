//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "utils/TempFile.h"

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/ObjectFile/Mach-O/ObjectFileMachO.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Process/elf-core/ProcessElfCore.h"
#include "Plugins/Process/mach-core/ProcessMachCore.h"
#include "Plugins/ScriptInterpreter/None/ScriptInterpreterNone.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Listener.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MachO.h"

#include <algorithm>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_fuzzer;

namespace {
Debugger &GetDebugger() {
  static DebuggerSP debugger_sp = Debugger::CreateInstance();
  return *debugger_sp;
}
} // namespace

extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
  FileSystem::Initialize();
  HostInfo::Initialize();
  PlatformMacOSX::Initialize();
  platform_linux::PlatformLinux::Initialize();
  ObjectFileMachO::Initialize();
  ObjectFileELF::Initialize();
  ProcessMachCore::Initialize();
  ProcessElfCore::Initialize();
  ScriptInterpreterNone::Initialize();

  Debugger::Initialize(nullptr);
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(uint8_t *data, size_t size) {
  // Neither format parses below the smaller of the two header sizes.
  if (size < std::min(sizeof(llvm::MachO::mach_header_64),
                      sizeof(llvm::ELF::Elf64_Ehdr)))
    return 0;

  std::unique_ptr<TempFile> file = TempFile::Create(data, size);
  if (!file)
    return 0;
  FileSpec core_file(file->GetPath());

  ModuleSpec module_spec(core_file);
  ModuleSP module_sp;
  ModuleList::GetSharedModule(module_spec, module_sp, nullptr, nullptr);
  if (!module_sp)
    return 0;

  ObjectFile *objfile = module_sp->GetObjectFile();
  if (!objfile)
    return 0;

  // For an ELF core these walk PT_NOTE into RefineModuleDetailsFromNote.
  objfile->GetArchitecture();
  objfile->GetUUID();

  objfile->GetAddressableBits();
  objfile->GetCorefileProcessMetadata();
  {
    addr_t value = LLDB_INVALID_ADDRESS;
    bool value_is_offset = false;
    UUID uuid;
    ObjectFile::BinaryType type = ObjectFile::eBinaryTypeInvalid;
    objfile->GetCorefileMainBinaryInfo(value, value_is_offset, uuid, type);
  }

  PlatformSP host_platform_sp = Platform::GetHostPlatform();
  if (host_platform_sp && objfile->GetType() == ObjectFile::eTypeCoreFile) {
    TargetSP target_sp;
    GetDebugger().GetTargetList().CreateTarget(
        GetDebugger(), /*user_exe_path=*/"", ArchSpec(), eLoadDependentsNo,
        host_platform_sp, target_sp);
    if (target_sp) {
      auto listener_sp = Listener::MakeListener("lldb.fuzzer.corefile");

      ProcessSP process_sp = target_sp->CreateProcess(
          listener_sp, llvm::StringRef(), &core_file, /*can_connect=*/false);
      if (process_sp)
        (void)process_sp->LoadCore();
    }
  }

  module_sp.reset();
  ModuleList::RemoveOrphanSharedModules(/*mandatory=*/true);

  return 0;
}
