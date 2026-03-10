//===-- TraceBundleLoader.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceBundleLoader.h"

#include "ThreadPostMortemTrace.h"
#include "TraceJSONStructs.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/ProcessTrace.h"
#include "lldb/Target/Target.h"
#include <optional>

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

FileSpec TraceBundleLoader::NormalizePath(const std::string &path) {
  FileSpec file_spec(path);
  if (file_spec.IsRelative())
    file_spec.PrependPathComponent(m_bundle_dir);
  return file_spec;
}

Error TraceBundleLoader::ParseModule(Target &target, const JSONModule &module) {
  auto do_parse = [&]() -> Error {
    FileSpec system_file_spec(module.system_path);

    FileSpec local_file_spec(module.file.has_value() ? *module.file
                                                     : module.system_path);

    ModuleSpec module_spec;
    module_spec.GetFileSpec() = local_file_spec;
    module_spec.GetPlatformFileSpec() = system_file_spec;

    if (module.uuid.has_value())
      module_spec.GetUUID().SetFromStringRef(*module.uuid);

    Status error;
    ModuleSP module_sp =
        target.GetOrCreateModule(module_spec, /*notify*/ false, &error);

    if (error.Fail())
      return error.ToError();

    bool load_addr_changed = false;
    module_sp->SetLoadAddress(target, module.load_address.value, false,
                              load_addr_changed);
    return Error::success();
  };
  if (Error err = do_parse())
    return createStringError(
        inconvertibleErrorCode(), "Error when parsing module %s. %s",
        module.system_path.c_str(), toString(std::move(err)).c_str());
  return Error::success();
}

Expected<TraceBundleLoader::ParsedProcess>
TraceBundleLoader::CreateEmptyProcess(lldb::pid_t pid, llvm::StringRef triple) {
  TargetSP target_sp;
  Status error = m_debugger.GetTargetList().CreateTarget(
      m_debugger, /*user_exe_path*/ StringRef(), triple, eLoadDependentsNo,
      /*platform_options*/ nullptr, target_sp);

  if (!target_sp)
    return error.ToError();

  ParsedProcess parsed_process;
  parsed_process.target_sp = target_sp;

  ProcessTrace::Initialize();
  ProcessSP process_sp = target_sp->CreateProcess(
      /*listener*/ nullptr, "trace",
      /*crash_file*/ nullptr,
      /*can_connect*/ false);

  process_sp->SetID(static_cast<lldb::pid_t>(pid));

  return parsed_process;
}
