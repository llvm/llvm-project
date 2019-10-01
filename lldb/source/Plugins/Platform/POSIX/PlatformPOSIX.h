//===-- PlatformPOSIX.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformPOSIX_h_
#define liblldb_PlatformPOSIX_h_

#include <map>
#include <memory>

#include "lldb/Interpreter/Options.h"
#include "lldb/Target/RemoteAwarePlatform.h"

class PlatformPOSIX : public lldb_private::RemoteAwarePlatform {
public:
  PlatformPOSIX(bool is_host);

  ~PlatformPOSIX() override;

  // lldb_private::Platform functions

  lldb_private::OptionGroupOptions *
  GetConnectionOptions(lldb_private::CommandInterpreter &interpreter) override;

  lldb_private::Status PutFile(const lldb_private::FileSpec &source,
                               const lldb_private::FileSpec &destination,
                               uint32_t uid = UINT32_MAX,
                               uint32_t gid = UINT32_MAX) override;

  lldb_private::Status
  GetFile(const lldb_private::FileSpec &source,
          const lldb_private::FileSpec &destination) override;

  const lldb::UnixSignalsSP &GetRemoteUnixSignals() override;

  lldb_private::Status ResolveExecutable(
      const lldb_private::ModuleSpec &module_spec, lldb::ModuleSP &module_sp,
      const lldb_private::FileSpecList *module_search_paths_ptr) override;

  lldb::ProcessSP Attach(lldb_private::ProcessAttachInfo &attach_info,
                         lldb_private::Debugger &debugger,
                         lldb_private::Target *target, // Can be nullptr, if
                                                       // nullptr create a new
                                                       // target, else use
                                                       // existing one
                         lldb_private::Status &error) override;

  lldb::ProcessSP DebugProcess(lldb_private::ProcessLaunchInfo &launch_info,
                               lldb_private::Debugger &debugger,
                               lldb_private::Target *target, // Can be nullptr,
                                                             // if nullptr
                                                             // create a new
                                                             // target, else use
                                                             // existing one
                               lldb_private::Status &error) override;

  std::string GetPlatformSpecificConnectionInformation() override;

  void CalculateTrapHandlerSymbolNames() override;

  lldb_private::Status ConnectRemote(lldb_private::Args &args) override;

  lldb_private::Status DisconnectRemote() override;

  uint32_t DoLoadImage(lldb_private::Process *process,
                       const lldb_private::FileSpec &remote_file,
                       const std::vector<std::string> *paths,
                       lldb_private::Status &error,
                       lldb_private::FileSpec *loaded_image) override;

  lldb_private::Status UnloadImage(lldb_private::Process *process,
                                   uint32_t image_token) override;

  size_t ConnectToWaitingProcesses(lldb_private::Debugger &debugger,
                                   lldb_private::Status &error) override;

  lldb_private::ConstString GetFullNameForDylib(lldb_private::ConstString basename) override;

protected:
  std::unique_ptr<lldb_private::OptionGroupPlatformRSync>
      m_option_group_platform_rsync;
  std::unique_ptr<lldb_private::OptionGroupPlatformSSH>
      m_option_group_platform_ssh;
  std::unique_ptr<lldb_private::OptionGroupPlatformCaching>
      m_option_group_platform_caching;

  std::map<lldb_private::CommandInterpreter *,
           std::unique_ptr<lldb_private::OptionGroupOptions>>
      m_options;

  lldb_private::Status
  EvaluateLibdlExpression(lldb_private::Process *process, const char *expr_cstr,
                          llvm::StringRef expr_prefix,
                          lldb::ValueObjectSP &result_valobj_sp);

  std::unique_ptr<lldb_private::UtilityFunction>
  MakeLoadImageUtilityFunction(lldb_private::ExecutionContext &exe_ctx,
                               lldb_private::Status &error);

  virtual
  llvm::StringRef GetLibdlFunctionDeclarations(lldb_private::Process *process);

private:
  DISALLOW_COPY_AND_ASSIGN(PlatformPOSIX);
};

#endif // liblldb_PlatformPOSIX_h_
