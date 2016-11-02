//===-- OptionGroupPlatform.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupPlatform.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Target/Platform.h"
#include "lldb/Utility/Utils.h"

using namespace lldb;
using namespace lldb_private;

PlatformSP OptionGroupPlatform::CreatePlatformWithOptions(
    CommandInterpreter &interpreter, const ArchSpec &arch, bool make_selected,
    Error &error, ArchSpec &platform_arch) const {
  PlatformSP platform_sp;

  if (!m_platform_name.empty()) {
    platform_sp = Platform::Create(ConstString(m_platform_name.c_str()), error);
    if (platform_sp) {
      if (platform_arch.IsValid() &&
          !platform_sp->IsCompatibleArchitecture(arch, false, &platform_arch)) {
        error.SetErrorStringWithFormat("platform '%s' doesn't support '%s'",
                                       platform_sp->GetName().GetCString(),
                                       arch.GetTriple().getTriple().c_str());
        platform_sp.reset();
        return platform_sp;
      }
    }
  } else if (arch.IsValid()) {
    platform_sp = Platform::Create(arch, &platform_arch, error);
  }

  if (platform_sp) {
    interpreter.GetDebugger().GetPlatformList().Append(platform_sp,
                                                       make_selected);
    if (m_os_version_major != UINT32_MAX) {
      platform_sp->SetOSVersion(m_os_version_major, m_os_version_minor,
                                m_os_version_update);
    }

    if (m_sdk_sysroot)
      platform_sp->SetSDKRootDirectory(m_sdk_sysroot);

    if (m_sdk_build)
      platform_sp->SetSDKBuild(m_sdk_build);
  }

  return platform_sp;
}

void OptionGroupPlatform::OptionParsingStarting(
    ExecutionContext *execution_context) {
  m_platform_name.clear();
  m_sdk_sysroot.Clear();
  m_sdk_build.Clear();
  m_os_version_major = UINT32_MAX;
  m_os_version_minor = UINT32_MAX;
  m_os_version_update = UINT32_MAX;
}

static OptionDefinition g_option_table[] = {
    {LLDB_OPT_SET_ALL, false, "platform", 'p', OptionParser::eRequiredArgument,
     nullptr, nullptr, 0, eArgTypePlatform, "Specify name of the platform to "
                                            "use for this target, creating the "
                                            "platform if necessary."},
    {LLDB_OPT_SET_ALL, false, "version", 'v', OptionParser::eRequiredArgument,
     nullptr, nullptr, 0, eArgTypeNone,
     "Specify the initial SDK version to use prior to connecting."},
    {LLDB_OPT_SET_ALL, false, "build", 'b', OptionParser::eRequiredArgument,
     nullptr, nullptr, 0, eArgTypeNone,
     "Specify the initial SDK build number."},
    {LLDB_OPT_SET_ALL, false, "sysroot", 'S', OptionParser::eRequiredArgument,
     nullptr, nullptr, 0, eArgTypeFilename, "Specify the SDK root directory "
                                            "that contains a root of all "
                                            "remote system files."}};

const OptionDefinition *OptionGroupPlatform::GetDefinitions() {
  if (m_include_platform_option)
    return g_option_table;
  return g_option_table + 1;
}

uint32_t OptionGroupPlatform::GetNumDefinitions() {
  if (m_include_platform_option)
    return llvm::array_lengthof(g_option_table);
  return llvm::array_lengthof(g_option_table) - 1;
}

Error OptionGroupPlatform::SetOptionValue(uint32_t option_idx,
                                          const char *option_arg,
                                          ExecutionContext *execution_context) {
  Error error;
  if (!m_include_platform_option)
    ++option_idx;

  const int short_option = g_option_table[option_idx].short_option;

  switch (short_option) {
  case 'p':
    m_platform_name.assign(option_arg);
    break;

  case 'v':
    if (Args::StringToVersion(option_arg, m_os_version_major,
                              m_os_version_minor,
                              m_os_version_update) == option_arg)
      error.SetErrorStringWithFormat("invalid version string '%s'", option_arg);
    break;

  case 'b':
    m_sdk_build.SetCString(option_arg);
    break;

  case 'S':
    m_sdk_sysroot.SetCString(option_arg);
    break;

  default:
    error.SetErrorStringWithFormat("unrecognized option '%c'", short_option);
    break;
  }
  return error;
}

bool OptionGroupPlatform::PlatformMatches(
    const lldb::PlatformSP &platform_sp) const {
  if (platform_sp) {
    if (!m_platform_name.empty()) {
      if (platform_sp->GetName() != ConstString(m_platform_name.c_str()))
        return false;
    }

    if (m_sdk_build && m_sdk_build != platform_sp->GetSDKBuild())
      return false;

    if (m_sdk_sysroot && m_sdk_sysroot != platform_sp->GetSDKRootDirectory())
      return false;

    if (m_os_version_major != UINT32_MAX) {
      uint32_t major, minor, update;
      if (platform_sp->GetOSVersion(major, minor, update)) {
        if (m_os_version_major != major)
          return false;
        if (m_os_version_minor != minor)
          return false;
        if (m_os_version_update != update)
          return false;
      }
    }
    return true;
  }
  return false;
}
