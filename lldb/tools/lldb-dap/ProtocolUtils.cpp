//===-- ProtocolUtils.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProtocolUtils.h"
#include "LLDBUtils.h"

#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBTarget.h"
#include "lldb/Host/PosixApi.h" // Adds PATH_MAX for windows

namespace lldb_dap {

static bool ShouldDisplayAssemblySource(
    lldb::SBAddress address,
    lldb::StopDisassemblyType stop_disassembly_display) {
  if (stop_disassembly_display == lldb::eStopDisassemblyTypeNever)
    return false;

  if (stop_disassembly_display == lldb::eStopDisassemblyTypeAlways)
    return true;

  // A line entry of 0 indicates the line is compiler generated i.e. no source
  // file is associated with the frame.
  auto line_entry = address.GetLineEntry();
  auto file_spec = line_entry.GetFileSpec();
  if (!file_spec.IsValid() || line_entry.GetLine() == 0 ||
      line_entry.GetLine() == LLDB_INVALID_LINE_NUMBER)
    return true;

  if (stop_disassembly_display == lldb::eStopDisassemblyTypeNoSource &&
      !file_spec.Exists()) {
    return true;
  }

  return false;
}

static protocol::Source CreateAssemblySource(const lldb::SBTarget &target,
                                             lldb::SBAddress address) {
  protocol::Source source;

  auto symbol = address.GetSymbol();
  std::string name;
  if (symbol.IsValid()) {
    source.sourceReference = symbol.GetStartAddress().GetLoadAddress(target);
    name = symbol.GetName();
  } else {
    const auto load_addr = address.GetLoadAddress(target);
    source.sourceReference = load_addr;
    name = GetLoadAddressString(load_addr);
  }

  lldb::SBModule module = address.GetModule();
  if (module.IsValid()) {
    lldb::SBFileSpec file_spec = module.GetFileSpec();
    if (file_spec.IsValid()) {
      std::string path = GetSBFileSpecPath(file_spec);
      if (!path.empty())
        source.path = path + '`' + name;
    }
  }

  source.name = std::move(name);

  // Mark the source as deemphasized since users will only be able to view
  // assembly for these frames.
  source.presentationHint =
      protocol::Source::PresentationHint::eSourcePresentationHintDeemphasize;

  return source;
}

protocol::Source CreateSource(const lldb::SBFileSpec &file) {
  protocol::Source source;
  if (file.IsValid()) {
    if (const char *name = file.GetFilename())
      source.name = name;
    char path[PATH_MAX] = "";
    if (file.GetPath(path, sizeof(path)) &&
        lldb::SBFileSpec::ResolvePath(path, path, PATH_MAX))
      source.path = path;
  }
  return source;
}

protocol::Source CreateSource(lldb::SBAddress address, lldb::SBTarget &target) {
  lldb::SBDebugger debugger = target.GetDebugger();
  lldb::StopDisassemblyType stop_disassembly_display =
      GetStopDisassemblyDisplay(debugger);
  if (ShouldDisplayAssemblySource(address, stop_disassembly_display))
    return CreateAssemblySource(target, address);

  lldb::SBLineEntry line_entry = GetLineEntryForAddress(target, address);
  return CreateSource(line_entry.GetFileSpec());
}

bool IsAssemblySource(const protocol::Source &source) {
  // According to the specification, a source must have either `path` or
  // `sourceReference` specified. We use `path` for sources with known source
  // code, and `sourceReferences` when falling back to assembly.
  return source.sourceReference.value_or(0) != 0;
}

std::string GetLoadAddressString(const lldb::addr_t addr) {
  return "0x" + llvm::utohexstr(addr, false, 16);
}

} // namespace lldb_dap
