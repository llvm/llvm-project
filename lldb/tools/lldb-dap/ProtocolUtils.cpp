//===-- ProtocolUtils.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProtocolUtils.h"
#include "JSONUtils.h"
#include "LLDBUtils.h"

#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBDeclaration.h"
#include "lldb/API/SBFormat.h"
#include "lldb/API/SBMutex.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"
#include "lldb/Host/PosixApi.h" // Adds PATH_MAX for windows

#include <iomanip>
#include <optional>
#include <sstream>

using namespace lldb_dap::protocol;
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

static uint64_t GetDebugInfoSizeInSection(lldb::SBSection section) {
  uint64_t debug_info_size = 0;
  const llvm::StringRef section_name(section.GetName());
  if (section_name.starts_with(".debug") ||
      section_name.starts_with("__debug") ||
      section_name.starts_with(".apple") || section_name.starts_with("__apple"))
    debug_info_size += section.GetFileByteSize();

  const size_t num_sub_sections = section.GetNumSubSections();
  for (size_t i = 0; i < num_sub_sections; i++)
    debug_info_size +=
        GetDebugInfoSizeInSection(section.GetSubSectionAtIndex(i));

  return debug_info_size;
}

static uint64_t GetDebugInfoSize(lldb::SBModule module) {
  uint64_t debug_info_size = 0;
  const size_t num_sections = module.GetNumSections();
  for (size_t i = 0; i < num_sections; i++)
    debug_info_size += GetDebugInfoSizeInSection(module.GetSectionAtIndex(i));

  return debug_info_size;
}

std::string ConvertDebugInfoSizeToString(uint64_t debug_size) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1);
  if (debug_size < 1024) {
    oss << debug_size << "B";
  } else if (debug_size < static_cast<uint64_t>(1024 * 1024)) {
    double kb = double(debug_size) / 1024.0;
    oss << kb << "KB";
  } else if (debug_size < 1024 * 1024 * 1024) {
    double mb = double(debug_size) / (1024.0 * 1024.0);
    oss << mb << "MB";
  } else {
    double gb = double(debug_size) / (1024.0 * 1024.0 * 1024.0);
    oss << gb << "GB";
  }
  return oss.str();
}

std::optional<protocol::Module> CreateModule(const lldb::SBTarget &target,
                                             lldb::SBModule &module,
                                             bool id_only) {
  if (!target.IsValid() || !module.IsValid())
    return std::nullopt;

  const llvm::StringRef uuid = module.GetUUIDString();
  if (uuid.empty())
    return std::nullopt;

  protocol::Module p_module;
  p_module.id = uuid;

  if (id_only)
    return p_module;

  std::array<char, PATH_MAX> path_buffer{};
  if (const lldb::SBFileSpec file_spec = module.GetFileSpec()) {
    p_module.name = file_spec.GetFilename();

    const uint32_t path_size =
        file_spec.GetPath(path_buffer.data(), path_buffer.size());
    p_module.path = std::string(path_buffer.data(), path_size);
  }

  if (const uint32_t num_compile_units = module.GetNumCompileUnits();
      num_compile_units > 0) {
    p_module.symbolStatus = "Symbols loaded.";

    p_module.debugInfoSizeBytes = GetDebugInfoSize(module);

    if (const lldb::SBFileSpec symbol_fspec = module.GetSymbolFileSpec()) {
      const uint32_t path_size =
          symbol_fspec.GetPath(path_buffer.data(), path_buffer.size());
      p_module.symbolFilePath = std::string(path_buffer.data(), path_size);
    }
  } else {
    p_module.symbolStatus = "Symbols not found.";
  }

  const auto load_address = module.GetObjectFileHeaderAddress();
  if (const lldb::addr_t raw_address = load_address.GetLoadAddress(target);
      raw_address != LLDB_INVALID_ADDRESS)
    p_module.addressRange = llvm::formatv("{0:x}", raw_address);

  std::array<uint32_t, 3> version_nums{};
  const uint32_t num_versions =
      module.GetVersion(version_nums.data(), version_nums.size());
  if (num_versions > 0) {
    p_module.version = llvm::formatv(
        "{:$[.]}", llvm::make_range(version_nums.begin(),
                                    version_nums.begin() + num_versions));
  }

  return p_module;
}

std::optional<protocol::Source> CreateSource(const lldb::SBFileSpec &file) {
  if (!file.IsValid())
    return std::nullopt;

  protocol::Source source;
  if (const char *name = file.GetFilename())
    source.name = name;
  char path[PATH_MAX] = "";
  if (file.GetPath(path, sizeof(path)) &&
      lldb::SBFileSpec::ResolvePath(path, path, PATH_MAX))
    source.path = path;
  return source;
}

bool IsAssemblySource(const protocol::Source &source) {
  // According to the specification, a source must have either `path` or
  // `sourceReference` specified. We use `path` for sources with known source
  // code, and `sourceReferences` when falling back to assembly.
  return source.sourceReference.value_or(LLDB_DAP_INVALID_SRC_REF) >
         LLDB_DAP_INVALID_SRC_REF;
}

bool DisplayAssemblySource(lldb::SBDebugger &debugger,
                           lldb::SBAddress address) {
  const lldb::StopDisassemblyType stop_disassembly_display =
      GetStopDisassemblyDisplay(debugger);
  return ShouldDisplayAssemblySource(address, stop_disassembly_display);
}

std::string GetLoadAddressString(const lldb::addr_t addr) {
  return "0x" + llvm::utohexstr(addr, false, 16);
}

protocol::Thread CreateThread(lldb::SBThread &thread, lldb::SBFormat &format) {
  std::string name;
  lldb::SBStream stream;
  if (format && thread.GetDescriptionWithFormat(format, stream).Success()) {
    name = stream.GetData();
  } else {
    llvm::StringRef thread_name(thread.GetName());
    llvm::StringRef queue_name(thread.GetQueueName());

    if (!thread_name.empty()) {
      name = thread_name.str();
    } else if (!queue_name.empty()) {
      auto kind = thread.GetQueue().GetKind();
      std::string queue_kind_label = "";
      if (kind == lldb::eQueueKindSerial)
        queue_kind_label = " (serial)";
      else if (kind == lldb::eQueueKindConcurrent)
        queue_kind_label = " (concurrent)";

      name = llvm::formatv("Thread {0} Queue: {1}{2}", thread.GetIndexID(),
                           queue_name, queue_kind_label)
                 .str();
    } else {
      name = llvm::formatv("Thread {0}", thread.GetIndexID()).str();
    }
  }
  return protocol::Thread{thread.GetThreadID(), name};
}

std::vector<protocol::Thread> GetThreads(lldb::SBProcess process,
                                         lldb::SBFormat &format) {
  lldb::SBMutex lock = process.GetTarget().GetAPIMutex();
  std::lock_guard<lldb::SBMutex> guard(lock);

  std::vector<protocol::Thread> threads;

  const uint32_t num_threads = process.GetNumThreads();
  threads.reserve(num_threads);
  for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    lldb::SBThread thread = process.GetThreadAtIndex(thread_idx);
    threads.emplace_back(CreateThread(thread, format));
  }
  return threads;
}

ExceptionBreakpointsFilter
CreateExceptionBreakpointFilter(const ExceptionBreakpoint &bp) {
  ExceptionBreakpointsFilter filter;
  filter.filter = bp.GetFilter();
  filter.label = bp.GetLabel();
  filter.description = bp.GetLabel();
  filter.defaultState = ExceptionBreakpoint::kDefaultValue;
  filter.supportsCondition = true;
  return filter;
}

Variable CreateVariable(lldb::SBValue v, int64_t var_ref, bool format_hex,
                        bool auto_variable_summaries,
                        bool synthetic_child_debugging, bool is_name_duplicated,
                        std::optional<std::string> custom_name) {
  VariableDescription desc(v, auto_variable_summaries, format_hex,
                           is_name_duplicated, custom_name);
  Variable var;
  var.name = desc.name;
  var.value = desc.display_value;
  var.type = desc.display_type_name;

  if (!desc.evaluate_name.empty())
    var.evaluateName = desc.evaluate_name;

  // If we have a type with many children, we would like to be able to
  // give a hint to the IDE that the type has indexed children so that the
  // request can be broken up in grabbing only a few children at a time. We
  // want to be careful and only call "v.GetNumChildren()" if we have an array
  // type or if we have a synthetic child provider producing indexed children.
  // We don't want to call "v.GetNumChildren()" on all objects as class, struct
  // and union types don't need to be completed if they are never expanded. So
  // we want to avoid calling this to only cases where we it makes sense to keep
  // performance high during normal debugging.

  // If we have an array type, say that it is indexed and provide the number
  // of children in case we have a huge array. If we don't do this, then we
  // might take a while to produce all children at onces which can delay your
  // debug session.
  if (desc.type_obj.IsArrayType()) {
    var.indexedVariables = v.GetNumChildren();
  } else if (v.IsSynthetic()) {
    // For a type with a synthetic child provider, the SBType of "v" won't tell
    // us anything about what might be displayed. Instead, we check if the first
    // child's name is "[0]" and then say it is indexed. We call
    // GetNumChildren() only if the child name matches to avoid a potentially
    // expensive operation.
    if (lldb::SBValue first_child = v.GetChildAtIndex(0)) {
      llvm::StringRef first_child_name = first_child.GetName();
      if (first_child_name == "[0]") {
        size_t num_children = v.GetNumChildren();
        // If we are creating a "[raw]" fake child for each synthetic type, we
        // have to account for it when returning indexed variables.
        if (synthetic_child_debugging)
          ++num_children;
        var.indexedVariables = num_children;
      }
    }
  }

  if (v.MightHaveChildren())
    var.variablesReference = var_ref;

  if (v.GetDeclaration().IsValid())
    var.declarationLocationReference = PackLocation(var_ref, false);

  if (ValuePointsToCode(v))
    var.valueLocationReference = PackLocation(var_ref, true);

  if (lldb::addr_t addr = v.GetLoadAddress(); addr != LLDB_INVALID_ADDRESS)
    var.memoryReference = addr;

  bool is_readonly = v.GetType().IsAggregateType() ||
                     v.GetValueType() == lldb::eValueTypeRegisterSet;
  if (is_readonly) {
    if (!var.presentationHint)
      var.presentationHint = {VariablePresentationHint()};
    var.presentationHint->attributes.push_back("readOnly");
  }

  return var;
}

} // namespace lldb_dap
