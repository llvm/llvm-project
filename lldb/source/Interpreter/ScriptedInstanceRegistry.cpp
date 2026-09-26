//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/ScriptedInstanceRegistry.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Utility/AnsiTerminal.h"
#include "lldb/Utility/Stream.h"

#include "llvm/ADT/StringMap.h"

#include <tuple>

using namespace lldb;
using namespace lldb_private;

static bool HasArgs(const ScriptedInstanceInfo &info) {
  return info.args_sp && info.args_sp->GetSize();
}

std::vector<ScriptedInstanceGroup> ScriptedInstanceRegistry::GetInstanceGroups(
    llvm::ArrayRef<ScriptedExtension> extensions) const {
  // Unknown names map to eScriptedExtensionInvalid, the zero value.
  llvm::StringMap<ScriptedExtension> extension_by_plugin;
  for (uint32_t i = 0; i < PluginManager::GetNumScriptedInterfaces(); i++)
    extension_by_plugin[PluginManager::GetScriptedInterfaceNameAtIndex(i)] =
        PluginManager::GetScriptedInterfaceExtensionAtIndex(i);

  std::vector<ScriptedInstanceGroup> groups;
  for (ScriptedInstanceInfo &info : CopyInstances()) {
    ScriptedExtension extension = extension_by_plugin.lookup(info.plugin_name);
    if (!extensions.empty() && !llvm::is_contained(extensions, extension))
      continue;

    auto it = llvm::find_if(groups, [&](const ScriptedInstanceGroup &group) {
      return group.class_name == info.class_name &&
             group.source_path == info.source_path &&
             group.extension == extension;
    });
    if (it == groups.end()) {
      it = groups.insert(groups.end(), ScriptedInstanceGroup());
      it->class_name = info.class_name;
      it->extension = extension;
      it->source_path = info.source_path;
    }
    it->instances.push_back(std::move(info));
  }

  for (ScriptedInstanceGroup &group : groups)
    llvm::sort(group.instances,
               [](const ScriptedInstanceInfo &lhs,
                  const ScriptedInstanceInfo &rhs) { return lhs.id < rhs.id; });
  llvm::sort(groups, [](const ScriptedInstanceGroup &lhs,
                        const ScriptedInstanceGroup &rhs) {
    return std::make_tuple(llvm::StringRef(lhs.class_name),
                           lhs.source_path.GetPath(), lhs.extension) <
           std::make_tuple(llvm::StringRef(rhs.class_name),
                           rhs.source_path.GetPath(), rhs.extension);
  });
  return groups;
}

StructuredData::DictionarySP ScriptedInstanceGroup::ToStructuredData() const {
  auto group_sp = std::make_shared<StructuredData::Dictionary>();
  group_sp->AddStringItem("class_name", class_name);
  group_sp->AddStringItem("extension",
                          ScriptInterpreter::ExtensionToString(extension));
  if (source_path)
    group_sp->AddStringItem("source_path", source_path.GetPath());

  auto instances_sp = std::make_shared<StructuredData::Array>();
  for (const ScriptedInstanceInfo &info : instances) {
    auto instance_sp = std::make_shared<StructuredData::Dictionary>();
    instance_sp->AddStringItem("uuid", info.uuid.GetAsString());
    if (info.object_address != LLDB_INVALID_ADDRESS)
      instance_sp->AddIntegerItem("address", info.object_address);
    if (HasArgs(info))
      instance_sp->AddItem("args", info.args_sp);
    instances_sp->AddItem(instance_sp);
  }
  group_sp->AddItem("instances", instances_sp);
  return group_sp;
}

void ScriptedInstanceGroup::Dump(Stream &s, const Debugger &debugger,
                                 bool use_color) const {
  auto ansi_code = [use_color](llvm::StringRef code) {
    return ansi::FormatAnsiTerminalCodes(code, use_color);
  };
  const std::string label_prefix = ansi_code(debugger.GetLabelAnsiPrefix());
  const std::string label_suffix = ansi_code(debugger.GetLabelAnsiSuffix());
  const std::string title_prefix = ansi_code(debugger.GetTitleAnsiPrefix());
  const std::string title_suffix = ansi_code(debugger.GetTitleAnsiSuffix());

  auto print_label = [&](llvm::StringRef indent, llvm::StringRef label) {
    s << indent << label_prefix << label << ':' << label_suffix;
  };
  auto print_instance = [&](const ScriptedInstanceInfo &info) {
    s << info.uuid.GetAsString();
    if (info.object_address != LLDB_INVALID_ADDRESS)
      s.Format(" ({0:x})", info.object_address);
    s << '\n';
  };
  auto print_args = [&](const ScriptedInstanceInfo &info,
                        llvm::StringRef indent) {
    if (!HasArgs(info))
      return;
    print_label(indent, "Args");
    s << '\n';
    info.args_sp->ForEachSorted(
        [&](llvm::StringRef key, StructuredData::Object *value) {
          s << indent << "  - " << key << ": ";
          if (StructuredData::String *str = value->GetAsString())
            s << str->GetValue();
          else
            value->Dump(s, /*pretty_print=*/false);
          s << '\n';
          return true;
        });
  };

  print_label("  ", "Class");
  s << ' ' << title_prefix << class_name << title_suffix << '\n';
  print_label("  ", "Extension");
  s << ' ' << ScriptInterpreter::ExtensionToString(extension) << '\n';
  if (source_path) {
    print_label("  ", "Path");
    s << ' ' << source_path.GetPath() << '\n';
  }

  if (instances.size() == 1) {
    print_label("  ", "Instance");
    s << ' ';
    print_instance(instances.front());
    print_args(instances.front(), "  ");
    return;
  }

  print_label("  ", "Instances");
  s << '\n';
  for (auto [idx, info] : llvm::enumerate(instances)) {
    s.Format("    [{0}] ", idx);
    print_instance(info);
    print_args(info, "        ");
  }
}
