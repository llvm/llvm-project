//===-- CommandHistoryWithContext.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/CommandHistoryWithContext.h"

#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace lldb;
using namespace lldb_private;

CommandHistoryWithContext::CommandHistoryWithContext(FileSpec history_file)
    : m_history_file(std::move(history_file)) {}

void CommandHistoryWithContext::AppendCommand(
    llvm::StringRef command, const CommandExecutionContext &context) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);

  // If this command was seen before, bump its invocation count, refresh its
  // context, and move it to the back so it becomes the most recently used.
  for (auto it = m_entries.begin(); it != m_entries.end(); ++it) {
    if (it->command != command)
      continue;
    Entry entry = std::move(*it);
    m_entries.erase(it);
    ++entry.invocation_count;
    entry.context = context;
    m_entries.push_back(std::move(entry));
    SaveLocked();
    return;
  }

  m_entries.push_back(Entry{command.str(), context, /*invocation_count=*/1});

  // Enforce the size cap by evicting the least frequently invoked entry. On a
  // tie, evict the oldest (front-most) of them: iterating from the front and
  // only replacing the victim on a strictly smaller count keeps the oldest
  // minimum-count entry selected.
  while (m_entries.size() > GetMaxEntries()) {
    auto victim = m_entries.begin();
    for (auto it = std::next(m_entries.begin()); it != m_entries.end(); ++it) {
      if (it->invocation_count < victim->invocation_count)
        victim = it;
    }
    m_entries.erase(victim);
  }

  SaveLocked();
}

std::optional<std::string>
CommandHistoryWithContext::FindSuggestion(
    llvm::StringRef line, const CommandExecutionContext &current) const {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);

  if (line.empty())
    return std::nullopt;

  // The best candidate for each tier of the preference list. Because we walk
  // the history from most recent to oldest and only assign each pointer once,
  // every pointer ends up referring to the most recent matching command.
  const Entry *by_file_line = nullptr;
  const Entry *by_function = nullptr;
  const Entry *by_target = nullptr;
  const Entry *by_prefix = nullptr;

  const bool have_file_line = !current.file.empty() && current.line != 0;

  for (auto it = m_entries.rbegin(); it != m_entries.rend(); ++it) {
    const Entry &entry = *it;
    // A suggestion must complete what the user already typed.
    if (!llvm::StringRef(entry.command).starts_with(line))
      continue;

    const CommandExecutionContext &ctx = entry.context;

    if (!by_prefix)
      by_prefix = &entry;

    if (!by_file_line && have_file_line && ctx.file == current.file &&
        ctx.line == current.line)
      by_file_line = &entry;

    if (!by_function && !current.function_name.empty() &&
        ctx.function_name == current.function_name)
      by_function = &entry;

    if (!by_target && !current.target_name.empty() &&
        ctx.target_name == current.target_name)
      by_target = &entry;
  }

  // Preference: same file/line, then same function, then same target, and
  // finally fall back to the most recent command sharing the prefix.
  const Entry *chosen = by_file_line ? by_file_line
                        : by_function ? by_function
                        : by_target   ? by_target
                                      : by_prefix;
  if (!chosen)
    return std::nullopt;

  return llvm::StringRef(chosen->command).drop_front(line.size()).str();
}

void CommandHistoryWithContext::Load() {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  m_entries.clear();

  if (!m_history_file)
    return;

  Log *log = GetLog(LLDBLog::Host);
  const std::string path = m_history_file.GetPath();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer_or_err =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer_or_err)
    return; // The file most likely does not exist yet.

  llvm::Expected<llvm::json::Value> value =
      llvm::json::parse((*buffer_or_err)->getBuffer());
  if (!value) {
    LLDB_LOG_ERROR(log, value.takeError(),
                   "failed to parse command history file: {0}");
    return;
  }

  const llvm::json::Object *root = value->getAsObject();
  if (!root)
    return;

  const llvm::json::Array *commands = root->getArray("commands");
  if (!commands)
    return;

  for (const llvm::json::Value &v : *commands) {
    const llvm::json::Object *obj = v.getAsObject();
    if (!obj)
      continue;

    Entry entry;
    entry.command = obj->getString("command").value_or("").str();
    if (entry.command.empty())
      continue;
    entry.context.target_name = obj->getString("target").value_or("").str();
    entry.context.file = obj->getString("file").value_or("").str();
    entry.context.line =
        static_cast<uint32_t>(obj->getInteger("line").value_or(0));
    entry.context.function_name =
        obj->getString("function").value_or("").str();
    entry.context.working_directory = obj->getString("cwd").value_or("").str();
    // Restore the invocation count, defaulting to 1 for older files or bad
    // values so every retained command counts as invoked at least once.
    uint64_t count = static_cast<uint64_t>(obj->getInteger("count").value_or(1));
    entry.invocation_count = count ? count : 1;
    m_entries.push_back(std::move(entry));
  }

  // Guard against an oversized file left behind by another version.
  while (m_entries.size() > GetMaxEntries())
    m_entries.pop_front();
}

void CommandHistoryWithContext::Save() const {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  SaveLocked();
}

void CommandHistoryWithContext::SaveLocked() const {
  if (!m_history_file)
    return;

  Log *log = GetLog(LLDBLog::Host);
  const std::string path = m_history_file.GetPath();

  // Make sure the containing (cache) directory exists.
  llvm::StringRef parent = llvm::sys::path::parent_path(path);
  if (!parent.empty()) {
    if (std::error_code ec = llvm::sys::fs::create_directories(parent)) {
      LLDB_LOG(log, "failed to create command history directory {0}: {1}",
               parent, ec.message());
      return;
    }
  }

  llvm::json::Array commands;
  for (const Entry &entry : m_entries) {
    commands.push_back(llvm::json::Object{
        {"command", entry.command},
        {"target", entry.context.target_name},
        {"file", entry.context.file},
        {"line", static_cast<int64_t>(entry.context.line)},
        {"function", entry.context.function_name},
        {"cwd", entry.context.working_directory},
        {"count", static_cast<int64_t>(entry.invocation_count)},
    });
  }

  llvm::json::Value root = llvm::json::Object{
      {"version", 1},
      {"commands", std::move(commands)},
  };

  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    LLDB_LOG(log, "failed to write command history file {0}: {1}", path,
             ec.message());
    return;
  }
  os << llvm::formatv("{0}", root);
}
