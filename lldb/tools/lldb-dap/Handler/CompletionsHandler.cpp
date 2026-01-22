//===-- CompletionsHandler.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "LLDBUtils.h"
#include "Protocol/ProtocolRequests.h"
#include "Protocol/ProtocolTypes.h"
#include "RequestHandler.h"
#include "lldb/API/SBStringList.h"
#include "llvm/Support/ConvertUTF.h"

using namespace llvm;
using namespace lldb_dap;
using namespace lldb;
using namespace lldb_dap::protocol;

namespace lldb_dap {
/// Gets the position in the UTF8 string where the specified line started.
static size_t GetLineStartPos(StringRef text, uint32_t line) {
  if (line == 0) // Invalid line.
    return StringRef::npos;

  if (line == 1)
    return 0;

  uint32_t cur_line = 1;
  size_t pos = 0;

  while (cur_line < line) {
    const size_t new_line_pos = text.find('\n', pos);

    if (new_line_pos == StringRef::npos)
      return new_line_pos;

    pos = new_line_pos + 1;
    // text may end with a new line
    if (pos >= text.size())
      return StringRef::npos;

    cur_line++;
  }

  assert(pos < text.size());
  return pos;
}

static std::optional<size_t> GetCursorPos(StringRef text, uint32_t line,
                                          uint32_t utf16_codeunits) {
  if (text.empty())
    return std::nullopt;

  const size_t line_start_pos = GetLineStartPos(text, line);
  if (line_start_pos == StringRef::npos)
    return std::nullopt;

  const StringRef completion_line =
      text.substr(line_start_pos, text.find('\n', line_start_pos));
  if (completion_line.empty())
    return std::nullopt;

  const std::optional<size_t> cursor_pos_opt =
      UTF16CodeunitToBytes(completion_line, utf16_codeunits);
  if (!cursor_pos_opt)
    return std::nullopt;

  const size_t cursor_pos = line_start_pos + *cursor_pos_opt;
  return cursor_pos;
}

/// Returns a list of possible completions for a given caret position and text.
///
/// Clients should only call this request if the corresponding capability
/// `supportsCompletionsRequest` is true.
Expected<CompletionsResponseBody>
CompletionsRequestHandler::Run(const CompletionsArguments &args) const {
  std::string text = args.text;
  const uint32_t line = args.line;
  // column starts at 1.
  const uint32_t utf16_codeunits = args.column - 1;

  const auto cursor_pos_opt = GetCursorPos(text, line, utf16_codeunits);
  if (!cursor_pos_opt)
    return CompletionsResponseBody{};

  size_t cursor_pos = *cursor_pos_opt;

  // If we have a frame, try to set the context for variable completions.
  lldb::SBFrame frame = dap.GetLLDBFrame(args.frameId);
  if (frame.IsValid()) {
    lldb::SBThread frame_thread = frame.GetThread();
    frame_thread.GetProcess().SetSelectedThread(frame_thread);
    frame_thread.SetSelectedFrame(frame.GetFrameID());
  }

  const StringRef escape_prefix = dap.configuration.commandEscapePrefix;
  const bool had_escape_prefix = StringRef(text).starts_with(escape_prefix);
  const ReplMode repl_mode = dap.DetectReplMode(frame, text, true);
  // Handle the cursor_pos change introduced by stripping out the
  // `command_escape_prefix`.
  if (had_escape_prefix) {
    if (cursor_pos < escape_prefix.size())
      return CompletionsResponseBody{};

    cursor_pos -= escape_prefix.size();
  }

  // While the user is typing then we likely have an incomplete input and cannot
  // reliably determine the precise intent (command vs variable), try completing
  // the text as both a command and variable expression, if applicable.
  const std::string expr_prefix = "expression -- ";
  const std::array<std::tuple<ReplMode, std::string, uint64_t>, 2> exprs = {
      {std::make_tuple(ReplMode::Command, text, cursor_pos),
       std::make_tuple(ReplMode::Variable, expr_prefix + text,
                       cursor_pos + expr_prefix.size())}};

  CompletionsResponseBody response;
  std::vector<CompletionItem> &targets = response.targets;
  lldb::SBCommandInterpreter interpreter = dap.debugger.GetCommandInterpreter();
  for (const auto &[mode, line, cursor] : exprs) {
    if (repl_mode != ReplMode::Auto && repl_mode != mode)
      continue;

    lldb::SBStringList matches;
    lldb::SBStringList descriptions;
    if (!interpreter.HandleCompletionWithDescriptions(
            line.c_str(), cursor, 0, 50, matches, descriptions))
      continue;

    // The first element is the common substring after the cursor position for
    // all the matches. The rest of the elements are the matches so ignore the
    // first result.
    for (uint32_t i = 1; i < matches.GetSize(); i++) {
      const StringRef match = matches.GetStringAtIndex(i);
      const StringRef description = descriptions.GetStringAtIndex(i);

      StringRef match_ref = match;
      for (const StringRef commit_point : {".", "->"}) {
        if (const size_t pos = match_ref.rfind(commit_point);
            pos != StringRef::npos) {
          match_ref = match_ref.substr(pos + commit_point.size());
        }
      }

      CompletionItem item;
      item.text = match_ref;
      item.label = match;
      if (!description.empty())
        item.detail = description;

      targets.emplace_back(std::move(item));
    }
  }

  return response;
}

} // namespace lldb_dap
