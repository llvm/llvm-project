//===-- CompletionsHandler.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "JSONUtils.h"
#include "Protocol/ProtocolRequests.h"
#include "Protocol/ProtocolTypes.h"
#include "RequestHandler.h"
#include "lldb/API/SBStringList.h"

using namespace llvm;
using namespace lldb_dap;
using namespace lldb_dap::protocol;

namespace lldb_dap {

/// Returns a list of possible completions for a given caret position and text.
///
/// Clients should only call this request if the corresponding capability
/// `supportsCompletionsRequest` is true.
Expected<CompletionsResponseBody>
CompletionsRequestHandler::Run(const CompletionsArguments &args) const {
  // If we have a frame, try to set the context for variable completions.
  lldb::SBFrame frame = dap.GetLLDBFrame(args.frameId);
  if (frame.IsValid()) {
    frame.GetThread().GetProcess().SetSelectedThread(frame.GetThread());
    frame.GetThread().SetSelectedFrame(frame.GetFrameID());
  }

  std::string text = args.text;
  auto original_column = args.column;
  auto original_line = args.line;
  auto offset = original_column - 1;
  if (original_line > 1) {
    SmallVector<StringRef, 2> lines;
    StringRef(text).split(lines, '\n');
    for (int i = 0; i < original_line - 1; i++) {
      offset += lines[i].size();
    }
  }

  std::vector<CompletionItem> targets;

  bool had_escape_prefix =
      StringRef(text).starts_with(dap.configuration.commandEscapePrefix);
  ReplMode completion_mode = dap.DetectReplMode(frame, text, true);

  // Handle the offset change introduced by stripping out the
  // `command_escape_prefix`.
  if (had_escape_prefix) {
    if (offset <
        static_cast<int64_t>(dap.configuration.commandEscapePrefix.size())) {
      return CompletionsResponseBody{std::move(targets)};
    }
    offset -= dap.configuration.commandEscapePrefix.size();
  }

  // While the user is typing then we likely have an incomplete input and cannot
  // reliably determine the precise intent (command vs variable), try completing
  // the text as both a command and variable expression, if applicable.
  const std::string expr_prefix = "expression -- ";
  std::array<std::tuple<ReplMode, std::string, uint64_t>, 2> exprs = {
      {std::make_tuple(ReplMode::Command, text, offset),
       std::make_tuple(ReplMode::Variable, expr_prefix + text,
                       offset + expr_prefix.size())}};
  for (const auto &[mode, line, cursor] : exprs) {
    if (completion_mode != ReplMode::Auto && completion_mode != mode)
      continue;

    lldb::SBStringList matches;
    lldb::SBStringList descriptions;
    if (!dap.debugger.GetCommandInterpreter().HandleCompletionWithDescriptions(
            line.c_str(), cursor, 0, 100, matches, descriptions))
      continue;

    // The first element is the common substring after the cursor position for
    // all the matches. The rest of the elements are the matches so ignore the
    // first result.
    for (size_t i = 1; i < matches.GetSize(); i++) {
      std::string match = matches.GetStringAtIndex(i);
      std::string description = descriptions.GetStringAtIndex(i);

      CompletionItem item;
      StringRef match_ref = match;
      for (StringRef commit_point : {".", "->"}) {
        if (match_ref.contains(commit_point)) {
          match_ref = match_ref.rsplit(commit_point).second;
        }
      }
      item.text = match_ref;

      if (description.empty())
        item.label = match;
      else
        item.label = match + " -- " + description;

      targets.emplace_back(std::move(item));
    }
  }

  return CompletionsResponseBody{std::move(targets)};
}

} // namespace lldb_dap
