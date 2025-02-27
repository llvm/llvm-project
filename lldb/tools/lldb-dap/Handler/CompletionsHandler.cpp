//===-- CompletionsHandler.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "JSONUtils.h"
#include "RequestHandler.h"
#include "lldb/API/SBStringList.h"

namespace lldb_dap {

// "CompletionsRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Returns a list of possible completions for a given caret
//     position and text.\nThe CompletionsRequest may only be called if the
//     'supportsCompletionsRequest' capability exists and is true.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "completions" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/CompletionsArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "CompletionsArguments": {
//   "type": "object",
//   "description": "Arguments for 'completions' request.",
//   "properties": {
//     "frameId": {
//       "type": "integer",
//       "description": "Returns completions in the scope of this stack frame.
//       If not specified, the completions are returned for the global scope."
//     },
//     "text": {
//       "type": "string",
//       "description": "One or more source lines. Typically this is the text a
//       user has typed into the debug console before he asked for completion."
//     },
//     "column": {
//       "type": "integer",
//       "description": "The character position for which to determine the
//       completion proposals."
//     },
//     "line": {
//       "type": "integer",
//       "description": "An optional line for which to determine the completion
//       proposals. If missing the first line of the text is assumed."
//     }
//   },
//   "required": [ "text", "column" ]
// },
// "CompletionsResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'completions' request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "targets": {
//             "type": "array",
//             "items": {
//               "$ref": "#/definitions/CompletionItem"
//             },
//             "description": "The possible completions for ."
//           }
//         },
//         "required": [ "targets" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// },
// "CompletionItem": {
//   "type": "object",
//   "description": "CompletionItems are the suggestions returned from the
//   CompletionsRequest.", "properties": {
//     "label": {
//       "type": "string",
//       "description": "The label of this completion item. By default this is
//       also the text that is inserted when selecting this completion."
//     },
//     "text": {
//       "type": "string",
//       "description": "If text is not falsy then it is inserted instead of the
//       label."
//     },
//     "sortText": {
//       "type": "string",
//       "description": "A string that should be used when comparing this item
//       with other items. When `falsy` the label is used."
//     },
//     "type": {
//       "$ref": "#/definitions/CompletionItemType",
//       "description": "The item's type. Typically the client uses this
//       information to render the item in the UI with an icon."
//     },
//     "start": {
//       "type": "integer",
//       "description": "This value determines the location (in the
//       CompletionsRequest's 'text' attribute) where the completion text is
//       added.\nIf missing the text is added at the location specified by the
//       CompletionsRequest's 'column' attribute."
//     },
//     "length": {
//       "type": "integer",
//       "description": "This value determines how many characters are
//       overwritten by the completion text.\nIf missing the value 0 is assumed
//       which results in the completion text being inserted."
//     }
//   },
//   "required": [ "label" ]
// },
// "CompletionItemType": {
//   "type": "string",
//   "description": "Some predefined types for the CompletionItem. Please note
//   that not all clients have specific icons for all of them.", "enum": [
//   "method", "function", "constructor", "field", "variable", "class",
//   "interface", "module", "property", "unit", "value", "enum", "keyword",
//   "snippet", "text", "color", "file", "reference", "customcolor" ]
// }
void CompletionsRequestHandler::operator()(
    const llvm::json::Object &request) const {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Object body;
  const auto *arguments = request.getObject("arguments");

  // If we have a frame, try to set the context for variable completions.
  lldb::SBFrame frame = dap.GetLLDBFrame(*arguments);
  if (frame.IsValid()) {
    frame.GetThread().GetProcess().SetSelectedThread(frame.GetThread());
    frame.GetThread().SetSelectedFrame(frame.GetFrameID());
  }

  std::string text = GetString(arguments, "text").str();
  auto original_column = GetSigned(arguments, "column", text.size());
  auto original_line = GetSigned(arguments, "line", 1);
  auto offset = original_column - 1;
  if (original_line > 1) {
    llvm::SmallVector<::llvm::StringRef, 2> lines;
    llvm::StringRef(text).split(lines, '\n');
    for (int i = 0; i < original_line - 1; i++) {
      offset += lines[i].size();
    }
  }
  llvm::json::Array targets;

  bool had_escape_prefix =
      llvm::StringRef(text).starts_with(dap.command_escape_prefix);
  ReplMode completion_mode = dap.DetectReplMode(frame, text, true);

  // Handle the offset change introduced by stripping out the
  // `command_escape_prefix`.
  if (had_escape_prefix) {
    if (offset < static_cast<int64_t>(dap.command_escape_prefix.size())) {
      body.try_emplace("targets", std::move(targets));
      response.try_emplace("body", std::move(body));
      dap.SendJSON(llvm::json::Value(std::move(response)));
      return;
    }
    offset -= dap.command_escape_prefix.size();
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

      llvm::json::Object item;
      llvm::StringRef match_ref = match;
      for (llvm::StringRef commit_point : {".", "->"}) {
        if (match_ref.contains(commit_point)) {
          match_ref = match_ref.rsplit(commit_point).second;
        }
      }
      EmplaceSafeString(item, "text", match_ref);

      if (description.empty())
        EmplaceSafeString(item, "label", match);
      else
        EmplaceSafeString(item, "label", match + " -- " + description);

      targets.emplace_back(std::move(item));
    }
  }

  body.try_emplace("targets", std::move(targets));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
