//===-- VariablesRequestHandler.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "JSONUtils.h"
#include "RequestHandler.h"

namespace lldb_dap {

// "VariablesRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Variables request; value of command field is 'variables'.
//     Retrieves all child variables for the given variable reference. An
//     optional filter can be used to limit the fetched children to either named
//     or indexed children.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "variables" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/VariablesArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "VariablesArguments": {
//   "type": "object",
//   "description": "Arguments for 'variables' request.",
//   "properties": {
//     "variablesReference": {
//       "type": "integer",
//       "description": "The Variable reference."
//     },
//     "filter": {
//       "type": "string",
//       "enum": [ "indexed", "named" ],
//       "description": "Optional filter to limit the child variables to either
//       named or indexed. If ommited, both types are fetched."
//     },
//     "start": {
//       "type": "integer",
//       "description": "The index of the first variable to return; if omitted
//       children start at 0."
//     },
//     "count": {
//       "type": "integer",
//       "description": "The number of variables to return. If count is missing
//       or 0, all variables are returned."
//     },
//     "format": {
//       "$ref": "#/definitions/ValueFormat",
//       "description": "Specifies details on how to format the Variable
//       values."
//     }
//   },
//   "required": [ "variablesReference" ]
// },
// "VariablesResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'variables' request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "variables": {
//             "type": "array",
//             "items": {
//               "$ref": "#/definitions/Variable"
//             },
//             "description": "All (or a range) of variables for the given
//             variable reference."
//           }
//         },
//         "required": [ "variables" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
void VariablesRequestHandler::operator()(
    const llvm::json::Object &request) const {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Array variables;
  const auto *arguments = request.getObject("arguments");
  const auto variablesReference =
      GetInteger<uint64_t>(arguments, "variablesReference").value_or(0);
  const auto start = GetInteger<int64_t>(arguments, "start").value_or(0);
  const auto count = GetInteger<int64_t>(arguments, "count").value_or(0);
  bool hex = false;
  const auto *format = arguments->getObject("format");
  if (format)
    hex = GetBoolean(format, "hex").value_or(false);

  if (lldb::SBValueList *top_scope =
          dap.variables.GetTopLevelScope(variablesReference)) {
    // variablesReference is one of our scopes, not an actual variable it is
    // asking for the list of args, locals or globals.
    int64_t start_idx = 0;
    int64_t num_children = 0;

    if (variablesReference == VARREF_REGS) {
      // Change the default format of any pointer sized registers in the first
      // register set to be the lldb::eFormatAddressInfo so we show the pointer
      // and resolve what the pointer resolves to. Only change the format if the
      // format was set to the default format or if it was hex as some registers
      // have formats set for them.
      const uint32_t addr_size = dap.target.GetProcess().GetAddressByteSize();
      lldb::SBValue reg_set = dap.variables.registers.GetValueAtIndex(0);
      const uint32_t num_regs = reg_set.GetNumChildren();
      for (uint32_t reg_idx = 0; reg_idx < num_regs; ++reg_idx) {
        lldb::SBValue reg = reg_set.GetChildAtIndex(reg_idx);
        const lldb::Format format = reg.GetFormat();
        if (format == lldb::eFormatDefault || format == lldb::eFormatHex) {
          if (reg.GetByteSize() == addr_size)
            reg.SetFormat(lldb::eFormatAddressInfo);
        }
      }
    }

    num_children = top_scope->GetSize();
    if (num_children == 0 && variablesReference == VARREF_LOCALS) {
      // Check for an error in the SBValueList that might explain why we don't
      // have locals. If we have an error display it as the sole value in the
      // the locals.

      // "error" owns the error string so we must keep it alive as long as we
      // want to use the returns "const char *"
      lldb::SBError error = top_scope->GetError();
      const char *var_err = error.GetCString();
      if (var_err) {
        // Create a fake variable named "error" to explain why variables were
        // not available. This new error will help let users know when there was
        // a problem that kept variables from being available for display and
        // allow users to fix this issue instead of seeing no variables. The
        // errors are only set when there is a problem that the user could
        // fix, so no error will show up when you have no debug info, only when
        // we do have debug info and something that is fixable can be done.
        llvm::json::Object object;
        EmplaceSafeString(object, "name", "<error>");
        EmplaceSafeString(object, "type", "const char *");
        EmplaceSafeString(object, "value", var_err);
        object.try_emplace("variablesReference", (int64_t)0);
        variables.emplace_back(std::move(object));
      }
    }
    const int64_t end_idx = start_idx + ((count == 0) ? num_children : count);

    // We first find out which variable names are duplicated
    std::map<std::string, int> variable_name_counts;
    for (auto i = start_idx; i < end_idx; ++i) {
      lldb::SBValue variable = top_scope->GetValueAtIndex(i);
      if (!variable.IsValid())
        break;
      variable_name_counts[GetNonNullVariableName(variable)]++;
    }

    // Show return value if there is any ( in the local top frame )
    if (variablesReference == VARREF_LOCALS) {
      auto process = dap.target.GetProcess();
      auto selected_thread = process.GetSelectedThread();
      lldb::SBValue stop_return_value = selected_thread.GetStopReturnValue();

      if (stop_return_value.IsValid() &&
          (selected_thread.GetSelectedFrame().GetFrameID() == 0)) {
        auto renamed_return_value = stop_return_value.Clone("(Return Value)");
        int64_t return_var_ref = 0;

        if (stop_return_value.MightHaveChildren() ||
            stop_return_value.IsSynthetic()) {
          return_var_ref = dap.variables.InsertVariable(stop_return_value,
                                                        /*is_permanent=*/false);
        }
        variables.emplace_back(
            CreateVariable(renamed_return_value, return_var_ref, hex,
                           dap.enable_auto_variable_summaries,
                           dap.enable_synthetic_child_debugging, false));
      }
    }

    // Now we construct the result with unique display variable names
    for (auto i = start_idx; i < end_idx; ++i) {
      lldb::SBValue variable = top_scope->GetValueAtIndex(i);

      if (!variable.IsValid())
        break;

      int64_t var_ref =
          dap.variables.InsertVariable(variable, /*is_permanent=*/false);
      variables.emplace_back(CreateVariable(
          variable, var_ref, hex, dap.enable_auto_variable_summaries,
          dap.enable_synthetic_child_debugging,
          variable_name_counts[GetNonNullVariableName(variable)] > 1));
    }
  } else {
    // We are expanding a variable that has children, so we will return its
    // children.
    lldb::SBValue variable = dap.variables.GetVariable(variablesReference);
    if (variable.IsValid()) {
      auto addChild = [&](lldb::SBValue child,
                          std::optional<std::string> custom_name = {}) {
        if (!child.IsValid())
          return;
        bool is_permanent =
            dap.variables.IsPermanentVariableReference(variablesReference);
        int64_t var_ref = dap.variables.InsertVariable(child, is_permanent);
        variables.emplace_back(CreateVariable(
            child, var_ref, hex, dap.enable_auto_variable_summaries,
            dap.enable_synthetic_child_debugging,
            /*is_name_duplicated=*/false, custom_name));
      };
      const int64_t num_children = variable.GetNumChildren();
      int64_t end_idx = start + ((count == 0) ? num_children : count);
      int64_t i = start;
      for (; i < end_idx && i < num_children; ++i)
        addChild(variable.GetChildAtIndex(i));

      // If we haven't filled the count quota from the request, we insert a new
      // "[raw]" child that can be used to inspect the raw version of a
      // synthetic member. That eliminates the need for the user to go to the
      // debug console and type `frame var <variable> to get these values.
      if (dap.enable_synthetic_child_debugging && variable.IsSynthetic() &&
          i == num_children)
        addChild(variable.GetNonSyntheticValue(), "[raw]");
    }
  }
  llvm::json::Object body;
  body.try_emplace("variables", std::move(variables));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
