//===-- VariablesRequestHandler.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "Handler/RequestHandler.h"
#include "JSONUtils.h"
#include "ProtocolUtils.h"

using namespace llvm;
using namespace lldb_dap::protocol;

namespace lldb_dap {

/// Retrieves all child variables for the given variable reference.
///
/// A filter can be used to limit the fetched children to either named or
/// indexed children.
Expected<VariablesResponseBody>
VariablesRequestHandler::Run(const VariablesArguments &arguments) const {
  const uint64_t var_ref = arguments.variablesReference;
  const uint64_t count = arguments.count;
  const uint64_t start = arguments.start;
  bool hex = false;
  if (arguments.format)
    hex = arguments.format->hex;

  std::vector<Variable> variables;

  if (lldb::SBValueList *top_scope = dap.variables.GetTopLevelScope(var_ref)) {
    // variablesReference is one of our scopes, not an actual variable it is
    // asking for the list of args, locals or globals.
    int64_t start_idx = 0;
    int64_t num_children = 0;

    if (var_ref == VARREF_REGS) {
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
    if (num_children == 0 && var_ref == VARREF_LOCALS) {
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
        Variable var;
        var.name = "<error>";
        var.type = "const char *";
        var.value = var_err;
        variables.emplace_back(var);
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
    if (var_ref == VARREF_LOCALS) {
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
        variables.emplace_back(CreateVariable(
            renamed_return_value, return_var_ref, hex,
            dap.configuration.enableAutoVariableSummaries,
            dap.configuration.enableSyntheticChildDebugging, false));
      }
    }

    // Now we construct the result with unique display variable names
    for (auto i = start_idx; i < end_idx; ++i) {
      lldb::SBValue variable = top_scope->GetValueAtIndex(i);

      if (!variable.IsValid())
        break;

      const int64_t frame_var_ref =
          dap.variables.InsertVariable(variable, /*is_permanent=*/false);
      variables.emplace_back(CreateVariable(
          variable, frame_var_ref, hex,
          dap.configuration.enableAutoVariableSummaries,
          dap.configuration.enableSyntheticChildDebugging,
          variable_name_counts[GetNonNullVariableName(variable)] > 1));
    }
  } else {
    // We are expanding a variable that has children, so we will return its
    // children.
    lldb::SBValue variable = dap.variables.GetVariable(var_ref);
    if (variable.IsValid()) {
      const bool is_permanent =
          dap.variables.IsPermanentVariableReference(var_ref);
      auto addChild = [&](lldb::SBValue child,
                          std::optional<std::string> custom_name = {}) {
        if (!child.IsValid())
          return;
        const int64_t child_var_ref =
            dap.variables.InsertVariable(child, is_permanent);
        variables.emplace_back(
            CreateVariable(child, child_var_ref, hex,
                           dap.configuration.enableAutoVariableSummaries,
                           dap.configuration.enableSyntheticChildDebugging,
                           /*is_name_duplicated=*/false, custom_name));
      };
      const int64_t num_children = variable.GetNumChildren();
      const int64_t end_idx = start + ((count == 0) ? num_children : count);
      int64_t i = start;
      for (; i < end_idx && i < num_children; ++i)
        addChild(variable.GetChildAtIndex(i));

      // If we haven't filled the count quota from the request, we insert a new
      // "[raw]" child that can be used to inspect the raw version of a
      // synthetic member. That eliminates the need for the user to go to the
      // debug console and type `frame var <variable> to get these values.
      if (dap.configuration.enableSyntheticChildDebugging &&
          variable.IsSynthetic() && i == num_children)
        addChild(variable.GetNonSyntheticValue(), "[raw]");
    }
  }

  return VariablesResponseBody{variables};
}

} // namespace lldb_dap
