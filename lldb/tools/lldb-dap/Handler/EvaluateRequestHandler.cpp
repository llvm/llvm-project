//===-- EvaluateRequestHandler.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "JSONUtils.h"
#include "LLDBUtils.h"
#include "RequestHandler.h"

namespace lldb_dap {

//  "EvaluateRequest": {
//    "allOf": [ { "$ref": "#/definitions/Request" }, {
//      "type": "object",
//      "description": "Evaluate request; value of command field is 'evaluate'.
//                      Evaluates the given expression in the context of the
//                      top most stack frame. The expression has access to any
//                      variables and arguments that are in scope.",
//      "properties": {
//        "command": {
//          "type": "string",
//          "enum": [ "evaluate" ]
//        },
//        "arguments": {
//          "$ref": "#/definitions/EvaluateArguments"
//        }
//      },
//      "required": [ "command", "arguments"  ]
//    }]
//  },
//  "EvaluateArguments": {
//    "type": "object",
//    "description": "Arguments for 'evaluate' request.",
//    "properties": {
//      "expression": {
//        "type": "string",
//        "description": "The expression to evaluate."
//      },
//      "frameId": {
//        "type": "integer",
//        "description": "Evaluate the expression in the scope of this stack
//                        frame. If not specified, the expression is evaluated
//                        in the global scope."
//      },
//      "context": {
//        "type": "string",
//        "_enum": [ "watch", "repl", "hover" ],
//        "enumDescriptions": [
//          "evaluate is run in a watch.",
//          "evaluate is run from REPL console.",
//          "evaluate is run from a data hover."
//        ],
//        "description": "The context in which the evaluate request is run."
//      },
//      "format": {
//        "$ref": "#/definitions/ValueFormat",
//        "description": "Specifies details on how to format the Evaluate
//                        result."
//      }
//    },
//    "required": [ "expression" ]
//  },
//  "EvaluateResponse": {
//    "allOf": [ { "$ref": "#/definitions/Response" }, {
//      "type": "object",
//      "description": "Response to 'evaluate' request.",
//      "properties": {
//        "body": {
//          "type": "object",
//          "properties": {
//            "result": {
//              "type": "string",
//              "description": "The result of the evaluate request."
//            },
//            "type": {
//              "type": "string",
//              "description": "The optional type of the evaluate result."
//            },
//            "presentationHint": {
//              "$ref": "#/definitions/VariablePresentationHint",
//              "description": "Properties of a evaluate result that can be
//                              used to determine how to render the result in
//                              the UI."
//            },
//            "variablesReference": {
//              "type": "number",
//              "description": "If variablesReference is > 0, the evaluate
//                              result is structured and its children can be
//                              retrieved by passing variablesReference to the
//                              VariablesRequest."
//            },
//            "namedVariables": {
//              "type": "number",
//              "description": "The number of named child variables. The
//                              client can use this optional information to
//                              present the variables in a paged UI and fetch
//                              them in chunks."
//            },
//            "indexedVariables": {
//              "type": "number",
//              "description": "The number of indexed child variables. The
//                              client can use this optional information to
//                              present the variables in a paged UI and fetch
//                              them in chunks."
//            },
//            "valueLocationReference": {
//              "type": "integer",
//              "description": "A reference that allows the client to request
//                              the location where the returned value is
//                              declared. For example, if a function pointer is
//                              returned, the adapter may be able to look up the
//                              function's location. This should be present only
//                              if the adapter is likely to be able to resolve
//                              the location.\n\nThis reference shares the same
//                              lifetime as the `variablesReference`. See
//                              'Lifetime of Object References' in the
//              Overview section for details."
//            }
//            "memoryReference": {
//               "type": "string",
//                "description": "A memory reference to a location appropriate
//                                for this result. For pointer type eval
//                                results, this is generally a reference to the
//                                memory address contained in the pointer. This
//                                attribute may be returned by a debug adapter
//                                if corresponding capability
//                                `supportsMemoryReferences` is true."
//             },
//          },
//          "required": [ "result", "variablesReference" ]
//        }
//      },
//      "required": [ "body" ]
//    }]
//  }
void EvaluateRequestHandler::operator()(
    const llvm::json::Object &request) const {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Object body;
  const auto *arguments = request.getObject("arguments");
  lldb::SBFrame frame = dap.GetLLDBFrame(*arguments);
  std::string expression = GetString(arguments, "expression").str();
  llvm::StringRef context = GetString(arguments, "context");
  bool repeat_last_command =
      expression.empty() && dap.last_nonempty_var_expression.empty();

  if (context == "repl" &&
      (repeat_last_command ||
       (!expression.empty() &&
        dap.DetectReplMode(frame, expression, false) == ReplMode::Command))) {
    // Since the current expression is not for a variable, clear the
    // last_nonempty_var_expression field.
    dap.last_nonempty_var_expression.clear();
    // If we're evaluating a command relative to the current frame, set the
    // focus_tid to the current frame for any thread related events.
    if (frame.IsValid()) {
      dap.focus_tid = frame.GetThread().GetThreadID();
    }
    auto result = RunLLDBCommandsVerbatim(dap.debugger, llvm::StringRef(),
                                          {std::string(expression)});
    EmplaceSafeString(body, "result", result);
    body.try_emplace("variablesReference", (int64_t)0);
  } else {
    if (context == "repl") {
      // If the expression is empty and the last expression was for a
      // variable, set the expression to the previous expression (repeat the
      // evaluation); otherwise save the current non-empty expression for the
      // next (possibly empty) variable expression.
      if (expression.empty())
        expression = dap.last_nonempty_var_expression;
      else
        dap.last_nonempty_var_expression = expression;
    }
    // Always try to get the answer from the local variables if possible. If
    // this fails, then if the context is not "hover", actually evaluate an
    // expression using the expression parser.
    //
    // "frame variable" is more reliable than the expression parser in
    // many cases and it is faster.
    lldb::SBValue value = frame.GetValueForVariablePath(
        expression.data(), lldb::eDynamicDontRunTarget);

    // Freeze dry the value in case users expand it later in the debug console
    if (value.GetError().Success() && context == "repl")
      value = value.Persist();

    if (value.GetError().Fail() && context != "hover")
      value = frame.EvaluateExpression(expression.data());

    if (value.GetError().Fail()) {
      response["success"] = llvm::json::Value(false);
      // This error object must live until we're done with the pointer returned
      // by GetCString().
      lldb::SBError error = value.GetError();
      const char *error_cstr = error.GetCString();
      if (error_cstr && error_cstr[0])
        EmplaceSafeString(response, "message", std::string(error_cstr));
      else
        EmplaceSafeString(response, "message", "evaluate failed");
    } else {
      VariableDescription desc(value, dap.enable_auto_variable_summaries);
      EmplaceSafeString(body, "result", desc.GetResult(context));
      EmplaceSafeString(body, "type", desc.display_type_name);
      int64_t var_ref = 0;
      if (value.MightHaveChildren() || ValuePointsToCode(value))
        var_ref = dap.variables.InsertVariable(
            value, /*is_permanent=*/context == "repl");
      if (value.MightHaveChildren())
        body.try_emplace("variablesReference", var_ref);
      else
        body.try_emplace("variablesReference", (int64_t)0);
      if (lldb::addr_t addr = value.GetLoadAddress();
          addr != LLDB_INVALID_ADDRESS)
        body.try_emplace("memoryReference", EncodeMemoryReference(addr));
      if (ValuePointsToCode(value))
        body.try_emplace("valueLocationReference", var_ref);
    }
  }
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}
} // namespace lldb_dap
