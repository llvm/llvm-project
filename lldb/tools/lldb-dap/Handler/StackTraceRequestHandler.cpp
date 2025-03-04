//===-- StackTraceRequestHandler.cpp --------------------------------------===//
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

/// Page size used for reporting addtional frames in the 'stackTrace' request.
static constexpr int StackPageSize = 20;

// Fill in the stack frames of the thread.
//
// Threads stacks may contain runtime specific extended backtraces, when
// constructing a stack trace first report the full thread stack trace then
// perform a breadth first traversal of any extended backtrace frames.
//
// For example:
//
// Thread (id=th0) stack=[s0, s1, s2, s3]
//   \ Extended backtrace "libdispatch" Thread (id=th1) stack=[s0, s1]
//     \ Extended backtrace "libdispatch" Thread (id=th2) stack=[s0, s1]
//   \ Extended backtrace "Application Specific Backtrace" Thread (id=th3)
//   stack=[s0, s1, s2]
//
// Which will flatten into:
//
//  0. th0->s0
//  1. th0->s1
//  2. th0->s2
//  3. th0->s3
//  4. label - Enqueued from th1, sf=-1, i=-4
//  5. th1->s0
//  6. th1->s1
//  7. label - Enqueued from th2
//  8. th2->s0
//  9. th2->s1
// 10. label - Application Specific Backtrace
// 11. th3->s0
// 12. th3->s1
// 13. th3->s2
//
// s=3,l=3 = [th0->s3, label1, th1->s0]
static bool FillStackFrames(DAP &dap, lldb::SBThread &thread,
                            llvm::json::Array &stack_frames, int64_t &offset,
                            const int64_t start_frame, const int64_t levels) {
  bool reached_end_of_stack = false;
  for (int64_t i = start_frame;
       static_cast<int64_t>(stack_frames.size()) < levels; i++) {
    if (i == -1) {
      stack_frames.emplace_back(
          CreateExtendedStackFrameLabel(thread, dap.frame_format));
      continue;
    }

    lldb::SBFrame frame = thread.GetFrameAtIndex(i);
    if (!frame.IsValid()) {
      offset += thread.GetNumFrames() + 1 /* label between threads */;
      reached_end_of_stack = true;
      break;
    }

    stack_frames.emplace_back(CreateStackFrame(frame, dap.frame_format));
  }

  if (dap.display_extended_backtrace && reached_end_of_stack) {
    // Check for any extended backtraces.
    for (uint32_t bt = 0;
         bt < thread.GetProcess().GetNumExtendedBacktraceTypes(); bt++) {
      lldb::SBThread backtrace = thread.GetExtendedBacktraceThread(
          thread.GetProcess().GetExtendedBacktraceTypeAtIndex(bt));
      if (!backtrace.IsValid())
        continue;

      reached_end_of_stack = FillStackFrames(
          dap, backtrace, stack_frames, offset,
          (start_frame - offset) > 0 ? start_frame - offset : -1, levels);
      if (static_cast<int64_t>(stack_frames.size()) >= levels)
        break;
    }
  }

  return reached_end_of_stack;
}

// "StackTraceRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "StackTrace request; value of command field is
//     'stackTrace'. The request returns a stacktrace from the current execution
//     state.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "stackTrace" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/StackTraceArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "StackTraceArguments": {
//   "type": "object",
//   "description": "Arguments for 'stackTrace' request.",
//   "properties": {
//     "threadId": {
//       "type": "integer",
//       "description": "Retrieve the stacktrace for this thread."
//     },
//     "startFrame": {
//       "type": "integer",
//       "description": "The index of the first frame to return; if omitted
//       frames start at 0."
//     },
//     "levels": {
//       "type": "integer",
//       "description": "The maximum number of frames to return. If levels is
//       not specified or 0, all frames are returned."
//     },
//     "format": {
//       "$ref": "#/definitions/StackFrameFormat",
//       "description": "Specifies details on how to format the stack frames.
//       The attribute is only honored by a debug adapter if the corresponding
//       capability `supportsValueFormattingOptions` is true."
//     }
//  },
//   "required": [ "threadId" ]
// },
// "StackTraceResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to `stackTrace` request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "stackFrames": {
//             "type": "array",
//             "items": {
//               "$ref": "#/definitions/StackFrame"
//             },
//             "description": "The frames of the stackframe. If the array has
//             length zero, there are no stackframes available. This means that
//             there is no location information available."
//           },
//           "totalFrames": {
//             "type": "integer",
//             "description": "The total number of frames available in the
//             stack. If omitted or if `totalFrames` is larger than the
//             available frames, a client is expected to request frames until
//             a request returns less frames than requested (which indicates
//             the end of the stack). Returning monotonically increasing
//             `totalFrames` values for subsequent requests can be used to
//             enforce paging in the client."
//           }
//         },
//         "required": [ "stackFrames" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
void StackTraceRequestHandler::operator()(
    const llvm::json::Object &request) const {
  llvm::json::Object response;
  FillResponse(request, response);
  lldb::SBError error;
  const auto *arguments = request.getObject("arguments");
  lldb::SBThread thread = dap.GetLLDBThread(*arguments);
  llvm::json::Array stack_frames;
  llvm::json::Object body;

  if (thread.IsValid()) {
    const auto start_frame = GetUnsigned(arguments, "startFrame", 0);
    const auto levels = GetUnsigned(arguments, "levels", 0);
    int64_t offset = 0;
    bool reached_end_of_stack =
        FillStackFrames(dap, thread, stack_frames, offset, start_frame,
                        levels == 0 ? INT64_MAX : levels);
    body.try_emplace("totalFrames",
                     start_frame + stack_frames.size() +
                         (reached_end_of_stack ? 0 : StackPageSize));
  }

  body.try_emplace("stackFrames", std::move(stack_frames));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
