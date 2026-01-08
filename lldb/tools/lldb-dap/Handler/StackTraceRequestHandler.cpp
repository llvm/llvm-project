//===-- StackTraceRequestHandler.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "DAPError.h"
#include "EventHelper.h"
#include "LLDBUtils.h"
#include "Protocol/ProtocolRequests.h"
#include "ProtocolUtils.h"
#include "RequestHandler.h"
#include "lldb/API/SBStream.h"

using namespace lldb_dap;
using namespace lldb_dap::protocol;

/// Page size used for reporting additional frames in the 'stackTrace' request.
static constexpr int k_stack_page_size = 20;

// Create a "StackFrame" object for a LLDB frame object.
static StackFrame CreateStackFrame(DAP &dap, lldb::SBFrame &frame,
                                   lldb::SBFormat &format) {
  StackFrame stack_frame;
  stack_frame.id = MakeDAPFrameID(frame);

  lldb::SBStream stream;
  if (format && frame.GetDescriptionWithFormat(format, stream).Success()) {
    stack_frame.name = llvm::StringRef(stream.GetData(), stream.GetSize());

    // `function_name` can be a nullptr, which throws an error when assigned to
    // an `std::string`.
  } else if (llvm::StringRef name = frame.GetDisplayFunctionName();
             !name.empty()) {
    stack_frame.name = name;
  }

  if (stack_frame.name.empty()) {
    // If the function name is unavailable, display the pc address as a 16-digit
    // hex string, e.g. "0x0000000000012345"
    stack_frame.name = GetLoadAddressString(frame.GetPC());
  }

  // We only include `[opt]` if a custom frame format is not specified.
  if (!format && frame.GetFunction().GetIsOptimized())
    stack_frame.name += " [opt]";

  std::optional<protocol::Source> source = dap.ResolveSource(frame);
  if (source && !IsAssemblySource(*source)) {
    // This is a normal source with a valid line entry.
    auto line_entry = frame.GetLineEntry();
    stack_frame.line = line_entry.GetLine();
    stack_frame.column = line_entry.GetColumn();
  } else if (frame.GetSymbol().IsValid()) {
    // This is a source where the disassembly is used, but there is a valid
    // symbol. Calculate the line of the current PC from the start of the
    // current symbol.
    lldb::SBInstructionList inst_list = dap.target.ReadInstructions(
        frame.GetSymbol().GetStartAddress(), frame.GetPCAddress(), nullptr);
    size_t inst_line = inst_list.GetSize();

    // Line numbers are 1-based.
    stack_frame.line = inst_line + 1;
    stack_frame.column = 1;
  } else {
    // No valid line entry or symbol.
    stack_frame.line = 0;
    stack_frame.column = 0;
  }

  stack_frame.source = std::move(source);
  stack_frame.instructionPointerReference = frame.GetPC();

  if (frame.IsArtificial() || frame.IsHidden())
    stack_frame.presentationHint = StackFrame::ePresentationHintSubtle;
  if (const lldb::SBModule module = frame.GetModule()) {
    if (llvm::StringRef uuid = module.GetUUIDString(); !uuid.empty())
      stack_frame.moduleId = uuid.str();
  }

  return stack_frame;
}

// Create a "StackFrame" label object for a LLDB thread.
static StackFrame CreateExtendedStackFrameLabel(lldb::SBThread &thread,
                                                lldb::SBFormat &format) {
  StackFrame stack_frame;
  lldb::SBStream stream;
  if (format && thread.GetDescriptionWithFormat(format, stream).Success()) {
    stack_frame.name = llvm::StringRef(stream.GetData(), stream.GetSize());
  } else {
    const uint32_t thread_idx = thread.GetExtendedBacktraceOriginatingIndexID();
    if (llvm::StringRef queue_name = thread.GetQueueName();
        !queue_name.empty()) {
      stack_frame.name = llvm::formatv("Enqueued from {0} (Thread {1})",
                                       queue_name, thread_idx);
    } else {
      stack_frame.name = llvm::formatv("Thread {0}", thread_idx);
    }
  }

  stack_frame.id = thread.GetThreadID() + 1;
  stack_frame.presentationHint = StackFrame::ePresentationHintLabel;
  stack_frame.line = 0;
  stack_frame.column = 0;

  return stack_frame;
}

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
                            lldb::SBFormat &frame_format,
                            std::vector<StackFrame> &stack_frames,
                            int64_t &offset, const int64_t start_frame,
                            const int64_t levels, const bool include_all) {
  bool reached_end_of_stack = false;
  for (int64_t i = start_frame;
       static_cast<int64_t>(stack_frames.size()) < levels; i++) {
    if (i == -1) {
      stack_frames.emplace_back(
          CreateExtendedStackFrameLabel(thread, frame_format));
      continue;
    }

    lldb::SBFrame frame = thread.GetFrameAtIndex(i);
    if (!frame.IsValid()) {
      offset += thread.GetNumFrames() + 1 /* label between threads */;
      reached_end_of_stack = true;
      break;
    }

    stack_frames.emplace_back(CreateStackFrame(dap, frame, frame_format));
  }

  if (include_all && reached_end_of_stack) {
    // Check for any extended backtraces.
    for (uint32_t bt = 0;
         bt < thread.GetProcess().GetNumExtendedBacktraceTypes(); bt++) {
      lldb::SBThread backtrace = thread.GetExtendedBacktraceThread(
          thread.GetProcess().GetExtendedBacktraceTypeAtIndex(bt));
      if (!backtrace.IsValid())
        continue;

      reached_end_of_stack = FillStackFrames(
          dap, backtrace, frame_format, stack_frames, offset,
          (start_frame - offset) > 0 ? start_frame - offset : -1, levels,
          include_all);
      if (static_cast<int64_t>(stack_frames.size()) >= levels)
        break;
    }
  }

  return reached_end_of_stack;
}

llvm::Expected<protocol::StackTraceResponseBody>
StackTraceRequestHandler::Run(const protocol::StackTraceArguments &args) const {
  lldb::SBThread thread = dap.GetLLDBThread(args.threadId);
  if (!thread.IsValid())
    return llvm::make_error<DAPError>("invalid thread");

  lldb::SBFormat frame_format = dap.frame_format;
  bool include_all = dap.configuration.displayExtendedBacktrace;

  if (args.format) {
    const StackFrameFormat &format = *args.format;

    include_all = format.includeAll;

    // FIXME: Support "parameterTypes" and "hex".
    // Only change the format string if we have to.
    if (format.module || format.line || format.parameters ||
        format.parameterNames || format.parameterValues) {
      std::string format_str;
      llvm::raw_string_ostream os(format_str);

      if (format.module)
        os << "{${module.file.basename} }";

      if (format.line)
        os << "{${line.file.basename}:${line.number}:${line.column} }";

      if (format.parameters || format.parameterNames || format.parameterValues)
        os << "{${function.name-with-args}}";
      else
        os << "{${function.name-without-args}}";

      lldb::SBError error;
      frame_format = lldb::SBFormat(format_str.c_str(), error);
      if (error.Fail())
        return ToError(error);
    }
  }

  StackTraceResponseBody body;
  const auto levels = args.levels == 0 ? INT64_MAX : args.levels;
  int64_t offset = 0;
  bool reached_end_of_stack =
      FillStackFrames(dap, thread, frame_format, body.stackFrames, offset,
                      args.startFrame, levels, include_all);
  body.totalFrames = args.startFrame + body.stackFrames.size() +
                     (reached_end_of_stack ? 0 : k_stack_page_size);

  return body;
}
