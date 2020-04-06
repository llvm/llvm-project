//===-- cli-wrapper-pt.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// CLI Wrapper of PTDecoder Tool to enable it to be used through LLDB's CLI. The
// wrapper provides a new command called processor-trace with 4 child
// subcommands as follows:
// processor-trace start
// processor-trace stop
// processor-trace show-trace-options
// processor-trace show-instr-log
//
//===----------------------------------------------------------------------===//

#include <cerrno>
#include <cinttypes>
#include <cstring>
#include <string>
#include <vector>

#include "PTDecoder.h"
#include "cli-wrapper-pt.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStructuredData.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"

static bool GetProcess(lldb::SBDebugger &debugger,
                       lldb::SBCommandReturnObject &result,
                       lldb::SBProcess &process) {
  if (!debugger.IsValid()) {
    result.Printf("error: invalid debugger\n");
    result.SetStatus(lldb::eReturnStatusFailed);
    return false;
  }

  lldb::SBTarget target = debugger.GetSelectedTarget();
  if (!target.IsValid()) {
    result.Printf("error: invalid target inside debugger\n");
    result.SetStatus(lldb::eReturnStatusFailed);
    return false;
  }

  process = target.GetProcess();
  if (!process.IsValid() ||
      (process.GetState() == lldb::StateType::eStateDetached) ||
      (process.GetState() == lldb::StateType::eStateExited) ||
      (process.GetState() == lldb::StateType::eStateInvalid)) {
    result.Printf("error: invalid process inside debugger's target\n");
    result.SetStatus(lldb::eReturnStatusFailed);
    return false;
  }

  return true;
}

static bool ParseCommandOption(char **command,
                               lldb::SBCommandReturnObject &result,
                               uint32_t &index, const std::string &arg,
                               uint32_t &parsed_result) {
  char *endptr;
  if (!command[++index]) {
    result.Printf("error: option \"%s\" requires an argument\n", arg.c_str());
    result.SetStatus(lldb::eReturnStatusFailed);
    return false;
  }

  errno = 0;
  unsigned long output = strtoul(command[index], &endptr, 0);
  if ((errno != 0) || (*endptr != '\0')) {
    result.Printf("error: invalid value \"%s\" provided for option \"%s\"\n",
                  command[index], arg.c_str());
    result.SetStatus(lldb::eReturnStatusFailed);
    return false;
  }
  if (output > UINT32_MAX) {
    result.Printf("error: value \"%s\" for option \"%s\" exceeds UINT32_MAX\n",
                  command[index], arg.c_str());
    result.SetStatus(lldb::eReturnStatusFailed);
    return false;
  }
  parsed_result = (uint32_t)output;
  return true;
}

static bool ParseCommandArgThread(char **command,
                                  lldb::SBCommandReturnObject &result,
                                  lldb::SBProcess &process, uint32_t &index,
                                  lldb::tid_t &thread_id) {
  char *endptr;
  if (!strcmp(command[index], "all"))
    thread_id = LLDB_INVALID_THREAD_ID;
  else {
    uint32_t thread_index_id;
    errno = 0;
    unsigned long output = strtoul(command[index], &endptr, 0);
    if ((errno != 0) || (*endptr != '\0') || (output > UINT32_MAX)) {
      result.Printf("error: invalid thread specification: \"%s\"\n",
                    command[index]);
      result.SetStatus(lldb::eReturnStatusFailed);
      return false;
    }
    thread_index_id = (uint32_t)output;

    lldb::SBThread thread = process.GetThreadByIndexID(thread_index_id);
    if (!thread.IsValid()) {
      result.Printf(
          "error: process has no thread with thread specification: \"%s\"\n",
          command[index]);
      result.SetStatus(lldb::eReturnStatusFailed);
      return false;
    }
    thread_id = thread.GetThreadID();
  }
  return true;
}

class ProcessorTraceStart : public lldb::SBCommandPluginInterface {
public:
  ProcessorTraceStart(std::shared_ptr<ptdecoder::PTDecoder> &pt_decoder)
      : SBCommandPluginInterface(), pt_decoder_sp(pt_decoder) {}

  ~ProcessorTraceStart() {}

  virtual bool DoExecute(lldb::SBDebugger debugger, char **command,
                         lldb::SBCommandReturnObject &result) {
    lldb::SBProcess process;
    lldb::SBThread thread;
    if (!GetProcess(debugger, result, process))
      return false;

    // Default initialize API's arguments
    lldb::SBTraceOptions lldb_SBTraceOptions;
    uint32_t trace_buffer_size = m_default_trace_buff_size;
    lldb::tid_t thread_id;

    // Parse Command line options
    bool thread_argument_provided = false;
    if (command) {
      for (uint32_t i = 0; command[i]; i++) {
        if (!strcmp(command[i], "-b")) {
          if (!ParseCommandOption(command, result, i, "-b", trace_buffer_size))
            return false;
        } else {
          thread_argument_provided = true;
          if (!ParseCommandArgThread(command, result, process, i, thread_id))
            return false;
        }
      }
    }

    if (!thread_argument_provided) {
      thread = process.GetSelectedThread();
      if (!thread.IsValid()) {
        result.Printf("error: invalid current selected thread\n");
        result.SetStatus(lldb::eReturnStatusFailed);
        return false;
      }
      thread_id = thread.GetThreadID();
    }

    if (trace_buffer_size > m_max_trace_buff_size)
      trace_buffer_size = m_max_trace_buff_size;

    // Set API's arguments with parsed values
    lldb_SBTraceOptions.setType(lldb::TraceType::eTraceTypeProcessorTrace);
    lldb_SBTraceOptions.setTraceBufferSize(trace_buffer_size);
    lldb_SBTraceOptions.setMetaDataBufferSize(0);
    lldb_SBTraceOptions.setThreadID(thread_id);
    lldb::SBStream sb_stream;
    sb_stream.Printf("{\"trace-tech\":\"intel-pt\"}");
    lldb::SBStructuredData custom_params;
    lldb::SBError error = custom_params.SetFromJSON(sb_stream);
    if (!error.Success()) {
      result.Printf("error: %s\n", error.GetCString());
      result.SetStatus(lldb::eReturnStatusFailed);
      return false;
    }
    lldb_SBTraceOptions.setTraceParams(custom_params);

    // Start trace
    pt_decoder_sp->StartProcessorTrace(process, lldb_SBTraceOptions, error);
    if (!error.Success()) {
      result.Printf("error: %s\n", error.GetCString());
      result.SetStatus(lldb::eReturnStatusFailed);
      return false;
    }
    result.SetStatus(lldb::eReturnStatusSuccessFinishResult);
    return true;
  }

private:
  std::shared_ptr<ptdecoder::PTDecoder> pt_decoder_sp;
  const uint32_t m_max_trace_buff_size = 0x3fff;
  const uint32_t m_default_trace_buff_size = 4096;
};

class ProcessorTraceInfo : public lldb::SBCommandPluginInterface {
public:
  ProcessorTraceInfo(std::shared_ptr<ptdecoder::PTDecoder> &pt_decoder)
      : SBCommandPluginInterface(), pt_decoder_sp(pt_decoder) {}

  ~ProcessorTraceInfo() {}

  virtual bool DoExecute(lldb::SBDebugger debugger, char **command,
                         lldb::SBCommandReturnObject &result) {
    lldb::SBProcess process;
    lldb::SBThread thread;
    if (!GetProcess(debugger, result, process))
      return false;

    lldb::tid_t thread_id;

    // Parse command line options
    bool thread_argument_provided = false;
    if (command) {
      for (uint32_t i = 0; command[i]; i++) {
        thread_argument_provided = true;
        if (!ParseCommandArgThread(command, result, process, i, thread_id))
          return false;
      }
    }

    if (!thread_argument_provided) {
      thread = process.GetSelectedThread();
      if (!thread.IsValid()) {
        result.Printf("error: invalid current selected thread\n");
        result.SetStatus(lldb::eReturnStatusFailed);
        return false;
      }
      thread_id = thread.GetThreadID();
    }

    size_t loop_count = 1;
    bool entire_process_tracing = false;
    if (thread_id == LLDB_INVALID_THREAD_ID) {
      entire_process_tracing = true;
      loop_count = process.GetNumThreads();
    }

    // Get trace information
    lldb::SBError error;
    lldb::SBCommandReturnObject res;
    for (size_t i = 0; i < loop_count; i++) {
      error.Clear();
      res.Clear();

      if (entire_process_tracing)
        thread = process.GetThreadAtIndex(i);
      else
        thread = process.GetThreadByID(thread_id);
      thread_id = thread.GetThreadID();

      ptdecoder::PTTraceOptions options;
      pt_decoder_sp->GetProcessorTraceInfo(process, thread_id, options, error);
      if (!error.Success()) {
        res.Printf("thread #%" PRIu32 ": tid=%" PRIu64 ", error: %s",
                   thread.GetIndexID(), thread_id, error.GetCString());
        result.AppendMessage(res.GetOutput());
        continue;
      }

      lldb::SBStructuredData data = options.GetTraceParams(error);
      if (!error.Success()) {
        res.Printf("thread #%" PRIu32 ": tid=%" PRIu64 ", error: %s",
                   thread.GetIndexID(), thread_id, error.GetCString());
        result.AppendMessage(res.GetOutput());
        continue;
      }

      lldb::SBStream s;
      error = data.GetAsJSON(s);
      if (!error.Success()) {
        res.Printf("thread #%" PRIu32 ": tid=%" PRIu64 ", error: %s",
                   thread.GetIndexID(), thread_id, error.GetCString());
        result.AppendMessage(res.GetOutput());
        continue;
      }

      res.Printf("thread #%" PRIu32 ": tid=%" PRIu64
                 ", trace buffer size=%" PRIu64 ", meta buffer size=%" PRIu64
                 ", trace type=%" PRIu32 ", custom trace params=%s",
                 thread.GetIndexID(), thread_id, options.GetTraceBufferSize(),
                 options.GetMetaDataBufferSize(), options.GetType(),
                 s.GetData());
      result.AppendMessage(res.GetOutput());
    }
    result.SetStatus(lldb::eReturnStatusSuccessFinishResult);
    return true;
  }

private:
  std::shared_ptr<ptdecoder::PTDecoder> pt_decoder_sp;
};

class ProcessorTraceShowInstrLog : public lldb::SBCommandPluginInterface {
public:
  ProcessorTraceShowInstrLog(std::shared_ptr<ptdecoder::PTDecoder> &pt_decoder)
      : SBCommandPluginInterface(), pt_decoder_sp(pt_decoder) {}

  ~ProcessorTraceShowInstrLog() {}

  virtual bool DoExecute(lldb::SBDebugger debugger, char **command,
                         lldb::SBCommandReturnObject &result) {
    lldb::SBProcess process;
    lldb::SBThread thread;
    if (!GetProcess(debugger, result, process))
      return false;

    // Default initialize API's arguments
    uint32_t offset;
    bool offset_provided = false;
    uint32_t count = m_default_count;
    lldb::tid_t thread_id;

    // Parse command line options
    bool thread_argument_provided = false;
    if (command) {
      for (uint32_t i = 0; command[i]; i++) {
        if (!strcmp(command[i], "-o")) {
          if (!ParseCommandOption(command, result, i, "-o", offset))
            return false;
          offset_provided = true;
        } else if (!strcmp(command[i], "-c")) {
          if (!ParseCommandOption(command, result, i, "-c", count))
            return false;
        } else {
          thread_argument_provided = true;
          if (!ParseCommandArgThread(command, result, process, i, thread_id))
            return false;
        }
      }
    }

    if (!thread_argument_provided) {
      thread = process.GetSelectedThread();
      if (!thread.IsValid()) {
        result.Printf("error: invalid current selected thread\n");
        result.SetStatus(lldb::eReturnStatusFailed);
        return false;
      }
      thread_id = thread.GetThreadID();
    }

    size_t loop_count = 1;
    bool entire_process_tracing = false;
    if (thread_id == LLDB_INVALID_THREAD_ID) {
      entire_process_tracing = true;
      loop_count = process.GetNumThreads();
    }

    // Get instruction log and disassemble it
    lldb::SBError error;
    lldb::SBCommandReturnObject res;
    for (size_t i = 0; i < loop_count; i++) {
      error.Clear();
      res.Clear();

      if (entire_process_tracing)
        thread = process.GetThreadAtIndex(i);
      else
        thread = process.GetThreadByID(thread_id);
      thread_id = thread.GetThreadID();

      // If offset is not provided then calculate a default offset (to display
      // last 'count' number of instructions)
      if (!offset_provided)
        offset = count - 1;

      // Get the instruction log
      ptdecoder::PTInstructionList insn_list;
      pt_decoder_sp->GetInstructionLogAtOffset(process, thread_id, offset,
                                               count, insn_list, error);
      if (!error.Success()) {
        res.Printf("thread #%" PRIu32 ": tid=%" PRIu64 ", error: %s",
                   thread.GetIndexID(), thread_id, error.GetCString());
        result.AppendMessage(res.GetOutput());
        continue;
      }

      // Disassemble the instruction log
      std::string disassembler_command("dis -c 1 -s ");
      res.Printf("thread #%" PRIu32 ": tid=%" PRIu64 "\n", thread.GetIndexID(),
                 thread_id);
      lldb::SBCommandInterpreter sb_cmnd_interpreter(
          debugger.GetCommandInterpreter());
      lldb::SBCommandReturnObject result_obj;
      for (size_t i = 0; i < insn_list.GetSize(); i++) {
        ptdecoder::PTInstruction insn = insn_list.GetInstructionAtIndex(i);
        uint64_t addr = insn.GetInsnAddress();
        std::string error = insn.GetError();
        if (!error.empty()) {
          res.AppendMessage(error.c_str());
          continue;
        }

        result_obj.Clear();
        std::string complete_disassembler_command =
            disassembler_command + std::to_string(addr);
        sb_cmnd_interpreter.HandleCommand(complete_disassembler_command.c_str(),
                                          result_obj, false);
        std::string result_str(result_obj.GetOutput());
        if (result_str.empty()) {
          lldb::SBCommandReturnObject output;
          output.Printf(" Disassembly not found for address: %" PRIu64, addr);
          res.AppendMessage(output.GetOutput());
          continue;
        }

        // LLDB's disassemble command displays assembly instructions along with
        // the names of the functions they belong to. Parse this result to
        // display only the assembly instructions and not the function names
        // in an instruction log
        std::size_t first_new_line_index = result_str.find_first_of('\n');
        std::size_t last_new_line_index = result_str.find_last_of('\n');
        if (first_new_line_index != last_new_line_index)
          res.AppendMessage((result_str.substr(first_new_line_index + 1,
                                               last_new_line_index -
                                                   first_new_line_index - 1))
                                .c_str());
        else
          res.AppendMessage(
              (result_str.substr(0, result_str.length() - 1)).c_str());
      }
      result.AppendMessage(res.GetOutput());
    }
    result.SetStatus(lldb::eReturnStatusSuccessFinishResult);
    return true;
  }

private:
  std::shared_ptr<ptdecoder::PTDecoder> pt_decoder_sp;
  const uint32_t m_default_count = 10;
};

class ProcessorTraceStop : public lldb::SBCommandPluginInterface {
public:
  ProcessorTraceStop(std::shared_ptr<ptdecoder::PTDecoder> &pt_decoder)
      : SBCommandPluginInterface(), pt_decoder_sp(pt_decoder) {}

  ~ProcessorTraceStop() {}

  virtual bool DoExecute(lldb::SBDebugger debugger, char **command,
                         lldb::SBCommandReturnObject &result) {
    lldb::SBProcess process;
    lldb::SBThread thread;
    if (!GetProcess(debugger, result, process))
      return false;

    lldb::tid_t thread_id;

    // Parse command line options
    bool thread_argument_provided = false;
    if (command) {
      for (uint32_t i = 0; command[i]; i++) {
        thread_argument_provided = true;
        if (!ParseCommandArgThread(command, result, process, i, thread_id))
          return false;
      }
    }

    if (!thread_argument_provided) {
      thread = process.GetSelectedThread();
      if (!thread.IsValid()) {
        result.Printf("error: invalid current selected thread\n");
        result.SetStatus(lldb::eReturnStatusFailed);
        return false;
      }
      thread_id = thread.GetThreadID();
    }

    // Stop trace
    lldb::SBError error;
    pt_decoder_sp->StopProcessorTrace(process, error, thread_id);
    if (!error.Success()) {
      result.Printf("error: %s\n", error.GetCString());
      result.SetStatus(lldb::eReturnStatusFailed);
      return false;
    }
    result.SetStatus(lldb::eReturnStatusSuccessFinishResult);
    return true;
  }

private:
  std::shared_ptr<ptdecoder::PTDecoder> pt_decoder_sp;
};

bool PTPluginInitialize(lldb::SBDebugger &debugger) {
  lldb::SBCommandInterpreter interpreter = debugger.GetCommandInterpreter();
  lldb::SBCommand proc_trace = interpreter.AddMultiwordCommand(
      "processor-trace", "Intel(R) Processor Trace for thread/process");

  std::shared_ptr<ptdecoder::PTDecoder> PTDecoderSP(
      new ptdecoder::PTDecoder(debugger));

  lldb::SBCommandPluginInterface *proc_trace_start =
      new ProcessorTraceStart(PTDecoderSP);
  const char *help_proc_trace_start = "start Intel(R) Processor Trace on a "
                                      "specific thread or on the whole process";
  const char *syntax_proc_trace_start =
      "processor-trace start  <cmd-options>\n\n"
      "\rcmd-options Usage:\n"
      "\r  processor-trace start [-b <buffer-size>] [<thread-index>]\n\n"
      "\t\b-b <buffer-size>\n"
      "\t    size of the trace buffer to store the trace data. If not "
      "specified then a default value will be taken\n\n"
      "\t\b<thread-index>\n"
      "\t    thread index of the thread. If no threads are specified, "
      "currently selected thread is taken.\n"
      "\t    Use the thread-index 'all' to start tracing the whole process\n";
  proc_trace.AddCommand("start", proc_trace_start, help_proc_trace_start,
                        syntax_proc_trace_start);

  lldb::SBCommandPluginInterface *proc_trace_stop =
      new ProcessorTraceStop(PTDecoderSP);
  const char *help_proc_trace_stop =
      "stop Intel(R) Processor Trace on a specific thread or on whole process";
  const char *syntax_proc_trace_stop =
      "processor-trace stop  <cmd-options>\n\n"
      "\rcmd-options Usage:\n"
      "\r  processor-trace stop [<thread-index>]\n\n"
      "\t\b<thread-index>\n"
      "\t    thread index of the thread. If no threads are specified, "
      "currently selected thread is taken.\n"
      "\t    Use the thread-index 'all' to stop tracing the whole process\n";
  proc_trace.AddCommand("stop", proc_trace_stop, help_proc_trace_stop,
                        syntax_proc_trace_stop);

  lldb::SBCommandPluginInterface *proc_trace_show_instr_log =
      new ProcessorTraceShowInstrLog(PTDecoderSP);
  const char *help_proc_trace_show_instr_log =
      "display a log of assembly instructions executed for a specific thread "
      "or for the whole process.\n"
      "The length of the log to be displayed and the offset in the whole "
      "instruction log from where the log needs to be displayed can also be "
      "provided. The offset is counted from the end of this whole "
      "instruction log which means the last executed instruction is at offset "
      "0 (zero)";
  const char *syntax_proc_trace_show_instr_log =
      "processor-trace show-instr-log  <cmd-options>\n\n"
      "\rcmd-options Usage:\n"
      "\r  processor-trace show-instr-log [-o <offset>] [-c <count>] "
      "[<thread-index>]\n\n"
      "\t\b-o <offset>\n"
      "\t    offset in the whole instruction log from where the log will be "
      "displayed. If not specified then a default value will be taken\n\n"
      "\t\b-c <count>\n"
      "\t    number of instructions to be displayed. If not specified then a "
      "default value will be taken\n\n"
      "\t\b<thread-index>\n"
      "\t    thread index of the thread. If no threads are specified, "
      "currently selected thread is taken.\n"
      "\t    Use the thread-index 'all' to show instruction log for all the "
      "threads of the process\n";
  proc_trace.AddCommand("show-instr-log", proc_trace_show_instr_log,
                        help_proc_trace_show_instr_log,
                        syntax_proc_trace_show_instr_log);

  lldb::SBCommandPluginInterface *proc_trace_options =
      new ProcessorTraceInfo(PTDecoderSP);
  const char *help_proc_trace_show_options =
      "display all the information regarding Intel(R) Processor Trace for a "
      "specific thread or for the whole process.\n"
      "The information contains trace buffer size and configuration options"
      " of Intel(R) Processor Trace.";
  const char *syntax_proc_trace_show_options =
      "processor-trace show-options <cmd-options>\n\n"
      "\rcmd-options Usage:\n"
      "\r  processor-trace show-options [<thread-index>]\n\n"
      "\t\b<thread-index>\n"
      "\t    thread index of the thread. If no threads are specified, "
      "currently selected thread is taken.\n"
      "\t    Use the thread-index 'all' to display information for all threads "
      "of the process\n";
  proc_trace.AddCommand("show-trace-options", proc_trace_options,
                        help_proc_trace_show_options,
                        syntax_proc_trace_show_options);

  return true;
}
