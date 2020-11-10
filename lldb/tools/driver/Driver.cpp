//===-- Driver.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Driver.h"

#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandInterpreterRunOptions.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBFile.h"
#include "lldb/API/SBHostOS.h"
#include "lldb/API/SBLanguageRuntime.h"
#include "lldb/API/SBReproducer.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStringList.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <atomic>
#include <bitset>
#include <csignal>
#include <string>
#include <thread>
#include <utility>

#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Includes for pipe()
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

#if !defined(__APPLE__)
#include "llvm/Support/DataTypes.h"
#endif

using namespace lldb;
using namespace llvm;

namespace {
enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  OPT_##ID,
#include "Options.inc"
#undef OPTION
};

#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Options.inc"
#undef PREFIX

const opt::OptTable::Info InfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  {                                                                            \
      PREFIX,      NAME,      HELPTEXT,                                        \
      METAVAR,     OPT_##ID,  opt::Option::KIND##Class,                        \
      PARAM,       FLAGS,     OPT_##GROUP,                                     \
      OPT_##ALIAS, ALIASARGS, VALUES},
#include "Options.inc"
#undef OPTION
};

class LLDBOptTable : public opt::OptTable {
public:
  LLDBOptTable() : OptTable(InfoTable) {}
};
} // namespace

static void reset_stdin_termios();
static bool g_old_stdin_termios_is_valid = false;
static struct termios g_old_stdin_termios;

static Driver *g_driver = nullptr;

// In the Driver::MainLoop, we change the terminal settings.  This function is
// added as an atexit handler to make sure we clean them up.
static void reset_stdin_termios() {
  if (g_old_stdin_termios_is_valid) {
    g_old_stdin_termios_is_valid = false;
    ::tcsetattr(STDIN_FILENO, TCSANOW, &g_old_stdin_termios);
  }
}

Driver::Driver()
    : SBBroadcaster("Driver"), m_debugger(SBDebugger::Create(false)) {
  // We want to be able to handle CTRL+D in the terminal to have it terminate
  // certain input
  m_debugger.SetCloseInputOnEOF(false);
  g_driver = this;
}

Driver::~Driver() {
  SBDebugger::Destroy(m_debugger);
  g_driver = nullptr;
}

void Driver::OptionData::AddInitialCommand(std::string command,
                                           CommandPlacement placement,
                                           bool is_file, SBError &error) {
  std::vector<InitialCmdEntry> *command_set;
  switch (placement) {
  case eCommandPlacementBeforeFile:
    command_set = &(m_initial_commands);
    break;
  case eCommandPlacementAfterFile:
    command_set = &(m_after_file_commands);
    break;
  case eCommandPlacementAfterCrash:
    command_set = &(m_after_crash_commands);
    break;
  }

  if (is_file) {
    SBFileSpec file(command.c_str());
    if (file.Exists())
      command_set->push_back(InitialCmdEntry(command, is_file));
    else if (file.ResolveExecutableLocation()) {
      char final_path[PATH_MAX];
      file.GetPath(final_path, sizeof(final_path));
      command_set->push_back(InitialCmdEntry(final_path, is_file));
    } else
      error.SetErrorStringWithFormat(
          "file specified in --source (-s) option doesn't exist: '%s'",
          command.c_str());
  } else
    command_set->push_back(InitialCmdEntry(command, is_file));
}

void Driver::WriteCommandsForSourcing(CommandPlacement placement,
                                      SBStream &strm) {
  std::vector<OptionData::InitialCmdEntry> *command_set;
  switch (placement) {
  case eCommandPlacementBeforeFile:
    command_set = &m_option_data.m_initial_commands;
    break;
  case eCommandPlacementAfterFile:
    command_set = &m_option_data.m_after_file_commands;
    break;
  case eCommandPlacementAfterCrash:
    command_set = &m_option_data.m_after_crash_commands;
    break;
  }

  for (const auto &command_entry : *command_set) {
    const char *command = command_entry.contents.c_str();
    if (command_entry.is_file) {
      bool source_quietly =
          m_option_data.m_source_quietly || command_entry.source_quietly;
      strm.Printf("command source -s %i '%s'\n",
                  static_cast<int>(source_quietly), command);
    } else
      strm.Printf("%s\n", command);
  }
}

// Check the arguments that were passed to this program to make sure they are
// valid and to get their argument values (if any).  Return a boolean value
// indicating whether or not to start up the full debugger (i.e. the Command
// Interpreter) or not.  Return FALSE if the arguments were invalid OR if the
// user only wanted help or version information.
SBError Driver::ProcessArgs(const opt::InputArgList &args, bool &exiting) {
  SBError error;

  // This is kind of a pain, but since we make the debugger in the Driver's
  // constructor, we can't know at that point whether we should read in init
  // files yet.  So we don't read them in in the Driver constructor, then set
  // the flags back to "read them in" here, and then if we see the "-n" flag,
  // we'll turn it off again.  Finally we have to read them in by hand later in
  // the main loop.
  m_debugger.SkipLLDBInitFiles(false);
  m_debugger.SkipAppInitFiles(false);

  if (args.hasArg(OPT_version)) {
    m_option_data.m_print_version = true;
  }

  if (args.hasArg(OPT_python_path)) {
    m_option_data.m_print_python_path = true;
  }

  if (args.hasArg(OPT_batch)) {
    m_option_data.m_batch = true;
  }

  if (auto *arg = args.getLastArg(OPT_core)) {
    auto arg_value = arg->getValue();
    SBFileSpec file(arg_value);
    if (!file.Exists()) {
      error.SetErrorStringWithFormat(
          "file specified in --core (-c) option doesn't exist: '%s'",
          arg_value);
      return error;
    }
    m_option_data.m_core_file = arg_value;
  }

  if (args.hasArg(OPT_editor)) {
    m_option_data.m_use_external_editor = true;
  }

  if (args.hasArg(OPT_no_lldbinit)) {
    m_debugger.SkipLLDBInitFiles(true);
    m_debugger.SkipAppInitFiles(true);
  }

  if (args.hasArg(OPT_local_lldbinit)) {
    lldb::SBDebugger::SetInternalVariable("target.load-cwd-lldbinit", "true",
                                          m_debugger.GetInstanceName());
  }

  if (args.hasArg(OPT_no_use_colors)) {
    m_debugger.SetUseColor(false);
    m_option_data.m_debug_mode = true;
  }

  if (auto *arg = args.getLastArg(OPT_file)) {
    auto arg_value = arg->getValue();
    SBFileSpec file(arg_value);
    if (file.Exists()) {
      m_option_data.m_args.emplace_back(arg_value);
    } else if (file.ResolveExecutableLocation()) {
      char path[PATH_MAX];
      file.GetPath(path, sizeof(path));
      m_option_data.m_args.emplace_back(path);
    } else {
      error.SetErrorStringWithFormat(
          "file specified in --file (-f) option doesn't exist: '%s'",
          arg_value);
      return error;
    }
  }

  if (auto *arg = args.getLastArg(OPT_arch)) {
    auto arg_value = arg->getValue();
    if (!lldb::SBDebugger::SetDefaultArchitecture(arg_value)) {
      error.SetErrorStringWithFormat(
          "invalid architecture in the -a or --arch option: '%s'", arg_value);
      return error;
    }
  }

  if (auto *arg = args.getLastArg(OPT_script_language)) {
    auto arg_value = arg->getValue();
    m_debugger.SetScriptLanguage(m_debugger.GetScriptingLanguage(arg_value));
  }

  if (args.hasArg(OPT_source_quietly)) {
    m_option_data.m_source_quietly = true;
  }

  if (auto *arg = args.getLastArg(OPT_attach_name)) {
    auto arg_value = arg->getValue();
    m_option_data.m_process_name = arg_value;
  }

  if (args.hasArg(OPT_wait_for)) {
    m_option_data.m_wait_for = true;
  }

  if (auto *arg = args.getLastArg(OPT_attach_pid)) {
    auto arg_value = arg->getValue();
    char *remainder;
    m_option_data.m_process_pid = strtol(arg_value, &remainder, 0);
    if (remainder == arg_value || *remainder != '\0') {
      error.SetErrorStringWithFormat(
          "Could not convert process PID: \"%s\" into a pid.", arg_value);
      return error;
    }
  }

  if (auto *arg = args.getLastArg(OPT_repl_language)) {
    auto arg_value = arg->getValue();
    m_option_data.m_repl_lang =
        SBLanguageRuntime::GetLanguageTypeFromString(arg_value);
    if (m_option_data.m_repl_lang == eLanguageTypeUnknown) {
      error.SetErrorStringWithFormat("Unrecognized language name: \"%s\"",
                                     arg_value);
      return error;
    }
  }

  if (args.hasArg(OPT_repl)) {
    m_option_data.m_repl = true;
  }

  if (auto *arg = args.getLastArg(OPT_repl_)) {
    m_option_data.m_repl = true;
    if (auto arg_value = arg->getValue())
      m_option_data.m_repl_options = arg_value;
  }

  // We need to process the options below together as their relative order
  // matters.
  for (auto *arg : args.filtered(OPT_source_on_crash, OPT_one_line_on_crash,
                                 OPT_source, OPT_source_before_file,
                                 OPT_one_line, OPT_one_line_before_file)) {
    auto arg_value = arg->getValue();
    if (arg->getOption().matches(OPT_source_on_crash)) {
      m_option_data.AddInitialCommand(arg_value, eCommandPlacementAfterCrash,
                                      true, error);
      if (error.Fail())
        return error;
    }

    if (arg->getOption().matches(OPT_one_line_on_crash)) {
      m_option_data.AddInitialCommand(arg_value, eCommandPlacementAfterCrash,
                                      false, error);
      if (error.Fail())
        return error;
    }

    if (arg->getOption().matches(OPT_source)) {
      m_option_data.AddInitialCommand(arg_value, eCommandPlacementAfterFile,
                                      true, error);
      if (error.Fail())
        return error;
    }

    if (arg->getOption().matches(OPT_source_before_file)) {
      m_option_data.AddInitialCommand(arg_value, eCommandPlacementBeforeFile,
                                      true, error);
      if (error.Fail())
        return error;
    }

    if (arg->getOption().matches(OPT_one_line)) {
      m_option_data.AddInitialCommand(arg_value, eCommandPlacementAfterFile,
                                      false, error);
      if (error.Fail())
        return error;
    }

    if (arg->getOption().matches(OPT_one_line_before_file)) {
      m_option_data.AddInitialCommand(arg_value, eCommandPlacementBeforeFile,
                                      false, error);
      if (error.Fail())
        return error;
    }
  }

  if (m_option_data.m_process_name.empty() &&
      m_option_data.m_process_pid == LLDB_INVALID_PROCESS_ID) {

    for (auto *arg : args.filtered(OPT_INPUT))
      m_option_data.m_args.push_back(arg->getAsString((args)));

    // Any argument following -- is an argument for the inferior.
    if (auto *arg = args.getLastArgNoClaim(OPT_REM)) {
      for (auto value : arg->getValues())
        m_option_data.m_args.emplace_back(value);
    }
  } else if (args.getLastArgNoClaim() != nullptr) {
    WithColor::warning() << "program arguments are ignored when attaching.\n";
  }

  if (m_option_data.m_print_version) {
    llvm::outs() << lldb::SBDebugger::GetVersionString() << '\n';
    exiting = true;
    return error;
  }

  if (m_option_data.m_print_python_path) {
    SBFileSpec python_file_spec = SBHostOS::GetLLDBPythonPath();
    if (python_file_spec.IsValid()) {
      char python_path[PATH_MAX];
      size_t num_chars = python_file_spec.GetPath(python_path, PATH_MAX);
      if (num_chars < PATH_MAX) {
        llvm::outs() << python_path << '\n';
      } else
        llvm::outs() << "<PATH TOO LONG>\n";
    } else
      llvm::outs() << "<COULD NOT FIND PATH>\n";
    exiting = true;
    return error;
  }

  return error;
}

static inline int OpenPipe(int fds[2], std::size_t size) {
#ifdef _WIN32
  return _pipe(fds, size, O_BINARY);
#else
  (void)size;
  return pipe(fds);
#endif
}

static ::FILE *PrepareCommandsForSourcing(const char *commands_data,
                                          size_t commands_size) {
  enum PIPES { READ, WRITE }; // Indexes for the read and write fds
  int fds[2] = {-1, -1};

  if (OpenPipe(fds, commands_size) != 0) {
    WithColor::error()
        << "can't create pipe file descriptors for LLDB commands\n";
    return nullptr;
  }

  ssize_t nrwr = write(fds[WRITE], commands_data, commands_size);
  if (size_t(nrwr) != commands_size) {
    WithColor::error()
        << format(
               "write(%i, %p, %" PRIu64
               ") failed (errno = %i) when trying to open LLDB commands pipe",
               fds[WRITE], static_cast<const void *>(commands_data),
               static_cast<uint64_t>(commands_size), errno)
        << '\n';
    llvm::sys::Process::SafelyCloseFileDescriptor(fds[READ]);
    llvm::sys::Process::SafelyCloseFileDescriptor(fds[WRITE]);
    return nullptr;
  }

  // Close the write end of the pipe, so that the command interpreter will exit
  // when it consumes all the data.
  llvm::sys::Process::SafelyCloseFileDescriptor(fds[WRITE]);

  // Open the read file descriptor as a FILE * that we can return as an input
  // handle.
  ::FILE *commands_file = fdopen(fds[READ], "rb");
  if (commands_file == nullptr) {
    WithColor::error() << format("fdopen(%i, \"rb\") failed (errno = %i) "
                                 "when trying to open LLDB commands pipe",
                                 fds[READ], errno)
                       << '\n';
    llvm::sys::Process::SafelyCloseFileDescriptor(fds[READ]);
    return nullptr;
  }

  // 'commands_file' now owns the read descriptor.
  return commands_file;
}

std::string EscapeString(std::string arg) {
  std::string::size_type pos = 0;
  while ((pos = arg.find_first_of("\"\\", pos)) != std::string::npos) {
    arg.insert(pos, 1, '\\');
    pos += 2;
  }
  return '"' + arg + '"';
}

int Driver::MainLoop() {
  if (::tcgetattr(STDIN_FILENO, &g_old_stdin_termios) == 0) {
    g_old_stdin_termios_is_valid = true;
    atexit(reset_stdin_termios);
  }

#ifndef _MSC_VER
  // Disabling stdin buffering with MSVC's 2015 CRT exposes a bug in fgets
  // which causes it to miss newlines depending on whether there have been an
  // odd or even number of characters.  Bug has been reported to MS via Connect.
  ::setbuf(stdin, nullptr);
#endif
  ::setbuf(stdout, nullptr);

  m_debugger.SetErrorFileHandle(stderr, false);
  m_debugger.SetOutputFileHandle(stdout, false);
  // Don't take ownership of STDIN yet...
  m_debugger.SetInputFileHandle(stdin, false);

  m_debugger.SetUseExternalEditor(m_option_data.m_use_external_editor);

  struct winsize window_size;
  if ((isatty(STDIN_FILENO) != 0) &&
      ::ioctl(STDIN_FILENO, TIOCGWINSZ, &window_size) == 0) {
    if (window_size.ws_col > 0)
      m_debugger.SetTerminalWidth(window_size.ws_col);
  }

  SBCommandInterpreter sb_interpreter = m_debugger.GetCommandInterpreter();

  // Before we handle any options from the command line, we parse the
  // REPL init file or the default file in the user's home directory.
  SBCommandReturnObject result;
  sb_interpreter.SourceInitFileInHomeDirectory(result, m_option_data.m_repl);
  if (m_option_data.m_debug_mode) {
    result.PutError(m_debugger.GetErrorFile());
    result.PutOutput(m_debugger.GetOutputFile());
  }

  // Source the local .lldbinit file if it exists and we're allowed to source.
  // Here we want to always print the return object because it contains the
  // warning and instructions to load local lldbinit files.
  sb_interpreter.SourceInitFileInCurrentWorkingDirectory(result);
  result.PutError(m_debugger.GetErrorFile());
  result.PutOutput(m_debugger.GetOutputFile());

  // We allow the user to specify an exit code when calling quit which we will
  // return when exiting.
  m_debugger.GetCommandInterpreter().AllowExitCodeOnQuit(true);

  // Now we handle options we got from the command line
  SBStream commands_stream;

  // First source in the commands specified to be run before the file arguments
  // are processed.
  WriteCommandsForSourcing(eCommandPlacementBeforeFile, commands_stream);

  // If we're not in --repl mode, add the commands to process the file
  // arguments, and the commands specified to run afterwards.
  if (!m_option_data.m_repl) {
    const size_t num_args = m_option_data.m_args.size();
    if (num_args > 0) {
      char arch_name[64];
      if (lldb::SBDebugger::GetDefaultArchitecture(arch_name,
                                                   sizeof(arch_name)))
        commands_stream.Printf("target create --arch=%s %s", arch_name,
                               EscapeString(m_option_data.m_args[0]).c_str());
      else
        commands_stream.Printf("target create %s",
                               EscapeString(m_option_data.m_args[0]).c_str());

      if (!m_option_data.m_core_file.empty()) {
        commands_stream.Printf(" --core %s",
                               EscapeString(m_option_data.m_core_file).c_str());
      }
      commands_stream.Printf("\n");

      if (num_args > 1) {
        commands_stream.Printf("settings set -- target.run-args ");
        for (size_t arg_idx = 1; arg_idx < num_args; ++arg_idx)
          commands_stream.Printf(
              " %s", EscapeString(m_option_data.m_args[arg_idx]).c_str());
        commands_stream.Printf("\n");
      }
    } else if (!m_option_data.m_core_file.empty()) {
      commands_stream.Printf("target create --core %s\n",
                             EscapeString(m_option_data.m_core_file).c_str());
    } else if (!m_option_data.m_process_name.empty()) {
      commands_stream.Printf(
          "process attach --name %s",
          EscapeString(m_option_data.m_process_name).c_str());

      if (m_option_data.m_wait_for)
        commands_stream.Printf(" --waitfor");

      commands_stream.Printf("\n");

    } else if (LLDB_INVALID_PROCESS_ID != m_option_data.m_process_pid) {
      commands_stream.Printf("process attach --pid %" PRIu64 "\n",
                             m_option_data.m_process_pid);
    }

    WriteCommandsForSourcing(eCommandPlacementAfterFile, commands_stream);
  } else if (!m_option_data.m_after_file_commands.empty()) {
    // We're in repl mode and after-file-load commands were specified.
    WithColor::warning() << "commands specified to run after file load (via -o "
                            "or -s) are ignored in REPL mode.\n";
  }

  if (m_option_data.m_debug_mode) {
    result.PutError(m_debugger.GetErrorFile());
    result.PutOutput(m_debugger.GetOutputFile());
  }

  const bool handle_events = true;
  const bool spawn_thread = false;

  // Check if we have any data in the commands stream, and if so, save it to a
  // temp file
  // so we can then run the command interpreter using the file contents.
  const char *commands_data = commands_stream.GetData();
  const size_t commands_size = commands_stream.GetSize();

  bool go_interactive = true;
  if ((commands_data != nullptr) && (commands_size != 0u)) {
    FILE *commands_file =
        PrepareCommandsForSourcing(commands_data, commands_size);

    if (commands_file == nullptr) {
      // We should have already printed an error in PrepareCommandsForSourcing.
      return 1;
    }

    m_debugger.SetInputFileHandle(commands_file, true);

    // Set the debugger into Sync mode when running the command file. Otherwise
    // command files that run the target won't run in a sensible way.
    bool old_async = m_debugger.GetAsync();
    m_debugger.SetAsync(false);

    SBCommandInterpreterRunOptions options;
    options.SetAutoHandleEvents(true);
    options.SetSpawnThread(false);
    options.SetStopOnError(true);
    options.SetStopOnCrash(m_option_data.m_batch);

    SBCommandInterpreterRunResult results =
        m_debugger.RunCommandInterpreter(options);
    if (results.GetResult() == lldb::eCommandInterpreterResultQuitRequested)
      go_interactive = false;
    if (m_option_data.m_batch &&
        results.GetResult() != lldb::eCommandInterpreterResultInferiorCrash)
      go_interactive = false;

    // When running in batch mode and stopped because of an error, exit with a
    // non-zero exit status.
    if (m_option_data.m_batch &&
        results.GetResult() == lldb::eCommandInterpreterResultCommandError)
      return 1;

    if (m_option_data.m_batch &&
        results.GetResult() == lldb::eCommandInterpreterResultInferiorCrash &&
        !m_option_data.m_after_crash_commands.empty()) {
      SBStream crash_commands_stream;
      WriteCommandsForSourcing(eCommandPlacementAfterCrash,
                               crash_commands_stream);
      const char *crash_commands_data = crash_commands_stream.GetData();
      const size_t crash_commands_size = crash_commands_stream.GetSize();
      commands_file =
          PrepareCommandsForSourcing(crash_commands_data, crash_commands_size);
      if (commands_file != nullptr) {
        m_debugger.SetInputFileHandle(commands_file, true);
        SBCommandInterpreterRunResult local_results =
            m_debugger.RunCommandInterpreter(options);
        if (local_results.GetResult() ==
            lldb::eCommandInterpreterResultQuitRequested)
          go_interactive = false;

        // When running in batch mode and an error occurred while sourcing
        // the crash commands, exit with a non-zero exit status.
        if (m_option_data.m_batch &&
            local_results.GetResult() ==
                lldb::eCommandInterpreterResultCommandError)
          return 1;
      }
    }
    m_debugger.SetAsync(old_async);
  }

  // Now set the input file handle to STDIN and run the command interpreter
  // again in interactive mode or repl mode and let the debugger take ownership
  // of stdin.
  if (go_interactive) {
    m_debugger.SetInputFileHandle(stdin, true);

    if (m_option_data.m_repl) {
      const char *repl_options = nullptr;
      if (!m_option_data.m_repl_options.empty())
        repl_options = m_option_data.m_repl_options.c_str();
      SBError error(
          m_debugger.RunREPL(m_option_data.m_repl_lang, repl_options));
      if (error.Fail()) {
        const char *error_cstr = error.GetCString();
        if ((error_cstr != nullptr) && (error_cstr[0] != 0))
          WithColor::error() << error_cstr << '\n';
        else
          WithColor::error() << error.GetError() << '\n';
      }
    } else {
      m_debugger.RunCommandInterpreter(handle_events, spawn_thread);
    }
  }

  reset_stdin_termios();
  fclose(stdin);

  return sb_interpreter.GetQuitStatus();
}

void Driver::ResizeWindow(unsigned short col) {
  GetDebugger().SetTerminalWidth(col);
}

void sigwinch_handler(int signo) {
  struct winsize window_size;
  if ((isatty(STDIN_FILENO) != 0) &&
      ::ioctl(STDIN_FILENO, TIOCGWINSZ, &window_size) == 0) {
    if ((window_size.ws_col > 0) && g_driver != nullptr) {
      g_driver->ResizeWindow(window_size.ws_col);
    }
  }
}

void sigint_handler(int signo) {
#ifdef _WIN32 // Restore handler as it is not persistent on Windows
  signal(SIGINT, sigint_handler);
#endif
  static std::atomic_flag g_interrupt_sent = ATOMIC_FLAG_INIT;
  if (g_driver != nullptr) {
    if (!g_interrupt_sent.test_and_set()) {
      g_driver->GetDebugger().DispatchInputInterrupt();
      g_interrupt_sent.clear();
      return;
    }
  }

  _exit(signo);
}

void sigtstp_handler(int signo) {
  if (g_driver != nullptr)
    g_driver->GetDebugger().SaveInputTerminalState();

  signal(signo, SIG_DFL);
  kill(getpid(), signo);
  signal(signo, sigtstp_handler);
}

void sigcont_handler(int signo) {
  if (g_driver != nullptr)
    g_driver->GetDebugger().RestoreInputTerminalState();

  signal(signo, SIG_DFL);
  kill(getpid(), signo);
  signal(signo, sigcont_handler);
}

void reproducer_handler(void *finalize_cmd) {
  if (SBReproducer::Generate()) {
    std::system(static_cast<const char *>(finalize_cmd));
    fflush(stdout);
  }
}

static void printHelp(LLDBOptTable &table, llvm::StringRef tool_name) {
  std::string usage_str = tool_name.str() + " [options]";
  table.PrintHelp(llvm::outs(), usage_str.c_str(), "LLDB", false);

  std::string examples = R"___(
EXAMPLES:
  The debugger can be started in several modes.

  Passing an executable as a positional argument prepares lldb to debug the
  given executable. To disambiguate between arguments passed to lldb and
  arguments passed to the debugged executable, arguments starting with a - must
  be passed after --.

    lldb --arch x86_64 /path/to/program program argument -- --arch arvm7

  For convenience, passing the executable after -- is also supported.

    lldb --arch x86_64 -- /path/to/program program argument --arch arvm7

  Passing one of the attach options causes lldb to immediately attach to the
  given process.

    lldb -p <pid>
    lldb -n <process-name>

  Passing --repl starts lldb in REPL mode.

    lldb -r

  Passing --core causes lldb to debug the core file.

    lldb -c /path/to/core

  Command options can be combined with these modes and cause lldb to run the
  specified commands before or after events, like loading the file or crashing,
  in the order provided on the command line.

    lldb -O 'settings set stop-disassembly-count 20' -o 'run' -o 'bt'
    lldb -S /source/before/file -s /source/after/file
    lldb -K /source/before/crash -k /source/after/crash

  Note: In REPL mode no file is loaded, so commands specified to run after
  loading the file (via -o or -s) will be ignored.)___";
  llvm::outs() << examples << '\n';
}

llvm::Optional<int> InitializeReproducer(llvm::StringRef argv0,
                                         opt::InputArgList &input_args) {
  if (auto *finalize_path = input_args.getLastArg(OPT_reproducer_finalize)) {
    if (const char *error = SBReproducer::Finalize(finalize_path->getValue())) {
      WithColor::error() << "reproducer finalization failed: " << error << '\n';
      return 1;
    }

    llvm::outs() << "********************\n";
    llvm::outs() << "Crash reproducer for ";
    llvm::outs() << lldb::SBDebugger::GetVersionString() << '\n';
    llvm::outs() << '\n';
    llvm::outs() << "Reproducer written to '" << SBReproducer::GetPath()
                 << "'\n";
    llvm::outs() << '\n';
    llvm::outs() << "Before attaching the reproducer to a bug report:\n";
    llvm::outs() << " - Look at the directory to ensure you're willing to "
                    "share its content.\n";
    llvm::outs()
        << " - Make sure the reproducer works by replaying the reproducer.\n";
    llvm::outs() << '\n';
    llvm::outs() << "Replay the reproducer with the following command:\n";
    llvm::outs() << argv0 << " -replay " << finalize_path->getValue() << "\n";
    llvm::outs() << "********************\n";
    return 0;
  }

  if (auto *replay_path = input_args.getLastArg(OPT_replay)) {
    SBReplayOptions replay_options;
    replay_options.SetCheckVersion(!input_args.hasArg(OPT_no_version_check));
    replay_options.SetVerify(!input_args.hasArg(OPT_no_verification));
    if (const char *error =
            SBReproducer::Replay(replay_path->getValue(), replay_options)) {
      WithColor::error() << "reproducer replay failed: " << error << '\n';
      return 1;
    }
    return 0;
  }

  bool capture = input_args.hasArg(OPT_capture);
  bool generate_on_exit = input_args.hasArg(OPT_generate_on_exit);
  auto *capture_path = input_args.getLastArg(OPT_capture_path);

  if (generate_on_exit && !capture) {
    WithColor::warning()
        << "-reproducer-generate-on-exit specified without -capture\n";
  }

  if (capture || capture_path) {
    if (capture_path) {
      if (!capture)
        WithColor::warning() << "-capture-path specified without -capture\n";
      if (const char *error = SBReproducer::Capture(capture_path->getValue())) {
        WithColor::error() << "reproducer capture failed: " << error << '\n';
        return 1;
      }
    } else {
      const char *error = SBReproducer::Capture();
      if (error) {
        WithColor::error() << "reproducer capture failed: " << error << '\n';
        return 1;
      }
    }
    if (generate_on_exit)
      SBReproducer::SetAutoGenerate(true);

    // Register the reproducer signal handler.
    if (!input_args.hasArg(OPT_no_generate_on_signal)) {
      if (const char *reproducer_path = SBReproducer::GetPath()) {
        // Leaking the string on purpose.
        std::string *finalize_cmd = new std::string(argv0);
        finalize_cmd->append(" --reproducer-finalize '");
        finalize_cmd->append(reproducer_path);
        finalize_cmd->append("'");
        llvm::sys::AddSignalHandler(reproducer_handler,
                                    const_cast<char *>(finalize_cmd->c_str()));
      }
    }
  }

  return llvm::None;
}

int main(int argc, char const *argv[]) {
  // Setup LLVM signal handlers and make sure we call llvm_shutdown() on
  // destruction.
  llvm::InitLLVM IL(argc, argv, /*InstallPipeSignalExitHandler=*/false);

  // Parse arguments.
  LLDBOptTable T;
  unsigned MissingArgIndex;
  unsigned MissingArgCount;
  ArrayRef<const char *> arg_arr = makeArrayRef(argv + 1, argc - 1);
  opt::InputArgList input_args =
      T.ParseArgs(arg_arr, MissingArgIndex, MissingArgCount);
  llvm::StringRef argv0 = llvm::sys::path::filename(argv[0]);

  if (input_args.hasArg(OPT_help)) {
    printHelp(T, argv0);
    return 0;
  }

  // Check for missing argument error.
  if (MissingArgCount) {
    WithColor::error() << "argument to '"
                       << input_args.getArgString(MissingArgIndex)
                       << "' is missing\n";
  }
  // Error out on unknown options.
  if (input_args.hasArg(OPT_UNKNOWN)) {
    for (auto *arg : input_args.filtered(OPT_UNKNOWN)) {
      WithColor::error() << "unknown option: " << arg->getSpelling() << '\n';
    }
  }
  if (MissingArgCount || input_args.hasArg(OPT_UNKNOWN)) {
    llvm::errs() << "Use '" << argv0
                 << " --help' for a complete list of options.\n";
    return 1;
  }

  if (auto exit_code = InitializeReproducer(argv[0], input_args)) {
    return *exit_code;
  }

  SBError error = SBDebugger::InitializeWithErrorHandling();
  if (error.Fail()) {
    WithColor::error() << "initialization failed: " << error.GetCString()
                       << '\n';
    return 1;
  }
  SBHostOS::ThreadCreated("<lldb.driver.main-thread>");

  signal(SIGINT, sigint_handler);
#if !defined(_MSC_VER)
  signal(SIGPIPE, SIG_IGN);
  signal(SIGWINCH, sigwinch_handler);
  signal(SIGTSTP, sigtstp_handler);
  signal(SIGCONT, sigcont_handler);
#endif

  int exit_code = 0;
  // Create a scope for driver so that the driver object will destroy itself
  // before SBDebugger::Terminate() is called.
  {
    Driver driver;

    bool exiting = false;
    SBError error(driver.ProcessArgs(input_args, exiting));
    if (error.Fail()) {
      exit_code = 1;
      if (const char *error_cstr = error.GetCString())
        WithColor::error() << error_cstr << '\n';
    } else if (!exiting) {
      exit_code = driver.MainLoop();
    }
  }

  SBDebugger::Terminate();
  return exit_code;
}
