//===-- Driver.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Driver_h_
#define lldb_Driver_h_

#include "Platform.h"
#include "lldb/Host/PseudoTerminal.h"

#include <bitset>
#include <set>
#include <string>
#include <vector>

#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBDefines.h"
#include "lldb/API/SBError.h"

class IOChannel;

class Driver : public lldb::SBBroadcaster {
public:
  typedef enum CommandPlacement {
    eCommandPlacementBeforeFile,
    eCommandPlacementAfterFile,
    eCommandPlacementAfterCrash,
  } CommandPlacement;

  Driver();

  virtual ~Driver();

  /// Runs the main loop.
  ///
  /// @return The exit code that the process should return.
  int MainLoop();

  lldb::SBError ParseArgs(int argc, const char *argv[], FILE *out_fh,
                          bool &do_exit);

  const char *GetFilename() const;

  const char *GetCrashLogFilename() const;

  const char *GetArchName() const;

  lldb::ScriptLanguage GetScriptLanguage() const;

  void WriteCommandsForSourcing(CommandPlacement placement,
                                lldb::SBStream &strm);

  bool GetDebugMode() const;

  class OptionData {
  public:
    OptionData();
    ~OptionData();

    void Clear();

    void AddInitialCommand(const char *command, CommandPlacement placement,
                           bool is_file, lldb::SBError &error);

    // static OptionDefinition m_cmd_option_table[];

    struct InitialCmdEntry {
      InitialCmdEntry(const char *in_contents, bool in_is_file,
                      bool is_cwd_lldbinit_file_read, bool in_quiet = false)
          : contents(in_contents), is_file(in_is_file),
            is_cwd_lldbinit_file_read(is_cwd_lldbinit_file_read),
            source_quietly(in_quiet) {}

      std::string contents;
      bool is_file;
      bool is_cwd_lldbinit_file_read; // if this is reading ./.lldbinit - so we
                                      // may skip if not permitted
      bool source_quietly;
    };

    std::vector<std::string> m_args;
    lldb::ScriptLanguage m_script_lang;
    std::string m_core_file;
    std::string m_crash_log;
    std::vector<InitialCmdEntry> m_initial_commands;
    std::vector<InitialCmdEntry> m_after_file_commands;
    std::vector<InitialCmdEntry> m_after_crash_commands;
    bool m_debug_mode;
    bool m_source_quietly;
    bool m_print_version;
    bool m_print_python_path;
    bool m_print_help;
    bool m_wait_for;
    bool m_repl;
    lldb::LanguageType m_repl_lang;
    std::string m_repl_options;
    std::string m_process_name;
    lldb::pid_t m_process_pid;
    bool m_use_external_editor; // FIXME: When we have set/show variables we can
                                // remove this from here.
    bool m_batch;
    typedef std::set<char> OptionSet;
    OptionSet m_seen_options;
  };

  static lldb::SBError SetOptionValue(int option_idx, const char *option_arg,
                                      Driver::OptionData &data);

  lldb::SBDebugger &GetDebugger() { return m_debugger; }

  void ResizeWindow(unsigned short col);

private:
  lldb::SBDebugger m_debugger;
  OptionData m_option_data;

  void ResetOptionValues();

  void ReadyForCommand();
};

#endif // lldb_Driver_h_
