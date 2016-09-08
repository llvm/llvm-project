//===-- CommandInterpreter.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>
#include <string>
#include <vector>

#include "CommandObjectScript.h"
#include "lldb/Interpreter/CommandObjectRegexCommand.h"

#include "../Commands/CommandObjectApropos.h"
#include "../Commands/CommandObjectArgs.h"
#include "../Commands/CommandObjectBreakpoint.h"
#include "../Commands/CommandObjectBugreport.h"
#include "../Commands/CommandObjectCommands.h"
#include "../Commands/CommandObjectDisassemble.h"
#include "../Commands/CommandObjectExpression.h"
#include "../Commands/CommandObjectFrame.h"
#include "../Commands/CommandObjectGUI.h"
#include "../Commands/CommandObjectHelp.h"
#include "../Commands/CommandObjectLanguage.h"
#include "../Commands/CommandObjectLog.h"
#include "../Commands/CommandObjectMemory.h"
#include "../Commands/CommandObjectPlatform.h"
#include "../Commands/CommandObjectPlugin.h"
#include "../Commands/CommandObjectProcess.h"
#include "../Commands/CommandObjectQuit.h"
#include "../Commands/CommandObjectRegister.h"
#include "../Commands/CommandObjectSettings.h"
#include "../Commands/CommandObjectSource.h"
#include "../Commands/CommandObjectSyntax.h"
#include "../Commands/CommandObjectTarget.h"
#include "../Commands/CommandObjectThread.h"
#include "../Commands/CommandObjectType.h"
#include "../Commands/CommandObjectVersion.h"
#include "../Commands/CommandObjectWatchpoint.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/State.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/Timer.h"

#ifndef LLDB_DISABLE_LIBEDIT
#include "lldb/Host/Editline.h"
#endif
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"

#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandCompletions.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/Property.h"

#include "lldb/Target/Process.h"
#include "lldb/Target/TargetList.h"
#include "lldb/Target/Thread.h"

#include "lldb/Utility/CleanUp.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"

using namespace lldb;
using namespace lldb_private;

static const char *k_white_space = " \t\v";

static PropertyDefinition g_properties[] = {
    {"expand-regex-aliases", OptionValue::eTypeBoolean, true, false, nullptr,
     nullptr, "If true, regular expression alias commands will show the "
              "expanded command that will be executed. This can be used to "
              "debug new regular expression alias commands."},
    {"prompt-on-quit", OptionValue::eTypeBoolean, true, true, nullptr, nullptr,
     "If true, LLDB will prompt you before quitting if there are any live "
     "processes being debugged. If false, LLDB will quit without asking in any "
     "case."},
    {"stop-command-source-on-error", OptionValue::eTypeBoolean, true, true,
     nullptr, nullptr, "If true, LLDB will stop running a 'command source' "
                       "script upon encountering an error."},
    {"space-repl-prompts", OptionValue::eTypeBoolean, true, false, nullptr,
     nullptr,
     "If true, blank lines will be printed between between REPL submissions."},
    {nullptr, OptionValue::eTypeInvalid, true, 0, nullptr, nullptr, nullptr}};

enum {
  ePropertyExpandRegexAliases = 0,
  ePropertyPromptOnQuit = 1,
  ePropertyStopCmdSourceOnError = 2,
  eSpaceReplPrompts = 3
};

ConstString &CommandInterpreter::GetStaticBroadcasterClass() {
  static ConstString class_name("lldb.commandInterpreter");
  return class_name;
}

CommandInterpreter::CommandInterpreter(Debugger &debugger,
                                       ScriptLanguage script_language,
                                       bool synchronous_execution)
    : Broadcaster(debugger.GetBroadcasterManager(),
                  CommandInterpreter::GetStaticBroadcasterClass().AsCString()),
      Properties(OptionValuePropertiesSP(
          new OptionValueProperties(ConstString("interpreter")))),
      IOHandlerDelegate(IOHandlerDelegate::Completion::LLDBCommand),
      m_debugger(debugger), m_synchronous_execution(synchronous_execution),
      m_skip_lldbinit_files(false), m_skip_app_init_files(false),
      m_script_interpreter_sp(), m_command_io_handler_sp(), m_comment_char('#'),
      m_batch_command_mode(false), m_truncation_warning(eNoTruncation),
      m_command_source_depth(0), m_num_errors(0), m_quit_requested(false),
      m_stopped_for_crash(false) {
  debugger.SetScriptLanguage(script_language);
  SetEventName(eBroadcastBitThreadShouldExit, "thread-should-exit");
  SetEventName(eBroadcastBitResetPrompt, "reset-prompt");
  SetEventName(eBroadcastBitQuitCommandReceived, "quit");
  CheckInWithManager();
  m_collection_sp->Initialize(g_properties);
}

bool CommandInterpreter::GetExpandRegexAliases() const {
  const uint32_t idx = ePropertyExpandRegexAliases;
  return m_collection_sp->GetPropertyAtIndexAsBoolean(
      nullptr, idx, g_properties[idx].default_uint_value != 0);
}

bool CommandInterpreter::GetPromptOnQuit() const {
  const uint32_t idx = ePropertyPromptOnQuit;
  return m_collection_sp->GetPropertyAtIndexAsBoolean(
      nullptr, idx, g_properties[idx].default_uint_value != 0);
}

void CommandInterpreter::SetPromptOnQuit(bool enable) {
  const uint32_t idx = ePropertyPromptOnQuit;
  m_collection_sp->SetPropertyAtIndexAsBoolean(nullptr, idx, enable);
}

void CommandInterpreter::ResolveCommand(const char *command_line,
                                        CommandReturnObject &result) {
  std::string command = command_line;
  if (ResolveCommandImpl(command, result) != nullptr) {
    result.AppendMessageWithFormat("%s", command.c_str());
    result.SetStatus(eReturnStatusSuccessFinishResult);
  }
}

bool CommandInterpreter::GetStopCmdSourceOnError() const {
  const uint32_t idx = ePropertyStopCmdSourceOnError;
  return m_collection_sp->GetPropertyAtIndexAsBoolean(
      nullptr, idx, g_properties[idx].default_uint_value != 0);
}

bool CommandInterpreter::GetSpaceReplPrompts() const {
  const uint32_t idx = eSpaceReplPrompts;
  return m_collection_sp->GetPropertyAtIndexAsBoolean(
      nullptr, idx, g_properties[idx].default_uint_value != 0);
}

void CommandInterpreter::Initialize() {
  Timer scoped_timer(__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);

  CommandReturnObject result;

  LoadCommandDictionary();

  // An alias arguments vector to reuse - reset it before use...
  OptionArgVectorSP alias_arguments_vector_sp(new OptionArgVector);

  // Set up some initial aliases.
  CommandObjectSP cmd_obj_sp = GetCommandSPExact("quit", false);
  if (cmd_obj_sp) {
    AddAlias("q", cmd_obj_sp);
    AddAlias("exit", cmd_obj_sp);
  }

  cmd_obj_sp = GetCommandSPExact("_regexp-attach", false);
  if (cmd_obj_sp)
    AddAlias("attach", cmd_obj_sp)->SetSyntax(cmd_obj_sp->GetSyntax());

  cmd_obj_sp = GetCommandSPExact("process detach", false);
  if (cmd_obj_sp) {
    AddAlias("detach", cmd_obj_sp);
  }

  cmd_obj_sp = GetCommandSPExact("process continue", false);
  if (cmd_obj_sp) {
    AddAlias("c", cmd_obj_sp);
    AddAlias("continue", cmd_obj_sp);
  }

  cmd_obj_sp = GetCommandSPExact("_regexp-break", false);
  if (cmd_obj_sp)
    AddAlias("b", cmd_obj_sp)->SetSyntax(cmd_obj_sp->GetSyntax());

  cmd_obj_sp = GetCommandSPExact("_regexp-tbreak", false);
  if (cmd_obj_sp)
    AddAlias("tbreak", cmd_obj_sp)->SetSyntax(cmd_obj_sp->GetSyntax());

  cmd_obj_sp = GetCommandSPExact("thread step-inst", false);
  if (cmd_obj_sp) {
    AddAlias("stepi", cmd_obj_sp);
    AddAlias("si", cmd_obj_sp);
  }

  cmd_obj_sp = GetCommandSPExact("thread step-inst-over", false);
  if (cmd_obj_sp) {
    AddAlias("nexti", cmd_obj_sp);
    AddAlias("ni", cmd_obj_sp);
  }

  cmd_obj_sp = GetCommandSPExact("thread step-in", false);
  if (cmd_obj_sp) {
    AddAlias("s", cmd_obj_sp);
    AddAlias("step", cmd_obj_sp);
    CommandAlias *sif_alias = AddAlias(
        "sif", cmd_obj_sp, "--end-linenumber block --step-in-target %1");
    if (sif_alias) {
      sif_alias->SetHelp("Step through the current block, stopping if you step "
                         "directly into a function whose name matches the "
                         "TargetFunctionName.");
      sif_alias->SetSyntax("sif <TargetFunctionName>");
    }
  }

  cmd_obj_sp = GetCommandSPExact("thread step-over", false);
  if (cmd_obj_sp) {
    AddAlias("n", cmd_obj_sp);
    AddAlias("next", cmd_obj_sp);
  }

  cmd_obj_sp = GetCommandSPExact("thread step-out", false);
  if (cmd_obj_sp) {
    AddAlias("finish", cmd_obj_sp);
  }

  cmd_obj_sp = GetCommandSPExact("frame select", false);
  if (cmd_obj_sp) {
    AddAlias("f", cmd_obj_sp);
  }

  cmd_obj_sp = GetCommandSPExact("thread select", false);
  if (cmd_obj_sp) {
    AddAlias("t", cmd_obj_sp);
  }

  cmd_obj_sp = GetCommandSPExact("_regexp-jump", false);
  if (cmd_obj_sp) {
    AddAlias("j", cmd_obj_sp)->SetSyntax(cmd_obj_sp->GetSyntax());
    AddAlias("jump", cmd_obj_sp)->SetSyntax(cmd_obj_sp->GetSyntax());
  }

  cmd_obj_sp = GetCommandSPExact("_regexp-list", false);
  if (cmd_obj_sp) {
    AddAlias("l", cmd_obj_sp)->SetSyntax(cmd_obj_sp->GetSyntax());
    AddAlias("list", cmd_obj_sp)->SetSyntax(cmd_obj_sp->GetSyntax());
  }

  cmd_obj_sp = GetCommandSPExact("_regexp-env", false);
  if (cmd_obj_sp)
    AddAlias("env", cmd_obj_sp)->SetSyntax(cmd_obj_sp->GetSyntax());

  cmd_obj_sp = GetCommandSPExact("memory read", false);
  if (cmd_obj_sp)
    AddAlias("x", cmd_obj_sp);

  cmd_obj_sp = GetCommandSPExact("_regexp-up", false);
  if (cmd_obj_sp)
    AddAlias("up", cmd_obj_sp)->SetSyntax(cmd_obj_sp->GetSyntax());

  cmd_obj_sp = GetCommandSPExact("_regexp-down", false);
  if (cmd_obj_sp)
    AddAlias("down", cmd_obj_sp)->SetSyntax(cmd_obj_sp->GetSyntax());

  cmd_obj_sp = GetCommandSPExact("_regexp-display", false);
  if (cmd_obj_sp)
    AddAlias("display", cmd_obj_sp)->SetSyntax(cmd_obj_sp->GetSyntax());

  cmd_obj_sp = GetCommandSPExact("disassemble", false);
  if (cmd_obj_sp)
    AddAlias("dis", cmd_obj_sp);

  cmd_obj_sp = GetCommandSPExact("disassemble", false);
  if (cmd_obj_sp)
    AddAlias("di", cmd_obj_sp);

  cmd_obj_sp = GetCommandSPExact("_regexp-undisplay", false);
  if (cmd_obj_sp)
    AddAlias("undisplay", cmd_obj_sp)->SetSyntax(cmd_obj_sp->GetSyntax());

  cmd_obj_sp = GetCommandSPExact("_regexp-bt", false);
  if (cmd_obj_sp)
    AddAlias("bt", cmd_obj_sp)->SetSyntax(cmd_obj_sp->GetSyntax());

  cmd_obj_sp = GetCommandSPExact("target create", false);
  if (cmd_obj_sp)
    AddAlias("file", cmd_obj_sp);

  cmd_obj_sp = GetCommandSPExact("target modules", false);
  if (cmd_obj_sp)
    AddAlias("image", cmd_obj_sp);

  alias_arguments_vector_sp.reset(new OptionArgVector);

  cmd_obj_sp = GetCommandSPExact("expression", false);
  if (cmd_obj_sp) {
    AddAlias("p", cmd_obj_sp, "--")->SetHelpLong("");
    AddAlias("print", cmd_obj_sp, "--")->SetHelpLong("");
    AddAlias("call", cmd_obj_sp, "--")->SetHelpLong("");

    if (auto po = AddAlias("po", cmd_obj_sp, "-O --")) {
      po->SetHelp("Evaluate an expression on the current thread.  Displays any "
                  "returned value with formatting "
                  "controlled by the type's author.");
      po->SetHelpLong("");
    }

    AddAlias("repl", cmd_obj_sp, "--repl -- ");

    AddAlias("parray", cmd_obj_sp, "--element-count %1 --")->SetHelpLong("");
    AddAlias("poarray", cmd_obj_sp,
             "--object-description --element-count %1 --")
        ->SetHelpLong("");
  }

  cmd_obj_sp = GetCommandSPExact("process kill", false);
  if (cmd_obj_sp) {
    AddAlias("kill", cmd_obj_sp);
  }

  cmd_obj_sp = GetCommandSPExact("process launch", false);
  if (cmd_obj_sp) {
    alias_arguments_vector_sp.reset(new OptionArgVector);
#if defined(__arm__) || defined(__arm64__) || defined(__aarch64__)
    AddAlias("r", cmd_obj_sp, "--");
    AddAlias("run", cmd_obj_sp, "--");
#else
#if defined(__APPLE__)
    std::string shell_option;
    shell_option.append("--shell-expand-args");
    shell_option.append(" true");
    shell_option.append(" --");
    AddAlias("r", cmd_obj_sp, "--shell-expand-args true --");
    AddAlias("run", cmd_obj_sp, "--shell-expand-args true --");
#else
    StreamString defaultshell;
    defaultshell.Printf("--shell=%s --",
                        HostInfo::GetDefaultShell().GetPath().c_str());
    AddAlias("r", cmd_obj_sp, defaultshell.GetData());
    AddAlias("run", cmd_obj_sp, defaultshell.GetData());
#endif
#endif
  }

  cmd_obj_sp = GetCommandSPExact("target symbols add", false);
  if (cmd_obj_sp) {
    AddAlias("add-dsym", cmd_obj_sp);
  }

  cmd_obj_sp = GetCommandSPExact("breakpoint set", false);
  if (cmd_obj_sp) {
    AddAlias("rbreak", cmd_obj_sp, "--func-regex %1");
  }
}

void CommandInterpreter::Clear() {
  m_command_io_handler_sp.reset();

  if (m_script_interpreter_sp)
    m_script_interpreter_sp->Clear();
}

const char *CommandInterpreter::ProcessEmbeddedScriptCommands(const char *arg) {
  // This function has not yet been implemented.

  // Look for any embedded script command
  // If found,
  //    get interpreter object from the command dictionary,
  //    call execute_one_command on it,
  //    get the results as a string,
  //    substitute that string for current stuff.

  return arg;
}

void CommandInterpreter::LoadCommandDictionary() {
  Timer scoped_timer(__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);

  lldb::ScriptLanguage script_language = m_debugger.GetScriptLanguage();

  m_command_dict["apropos"] = CommandObjectSP(new CommandObjectApropos(*this));
  m_command_dict["breakpoint"] =
      CommandObjectSP(new CommandObjectMultiwordBreakpoint(*this));
  m_command_dict["bugreport"] =
      CommandObjectSP(new CommandObjectMultiwordBugreport(*this));
  m_command_dict["command"] =
      CommandObjectSP(new CommandObjectMultiwordCommands(*this));
  m_command_dict["disassemble"] =
      CommandObjectSP(new CommandObjectDisassemble(*this));
  m_command_dict["expression"] =
      CommandObjectSP(new CommandObjectExpression(*this));
  m_command_dict["frame"] =
      CommandObjectSP(new CommandObjectMultiwordFrame(*this));
  m_command_dict["gui"] = CommandObjectSP(new CommandObjectGUI(*this));
  m_command_dict["help"] = CommandObjectSP(new CommandObjectHelp(*this));
  m_command_dict["log"] = CommandObjectSP(new CommandObjectLog(*this));
  m_command_dict["memory"] = CommandObjectSP(new CommandObjectMemory(*this));
  m_command_dict["platform"] =
      CommandObjectSP(new CommandObjectPlatform(*this));
  m_command_dict["plugin"] = CommandObjectSP(new CommandObjectPlugin(*this));
  m_command_dict["process"] =
      CommandObjectSP(new CommandObjectMultiwordProcess(*this));
  m_command_dict["quit"] = CommandObjectSP(new CommandObjectQuit(*this));
  m_command_dict["register"] =
      CommandObjectSP(new CommandObjectRegister(*this));
  m_command_dict["script"] =
      CommandObjectSP(new CommandObjectScript(*this, script_language));
  m_command_dict["settings"] =
      CommandObjectSP(new CommandObjectMultiwordSettings(*this));
  m_command_dict["source"] =
      CommandObjectSP(new CommandObjectMultiwordSource(*this));
  m_command_dict["target"] =
      CommandObjectSP(new CommandObjectMultiwordTarget(*this));
  m_command_dict["thread"] =
      CommandObjectSP(new CommandObjectMultiwordThread(*this));
  m_command_dict["type"] = CommandObjectSP(new CommandObjectType(*this));
  m_command_dict["version"] = CommandObjectSP(new CommandObjectVersion(*this));
  m_command_dict["watchpoint"] =
      CommandObjectSP(new CommandObjectMultiwordWatchpoint(*this));
  m_command_dict["language"] =
      CommandObjectSP(new CommandObjectLanguage(*this));

  const char *break_regexes[][2] = {
      {"^(.*[^[:space:]])[[:space:]]*:[[:space:]]*([[:digit:]]+)[[:space:]]*$",
       "breakpoint set --file '%1' --line %2"},
      {"^/([^/]+)/$", "breakpoint set --source-pattern-regexp '%1'"},
      {"^([[:digit:]]+)[[:space:]]*$", "breakpoint set --line %1"},
      {"^\\*?(0x[[:xdigit:]]+)[[:space:]]*$", "breakpoint set --address %1"},
      {"^[\"']?([-+]?\\[.*\\])[\"']?[[:space:]]*$",
       "breakpoint set --name '%1'"},
      {"^(-.*)$", "breakpoint set %1"},
      {"^(.*[^[:space:]])`(.*[^[:space:]])[[:space:]]*$",
       "breakpoint set --name '%2' --shlib '%1'"},
      {"^\\&(.*[^[:space:]])[[:space:]]*$",
       "breakpoint set --name '%1' --skip-prologue=0"},
      {"^[\"']?(.*[^[:space:]\"'])[\"']?[[:space:]]*$",
       "breakpoint set --name '%1'"}};

  size_t num_regexes = llvm::array_lengthof(break_regexes);

  std::unique_ptr<CommandObjectRegexCommand> break_regex_cmd_ap(
      new CommandObjectRegexCommand(
          *this, "_regexp-break",
          "Set a breakpoint using one of several shorthand formats.\n",
          "\n"
          "_regexp-break <filename>:<linenum>\n"
          "              main.c:12             // Break at line 12 of "
          "main.c\n\n"
          "_regexp-break <linenum>\n"
          "              12                    // Break at line 12 of current "
          "file\n\n"
          "_regexp-break 0x<address>\n"
          "              0x1234000             // Break at address "
          "0x1234000\n\n"
          "_regexp-break <name>\n"
          "              main                  // Break in 'main' after the "
          "prologue\n\n"
          "_regexp-break &<name>\n"
          "              &main                 // Break at first instruction "
          "in 'main'\n\n"
          "_regexp-break <module>`<name>\n"
          "              libc.so`malloc        // Break in 'malloc' from "
          "'libc.so'\n\n"
          "_regexp-break /<source-regex>/\n"
          "              /break here/          // Break on source lines in "
          "current file\n"
          "                                    // containing text 'break "
          "here'.\n",
          2, CommandCompletions::eSymbolCompletion |
                 CommandCompletions::eSourceFileCompletion,
          false));

  if (break_regex_cmd_ap.get()) {
    bool success = true;
    for (size_t i = 0; i < num_regexes; i++) {
      success = break_regex_cmd_ap->AddRegexCommand(break_regexes[i][0],
                                                    break_regexes[i][1]);
      if (!success)
        break;
    }
    success =
        break_regex_cmd_ap->AddRegexCommand("^$", "breakpoint list --full");

    if (success) {
      CommandObjectSP break_regex_cmd_sp(break_regex_cmd_ap.release());
      m_command_dict[break_regex_cmd_sp->GetCommandName()] = break_regex_cmd_sp;
    }
  }

  std::unique_ptr<CommandObjectRegexCommand> tbreak_regex_cmd_ap(
      new CommandObjectRegexCommand(
          *this, "_regexp-tbreak",
          "Set a one-shot breakpoint using one of several shorthand formats.\n",
          "\n"
          "_regexp-break <filename>:<linenum>\n"
          "              main.c:12             // Break at line 12 of "
          "main.c\n\n"
          "_regexp-break <linenum>\n"
          "              12                    // Break at line 12 of current "
          "file\n\n"
          "_regexp-break 0x<address>\n"
          "              0x1234000             // Break at address "
          "0x1234000\n\n"
          "_regexp-break <name>\n"
          "              main                  // Break in 'main' after the "
          "prologue\n\n"
          "_regexp-break &<name>\n"
          "              &main                 // Break at first instruction "
          "in 'main'\n\n"
          "_regexp-break <module>`<name>\n"
          "              libc.so`malloc        // Break in 'malloc' from "
          "'libc.so'\n\n"
          "_regexp-break /<source-regex>/\n"
          "              /break here/          // Break on source lines in "
          "current file\n"
          "                                    // containing text 'break "
          "here'.\n",
          2, CommandCompletions::eSymbolCompletion |
                 CommandCompletions::eSourceFileCompletion,
          false));

  if (tbreak_regex_cmd_ap.get()) {
    bool success = true;
    for (size_t i = 0; i < num_regexes; i++) {
      // If you add a resultant command string longer than 1024 characters be
      // sure to increase the size of this buffer.
      char buffer[1024];
      int num_printed =
          snprintf(buffer, 1024, "%s %s", break_regexes[i][1], "-o");
      assert(num_printed < 1024);
      UNUSED_IF_ASSERT_DISABLED(num_printed);
      success =
          tbreak_regex_cmd_ap->AddRegexCommand(break_regexes[i][0], buffer);
      if (!success)
        break;
    }
    success =
        tbreak_regex_cmd_ap->AddRegexCommand("^$", "breakpoint list --full");

    if (success) {
      CommandObjectSP tbreak_regex_cmd_sp(tbreak_regex_cmd_ap.release());
      m_command_dict[tbreak_regex_cmd_sp->GetCommandName()] =
          tbreak_regex_cmd_sp;
    }
  }

  std::unique_ptr<CommandObjectRegexCommand> attach_regex_cmd_ap(
      new CommandObjectRegexCommand(
          *this, "_regexp-attach", "Attach to process by ID or name.",
          "_regexp-attach <pid> | <process-name>", 2, 0, false));
  if (attach_regex_cmd_ap.get()) {
    if (attach_regex_cmd_ap->AddRegexCommand("^([0-9]+)[[:space:]]*$",
                                             "process attach --pid %1") &&
        attach_regex_cmd_ap->AddRegexCommand(
            "^(-.*|.* -.*)$", "process attach %1") && // Any options that are
                                                      // specified get passed to
                                                      // 'process attach'
        attach_regex_cmd_ap->AddRegexCommand("^(.+)$",
                                             "process attach --name '%1'") &&
        attach_regex_cmd_ap->AddRegexCommand("^$", "process attach")) {
      CommandObjectSP attach_regex_cmd_sp(attach_regex_cmd_ap.release());
      m_command_dict[attach_regex_cmd_sp->GetCommandName()] =
          attach_regex_cmd_sp;
    }
  }

  std::unique_ptr<CommandObjectRegexCommand> down_regex_cmd_ap(
      new CommandObjectRegexCommand(*this, "_regexp-down",
                                    "Select a newer stack frame.  Defaults to "
                                    "moving one frame, a numeric argument can "
                                    "specify an arbitrary number.",
                                    "_regexp-down [<count>]", 2, 0, false));
  if (down_regex_cmd_ap.get()) {
    if (down_regex_cmd_ap->AddRegexCommand("^$", "frame select -r -1") &&
        down_regex_cmd_ap->AddRegexCommand("^([0-9]+)$",
                                           "frame select -r -%1")) {
      CommandObjectSP down_regex_cmd_sp(down_regex_cmd_ap.release());
      m_command_dict[down_regex_cmd_sp->GetCommandName()] = down_regex_cmd_sp;
    }
  }

  std::unique_ptr<CommandObjectRegexCommand> up_regex_cmd_ap(
      new CommandObjectRegexCommand(
          *this, "_regexp-up",
          "Select an older stack frame.  Defaults to moving one "
          "frame, a numeric argument can specify an arbitrary number.",
          "_regexp-up [<count>]", 2, 0, false));
  if (up_regex_cmd_ap.get()) {
    if (up_regex_cmd_ap->AddRegexCommand("^$", "frame select -r 1") &&
        up_regex_cmd_ap->AddRegexCommand("^([0-9]+)$", "frame select -r %1")) {
      CommandObjectSP up_regex_cmd_sp(up_regex_cmd_ap.release());
      m_command_dict[up_regex_cmd_sp->GetCommandName()] = up_regex_cmd_sp;
    }
  }

  std::unique_ptr<CommandObjectRegexCommand> display_regex_cmd_ap(
      new CommandObjectRegexCommand(
          *this, "_regexp-display",
          "Evaluate an expression at every stop (see 'help target stop-hook'.)",
          "_regexp-display expression", 2, 0, false));
  if (display_regex_cmd_ap.get()) {
    if (display_regex_cmd_ap->AddRegexCommand(
            "^(.+)$", "target stop-hook add -o \"expr -- %1\"")) {
      CommandObjectSP display_regex_cmd_sp(display_regex_cmd_ap.release());
      m_command_dict[display_regex_cmd_sp->GetCommandName()] =
          display_regex_cmd_sp;
    }
  }

  std::unique_ptr<CommandObjectRegexCommand> undisplay_regex_cmd_ap(
      new CommandObjectRegexCommand(
          *this, "_regexp-undisplay", "Stop displaying expression at every "
                                      "stop (specified by stop-hook index.)",
          "_regexp-undisplay stop-hook-number", 2, 0, false));
  if (undisplay_regex_cmd_ap.get()) {
    if (undisplay_regex_cmd_ap->AddRegexCommand("^([0-9]+)$",
                                                "target stop-hook delete %1")) {
      CommandObjectSP undisplay_regex_cmd_sp(undisplay_regex_cmd_ap.release());
      m_command_dict[undisplay_regex_cmd_sp->GetCommandName()] =
          undisplay_regex_cmd_sp;
    }
  }

  std::unique_ptr<CommandObjectRegexCommand> connect_gdb_remote_cmd_ap(
      new CommandObjectRegexCommand(
          *this, "gdb-remote", "Connect to a process via remote GDB server.  "
                               "If no host is specifed, localhost is assumed.",
          "gdb-remote [<hostname>:]<portnum>", 2, 0, false));
  if (connect_gdb_remote_cmd_ap.get()) {
    if (connect_gdb_remote_cmd_ap->AddRegexCommand(
            "^([^:]+:[[:digit:]]+)$",
            "process connect --plugin gdb-remote connect://%1") &&
        connect_gdb_remote_cmd_ap->AddRegexCommand(
            "^([[:digit:]]+)$",
            "process connect --plugin gdb-remote connect://localhost:%1")) {
      CommandObjectSP command_sp(connect_gdb_remote_cmd_ap.release());
      m_command_dict[command_sp->GetCommandName()] = command_sp;
    }
  }

  std::unique_ptr<CommandObjectRegexCommand> connect_kdp_remote_cmd_ap(
      new CommandObjectRegexCommand(
          *this, "kdp-remote", "Connect to a process via remote KDP server.  "
                               "If no UDP port is specified, port 41139 is "
                               "assumed.",
          "kdp-remote <hostname>[:<portnum>]", 2, 0, false));
  if (connect_kdp_remote_cmd_ap.get()) {
    if (connect_kdp_remote_cmd_ap->AddRegexCommand(
            "^([^:]+:[[:digit:]]+)$",
            "process connect --plugin kdp-remote udp://%1") &&
        connect_kdp_remote_cmd_ap->AddRegexCommand(
            "^(.+)$", "process connect --plugin kdp-remote udp://%1:41139")) {
      CommandObjectSP command_sp(connect_kdp_remote_cmd_ap.release());
      m_command_dict[command_sp->GetCommandName()] = command_sp;
    }
  }

  std::unique_ptr<CommandObjectRegexCommand> bt_regex_cmd_ap(
      new CommandObjectRegexCommand(
          *this, "_regexp-bt",
          "Show the current thread's call stack.  Any numeric argument "
          "displays at most that many "
          "frames.  The argument 'all' displays all threads.",
          "bt [<digit> | all]", 2, 0, false));
  if (bt_regex_cmd_ap.get()) {
    // accept but don't document "bt -c <number>" -- before bt was a regex
    // command if you wanted to backtrace
    // three frames you would do "bt -c 3" but the intention is to have this
    // emulate the gdb "bt" command and
    // so now "bt 3" is the preferred form, in line with gdb.
    if (bt_regex_cmd_ap->AddRegexCommand("^([[:digit:]]+)$",
                                         "thread backtrace -c %1") &&
        bt_regex_cmd_ap->AddRegexCommand("^-c ([[:digit:]]+)$",
                                         "thread backtrace -c %1") &&
        bt_regex_cmd_ap->AddRegexCommand("^all$", "thread backtrace all") &&
        bt_regex_cmd_ap->AddRegexCommand("^$", "thread backtrace")) {
      CommandObjectSP command_sp(bt_regex_cmd_ap.release());
      m_command_dict[command_sp->GetCommandName()] = command_sp;
    }
  }

  std::unique_ptr<CommandObjectRegexCommand> list_regex_cmd_ap(
      new CommandObjectRegexCommand(
          *this, "_regexp-list",
          "List relevant source code using one of several shorthand formats.",
          "\n"
          "_regexp-list <file>:<line>   // List around specific file/line\n"
          "_regexp-list <line>          // List current file around specified "
          "line\n"
          "_regexp-list <function-name> // List specified function\n"
          "_regexp-list 0x<address>     // List around specified address\n"
          "_regexp-list -[<count>]      // List previous <count> lines\n"
          "_regexp-list                 // List subsequent lines",
          2, CommandCompletions::eSourceFileCompletion, false));
  if (list_regex_cmd_ap.get()) {
    if (list_regex_cmd_ap->AddRegexCommand("^([0-9]+)[[:space:]]*$",
                                           "source list --line %1") &&
        list_regex_cmd_ap->AddRegexCommand(
            "^(.*[^[:space:]])[[:space:]]*:[[:space:]]*([[:digit:]]+)[[:space:]"
            "]*$",
            "source list --file '%1' --line %2") &&
        list_regex_cmd_ap->AddRegexCommand(
            "^\\*?(0x[[:xdigit:]]+)[[:space:]]*$",
            "source list --address %1") &&
        list_regex_cmd_ap->AddRegexCommand("^-[[:space:]]*$",
                                           "source list --reverse") &&
        list_regex_cmd_ap->AddRegexCommand(
            "^-([[:digit:]]+)[[:space:]]*$",
            "source list --reverse --count %1") &&
        list_regex_cmd_ap->AddRegexCommand("^([^.]+)\\.([^.]+)$",
                                           "source list --file \"%1.%2\"") &&
        list_regex_cmd_ap->AddRegexCommand("^(.+)$",
                                           "source list --name \"%1\"") &&
        list_regex_cmd_ap->AddRegexCommand("^$", "source list")) {
      CommandObjectSP list_regex_cmd_sp(list_regex_cmd_ap.release());
      m_command_dict[list_regex_cmd_sp->GetCommandName()] = list_regex_cmd_sp;
    }
  }

  std::unique_ptr<CommandObjectRegexCommand> env_regex_cmd_ap(
      new CommandObjectRegexCommand(
          *this, "_regexp-env",
          "Shorthand for viewing and setting environment variables.",
          "\n"
          "_regexp-env                  // Show enrivonment\n"
          "_regexp-env <name>=<value>   // Set an environment variable",
          2, 0, false));
  if (env_regex_cmd_ap.get()) {
    if (env_regex_cmd_ap->AddRegexCommand("^$",
                                          "settings show target.env-vars") &&
        env_regex_cmd_ap->AddRegexCommand("^([A-Za-z_][A-Za-z_0-9]*=.*)$",
                                          "settings set target.env-vars %1")) {
      CommandObjectSP env_regex_cmd_sp(env_regex_cmd_ap.release());
      m_command_dict[env_regex_cmd_sp->GetCommandName()] = env_regex_cmd_sp;
    }
  }

  std::unique_ptr<CommandObjectRegexCommand> jump_regex_cmd_ap(
      new CommandObjectRegexCommand(
          *this, "_regexp-jump", "Set the program counter to a new address.",
          "\n"
          "_regexp-jump <line>\n"
          "_regexp-jump +<line-offset> | -<line-offset>\n"
          "_regexp-jump <file>:<line>\n"
          "_regexp-jump *<addr>\n",
          2, 0, false));
  if (jump_regex_cmd_ap.get()) {
    if (jump_regex_cmd_ap->AddRegexCommand("^\\*(.*)$",
                                           "thread jump --addr %1") &&
        jump_regex_cmd_ap->AddRegexCommand("^([0-9]+)$",
                                           "thread jump --line %1") &&
        jump_regex_cmd_ap->AddRegexCommand("^([^:]+):([0-9]+)$",
                                           "thread jump --file %1 --line %2") &&
        jump_regex_cmd_ap->AddRegexCommand("^([+\\-][0-9]+)$",
                                           "thread jump --by %1")) {
      CommandObjectSP jump_regex_cmd_sp(jump_regex_cmd_ap.release());
      m_command_dict[jump_regex_cmd_sp->GetCommandName()] = jump_regex_cmd_sp;
    }
  }
}

int CommandInterpreter::GetCommandNamesMatchingPartialString(
    const char *cmd_str, bool include_aliases, StringList &matches) {
  AddNamesMatchingPartialString(m_command_dict, cmd_str, matches);

  if (include_aliases) {
    AddNamesMatchingPartialString(m_alias_dict, cmd_str, matches);
  }

  return matches.GetSize();
}

CommandObjectSP CommandInterpreter::GetCommandSP(const char *cmd_cstr,
                                                 bool include_aliases,
                                                 bool exact,
                                                 StringList *matches) {
  CommandObject::CommandMap::iterator pos;
  CommandObjectSP command_sp;

  std::string cmd(cmd_cstr);

  if (HasCommands()) {
    pos = m_command_dict.find(cmd);
    if (pos != m_command_dict.end())
      command_sp = pos->second;
  }

  if (include_aliases && HasAliases()) {
    auto alias_pos = m_alias_dict.find(cmd);
    if (alias_pos != m_alias_dict.end())
      command_sp = alias_pos->second;
  }

  if (HasUserCommands()) {
    pos = m_user_dict.find(cmd);
    if (pos != m_user_dict.end())
      command_sp = pos->second;
  }

  if (!exact && !command_sp) {
    // We will only get into here if we didn't find any exact matches.

    CommandObjectSP user_match_sp, alias_match_sp, real_match_sp;

    StringList local_matches;
    if (matches == nullptr)
      matches = &local_matches;

    unsigned int num_cmd_matches = 0;
    unsigned int num_alias_matches = 0;
    unsigned int num_user_matches = 0;

    // Look through the command dictionaries one by one, and if we get only one
    // match from any of
    // them in toto, then return that, otherwise return an empty CommandObjectSP
    // and the list of matches.

    if (HasCommands()) {
      num_cmd_matches =
          AddNamesMatchingPartialString(m_command_dict, cmd_cstr, *matches);
    }

    if (num_cmd_matches == 1) {
      cmd.assign(matches->GetStringAtIndex(0));
      pos = m_command_dict.find(cmd);
      if (pos != m_command_dict.end())
        real_match_sp = pos->second;
    }

    if (include_aliases && HasAliases()) {
      num_alias_matches =
          AddNamesMatchingPartialString(m_alias_dict, cmd_cstr, *matches);
    }

    if (num_alias_matches == 1) {
      cmd.assign(matches->GetStringAtIndex(num_cmd_matches));
      auto alias_pos = m_alias_dict.find(cmd);
      if (alias_pos != m_alias_dict.end())
        alias_match_sp = alias_pos->second;
    }

    if (HasUserCommands()) {
      num_user_matches =
          AddNamesMatchingPartialString(m_user_dict, cmd_cstr, *matches);
    }

    if (num_user_matches == 1) {
      cmd.assign(
          matches->GetStringAtIndex(num_cmd_matches + num_alias_matches));

      pos = m_user_dict.find(cmd);
      if (pos != m_user_dict.end())
        user_match_sp = pos->second;
    }

    // If we got exactly one match, return that, otherwise return the match
    // list.

    if (num_user_matches + num_cmd_matches + num_alias_matches == 1) {
      if (num_cmd_matches)
        return real_match_sp;
      else if (num_alias_matches)
        return alias_match_sp;
      else
        return user_match_sp;
    }
  } else if (matches && command_sp) {
    matches->AppendString(cmd_cstr);
  }

  return command_sp;
}

bool CommandInterpreter::AddCommand(const char *name,
                                    const lldb::CommandObjectSP &cmd_sp,
                                    bool can_replace) {
  if (cmd_sp.get())
    assert((this == &cmd_sp->GetCommandInterpreter()) &&
           "tried to add a CommandObject from a different interpreter");

  if (name && name[0]) {
    std::string name_sstr(name);
    bool found = (m_command_dict.find(name_sstr) != m_command_dict.end());
    if (found && !can_replace)
      return false;
    if (found && m_command_dict[name_sstr]->IsRemovable() == false)
      return false;
    m_command_dict[name_sstr] = cmd_sp;
    return true;
  }
  return false;
}

bool CommandInterpreter::AddUserCommand(std::string name,
                                        const lldb::CommandObjectSP &cmd_sp,
                                        bool can_replace) {
  if (cmd_sp.get())
    assert((this == &cmd_sp->GetCommandInterpreter()) &&
           "tried to add a CommandObject from a different interpreter");

  if (!name.empty()) {
    const char *name_cstr = name.c_str();

    // do not allow replacement of internal commands
    if (CommandExists(name_cstr)) {
      if (can_replace == false)
        return false;
      if (m_command_dict[name]->IsRemovable() == false)
        return false;
    }

    if (UserCommandExists(name_cstr)) {
      if (can_replace == false)
        return false;
      if (m_user_dict[name]->IsRemovable() == false)
        return false;
    }

    m_user_dict[name] = cmd_sp;
    return true;
  }
  return false;
}

CommandObjectSP CommandInterpreter::GetCommandSPExact(const char *cmd_cstr,
                                                      bool include_aliases) {
  Args cmd_words(cmd_cstr); // Break up the command string into words, in case
                            // it's a multi-word command.
  CommandObjectSP ret_val;  // Possibly empty return value.

  if (cmd_cstr == nullptr)
    return ret_val;

  if (cmd_words.GetArgumentCount() == 1)
    return GetCommandSP(cmd_cstr, include_aliases, true, nullptr);
  else {
    // We have a multi-word command (seemingly), so we need to do more work.
    // First, get the cmd_obj_sp for the first word in the command.
    CommandObjectSP cmd_obj_sp = GetCommandSP(cmd_words.GetArgumentAtIndex(0),
                                              include_aliases, true, nullptr);
    if (cmd_obj_sp.get() != nullptr) {
      // Loop through the rest of the words in the command (everything passed in
      // was supposed to be part of a
      // command name), and find the appropriate sub-command SP for each command
      // word....
      size_t end = cmd_words.GetArgumentCount();
      for (size_t j = 1; j < end; ++j) {
        if (cmd_obj_sp->IsMultiwordObject()) {
          cmd_obj_sp =
              cmd_obj_sp->GetSubcommandSP(cmd_words.GetArgumentAtIndex(j));
          if (cmd_obj_sp.get() == nullptr)
            // The sub-command name was invalid.  Fail and return the empty
            // 'ret_val'.
            return ret_val;
        } else
          // We have more words in the command name, but we don't have a
          // multiword object. Fail and return
          // empty 'ret_val'.
          return ret_val;
      }
      // We successfully looped through all the command words and got valid
      // command objects for them.  Assign the
      // last object retrieved to 'ret_val'.
      ret_val = cmd_obj_sp;
    }
  }
  return ret_val;
}

CommandObject *CommandInterpreter::GetCommandObjectExact(const char *cmd_cstr,
                                                         bool include_aliases) {
  return GetCommandSPExact(cmd_cstr, include_aliases).get();
}

CommandObject *CommandInterpreter::GetCommandObject(const char *cmd_cstr,
                                                    StringList *matches) {
  CommandObject *command_obj =
      GetCommandSP(cmd_cstr, false, true, matches).get();

  // If we didn't find an exact match to the command string in the commands,
  // look in
  // the aliases.

  if (command_obj)
    return command_obj;

  command_obj = GetCommandSP(cmd_cstr, true, true, matches).get();

  if (command_obj)
    return command_obj;

  // If there wasn't an exact match then look for an inexact one in just the
  // commands
  command_obj = GetCommandSP(cmd_cstr, false, false, nullptr).get();

  // Finally, if there wasn't an inexact match among the commands, look for an
  // inexact
  // match in both the commands and aliases.

  if (command_obj) {
    if (matches)
      matches->AppendString(command_obj->GetCommandName());
    return command_obj;
  }

  return GetCommandSP(cmd_cstr, true, false, matches).get();
}

bool CommandInterpreter::CommandExists(const char *cmd) {
  return m_command_dict.find(cmd) != m_command_dict.end();
}

bool CommandInterpreter::GetAliasFullName(const char *cmd,
                                          std::string &full_name) {
  bool exact_match = (m_alias_dict.find(cmd) != m_alias_dict.end());
  if (exact_match) {
    full_name.assign(cmd);
    return exact_match;
  } else {
    StringList matches;
    size_t num_alias_matches;
    num_alias_matches =
        AddNamesMatchingPartialString(m_alias_dict, cmd, matches);
    if (num_alias_matches == 1) {
      // Make sure this isn't shadowing a command in the regular command space:
      StringList regular_matches;
      const bool include_aliases = false;
      const bool exact = false;
      CommandObjectSP cmd_obj_sp(
          GetCommandSP(cmd, include_aliases, exact, &regular_matches));
      if (cmd_obj_sp || regular_matches.GetSize() > 0)
        return false;
      else {
        full_name.assign(matches.GetStringAtIndex(0));
        return true;
      }
    } else
      return false;
  }
}

bool CommandInterpreter::AliasExists(const char *cmd) {
  return m_alias_dict.find(cmd) != m_alias_dict.end();
}

bool CommandInterpreter::UserCommandExists(const char *cmd) {
  return m_user_dict.find(cmd) != m_user_dict.end();
}

CommandAlias *
CommandInterpreter::AddAlias(const char *alias_name,
                             lldb::CommandObjectSP &command_obj_sp,
                             const char *args_string) {
  if (command_obj_sp.get())
    assert((this == &command_obj_sp->GetCommandInterpreter()) &&
           "tried to add a CommandObject from a different interpreter");

  std::unique_ptr<CommandAlias> command_alias_up(
      new CommandAlias(*this, command_obj_sp, args_string, alias_name));

  if (command_alias_up && command_alias_up->IsValid()) {
    m_alias_dict[alias_name] = CommandObjectSP(command_alias_up.get());
    return command_alias_up.release();
  }

  return nullptr;
}

bool CommandInterpreter::RemoveAlias(const char *alias_name) {
  auto pos = m_alias_dict.find(alias_name);
  if (pos != m_alias_dict.end()) {
    m_alias_dict.erase(pos);
    return true;
  }
  return false;
}

bool CommandInterpreter::RemoveCommand(const char *cmd) {
  auto pos = m_command_dict.find(cmd);
  if (pos != m_command_dict.end()) {
    if (pos->second->IsRemovable()) {
      // Only regular expression objects or python commands are removable
      m_command_dict.erase(pos);
      return true;
    }
  }
  return false;
}
bool CommandInterpreter::RemoveUser(const char *alias_name) {
  CommandObject::CommandMap::iterator pos = m_user_dict.find(alias_name);
  if (pos != m_user_dict.end()) {
    m_user_dict.erase(pos);
    return true;
  }
  return false;
}

void CommandInterpreter::GetHelp(CommandReturnObject &result,
                                 uint32_t cmd_types) {
  const char *help_prologue = GetDebugger().GetIOHandlerHelpPrologue();
  if (help_prologue != NULL) {
    OutputFormattedHelpText(result.GetOutputStream(), NULL, help_prologue);
  }

  CommandObject::CommandMap::const_iterator pos;
  size_t max_len = FindLongestCommandWord(m_command_dict);

  if ((cmd_types & eCommandTypesBuiltin) == eCommandTypesBuiltin) {
    result.AppendMessage("Debugger commands:");
    result.AppendMessage("");

    for (pos = m_command_dict.begin(); pos != m_command_dict.end(); ++pos) {
      if (!(cmd_types & eCommandTypesHidden) &&
          (pos->first.compare(0, 1, "_") == 0))
        continue;

      OutputFormattedHelpText(result.GetOutputStream(), pos->first.c_str(),
                              "--", pos->second->GetHelp(), max_len);
    }
    result.AppendMessage("");
  }

  if (!m_alias_dict.empty() &&
      ((cmd_types & eCommandTypesAliases) == eCommandTypesAliases)) {
    result.AppendMessageWithFormat(
        "Current command abbreviations "
        "(type '%shelp command alias' for more info):\n",
        GetCommandPrefix());
    result.AppendMessage("");
    max_len = FindLongestCommandWord(m_alias_dict);

    for (auto alias_pos = m_alias_dict.begin(); alias_pos != m_alias_dict.end();
         ++alias_pos) {
      OutputFormattedHelpText(result.GetOutputStream(),
                              alias_pos->first.c_str(), "--",
                              alias_pos->second->GetHelp(), max_len);
    }
    result.AppendMessage("");
  }

  if (!m_user_dict.empty() &&
      ((cmd_types & eCommandTypesUserDef) == eCommandTypesUserDef)) {
    result.AppendMessage("Current user-defined commands:");
    result.AppendMessage("");
    max_len = FindLongestCommandWord(m_user_dict);
    for (pos = m_user_dict.begin(); pos != m_user_dict.end(); ++pos) {
      OutputFormattedHelpText(result.GetOutputStream(), pos->first.c_str(),
                              "--", pos->second->GetHelp(), max_len);
    }
    result.AppendMessage("");
  }

  result.AppendMessageWithFormat(
      "For more information on any command, type '%shelp <command-name>'.\n",
      GetCommandPrefix());
}

CommandObject *
CommandInterpreter::GetCommandObjectForCommand(std::string &command_string) {
  // This function finds the final, lowest-level, alias-resolved command object
  // whose 'Execute' function will
  // eventually be invoked by the given command line.

  CommandObject *cmd_obj = nullptr;
  size_t start = command_string.find_first_not_of(k_white_space);
  size_t end = 0;
  bool done = false;
  while (!done) {
    if (start != std::string::npos) {
      // Get the next word from command_string.
      end = command_string.find_first_of(k_white_space, start);
      if (end == std::string::npos)
        end = command_string.size();
      std::string cmd_word = command_string.substr(start, end - start);

      if (cmd_obj == nullptr)
        // Since cmd_obj is NULL we are on our first time through this loop.
        // Check to see if cmd_word is a valid
        // command or alias.
        cmd_obj = GetCommandObject(cmd_word.c_str());
      else if (cmd_obj->IsMultiwordObject()) {
        // Our current object is a multi-word object; see if the cmd_word is a
        // valid sub-command for our object.
        CommandObject *sub_cmd_obj =
            cmd_obj->GetSubcommandObject(cmd_word.c_str());
        if (sub_cmd_obj)
          cmd_obj = sub_cmd_obj;
        else // cmd_word was not a valid sub-command word, so we are done
          done = true;
      } else
        // We have a cmd_obj and it is not a multi-word object, so we are done.
        done = true;

      // If we didn't find a valid command object, or our command object is not
      // a multi-word object, or
      // we are at the end of the command_string, then we are done.  Otherwise,
      // find the start of the
      // next word.

      if (!cmd_obj || !cmd_obj->IsMultiwordObject() ||
          end >= command_string.size())
        done = true;
      else
        start = command_string.find_first_not_of(k_white_space, end);
    } else
      // Unable to find any more words.
      done = true;
  }

  if (end == command_string.size())
    command_string.clear();
  else
    command_string = command_string.substr(end);

  return cmd_obj;
}

static const char *k_valid_command_chars =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_";
static void StripLeadingSpaces(std::string &s) {
  if (!s.empty()) {
    size_t pos = s.find_first_not_of(k_white_space);
    if (pos == std::string::npos)
      s.clear();
    else if (pos == 0)
      return;
    s.erase(0, pos);
  }
}

static size_t FindArgumentTerminator(const std::string &s) {
  const size_t s_len = s.size();
  size_t offset = 0;
  while (offset < s_len) {
    size_t pos = s.find("--", offset);
    if (pos == std::string::npos)
      break;
    if (pos > 0) {
      if (isspace(s[pos - 1])) {
        // Check if the string ends "\s--" (where \s is a space character)
        // or if we have "\s--\s".
        if ((pos + 2 >= s_len) || isspace(s[pos + 2])) {
          return pos;
        }
      }
    }
    offset = pos + 2;
  }
  return std::string::npos;
}

static bool ExtractCommand(std::string &command_string, std::string &command,
                           std::string &suffix, char &quote_char) {
  command.clear();
  suffix.clear();
  StripLeadingSpaces(command_string);

  bool result = false;
  quote_char = '\0';

  if (!command_string.empty()) {
    const char first_char = command_string[0];
    if (first_char == '\'' || first_char == '"') {
      quote_char = first_char;
      const size_t end_quote_pos = command_string.find(quote_char, 1);
      if (end_quote_pos == std::string::npos) {
        command.swap(command_string);
        command_string.erase();
      } else {
        command.assign(command_string, 1, end_quote_pos - 1);
        if (end_quote_pos + 1 < command_string.size())
          command_string.erase(0, command_string.find_first_not_of(
                                      k_white_space, end_quote_pos + 1));
        else
          command_string.erase();
      }
    } else {
      const size_t first_space_pos =
          command_string.find_first_of(k_white_space);
      if (first_space_pos == std::string::npos) {
        command.swap(command_string);
        command_string.erase();
      } else {
        command.assign(command_string, 0, first_space_pos);
        command_string.erase(0, command_string.find_first_not_of(
                                    k_white_space, first_space_pos));
      }
    }
    result = true;
  }

  if (!command.empty()) {
    // actual commands can't start with '-' or '_'
    if (command[0] != '-' && command[0] != '_') {
      size_t pos = command.find_first_not_of(k_valid_command_chars);
      if (pos > 0 && pos != std::string::npos) {
        suffix.assign(command.begin() + pos, command.end());
        command.erase(pos);
      }
    }
  }

  return result;
}

CommandObject *CommandInterpreter::BuildAliasResult(
    const char *alias_name, std::string &raw_input_string,
    std::string &alias_result, CommandReturnObject &result) {
  CommandObject *alias_cmd_obj = nullptr;
  Args cmd_args(raw_input_string);
  alias_cmd_obj = GetCommandObject(alias_name);
  StreamString result_str;

  if (alias_cmd_obj && alias_cmd_obj->IsAlias()) {
    std::pair<CommandObjectSP, OptionArgVectorSP> desugared =
        ((CommandAlias *)alias_cmd_obj)->Desugar();
    OptionArgVectorSP option_arg_vector_sp = desugared.second;
    alias_cmd_obj = desugared.first.get();
    std::string alias_name_str = alias_name;
    if ((cmd_args.GetArgumentCount() == 0) ||
        (alias_name_str.compare(cmd_args.GetArgumentAtIndex(0)) != 0))
      cmd_args.Unshift(alias_name);

    result_str.Printf("%s", alias_cmd_obj->GetCommandName());

    if (option_arg_vector_sp.get()) {
      OptionArgVector *option_arg_vector = option_arg_vector_sp.get();

      for (size_t i = 0; i < option_arg_vector->size(); ++i) {
        OptionArgPair option_pair = (*option_arg_vector)[i];
        OptionArgValue value_pair = option_pair.second;
        int value_type = value_pair.first;
        std::string option = option_pair.first;
        std::string value = value_pair.second;
        if (option.compare("<argument>") == 0)
          result_str.Printf(" %s", value.c_str());
        else {
          result_str.Printf(" %s", option.c_str());
          if (value_type != OptionParser::eNoArgument) {
            if (value_type != OptionParser::eOptionalArgument)
              result_str.Printf(" ");
            int index = GetOptionArgumentPosition(value.c_str());
            if (index == 0)
              result_str.Printf("%s", value.c_str());
            else if (static_cast<size_t>(index) >=
                     cmd_args.GetArgumentCount()) {

              result.AppendErrorWithFormat("Not enough arguments provided; you "
                                           "need at least %d arguments to use "
                                           "this alias.\n",
                                           index);
              result.SetStatus(eReturnStatusFailed);
              return nullptr;
            } else {
              size_t strpos =
                  raw_input_string.find(cmd_args.GetArgumentAtIndex(index));
              if (strpos != std::string::npos)
                raw_input_string = raw_input_string.erase(
                    strpos, strlen(cmd_args.GetArgumentAtIndex(index)));
              result_str.Printf("%s", cmd_args.GetArgumentAtIndex(index));
            }
          }
        }
      }
    }

    alias_result = result_str.GetData();
  }
  return alias_cmd_obj;
}

Error CommandInterpreter::PreprocessCommand(std::string &command) {
  // The command preprocessor needs to do things to the command
  // line before any parsing of arguments or anything else is done.
  // The only current stuff that gets preprocessed is anything enclosed
  // in backtick ('`') characters is evaluated as an expression and
  // the result of the expression must be a scalar that can be substituted
  // into the command. An example would be:
  // (lldb) memory read `$rsp + 20`
  Error error; // Error for any expressions that might not evaluate
  size_t start_backtick;
  size_t pos = 0;
  while ((start_backtick = command.find('`', pos)) != std::string::npos) {
    if (start_backtick > 0 && command[start_backtick - 1] == '\\') {
      // The backtick was preceded by a '\' character, remove the slash
      // and don't treat the backtick as the start of an expression
      command.erase(start_backtick - 1, 1);
      // No need to add one to start_backtick since we just deleted a char
      pos = start_backtick;
    } else {
      const size_t expr_content_start = start_backtick + 1;
      const size_t end_backtick = command.find('`', expr_content_start);
      if (end_backtick == std::string::npos)
        return error;
      else if (end_backtick == expr_content_start) {
        // Empty expression (two backticks in a row)
        command.erase(start_backtick, 2);
      } else {
        std::string expr_str(command, expr_content_start,
                             end_backtick - expr_content_start);

        ExecutionContext exe_ctx(GetExecutionContext());
        Target *target = exe_ctx.GetTargetPtr();
        // Get a dummy target to allow for calculator mode while processing
        // backticks.
        // This also helps break the infinite loop caused when target is null.
        if (!target)
          target = m_debugger.GetDummyTarget();
        if (target) {
          ValueObjectSP expr_result_valobj_sp;

          EvaluateExpressionOptions options;
          options.SetCoerceToId(false);
          options.SetUnwindOnError(true);
          options.SetIgnoreBreakpoints(true);
          options.SetKeepInMemory(false);
          options.SetTryAllThreads(true);
          options.SetTimeoutUsec(0);

          ExpressionResults expr_result = target->EvaluateExpression(
              expr_str.c_str(), exe_ctx.GetFramePtr(), expr_result_valobj_sp,
              options);

          if (expr_result == eExpressionCompleted) {
            Scalar scalar;
            if (expr_result_valobj_sp)
              expr_result_valobj_sp =
                  expr_result_valobj_sp->GetQualifiedRepresentationIfAvailable(
                      expr_result_valobj_sp->GetDynamicValueType(), true);
            if (expr_result_valobj_sp->ResolveValue(scalar)) {
              command.erase(start_backtick, end_backtick - start_backtick + 1);
              StreamString value_strm;
              const bool show_type = false;
              scalar.GetValue(&value_strm, show_type);
              size_t value_string_size = value_strm.GetSize();
              if (value_string_size) {
                command.insert(start_backtick, value_strm.GetData(),
                               value_string_size);
                pos = start_backtick + value_string_size;
                continue;
              } else {
                error.SetErrorStringWithFormat("expression value didn't result "
                                               "in a scalar value for the "
                                               "expression '%s'",
                                               expr_str.c_str());
              }
            } else {
              error.SetErrorStringWithFormat("expression value didn't result "
                                             "in a scalar value for the "
                                             "expression '%s'",
                                             expr_str.c_str());
            }
          } else {
            if (expr_result_valobj_sp)
              error = expr_result_valobj_sp->GetError();
            if (error.Success()) {

              switch (expr_result) {
              case eExpressionSetupError:
                error.SetErrorStringWithFormat(
                    "expression setup error for the expression '%s'",
                    expr_str.c_str());
                break;
              case eExpressionParseError:
                error.SetErrorStringWithFormat(
                    "expression parse error for the expression '%s'",
                    expr_str.c_str());
                break;
              case eExpressionResultUnavailable:
                error.SetErrorStringWithFormat(
                    "expression error fetching result for the expression '%s'",
                    expr_str.c_str());
                break;
              case eExpressionCompleted:
                break;
              case eExpressionDiscarded:
                error.SetErrorStringWithFormat(
                    "expression discarded for the expression '%s'",
                    expr_str.c_str());
                break;
              case eExpressionInterrupted:
                error.SetErrorStringWithFormat(
                    "expression interrupted for the expression '%s'",
                    expr_str.c_str());
                break;
              case eExpressionHitBreakpoint:
                error.SetErrorStringWithFormat(
                    "expression hit breakpoint for the expression '%s'",
                    expr_str.c_str());
                break;
              case eExpressionTimedOut:
                error.SetErrorStringWithFormat(
                    "expression timed out for the expression '%s'",
                    expr_str.c_str());
                break;
              case eExpressionStoppedForDebug:
                error.SetErrorStringWithFormat("expression stop at entry point "
                                               "for debugging for the "
                                               "expression '%s'",
                                               expr_str.c_str());
                break;
              }
            }
          }
        }
      }
      if (error.Fail())
        break;
    }
  }
  return error;
}

bool CommandInterpreter::HandleCommand(const char *command_line,
                                       LazyBool lazy_add_to_history,
                                       CommandReturnObject &result,
                                       ExecutionContext *override_context,
                                       bool repeat_on_empty_command,
                                       bool no_context_switching)

{

  std::string command_string(command_line);
  std::string original_command_string(command_line);

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_COMMANDS));
  Host::SetCrashDescriptionWithFormat("HandleCommand(command = \"%s\")",
                                      command_line);

  // Make a scoped cleanup object that will clear the crash description string
  // on exit of this function.
  lldb_utility::CleanUp<const char *> crash_description_cleanup(
      nullptr, Host::SetCrashDescription);

  if (log)
    log->Printf("Processing command: %s", command_line);

  Timer scoped_timer(__PRETTY_FUNCTION__, "Handling command: %s.",
                     command_line);

  if (!no_context_switching)
    UpdateExecutionContext(override_context);

  bool add_to_history;
  if (lazy_add_to_history == eLazyBoolCalculate)
    add_to_history = (m_command_source_depth == 0);
  else
    add_to_history = (lazy_add_to_history == eLazyBoolYes);

  bool empty_command = false;
  bool comment_command = false;
  if (command_string.empty())
    empty_command = true;
  else {
    const char *k_space_characters = "\t\n\v\f\r ";

    size_t non_space = command_string.find_first_not_of(k_space_characters);
    // Check for empty line or comment line (lines whose first
    // non-space character is the comment character for this interpreter)
    if (non_space == std::string::npos)
      empty_command = true;
    else if (command_string[non_space] == m_comment_char)
      comment_command = true;
    else if (command_string[non_space] == CommandHistory::g_repeat_char) {
      const char *history_string =
          m_command_history.FindString(command_string.c_str() + non_space);
      if (history_string == nullptr) {
        result.AppendErrorWithFormat("Could not find entry: %s in history",
                                     command_string.c_str());
        result.SetStatus(eReturnStatusFailed);
        return false;
      }
      add_to_history = false;
      command_string = history_string;
      original_command_string = history_string;
    }
  }

  if (empty_command) {
    if (repeat_on_empty_command) {
      if (m_command_history.IsEmpty()) {
        result.AppendError("empty command");
        result.SetStatus(eReturnStatusFailed);
        return false;
      } else {
        command_line = m_repeat_command.c_str();
        command_string = command_line;
        original_command_string = command_line;
        if (m_repeat_command.empty()) {
          result.AppendErrorWithFormat("No auto repeat.\n");
          result.SetStatus(eReturnStatusFailed);
          return false;
        }
      }
      add_to_history = false;
    } else {
      result.SetStatus(eReturnStatusSuccessFinishNoResult);
      return true;
    }
  } else if (comment_command) {
    result.SetStatus(eReturnStatusSuccessFinishNoResult);
    return true;
  }

  Error error(PreprocessCommand(command_string));

  if (error.Fail()) {
    result.AppendError(error.AsCString());
    result.SetStatus(eReturnStatusFailed);
    return false;
  }

  // Phase 1.

  // Before we do ANY kind of argument processing, we need to figure out what
  // the real/final command object is for the specified command.  This gets
  // complicated by the fact that the user could have specified an alias, and,
  // in translating the alias, there may also be command options and/or even
  // data (including raw text strings) that need to be found and inserted into
  // the command line as part of the translation.  So this first step is plain
  // look-up and replacement, resulting in:
  //    1. the command object whose Execute method will actually be called
  //    2. a revised command string, with all substitutions and replacements
  //       taken care of
  // From 1 above, we can determine whether the Execute function wants raw
  // input or not.

  CommandObject *cmd_obj = ResolveCommandImpl(command_string, result);

  // Although the user may have abbreviated the command, the command_string now
  // has the command expanded to the full name.  For example, if the input
  // was "br s -n main", command_string is now "breakpoint set -n main".

  if (log) {
    log->Printf("HandleCommand, cmd_obj : '%s'",
                cmd_obj ? cmd_obj->GetCommandName() : "<not found>");
    log->Printf("HandleCommand, (revised) command_string: '%s'",
                command_string.c_str());
    const bool wants_raw_input =
        (cmd_obj != NULL) ? cmd_obj->WantsRawCommandString() : false;
    log->Printf("HandleCommand, wants_raw_input:'%s'",
                wants_raw_input ? "True" : "False");
  }

  // Phase 2.
  // Take care of things like setting up the history command & calling the
  // appropriate Execute method on the
  // CommandObject, with the appropriate arguments.

  if (cmd_obj != nullptr) {
    if (add_to_history) {
      Args command_args(command_string);
      const char *repeat_command = cmd_obj->GetRepeatCommand(command_args, 0);
      if (repeat_command != nullptr)
        m_repeat_command.assign(repeat_command);
      else
        m_repeat_command.assign(original_command_string.c_str());

      m_command_history.AppendString(original_command_string);
    }

    std::string remainder;
    const std::size_t actual_cmd_name_len = strlen(cmd_obj->GetCommandName());
    if (actual_cmd_name_len < command_string.length())
      remainder = command_string.substr(actual_cmd_name_len);

    // Remove any initial spaces
    size_t pos = remainder.find_first_not_of(k_white_space);
    if (pos != 0 && pos != std::string::npos)
      remainder.erase(0, pos);

    if (log)
      log->Printf(
          "HandleCommand, command line after removing command name(s): '%s'",
          remainder.c_str());

    cmd_obj->Execute(remainder.c_str(), result);
  } else {
    // We didn't find the first command object, so complete the first argument.
    Args command_args(command_string);
    StringList matches;
    int num_matches;
    int cursor_index = 0;
    int cursor_char_position = strlen(command_args.GetArgumentAtIndex(0));
    bool word_complete;
    num_matches = HandleCompletionMatches(command_args, cursor_index,
                                          cursor_char_position, 0, -1,
                                          word_complete, matches);

    if (num_matches > 0) {
      std::string error_msg;
      error_msg.assign("ambiguous command '");
      error_msg.append(command_args.GetArgumentAtIndex(0));
      error_msg.append("'.");

      error_msg.append(" Possible completions:");
      for (int i = 0; i < num_matches; i++) {
        error_msg.append("\n\t");
        error_msg.append(matches.GetStringAtIndex(i));
      }
      error_msg.append("\n");
      result.AppendRawError(error_msg.c_str());
    } else
      result.AppendErrorWithFormat("Unrecognized command '%s'.\n",
                                   command_args.GetArgumentAtIndex(0));

    result.SetStatus(eReturnStatusFailed);
  }

  if (log)
    log->Printf("HandleCommand, command %s",
                (result.Succeeded() ? "succeeded" : "did not succeed"));

  return result.Succeeded();
}

int CommandInterpreter::HandleCompletionMatches(
    Args &parsed_line, int &cursor_index, int &cursor_char_position,
    int match_start_point, int max_return_elements, bool &word_complete,
    StringList &matches) {
  int num_command_matches = 0;
  bool look_for_subcommand = false;

  // For any of the command completions a unique match will be a complete word.
  word_complete = true;

  if (cursor_index == -1) {
    // We got nothing on the command line, so return the list of commands
    bool include_aliases = true;
    num_command_matches =
        GetCommandNamesMatchingPartialString("", include_aliases, matches);
  } else if (cursor_index == 0) {
    // The cursor is in the first argument, so just do a lookup in the
    // dictionary.
    CommandObject *cmd_obj =
        GetCommandObject(parsed_line.GetArgumentAtIndex(0), &matches);
    num_command_matches = matches.GetSize();

    if (num_command_matches == 1 && cmd_obj && cmd_obj->IsMultiwordObject() &&
        matches.GetStringAtIndex(0) != nullptr &&
        strcmp(parsed_line.GetArgumentAtIndex(0),
               matches.GetStringAtIndex(0)) == 0) {
      if (parsed_line.GetArgumentCount() == 1) {
        word_complete = true;
      } else {
        look_for_subcommand = true;
        num_command_matches = 0;
        matches.DeleteStringAtIndex(0);
        parsed_line.AppendArgument("");
        cursor_index++;
        cursor_char_position = 0;
      }
    }
  }

  if (cursor_index > 0 || look_for_subcommand) {
    // We are completing further on into a commands arguments, so find the
    // command and tell it
    // to complete the command.
    // First see if there is a matching initial command:
    CommandObject *command_object =
        GetCommandObject(parsed_line.GetArgumentAtIndex(0));
    if (command_object == nullptr) {
      return 0;
    } else {
      parsed_line.Shift();
      cursor_index--;
      num_command_matches = command_object->HandleCompletion(
          parsed_line, cursor_index, cursor_char_position, match_start_point,
          max_return_elements, word_complete, matches);
    }
  }

  return num_command_matches;
}

int CommandInterpreter::HandleCompletion(
    const char *current_line, const char *cursor, const char *last_char,
    int match_start_point, int max_return_elements, StringList &matches) {
  // We parse the argument up to the cursor, so the last argument in parsed_line
  // is
  // the one containing the cursor, and the cursor is after the last character.

  Args parsed_line(llvm::StringRef(current_line, last_char - current_line));
  Args partial_parsed_line(
      llvm::StringRef(current_line, cursor - current_line));

  // Don't complete comments, and if the line we are completing is just the
  // history repeat character,
  // substitute the appropriate history line.
  const char *first_arg = parsed_line.GetArgumentAtIndex(0);
  if (first_arg) {
    if (first_arg[0] == m_comment_char)
      return 0;
    else if (first_arg[0] == CommandHistory::g_repeat_char) {
      const char *history_string = m_command_history.FindString(first_arg);
      if (history_string != nullptr) {
        matches.Clear();
        matches.InsertStringAtIndex(0, history_string);
        return -2;
      } else
        return 0;
    }
  }

  int num_args = partial_parsed_line.GetArgumentCount();
  int cursor_index = partial_parsed_line.GetArgumentCount() - 1;
  int cursor_char_position;

  if (cursor_index == -1)
    cursor_char_position = 0;
  else
    cursor_char_position =
        strlen(partial_parsed_line.GetArgumentAtIndex(cursor_index));

  if (cursor > current_line && cursor[-1] == ' ') {
    // We are just after a space.  If we are in an argument, then we will
    // continue
    // parsing, but if we are between arguments, then we have to complete
    // whatever the next
    // element would be.
    // We can distinguish the two cases because if we are in an argument (e.g.
    // because the space is
    // protected by a quote) then the space will also be in the parsed
    // argument...

    const char *current_elem =
        partial_parsed_line.GetArgumentAtIndex(cursor_index);
    if (cursor_char_position == 0 ||
        current_elem[cursor_char_position - 1] != ' ') {
      parsed_line.InsertArgumentAtIndex(cursor_index + 1, "", '\0');
      cursor_index++;
      cursor_char_position = 0;
    }
  }

  int num_command_matches;

  matches.Clear();

  // Only max_return_elements == -1 is supported at present:
  assert(max_return_elements == -1);
  bool word_complete;
  num_command_matches = HandleCompletionMatches(
      parsed_line, cursor_index, cursor_char_position, match_start_point,
      max_return_elements, word_complete, matches);

  if (num_command_matches <= 0)
    return num_command_matches;

  if (num_args == 0) {
    // If we got an empty string, insert nothing.
    matches.InsertStringAtIndex(0, "");
  } else {
    // Now figure out if there is a common substring, and if so put that in
    // element 0, otherwise
    // put an empty string in element 0.
    std::string command_partial_str;
    if (cursor_index >= 0)
      command_partial_str.assign(parsed_line.GetArgumentAtIndex(cursor_index),
                                 parsed_line.GetArgumentAtIndex(cursor_index) +
                                     cursor_char_position);

    std::string common_prefix;
    matches.LongestCommonPrefix(common_prefix);
    const size_t partial_name_len = command_partial_str.size();
    common_prefix.erase(0, partial_name_len);

    // If we matched a unique single command, add a space...
    // Only do this if the completer told us this was a complete word,
    // however...
    if (num_command_matches == 1 && word_complete) {
      char quote_char = parsed_line.GetArgumentQuoteCharAtIndex(cursor_index);
      common_prefix =
          Args::EscapeLLDBCommandArgument(common_prefix, quote_char);
      if (quote_char != '\0')
        common_prefix.push_back(quote_char);
      common_prefix.push_back(' ');
    }
    matches.InsertStringAtIndex(0, common_prefix.c_str());
  }
  return num_command_matches;
}

CommandInterpreter::~CommandInterpreter() {}

void CommandInterpreter::UpdatePrompt(const char *new_prompt) {
  EventSP prompt_change_event_sp(
      new Event(eBroadcastBitResetPrompt, new EventDataBytes(new_prompt)));
  ;
  BroadcastEvent(prompt_change_event_sp);
  if (m_command_io_handler_sp)
    m_command_io_handler_sp->SetPrompt(new_prompt);
}

bool CommandInterpreter::Confirm(const char *message, bool default_answer) {
  // Check AutoConfirm first:
  if (m_debugger.GetAutoConfirm())
    return default_answer;

  IOHandlerConfirm *confirm =
      new IOHandlerConfirm(m_debugger, message, default_answer);
  IOHandlerSP io_handler_sp(confirm);
  m_debugger.RunIOHandler(io_handler_sp);
  return confirm->GetResponse();
}

CommandAlias *CommandInterpreter::GetAlias(const char *alias_name) {
  OptionArgVectorSP ret_val;

  std::string alias(alias_name);

  auto pos = m_alias_dict.find(alias);
  if (pos != m_alias_dict.end())
    return (CommandAlias *)pos->second.get();

  return nullptr;
}

bool CommandInterpreter::HasCommands() { return (!m_command_dict.empty()); }

bool CommandInterpreter::HasAliases() { return (!m_alias_dict.empty()); }

bool CommandInterpreter::HasUserCommands() { return (!m_user_dict.empty()); }

bool CommandInterpreter::HasAliasOptions() { return HasAliases(); }

void CommandInterpreter::BuildAliasCommandArgs(CommandObject *alias_cmd_obj,
                                               const char *alias_name,
                                               Args &cmd_args,
                                               std::string &raw_input_string,
                                               CommandReturnObject &result) {
  OptionArgVectorSP option_arg_vector_sp =
      GetAlias(alias_name)->GetOptionArguments();

  bool wants_raw_input = alias_cmd_obj->WantsRawCommandString();

  // Make sure that the alias name is the 0th element in cmd_args
  std::string alias_name_str = alias_name;
  if (alias_name_str.compare(cmd_args.GetArgumentAtIndex(0)) != 0)
    cmd_args.Unshift(alias_name);

  Args new_args(alias_cmd_obj->GetCommandName());
  if (new_args.GetArgumentCount() == 2)
    new_args.Shift();

  if (option_arg_vector_sp.get()) {
    if (wants_raw_input) {
      // We have a command that both has command options and takes raw input.
      // Make *sure* it has a
      // " -- " in the right place in the raw_input_string.
      size_t pos = raw_input_string.find(" -- ");
      if (pos == std::string::npos) {
        // None found; assume it goes at the beginning of the raw input string
        raw_input_string.insert(0, " -- ");
      }
    }

    OptionArgVector *option_arg_vector = option_arg_vector_sp.get();
    const size_t old_size = cmd_args.GetArgumentCount();
    std::vector<bool> used(old_size + 1, false);

    used[0] = true;

    for (size_t i = 0; i < option_arg_vector->size(); ++i) {
      OptionArgPair option_pair = (*option_arg_vector)[i];
      OptionArgValue value_pair = option_pair.second;
      int value_type = value_pair.first;
      std::string option = option_pair.first;
      std::string value = value_pair.second;
      if (option.compare("<argument>") == 0) {
        if (!wants_raw_input || (value.compare("--") != 0)) // Since we inserted
                                                            // this above, make
                                                            // sure we don't
                                                            // insert it twice
          new_args.AppendArgument(value.c_str());
      } else {
        if (value_type != OptionParser::eOptionalArgument)
          new_args.AppendArgument(option.c_str());
        if (value.compare("<no-argument>") != 0) {
          int index = GetOptionArgumentPosition(value.c_str());
          if (index == 0) {
            // value was NOT a positional argument; must be a real value
            if (value_type != OptionParser::eOptionalArgument)
              new_args.AppendArgument(value.c_str());
            else {
              char buffer[255];
              ::snprintf(buffer, sizeof(buffer), "%s%s", option.c_str(),
                         value.c_str());
              new_args.AppendArgument(buffer);
            }

          } else if (static_cast<size_t>(index) >=
                     cmd_args.GetArgumentCount()) {
            result.AppendErrorWithFormat("Not enough arguments provided; you "
                                         "need at least %d arguments to use "
                                         "this alias.\n",
                                         index);
            result.SetStatus(eReturnStatusFailed);
            return;
          } else {
            // Find and remove cmd_args.GetArgumentAtIndex(i) from
            // raw_input_string
            size_t strpos =
                raw_input_string.find(cmd_args.GetArgumentAtIndex(index));
            if (strpos != std::string::npos) {
              raw_input_string = raw_input_string.erase(
                  strpos, strlen(cmd_args.GetArgumentAtIndex(index)));
            }

            if (value_type != OptionParser::eOptionalArgument)
              new_args.AppendArgument(cmd_args.GetArgumentAtIndex(index));
            else {
              char buffer[255];
              ::snprintf(buffer, sizeof(buffer), "%s%s", option.c_str(),
                         cmd_args.GetArgumentAtIndex(index));
              new_args.AppendArgument(buffer);
            }
            used[index] = true;
          }
        }
      }
    }

    for (size_t j = 0; j < cmd_args.GetArgumentCount(); ++j) {
      if (!used[j] && !wants_raw_input)
        new_args.AppendArgument(cmd_args.GetArgumentAtIndex(j));
    }

    cmd_args.Clear();
    cmd_args.SetArguments(new_args.GetArgumentCount(),
                          new_args.GetConstArgumentVector());
  } else {
    result.SetStatus(eReturnStatusSuccessFinishNoResult);
    // This alias was not created with any options; nothing further needs to be
    // done, unless it is a command that
    // wants raw input, in which case we need to clear the rest of the data from
    // cmd_args, since its in the raw
    // input string.
    if (wants_raw_input) {
      cmd_args.Clear();
      cmd_args.SetArguments(new_args.GetArgumentCount(),
                            new_args.GetConstArgumentVector());
    }
    return;
  }

  result.SetStatus(eReturnStatusSuccessFinishNoResult);
  return;
}

int CommandInterpreter::GetOptionArgumentPosition(const char *in_string) {
  int position = 0; // Any string that isn't an argument position, i.e. '%'
                    // followed by an integer, gets a position
                    // of zero.

  const char *cptr = in_string;

  // Does it start with '%'
  if (cptr[0] == '%') {
    ++cptr;

    // Is the rest of it entirely digits?
    if (isdigit(cptr[0])) {
      const char *start = cptr;
      while (isdigit(cptr[0]))
        ++cptr;

      // We've gotten to the end of the digits; are we at the end of the string?
      if (cptr[0] == '\0')
        position = atoi(start);
    }
  }

  return position;
}

void CommandInterpreter::SourceInitFile(bool in_cwd,
                                        CommandReturnObject &result) {
  FileSpec init_file;
  if (in_cwd) {
    ExecutionContext exe_ctx(GetExecutionContext());
    Target *target = exe_ctx.GetTargetPtr();
    if (target) {
      // In the current working directory we don't load any program specific
      // .lldbinit files, we only look for a ".lldbinit" file.
      if (m_skip_lldbinit_files)
        return;

      LoadCWDlldbinitFile should_load =
          target->TargetProperties::GetLoadCWDlldbinitFile();
      if (should_load == eLoadCWDlldbinitWarn) {
        FileSpec dot_lldb(".lldbinit", true);
        llvm::SmallString<64> home_dir_path;
        llvm::sys::path::home_directory(home_dir_path);
        FileSpec homedir_dot_lldb(home_dir_path.c_str(), false);
        homedir_dot_lldb.AppendPathComponent(".lldbinit");
        homedir_dot_lldb.ResolvePath();
        if (dot_lldb.Exists() &&
            dot_lldb.GetDirectory() != homedir_dot_lldb.GetDirectory()) {
          result.AppendErrorWithFormat(
              "There is a .lldbinit file in the current directory which is not "
              "being read.\n"
              "To silence this warning without sourcing in the local "
              ".lldbinit,\n"
              "add the following to the lldbinit file in your home directory:\n"
              "    settings set target.load-cwd-lldbinit false\n"
              "To allow lldb to source .lldbinit files in the current working "
              "directory,\n"
              "set the value of this variable to true.  Only do so if you "
              "understand and\n"
              "accept the security risk.");
          result.SetStatus(eReturnStatusFailed);
          return;
        }
      } else if (should_load == eLoadCWDlldbinitTrue) {
        init_file.SetFile("./.lldbinit", true);
      }
    }
  } else {
    // If we aren't looking in the current working directory we are looking
    // in the home directory. We will first see if there is an application
    // specific ".lldbinit" file whose name is "~/.lldbinit" followed by a
    // "-" and the name of the program. If this file doesn't exist, we fall
    // back to just the "~/.lldbinit" file. We also obey any requests to not
    // load the init files.
    llvm::SmallString<64> home_dir_path;
    llvm::sys::path::home_directory(home_dir_path);
    FileSpec profilePath(home_dir_path.c_str(), false);
    profilePath.AppendPathComponent(".lldbinit");
    std::string init_file_path = profilePath.GetPath();

    if (m_skip_app_init_files == false) {
      FileSpec program_file_spec(HostInfo::GetProgramFileSpec());
      const char *program_name = program_file_spec.GetFilename().AsCString();

      if (program_name) {
        char program_init_file_name[PATH_MAX];
        ::snprintf(program_init_file_name, sizeof(program_init_file_name),
                   "%s-%s", init_file_path.c_str(), program_name);
        init_file.SetFile(program_init_file_name, true);
        if (!init_file.Exists())
          init_file.Clear();
      }
    }

    if (!init_file && !m_skip_lldbinit_files)
      init_file.SetFile(init_file_path.c_str(), false);
  }

  // If the file exists, tell HandleCommand to 'source' it; this will do the
  // actual broadcasting
  // of the commands back to any appropriate listener (see
  // CommandObjectSource::Execute for more details).

  if (init_file.Exists()) {
    const bool saved_batch = SetBatchCommandMode(true);
    CommandInterpreterRunOptions options;
    options.SetSilent(true);
    options.SetStopOnError(false);
    options.SetStopOnContinue(true);

    HandleCommandsFromFile(init_file,
                           nullptr, // Execution context
                           options, result);
    SetBatchCommandMode(saved_batch);
  } else {
    // nothing to be done if the file doesn't exist
    result.SetStatus(eReturnStatusSuccessFinishNoResult);
  }
}

const char *CommandInterpreter::GetCommandPrefix() {
  const char *prefix = GetDebugger().GetIOHandlerCommandPrefix();
  return prefix == NULL ? "" : prefix;
}

PlatformSP CommandInterpreter::GetPlatform(bool prefer_target_platform) {
  PlatformSP platform_sp;
  if (prefer_target_platform) {
    ExecutionContext exe_ctx(GetExecutionContext());
    Target *target = exe_ctx.GetTargetPtr();
    if (target)
      platform_sp = target->GetPlatform();
  }

  if (!platform_sp)
    platform_sp = m_debugger.GetPlatformList().GetSelectedPlatform();
  return platform_sp;
}

void CommandInterpreter::HandleCommands(const StringList &commands,
                                        ExecutionContext *override_context,
                                        CommandInterpreterRunOptions &options,
                                        CommandReturnObject &result) {
  size_t num_lines = commands.GetSize();

  // If we are going to continue past a "continue" then we need to run the
  // commands synchronously.
  // Make sure you reset this value anywhere you return from the function.

  bool old_async_execution = m_debugger.GetAsyncExecution();

  // If we've been given an execution context, set it at the start, but don't
  // keep resetting it or we will
  // cause series of commands that change the context, then do an operation that
  // relies on that context to fail.

  if (override_context != nullptr)
    UpdateExecutionContext(override_context);

  if (!options.GetStopOnContinue()) {
    m_debugger.SetAsyncExecution(false);
  }

  for (size_t idx = 0; idx < num_lines; idx++) {
    const char *cmd = commands.GetStringAtIndex(idx);
    if (cmd[0] == '\0')
      continue;

    if (options.GetEchoCommands()) {
      result.AppendMessageWithFormat("%s %s\n", m_debugger.GetPrompt(), cmd);
    }

    CommandReturnObject tmp_result;
    // If override_context is not NULL, pass no_context_switching = true for
    // HandleCommand() since we updated our context already.

    // We might call into a regex or alias command, in which case the
    // add_to_history will get lost.  This
    // m_command_source_depth dingus is the way we turn off adding to the
    // history in that case, so set it up here.
    if (!options.GetAddToHistory())
      m_command_source_depth++;
    bool success =
        HandleCommand(cmd, options.m_add_to_history, tmp_result,
                      nullptr, /* override_context */
                      true,    /* repeat_on_empty_command */
                      override_context != nullptr /* no_context_switching */);
    if (!options.GetAddToHistory())
      m_command_source_depth--;

    if (options.GetPrintResults()) {
      if (tmp_result.Succeeded())
        result.AppendMessageWithFormat("%s", tmp_result.GetOutputData());
    }

    if (!success || !tmp_result.Succeeded()) {
      const char *error_msg = tmp_result.GetErrorData();
      if (error_msg == nullptr || error_msg[0] == '\0')
        error_msg = "<unknown error>.\n";
      if (options.GetStopOnError()) {
        result.AppendErrorWithFormat(
            "Aborting reading of commands after command #%" PRIu64
            ": '%s' failed with %s",
            (uint64_t)idx, cmd, error_msg);
        result.SetStatus(eReturnStatusFailed);
        m_debugger.SetAsyncExecution(old_async_execution);
        return;
      } else if (options.GetPrintResults()) {
        result.AppendMessageWithFormat("Command #%" PRIu64
                                       " '%s' failed with %s",
                                       (uint64_t)idx + 1, cmd, error_msg);
      }
    }

    if (result.GetImmediateOutputStream())
      result.GetImmediateOutputStream()->Flush();

    if (result.GetImmediateErrorStream())
      result.GetImmediateErrorStream()->Flush();

    // N.B. Can't depend on DidChangeProcessState, because the state coming into
    // the command execution
    // could be running (for instance in Breakpoint Commands.
    // So we check the return value to see if it is has running in it.
    if ((tmp_result.GetStatus() == eReturnStatusSuccessContinuingNoResult) ||
        (tmp_result.GetStatus() == eReturnStatusSuccessContinuingResult)) {
      if (options.GetStopOnContinue()) {
        // If we caused the target to proceed, and we're going to stop in that
        // case, set the
        // status in our real result before returning.  This is an error if the
        // continue was not the
        // last command in the set of commands to be run.
        if (idx != num_lines - 1)
          result.AppendErrorWithFormat(
              "Aborting reading of commands after command #%" PRIu64
              ": '%s' continued the target.\n",
              (uint64_t)idx + 1, cmd);
        else
          result.AppendMessageWithFormat("Command #%" PRIu64
                                         " '%s' continued the target.\n",
                                         (uint64_t)idx + 1, cmd);

        result.SetStatus(tmp_result.GetStatus());
        m_debugger.SetAsyncExecution(old_async_execution);

        return;
      }
    }

    // Also check for "stop on crash here:
    bool should_stop = false;
    if (tmp_result.GetDidChangeProcessState() && options.GetStopOnCrash()) {
      TargetSP target_sp(m_debugger.GetTargetList().GetSelectedTarget());
      if (target_sp) {
        ProcessSP process_sp(target_sp->GetProcessSP());
        if (process_sp) {
          for (ThreadSP thread_sp : process_sp->GetThreadList().Threads()) {
            StopReason reason = thread_sp->GetStopReason();
            if (reason == eStopReasonSignal || reason == eStopReasonException ||
                reason == eStopReasonInstrumentation) {
              should_stop = true;
              break;
            }
          }
        }
      }
      if (should_stop) {
        if (idx != num_lines - 1)
          result.AppendErrorWithFormat(
              "Aborting reading of commands after command #%" PRIu64
              ": '%s' stopped with a signal or exception.\n",
              (uint64_t)idx + 1, cmd);
        else
          result.AppendMessageWithFormat(
              "Command #%" PRIu64 " '%s' stopped with a signal or exception.\n",
              (uint64_t)idx + 1, cmd);

        result.SetStatus(tmp_result.GetStatus());
        m_debugger.SetAsyncExecution(old_async_execution);

        return;
      }
    }
  }

  result.SetStatus(eReturnStatusSuccessFinishResult);
  m_debugger.SetAsyncExecution(old_async_execution);

  return;
}

// Make flags that we can pass into the IOHandler so our delegates can do the
// right thing
enum {
  eHandleCommandFlagStopOnContinue = (1u << 0),
  eHandleCommandFlagStopOnError = (1u << 1),
  eHandleCommandFlagEchoCommand = (1u << 2),
  eHandleCommandFlagPrintResult = (1u << 3),
  eHandleCommandFlagStopOnCrash = (1u << 4)
};

void CommandInterpreter::HandleCommandsFromFile(
    FileSpec &cmd_file, ExecutionContext *context,
    CommandInterpreterRunOptions &options, CommandReturnObject &result) {
  if (cmd_file.Exists()) {
    StreamFileSP input_file_sp(new StreamFile());

    std::string cmd_file_path = cmd_file.GetPath();
    Error error = input_file_sp->GetFile().Open(cmd_file_path.c_str(),
                                                File::eOpenOptionRead);

    if (error.Success()) {
      Debugger &debugger = GetDebugger();

      uint32_t flags = 0;

      if (options.m_stop_on_continue == eLazyBoolCalculate) {
        if (m_command_source_flags.empty()) {
          // Stop on continue by default
          flags |= eHandleCommandFlagStopOnContinue;
        } else if (m_command_source_flags.back() &
                   eHandleCommandFlagStopOnContinue) {
          flags |= eHandleCommandFlagStopOnContinue;
        }
      } else if (options.m_stop_on_continue == eLazyBoolYes) {
        flags |= eHandleCommandFlagStopOnContinue;
      }

      if (options.m_stop_on_error == eLazyBoolCalculate) {
        if (m_command_source_flags.empty()) {
          if (GetStopCmdSourceOnError())
            flags |= eHandleCommandFlagStopOnError;
        } else if (m_command_source_flags.back() &
                   eHandleCommandFlagStopOnError) {
          flags |= eHandleCommandFlagStopOnError;
        }
      } else if (options.m_stop_on_error == eLazyBoolYes) {
        flags |= eHandleCommandFlagStopOnError;
      }

      if (options.GetStopOnCrash()) {
        if (m_command_source_flags.empty()) {
          // Echo command by default
          flags |= eHandleCommandFlagStopOnCrash;
        } else if (m_command_source_flags.back() &
                   eHandleCommandFlagStopOnCrash) {
          flags |= eHandleCommandFlagStopOnCrash;
        }
      }

      if (options.m_echo_commands == eLazyBoolCalculate) {
        if (m_command_source_flags.empty()) {
          // Echo command by default
          flags |= eHandleCommandFlagEchoCommand;
        } else if (m_command_source_flags.back() &
                   eHandleCommandFlagEchoCommand) {
          flags |= eHandleCommandFlagEchoCommand;
        }
      } else if (options.m_echo_commands == eLazyBoolYes) {
        flags |= eHandleCommandFlagEchoCommand;
      }

      if (options.m_print_results == eLazyBoolCalculate) {
        if (m_command_source_flags.empty()) {
          // Print output by default
          flags |= eHandleCommandFlagPrintResult;
        } else if (m_command_source_flags.back() &
                   eHandleCommandFlagPrintResult) {
          flags |= eHandleCommandFlagPrintResult;
        }
      } else if (options.m_print_results == eLazyBoolYes) {
        flags |= eHandleCommandFlagPrintResult;
      }

      if (flags & eHandleCommandFlagPrintResult) {
        debugger.GetOutputFile()->Printf("Executing commands in '%s'.\n",
                                         cmd_file_path.c_str());
      }

      // Used for inheriting the right settings when "command source" might have
      // nested "command source" commands
      lldb::StreamFileSP empty_stream_sp;
      m_command_source_flags.push_back(flags);
      IOHandlerSP io_handler_sp(new IOHandlerEditline(
          debugger, IOHandler::Type::CommandInterpreter, input_file_sp,
          empty_stream_sp, // Pass in an empty stream so we inherit the top
                           // input reader output stream
          empty_stream_sp, // Pass in an empty stream so we inherit the top
                           // input reader error stream
          flags,
          nullptr, // Pass in NULL for "editline_name" so no history is saved,
                   // or written
          debugger.GetPrompt(), NULL,
          false, // Not multi-line
          debugger.GetUseColor(), 0, *this));
      const bool old_async_execution = debugger.GetAsyncExecution();

      // Set synchronous execution if we are not stopping on continue
      if ((flags & eHandleCommandFlagStopOnContinue) == 0)
        debugger.SetAsyncExecution(false);

      m_command_source_depth++;

      debugger.RunIOHandler(io_handler_sp);
      if (!m_command_source_flags.empty())
        m_command_source_flags.pop_back();
      m_command_source_depth--;
      result.SetStatus(eReturnStatusSuccessFinishNoResult);
      debugger.SetAsyncExecution(old_async_execution);
    } else {
      result.AppendErrorWithFormat(
          "error: an error occurred read file '%s': %s\n",
          cmd_file_path.c_str(), error.AsCString());
      result.SetStatus(eReturnStatusFailed);
    }

  } else {
    result.AppendErrorWithFormat(
        "Error reading commands from file %s - file not found.\n",
        cmd_file.GetFilename().AsCString("<Unknown>"));
    result.SetStatus(eReturnStatusFailed);
    return;
  }
}

ScriptInterpreter *CommandInterpreter::GetScriptInterpreter(bool can_create) {
  if (m_script_interpreter_sp)
    return m_script_interpreter_sp.get();

  if (!can_create)
    return nullptr;

  lldb::ScriptLanguage script_lang = GetDebugger().GetScriptLanguage();
  m_script_interpreter_sp =
      PluginManager::GetScriptInterpreterForLanguage(script_lang, *this);
  return m_script_interpreter_sp.get();
}

bool CommandInterpreter::GetSynchronous() { return m_synchronous_execution; }

void CommandInterpreter::SetSynchronous(bool value) {
  m_synchronous_execution = value;
}

void CommandInterpreter::OutputFormattedHelpText(Stream &strm,
                                                 const char *prefix,
                                                 const char *help_text) {
  const uint32_t max_columns = m_debugger.GetTerminalWidth();
  if (prefix == NULL)
    prefix = "";

  size_t prefix_width = strlen(prefix);
  size_t line_width_max = max_columns - prefix_width;
  const char *help_text_end = help_text + strlen(help_text);
  const char *line_start = help_text;
  if (line_width_max < 16)
    line_width_max = help_text_end - help_text + prefix_width;

  strm.IndentMore(prefix_width);
  while (line_start < help_text_end) {
    // Break each line at the first newline or last space/tab before
    // the maximum number of characters that fit on a line.  Lines with no
    // natural break are left unbroken to wrap.
    const char *line_end = help_text_end;
    const char *line_scan = line_start;
    const char *line_scan_end = help_text_end;
    while (line_scan < line_scan_end) {
      char next = *line_scan;
      if (next == '\t' || next == ' ') {
        line_end = line_scan;
        line_scan_end = line_start + line_width_max;
      } else if (next == '\n' || next == '\0') {
        line_end = line_scan;
        break;
      }
      ++line_scan;
    }

    // Prefix the first line, indent subsequent lines to line up
    if (line_start == help_text)
      strm.Write(prefix, prefix_width);
    else
      strm.Indent();
    strm.Write(line_start, line_end - line_start);
    strm.EOL();

    // When a line breaks at whitespace consume it before continuing
    line_start = line_end;
    char next = *line_start;
    if (next == '\n')
      ++line_start;
    else
      while (next == ' ' || next == '\t')
        next = *(++line_start);
  }
  strm.IndentLess(prefix_width);
}

void CommandInterpreter::OutputFormattedHelpText(Stream &strm,
                                                 const char *word_text,
                                                 const char *separator,
                                                 const char *help_text,
                                                 size_t max_word_len) {
  StreamString prefix_stream;
  prefix_stream.Printf("  %-*s %s ", (int)max_word_len, word_text, separator);
  OutputFormattedHelpText(strm, prefix_stream.GetData(), help_text);
}

void CommandInterpreter::OutputHelpText(Stream &strm, const char *word_text,
                                        const char *separator,
                                        const char *help_text,
                                        uint32_t max_word_len) {
  int indent_size = max_word_len + strlen(separator) + 2;

  strm.IndentMore(indent_size);

  StreamString text_strm;
  text_strm.Printf("%-*s %s %s", max_word_len, word_text, separator, help_text);

  const uint32_t max_columns = m_debugger.GetTerminalWidth();

  size_t len = text_strm.GetSize();
  const char *text = text_strm.GetData();

  uint32_t chars_left = max_columns;

  for (uint32_t i = 0; i < len; i++) {
    if ((text[i] == ' ' && ::strchr((text + i + 1), ' ') &&
         chars_left < ::strchr((text + i + 1), ' ') - (text + i)) ||
        text[i] == '\n') {
      chars_left = max_columns - indent_size;
      strm.EOL();
      strm.Indent();
    } else {
      strm.PutChar(text[i]);
      chars_left--;
    }
  }

  strm.EOL();
  strm.IndentLess(indent_size);
}

void CommandInterpreter::FindCommandsForApropos(
    const char *search_word, StringList &commands_found,
    StringList &commands_help, CommandObject::CommandMap &command_map) {
  CommandObject::CommandMap::const_iterator pos;

  for (pos = command_map.begin(); pos != command_map.end(); ++pos) {
    const char *command_name = pos->first.c_str();
    CommandObject *cmd_obj = pos->second.get();

    const bool search_short_help = true;
    const bool search_long_help = false;
    const bool search_syntax = false;
    const bool search_options = false;
    if (strcasestr(command_name, search_word) ||
        cmd_obj->HelpTextContainsWord(search_word, search_short_help,
                                      search_long_help, search_syntax,
                                      search_options)) {
      commands_found.AppendString(cmd_obj->GetCommandName());
      commands_help.AppendString(cmd_obj->GetHelp());
    }

    if (cmd_obj->IsMultiwordObject()) {
      CommandObjectMultiword *cmd_multiword = cmd_obj->GetAsMultiwordCommand();
      FindCommandsForApropos(search_word, commands_found, commands_help,
                             cmd_multiword->GetSubcommandDictionary());
    }
  }
}

void CommandInterpreter::FindCommandsForApropos(const char *search_word,
                                                StringList &commands_found,
                                                StringList &commands_help,
                                                bool search_builtin_commands,
                                                bool search_user_commands,
                                                bool search_alias_commands) {
  CommandObject::CommandMap::const_iterator pos;

  if (search_builtin_commands)
    FindCommandsForApropos(search_word, commands_found, commands_help,
                           m_command_dict);

  if (search_user_commands)
    FindCommandsForApropos(search_word, commands_found, commands_help,
                           m_user_dict);

  if (search_alias_commands)
    FindCommandsForApropos(search_word, commands_found, commands_help,
                           m_alias_dict);
}

void CommandInterpreter::UpdateExecutionContext(
    ExecutionContext *override_context) {
  if (override_context != nullptr) {
    m_exe_ctx_ref = *override_context;
  } else {
    const bool adopt_selected = true;
    m_exe_ctx_ref.SetTargetPtr(m_debugger.GetSelectedTarget().get(),
                               adopt_selected);
  }
}

size_t CommandInterpreter::GetProcessOutput() {
  //  The process has stuff waiting for stderr; get it and write it out to the
  //  appropriate place.
  char stdio_buffer[1024];
  size_t len;
  size_t total_bytes = 0;
  Error error;
  TargetSP target_sp(m_debugger.GetTargetList().GetSelectedTarget());
  if (target_sp) {
    ProcessSP process_sp(target_sp->GetProcessSP());
    if (process_sp) {
      while ((len = process_sp->GetSTDOUT(stdio_buffer, sizeof(stdio_buffer),
                                          error)) > 0) {
        size_t bytes_written = len;
        m_debugger.GetOutputFile()->Write(stdio_buffer, bytes_written);
        total_bytes += len;
      }
      while ((len = process_sp->GetSTDERR(stdio_buffer, sizeof(stdio_buffer),
                                          error)) > 0) {
        size_t bytes_written = len;
        m_debugger.GetErrorFile()->Write(stdio_buffer, bytes_written);
        total_bytes += len;
      }
    }
  }
  return total_bytes;
}

void CommandInterpreter::IOHandlerInputComplete(IOHandler &io_handler,
                                                std::string &line) {
  const bool is_interactive = io_handler.GetIsInteractive();
  if (is_interactive == false) {
    // When we are not interactive, don't execute blank lines. This will happen
    // sourcing a commands file. We don't want blank lines to repeat the
    // previous
    // command and cause any errors to occur (like redefining an alias, get an
    // error
    // and stop parsing the commands file).
    if (line.empty())
      return;

    // When using a non-interactive file handle (like when sourcing commands
    // from a file)
    // we need to echo the command out so we don't just see the command output
    // and no
    // command...
    if (io_handler.GetFlags().Test(eHandleCommandFlagEchoCommand))
      io_handler.GetOutputStreamFile()->Printf("%s%s\n", io_handler.GetPrompt(),
                                               line.c_str());
  }

  lldb_private::CommandReturnObject result;
  HandleCommand(line.c_str(), eLazyBoolCalculate, result);

  // Now emit the command output text from the command we just executed
  if (io_handler.GetFlags().Test(eHandleCommandFlagPrintResult)) {
    // Display any STDOUT/STDERR _prior_ to emitting the command result text
    GetProcessOutput();

    if (!result.GetImmediateOutputStream()) {
      const char *output = result.GetOutputData();
      if (output && output[0])
        io_handler.GetOutputStreamFile()->PutCString(output);
    }

    // Now emit the command error text from the command we just executed
    if (!result.GetImmediateErrorStream()) {
      const char *error = result.GetErrorData();
      if (error && error[0])
        io_handler.GetErrorStreamFile()->PutCString(error);
    }
  }

  switch (result.GetStatus()) {
  case eReturnStatusInvalid:
  case eReturnStatusSuccessFinishNoResult:
  case eReturnStatusSuccessFinishResult:
  case eReturnStatusStarted:
    break;

  case eReturnStatusSuccessContinuingNoResult:
  case eReturnStatusSuccessContinuingResult:
    if (io_handler.GetFlags().Test(eHandleCommandFlagStopOnContinue))
      io_handler.SetIsDone(true);
    break;

  case eReturnStatusFailed:
    m_num_errors++;
    if (io_handler.GetFlags().Test(eHandleCommandFlagStopOnError))
      io_handler.SetIsDone(true);
    break;

  case eReturnStatusQuit:
    m_quit_requested = true;
    io_handler.SetIsDone(true);
    break;
  }

  // Finally, if we're going to stop on crash, check that here:
  if (!m_quit_requested && result.GetDidChangeProcessState() &&
      io_handler.GetFlags().Test(eHandleCommandFlagStopOnCrash)) {
    bool should_stop = false;
    TargetSP target_sp(m_debugger.GetTargetList().GetSelectedTarget());
    if (target_sp) {
      ProcessSP process_sp(target_sp->GetProcessSP());
      if (process_sp) {
        for (ThreadSP thread_sp : process_sp->GetThreadList().Threads()) {
          StopReason reason = thread_sp->GetStopReason();
          if ((reason == eStopReasonSignal || reason == eStopReasonException ||
               reason == eStopReasonInstrumentation) &&
              !result.GetAbnormalStopWasExpected()) {
            should_stop = true;
            break;
          }
        }
      }
    }
    if (should_stop) {
      io_handler.SetIsDone(true);
      m_stopped_for_crash = true;
    }
  }
}

bool CommandInterpreter::IOHandlerInterrupt(IOHandler &io_handler) {
  ExecutionContext exe_ctx(GetExecutionContext());
  Process *process = exe_ctx.GetProcessPtr();

  if (process) {
    StateType state = process->GetState();
    if (StateIsRunningState(state)) {
      process->Halt();
      return true; // Don't do any updating when we are running
    }
  }

  ScriptInterpreter *script_interpreter = GetScriptInterpreter(false);
  if (script_interpreter) {
    if (script_interpreter->Interrupt())
      return true;
  }
  return false;
}

void CommandInterpreter::GetLLDBCommandsFromIOHandler(
    const char *prompt, IOHandlerDelegate &delegate, bool asynchronously,
    void *baton) {
  Debugger &debugger = GetDebugger();
  IOHandlerSP io_handler_sp(
      new IOHandlerEditline(debugger, IOHandler::Type::CommandList,
                            "lldb", // Name of input reader for history
                            prompt, // Prompt
                            NULL,   // Continuation prompt
                            true,   // Get multiple lines
                            debugger.GetUseColor(),
                            0,          // Don't show line numbers
                            delegate)); // IOHandlerDelegate

  if (io_handler_sp) {
    io_handler_sp->SetUserData(baton);
    if (asynchronously)
      debugger.PushIOHandler(io_handler_sp);
    else
      debugger.RunIOHandler(io_handler_sp);
  }
}

void CommandInterpreter::GetPythonCommandsFromIOHandler(
    const char *prompt, IOHandlerDelegate &delegate, bool asynchronously,
    void *baton) {
  Debugger &debugger = GetDebugger();
  IOHandlerSP io_handler_sp(
      new IOHandlerEditline(debugger, IOHandler::Type::PythonCode,
                            "lldb-python", // Name of input reader for history
                            prompt,        // Prompt
                            NULL,          // Continuation prompt
                            true,          // Get multiple lines
                            debugger.GetUseColor(),
                            0,          // Don't show line numbers
                            delegate)); // IOHandlerDelegate

  if (io_handler_sp) {
    io_handler_sp->SetUserData(baton);
    if (asynchronously)
      debugger.PushIOHandler(io_handler_sp);
    else
      debugger.RunIOHandler(io_handler_sp);
  }
}

bool CommandInterpreter::IsActive() {
  return m_debugger.IsTopIOHandler(m_command_io_handler_sp);
}

lldb::IOHandlerSP
CommandInterpreter::GetIOHandler(bool force_create,
                                 CommandInterpreterRunOptions *options) {
  // Always re-create the IOHandlerEditline in case the input
  // changed. The old instance might have had a non-interactive
  // input and now it does or vice versa.
  if (force_create || !m_command_io_handler_sp) {
    // Always re-create the IOHandlerEditline in case the input
    // changed. The old instance might have had a non-interactive
    // input and now it does or vice versa.
    uint32_t flags = 0;

    if (options) {
      if (options->m_stop_on_continue == eLazyBoolYes)
        flags |= eHandleCommandFlagStopOnContinue;
      if (options->m_stop_on_error == eLazyBoolYes)
        flags |= eHandleCommandFlagStopOnError;
      if (options->m_stop_on_crash == eLazyBoolYes)
        flags |= eHandleCommandFlagStopOnCrash;
      if (options->m_echo_commands != eLazyBoolNo)
        flags |= eHandleCommandFlagEchoCommand;
      if (options->m_print_results != eLazyBoolNo)
        flags |= eHandleCommandFlagPrintResult;
    } else {
      flags = eHandleCommandFlagEchoCommand | eHandleCommandFlagPrintResult;
    }

    m_command_io_handler_sp.reset(new IOHandlerEditline(
        m_debugger, IOHandler::Type::CommandInterpreter,
        m_debugger.GetInputFile(), m_debugger.GetOutputFile(),
        m_debugger.GetErrorFile(), flags, "lldb", m_debugger.GetPrompt(),
        NULL,  // Continuation prompt
        false, // Don't enable multiple line input, just single line commands
        m_debugger.GetUseColor(),
        0, // Don't show line numbers
        *this));
  }
  return m_command_io_handler_sp;
}

void CommandInterpreter::RunCommandInterpreter(
    bool auto_handle_events, bool spawn_thread,
    CommandInterpreterRunOptions &options) {
  // Always re-create the command interpreter when we run it in case
  // any file handles have changed.
  bool force_create = true;
  m_debugger.PushIOHandler(GetIOHandler(force_create, &options));
  m_stopped_for_crash = false;

  if (auto_handle_events)
    m_debugger.StartEventHandlerThread();

  if (spawn_thread) {
    m_debugger.StartIOHandlerThread();
  } else {
    m_debugger.ExecuteIOHandlers();

    if (auto_handle_events)
      m_debugger.StopEventHandlerThread();
  }
}

CommandObject *
CommandInterpreter::ResolveCommandImpl(std::string &command_line,
                                       CommandReturnObject &result) {
  std::string scratch_command(command_line); // working copy so we don't modify
                                             // command_line unless we succeed
  CommandObject *cmd_obj = nullptr;
  StreamString revised_command_line;
  bool wants_raw_input = false;
  size_t actual_cmd_name_len = 0;
  std::string next_word;
  StringList matches;
  bool done = false;
  while (!done) {
    char quote_char = '\0';
    std::string suffix;
    ExtractCommand(scratch_command, next_word, suffix, quote_char);
    if (cmd_obj == nullptr) {
      std::string full_name;
      if (GetAliasFullName(next_word.c_str(), full_name)) {
        std::string alias_result;
        cmd_obj = BuildAliasResult(full_name.c_str(), scratch_command,
                                   alias_result, result);
        revised_command_line.Printf("%s", alias_result.c_str());
        if (cmd_obj) {
          wants_raw_input = cmd_obj->WantsRawCommandString();
          actual_cmd_name_len = strlen(cmd_obj->GetCommandName());
        }
      } else {
        cmd_obj = GetCommandObject(next_word.c_str(), &matches);
        if (cmd_obj) {
          actual_cmd_name_len += strlen(cmd_obj->GetCommandName());
          revised_command_line.Printf("%s", cmd_obj->GetCommandName());
          wants_raw_input = cmd_obj->WantsRawCommandString();
        } else {
          revised_command_line.Printf("%s", next_word.c_str());
        }
      }
    } else {
      if (cmd_obj->IsMultiwordObject()) {
        CommandObject *sub_cmd_obj =
            cmd_obj->GetSubcommandObject(next_word.c_str());
        if (sub_cmd_obj) {
          // The subcommand's name includes the parent command's name,
          // so restart rather than append to the revised_command_line.
          actual_cmd_name_len = strlen(sub_cmd_obj->GetCommandName()) + 1;
          revised_command_line.Clear();
          revised_command_line.Printf("%s", sub_cmd_obj->GetCommandName());
          cmd_obj = sub_cmd_obj;
          wants_raw_input = cmd_obj->WantsRawCommandString();
        } else {
          if (quote_char)
            revised_command_line.Printf(" %c%s%s%c", quote_char,
                                        next_word.c_str(), suffix.c_str(),
                                        quote_char);
          else
            revised_command_line.Printf(" %s%s", next_word.c_str(),
                                        suffix.c_str());
          done = true;
        }
      } else {
        if (quote_char)
          revised_command_line.Printf(" %c%s%s%c", quote_char,
                                      next_word.c_str(), suffix.c_str(),
                                      quote_char);
        else
          revised_command_line.Printf(" %s%s", next_word.c_str(),
                                      suffix.c_str());
        done = true;
      }
    }

    if (cmd_obj == nullptr) {
      const size_t num_matches = matches.GetSize();
      if (matches.GetSize() > 1) {
        StreamString error_msg;
        error_msg.Printf("Ambiguous command '%s'. Possible matches:\n",
                         next_word.c_str());

        for (uint32_t i = 0; i < num_matches; ++i) {
          error_msg.Printf("\t%s\n", matches.GetStringAtIndex(i));
        }
        result.AppendRawError(error_msg.GetString().c_str());
      } else {
        // We didn't have only one match, otherwise we wouldn't get here.
        assert(num_matches == 0);
        result.AppendErrorWithFormat("'%s' is not a valid command.\n",
                                     next_word.c_str());
      }
      result.SetStatus(eReturnStatusFailed);
      return nullptr;
    }

    if (cmd_obj->IsMultiwordObject()) {
      if (!suffix.empty()) {
        result.AppendErrorWithFormat(
            "command '%s' did not recognize '%s%s%s' as valid (subcommand "
            "might be invalid).\n",
            cmd_obj->GetCommandName(),
            next_word.empty() ? "" : next_word.c_str(),
            next_word.empty() ? " -- " : " ", suffix.c_str());
        result.SetStatus(eReturnStatusFailed);
        return nullptr;
      }
    } else {
      // If we found a normal command, we are done
      done = true;
      if (!suffix.empty()) {
        switch (suffix[0]) {
        case '/':
          // GDB format suffixes
          {
            Options *command_options = cmd_obj->GetOptions();
            if (command_options &&
                command_options->SupportsLongOption("gdb-format")) {
              std::string gdb_format_option("--gdb-format=");
              gdb_format_option += (suffix.c_str() + 1);

              bool inserted = false;
              std::string &cmd = revised_command_line.GetString();
              size_t arg_terminator_idx = FindArgumentTerminator(cmd);
              if (arg_terminator_idx != std::string::npos) {
                // Insert the gdb format option before the "--" that terminates
                // options
                gdb_format_option.append(1, ' ');
                cmd.insert(arg_terminator_idx, gdb_format_option);
                inserted = true;
              }

              if (!inserted)
                revised_command_line.Printf(" %s", gdb_format_option.c_str());

              if (wants_raw_input &&
                  FindArgumentTerminator(cmd) == std::string::npos)
                revised_command_line.PutCString(" --");
            } else {
              result.AppendErrorWithFormat(
                  "the '%s' command doesn't support the --gdb-format option\n",
                  cmd_obj->GetCommandName());
              result.SetStatus(eReturnStatusFailed);
              return nullptr;
            }
          }
          break;

        default:
          result.AppendErrorWithFormat(
              "unknown command shorthand suffix: '%s'\n", suffix.c_str());
          result.SetStatus(eReturnStatusFailed);
          return nullptr;
        }
      }
    }
    if (scratch_command.empty())
      done = true;
  }

  if (!scratch_command.empty())
    revised_command_line.Printf(" %s", scratch_command.c_str());

  if (cmd_obj != NULL)
    command_line = revised_command_line.GetData();

  return cmd_obj;
}
