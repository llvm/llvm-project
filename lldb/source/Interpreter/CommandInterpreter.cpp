//===-- CommandInterpreter.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <string>
#include <vector>

#include <getopt.h>
#include <stdlib.h>

#include "CommandObjectScript.h"
#include "lldb/Interpreter/CommandObjectRegexCommand.h"

#include "../Commands/CommandObjectApropos.h"
#include "../Commands/CommandObjectArgs.h"
#include "../Commands/CommandObjectBreakpoint.h"
#include "../Commands/CommandObjectDisassemble.h"
#include "../Commands/CommandObjectExpression.h"
#include "../Commands/CommandObjectFrame.h"
#include "../Commands/CommandObjectHelp.h"
#include "../Commands/CommandObjectLog.h"
#include "../Commands/CommandObjectMemory.h"
#include "../Commands/CommandObjectPlatform.h"
#include "../Commands/CommandObjectPlugin.h"
#include "../Commands/CommandObjectProcess.h"
#include "../Commands/CommandObjectQuit.h"
#include "../Commands/CommandObjectRegister.h"
#include "../Commands/CommandObjectSettings.h"
#include "../Commands/CommandObjectSource.h"
#include "../Commands/CommandObjectCommands.h"
#include "../Commands/CommandObjectSyntax.h"
#include "../Commands/CommandObjectTarget.h"
#include "../Commands/CommandObjectThread.h"
#include "../Commands/CommandObjectType.h"
#include "../Commands/CommandObjectVersion.h"
#include "../Commands/CommandObjectWatchpoint.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/InputReader.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/Timer.h"

#include "lldb/Host/Host.h"

#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/ScriptInterpreterNone.h"
#include "lldb/Interpreter/ScriptInterpreterPython.h"


#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/TargetList.h"

#include "lldb/Utility/CleanUp.h"

using namespace lldb;
using namespace lldb_private;


static PropertyDefinition
g_properties[] =
{
    { "expand-regex-aliases", OptionValue::eTypeBoolean, true, false, NULL, NULL, "If true, regular expression alias commands will show the expanded command that will be executed. This can be used to debug new regular expression alias commands." },
    { NULL                  , OptionValue::eTypeInvalid, true, 0    , NULL, NULL, NULL }
};

enum
{
    ePropertyExpandRegexAliases = 0
};

ConstString &
CommandInterpreter::GetStaticBroadcasterClass ()
{
    static ConstString class_name ("lldb.commandInterpreter");
    return class_name;
}

CommandInterpreter::CommandInterpreter
(
    Debugger &debugger,
    ScriptLanguage script_language,
    bool synchronous_execution
) :
    Broadcaster (&debugger, "lldb.command-interpreter"),
    Properties(OptionValuePropertiesSP(new OptionValueProperties(ConstString("interpreter")))),
    m_debugger (debugger),
    m_synchronous_execution (synchronous_execution),
    m_skip_lldbinit_files (false),
    m_skip_app_init_files (false),
    m_script_interpreter_ap (),
    m_comment_char ('#'),
    m_repeat_char ('!'),
    m_batch_command_mode (false),
    m_truncation_warning(eNoTruncation),
    m_command_source_depth (0)
{
    debugger.SetScriptLanguage (script_language);
    SetEventName (eBroadcastBitThreadShouldExit, "thread-should-exit");
    SetEventName (eBroadcastBitResetPrompt, "reset-prompt");
    SetEventName (eBroadcastBitQuitCommandReceived, "quit");    
    CheckInWithManager ();
    m_collection_sp->Initialize (g_properties);
}

bool
CommandInterpreter::GetExpandRegexAliases () const
{
    const uint32_t idx = ePropertyExpandRegexAliases;
    return m_collection_sp->GetPropertyAtIndexAsBoolean (NULL, idx, g_properties[idx].default_uint_value != 0);
}



void
CommandInterpreter::Initialize ()
{
    Timer scoped_timer (__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);

    CommandReturnObject result;

    LoadCommandDictionary ();

    // Set up some initial aliases.
    CommandObjectSP cmd_obj_sp = GetCommandSPExact ("quit", false);
    if (cmd_obj_sp)
    {
        AddAlias ("q", cmd_obj_sp);
        AddAlias ("exit", cmd_obj_sp);
    }
    
    cmd_obj_sp = GetCommandSPExact ("_regexp-attach",false);
    if (cmd_obj_sp)
    {
        AddAlias ("attach", cmd_obj_sp);
    }

    cmd_obj_sp = GetCommandSPExact ("process detach",false);
    if (cmd_obj_sp)
    {
        AddAlias ("detach", cmd_obj_sp);
    }

    cmd_obj_sp = GetCommandSPExact ("process continue", false);
    if (cmd_obj_sp)
    {
        AddAlias ("c", cmd_obj_sp);
        AddAlias ("continue", cmd_obj_sp);
    }

    cmd_obj_sp = GetCommandSPExact ("_regexp-break",false);
    if (cmd_obj_sp)
        AddAlias ("b", cmd_obj_sp);

    cmd_obj_sp = GetCommandSPExact ("_regexp-tbreak",false);
    if (cmd_obj_sp)
        AddAlias ("tbreak", cmd_obj_sp);

    cmd_obj_sp = GetCommandSPExact ("thread step-inst", false);
    if (cmd_obj_sp)
    {
        AddAlias ("stepi", cmd_obj_sp);
        AddAlias ("si", cmd_obj_sp);
    }

    cmd_obj_sp = GetCommandSPExact ("thread step-inst-over", false);
    if (cmd_obj_sp)
    {
        AddAlias ("nexti", cmd_obj_sp);
        AddAlias ("ni", cmd_obj_sp);
    }

    cmd_obj_sp = GetCommandSPExact ("thread step-in", false);
    if (cmd_obj_sp)
    {
        AddAlias ("s", cmd_obj_sp);
        AddAlias ("step", cmd_obj_sp);
    }

    cmd_obj_sp = GetCommandSPExact ("thread step-over", false);
    if (cmd_obj_sp)
    {
        AddAlias ("n", cmd_obj_sp);
        AddAlias ("next", cmd_obj_sp);
    }

    cmd_obj_sp = GetCommandSPExact ("thread step-out", false);
    if (cmd_obj_sp)
    {
        AddAlias ("finish", cmd_obj_sp);
    }

    cmd_obj_sp = GetCommandSPExact ("frame select", false);
    if (cmd_obj_sp)
    {
        AddAlias ("f", cmd_obj_sp);
    }

    cmd_obj_sp = GetCommandSPExact ("thread select", false);
    if (cmd_obj_sp)
    {
        AddAlias ("t", cmd_obj_sp);
    }

    cmd_obj_sp = GetCommandSPExact ("source list", false);
    if (cmd_obj_sp)
    {
        AddAlias ("l", cmd_obj_sp);
        AddAlias ("list", cmd_obj_sp);
    }

    cmd_obj_sp = GetCommandSPExact ("memory read", false);
    if (cmd_obj_sp)
        AddAlias ("x", cmd_obj_sp);

    cmd_obj_sp = GetCommandSPExact ("_regexp-up", false);
    if (cmd_obj_sp)
        AddAlias ("up", cmd_obj_sp);

    cmd_obj_sp = GetCommandSPExact ("_regexp-down", false);
    if (cmd_obj_sp)
        AddAlias ("down", cmd_obj_sp);

    cmd_obj_sp = GetCommandSPExact ("_regexp-display", false);
    if (cmd_obj_sp)
        AddAlias ("display", cmd_obj_sp);
        
    cmd_obj_sp = GetCommandSPExact ("disassemble", false);
    if (cmd_obj_sp)
        AddAlias ("dis", cmd_obj_sp);

    cmd_obj_sp = GetCommandSPExact ("disassemble", false);
    if (cmd_obj_sp)
        AddAlias ("di", cmd_obj_sp);



    cmd_obj_sp = GetCommandSPExact ("_regexp-undisplay", false);
    if (cmd_obj_sp)
        AddAlias ("undisplay", cmd_obj_sp);

    cmd_obj_sp = GetCommandSPExact ("_regexp-bt", false);
    if (cmd_obj_sp)
        AddAlias ("bt", cmd_obj_sp);

    cmd_obj_sp = GetCommandSPExact ("target create", false);
    if (cmd_obj_sp)
        AddAlias ("file", cmd_obj_sp);

    cmd_obj_sp = GetCommandSPExact ("target modules", false);
    if (cmd_obj_sp)
        AddAlias ("image", cmd_obj_sp);


    OptionArgVectorSP alias_arguments_vector_sp (new OptionArgVector);
    
    cmd_obj_sp = GetCommandSPExact ("expression", false);
    if (cmd_obj_sp)
    {
        AddAlias ("expr", cmd_obj_sp);
        
        ProcessAliasOptionsArgs (cmd_obj_sp, "--", alias_arguments_vector_sp);
        AddAlias ("p", cmd_obj_sp);
        AddAlias ("print", cmd_obj_sp);
        AddAlias ("call", cmd_obj_sp);
        AddOrReplaceAliasOptions ("p", alias_arguments_vector_sp);
        AddOrReplaceAliasOptions ("print", alias_arguments_vector_sp);
        AddOrReplaceAliasOptions ("call", alias_arguments_vector_sp);

        alias_arguments_vector_sp.reset (new OptionArgVector);
        ProcessAliasOptionsArgs (cmd_obj_sp, "-o --", alias_arguments_vector_sp);
        AddAlias ("po", cmd_obj_sp);
        AddOrReplaceAliasOptions ("po", alias_arguments_vector_sp);
    }
    
    cmd_obj_sp = GetCommandSPExact ("process kill", false);
    if (cmd_obj_sp)
    {
        AddAlias ("kill", cmd_obj_sp);
    }
    
    cmd_obj_sp = GetCommandSPExact ("process launch", false);
    if (cmd_obj_sp)
    {
        alias_arguments_vector_sp.reset (new OptionArgVector);
#if defined (__arm__)
        ProcessAliasOptionsArgs (cmd_obj_sp, "--", alias_arguments_vector_sp);
#else
        ProcessAliasOptionsArgs (cmd_obj_sp, "--shell=/bin/bash --", alias_arguments_vector_sp);
#endif
        AddAlias ("r", cmd_obj_sp);
        AddAlias ("run", cmd_obj_sp);
        AddOrReplaceAliasOptions ("r", alias_arguments_vector_sp);
        AddOrReplaceAliasOptions ("run", alias_arguments_vector_sp);
    }
    
    cmd_obj_sp = GetCommandSPExact ("target symbols add", false);
    if (cmd_obj_sp)
    {
        AddAlias ("add-dsym", cmd_obj_sp);
    }
    
    cmd_obj_sp = GetCommandSPExact ("breakpoint set", false);
    if (cmd_obj_sp)
    {
        alias_arguments_vector_sp.reset (new OptionArgVector);
        ProcessAliasOptionsArgs (cmd_obj_sp, "--func-regex %1", alias_arguments_vector_sp);
        AddAlias ("rbreak", cmd_obj_sp);
        AddOrReplaceAliasOptions("rbreak", alias_arguments_vector_sp);
    }
}

const char *
CommandInterpreter::ProcessEmbeddedScriptCommands (const char *arg)
{
    // This function has not yet been implemented.

    // Look for any embedded script command
    // If found,
    //    get interpreter object from the command dictionary,
    //    call execute_one_command on it,
    //    get the results as a string,
    //    substitute that string for current stuff.

    return arg;
}


void
CommandInterpreter::LoadCommandDictionary ()
{
    Timer scoped_timer (__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);

    // **** IMPORTANT **** IMPORTANT *** IMPORTANT *** **** IMPORTANT **** IMPORTANT *** IMPORTANT ***
    //
    // Command objects that are used as cross reference objects (i.e. they inherit from CommandObjectCrossref)
    // *MUST* be created and put into the command dictionary *BEFORE* any multi-word commands (which may use
    // the cross-referencing stuff) are created!!!
    //
    // **** IMPORTANT **** IMPORTANT *** IMPORTANT *** **** IMPORTANT **** IMPORTANT *** IMPORTANT ***


    // Command objects that inherit from CommandObjectCrossref must be created before other command objects
    // are created.  This is so that when another command is created that needs to go into a crossref object,
    // the crossref object exists and is ready to take the cross reference. Put the cross referencing command
    // objects into the CommandDictionary now, so they are ready for use when the other commands get created.

    // Non-CommandObjectCrossref commands can now be created.

    lldb::ScriptLanguage script_language = m_debugger.GetScriptLanguage();
    
    m_command_dict["apropos"]   = CommandObjectSP (new CommandObjectApropos (*this));
    m_command_dict["breakpoint"]= CommandObjectSP (new CommandObjectMultiwordBreakpoint (*this));
    //m_command_dict["call"]      = CommandObjectSP (new CommandObjectCall (*this));
    m_command_dict["command"]   = CommandObjectSP (new CommandObjectMultiwordCommands (*this));
    m_command_dict["disassemble"] = CommandObjectSP (new CommandObjectDisassemble (*this));
    m_command_dict["expression"]= CommandObjectSP (new CommandObjectExpression (*this));
//    m_command_dict["file"]      = CommandObjectSP (new CommandObjectFile (*this));
    m_command_dict["frame"]     = CommandObjectSP (new CommandObjectMultiwordFrame (*this));
    m_command_dict["help"]      = CommandObjectSP (new CommandObjectHelp (*this));
    ///    m_command_dict["image"]     = CommandObjectSP (new CommandObjectImage (*this));
    m_command_dict["log"]       = CommandObjectSP (new CommandObjectLog (*this));
    m_command_dict["memory"]    = CommandObjectSP (new CommandObjectMemory (*this));
    m_command_dict["platform"]  = CommandObjectSP (new CommandObjectPlatform (*this));
    m_command_dict["plugin"]    = CommandObjectSP (new CommandObjectPlugin (*this));
    m_command_dict["process"]   = CommandObjectSP (new CommandObjectMultiwordProcess (*this));
    m_command_dict["quit"]      = CommandObjectSP (new CommandObjectQuit (*this));
    m_command_dict["register"]  = CommandObjectSP (new CommandObjectRegister (*this));
    m_command_dict["script"]    = CommandObjectSP (new CommandObjectScript (*this, script_language));
    m_command_dict["settings"]  = CommandObjectSP (new CommandObjectMultiwordSettings (*this));
    m_command_dict["source"]    = CommandObjectSP (new CommandObjectMultiwordSource (*this));
    m_command_dict["target"]    = CommandObjectSP (new CommandObjectMultiwordTarget (*this));
    m_command_dict["thread"]    = CommandObjectSP (new CommandObjectMultiwordThread (*this));
    m_command_dict["type"]      = CommandObjectSP (new CommandObjectType (*this));
    m_command_dict["version"]   = CommandObjectSP (new CommandObjectVersion (*this));
    m_command_dict["watchpoint"]= CommandObjectSP (new CommandObjectMultiwordWatchpoint (*this));

    const char *break_regexes[][2] = {{"^(.*[^[:space:]])[[:space:]]*:[[:space:]]*([[:digit:]]+)[[:space:]]*$", "breakpoint set --file '%1' --line %2"},
                                      {"^([[:digit:]]+)[[:space:]]*$", "breakpoint set --line %1"},
                                      {"^(0x[[:xdigit:]]+)[[:space:]]*$", "breakpoint set --address %1"},
                                      {"^[\"']?([-+]\\[.*\\])[\"']?[[:space:]]*$", "breakpoint set --name '%1'"},
                                      {"^(-.*)$", "breakpoint set %1"},
                                      {"^(.*[^[:space:]])`(.*[^[:space:]])[[:space:]]*$", "breakpoint set --name '%2' --shlib '%1'"},
                                      {"^(.*[^[:space:]])[[:space:]]*$", "breakpoint set --name '%1'"}};
    
    size_t num_regexes = sizeof break_regexes/sizeof(char *[2]);
        
    std::auto_ptr<CommandObjectRegexCommand>
    break_regex_cmd_ap(new CommandObjectRegexCommand (*this,
                                                      "_regexp-break",
                                                      "Set a breakpoint using a regular expression to specify the location, where <linenum> is in decimal and <address> is in hex.",
                                                      "_regexp-break [<filename>:<linenum>]\n_regexp-break [<linenum>]\n_regexp-break [<address>]\n_regexp-break <...>", 2));

    if (break_regex_cmd_ap.get())
    {
        bool success = true;
        for (size_t i = 0; i < num_regexes; i++)
        {
            success = break_regex_cmd_ap->AddRegexCommand (break_regexes[i][0], break_regexes[i][1]);
            if (!success)
                break;
        }
        success = break_regex_cmd_ap->AddRegexCommand("^$", "breakpoint list --full");

        if (success)
        {
            CommandObjectSP break_regex_cmd_sp(break_regex_cmd_ap.release());
            m_command_dict[break_regex_cmd_sp->GetCommandName ()] = break_regex_cmd_sp;
        }
    }

    std::auto_ptr<CommandObjectRegexCommand>
    tbreak_regex_cmd_ap(new CommandObjectRegexCommand (*this,
                                                      "_regexp-tbreak",
                                                      "Set a one shot breakpoint using a regular expression to specify the location, where <linenum> is in decimal and <address> is in hex.",
                                                      "_regexp-tbreak [<filename>:<linenum>]\n_regexp-break [<linenum>]\n_regexp-break [<address>]\n_regexp-break <...>", 2));

    if (tbreak_regex_cmd_ap.get())
    {
        bool success = true;
        for (size_t i = 0; i < num_regexes; i++)
        {
            // If you add a resultant command string longer than 1024 characters be sure to increase the size of this buffer.
            char buffer[1024];
            int num_printed = snprintf(buffer, 1024, "%s %s", break_regexes[i][1], "-o");
            assert (num_printed < 1024);
            success = tbreak_regex_cmd_ap->AddRegexCommand (break_regexes[i][0], buffer);
            if (!success)
                break;
        }
        success = tbreak_regex_cmd_ap->AddRegexCommand("^$", "breakpoint list --full");

        if (success)
        {
            CommandObjectSP tbreak_regex_cmd_sp(tbreak_regex_cmd_ap.release());
            m_command_dict[tbreak_regex_cmd_sp->GetCommandName ()] = tbreak_regex_cmd_sp;
        }
    }

    std::auto_ptr<CommandObjectRegexCommand>
    attach_regex_cmd_ap(new CommandObjectRegexCommand (*this,
                                                       "_regexp-attach",
                                                       "Attach to a process id if in decimal, otherwise treat the argument as a process name to attach to.",
                                                       "_regexp-attach [<pid>]\n_regexp-attach [<process-name>]", 2));
    if (attach_regex_cmd_ap.get())
    {
        if (attach_regex_cmd_ap->AddRegexCommand("^([0-9]+)$", "process attach --pid %1") &&
            attach_regex_cmd_ap->AddRegexCommand("^(.*[^[:space:]])[[:space:]]*$", "process attach --name '%1'"))
        {
            CommandObjectSP attach_regex_cmd_sp(attach_regex_cmd_ap.release());
            m_command_dict[attach_regex_cmd_sp->GetCommandName ()] = attach_regex_cmd_sp;
        }
    }
    
    std::auto_ptr<CommandObjectRegexCommand>
    down_regex_cmd_ap(new CommandObjectRegexCommand (*this,
                                                     "_regexp-down",
                                                     "Go down \"n\" frames in the stack (1 frame by default).",
                                                     "_regexp-down [n]", 2));
    if (down_regex_cmd_ap.get())
    {
        if (down_regex_cmd_ap->AddRegexCommand("^$", "frame select -r -1") &&
            down_regex_cmd_ap->AddRegexCommand("^([0-9]+)$", "frame select -r -%1"))
        {
            CommandObjectSP down_regex_cmd_sp(down_regex_cmd_ap.release());
            m_command_dict[down_regex_cmd_sp->GetCommandName ()] = down_regex_cmd_sp;
        }
    }
    
    std::auto_ptr<CommandObjectRegexCommand>
    up_regex_cmd_ap(new CommandObjectRegexCommand (*this,
                                                   "_regexp-up",
                                                   "Go up \"n\" frames in the stack (1 frame by default).",
                                                   "_regexp-up [n]", 2));
    if (up_regex_cmd_ap.get())
    {
        if (up_regex_cmd_ap->AddRegexCommand("^$", "frame select -r 1") &&
            up_regex_cmd_ap->AddRegexCommand("^([0-9]+)$", "frame select -r %1"))
        {
            CommandObjectSP up_regex_cmd_sp(up_regex_cmd_ap.release());
            m_command_dict[up_regex_cmd_sp->GetCommandName ()] = up_regex_cmd_sp;
        }
    }

    std::auto_ptr<CommandObjectRegexCommand>
    display_regex_cmd_ap(new CommandObjectRegexCommand (*this,
                                                   "_regexp-display",
                                                   "Add an expression evaluation stop-hook.",
                                                   "_regexp-display expression", 2));
    if (display_regex_cmd_ap.get())
    {
        if (display_regex_cmd_ap->AddRegexCommand("^(.+)$", "target stop-hook add -o \"expr -- %1\""))
        {
            CommandObjectSP display_regex_cmd_sp(display_regex_cmd_ap.release());
            m_command_dict[display_regex_cmd_sp->GetCommandName ()] = display_regex_cmd_sp;
        }
    }

    std::auto_ptr<CommandObjectRegexCommand>
    undisplay_regex_cmd_ap(new CommandObjectRegexCommand (*this,
                                                   "_regexp-undisplay",
                                                   "Remove an expression evaluation stop-hook.",
                                                   "_regexp-undisplay stop-hook-number", 2));
    if (undisplay_regex_cmd_ap.get())
    {
        if (undisplay_regex_cmd_ap->AddRegexCommand("^([0-9]+)$", "target stop-hook delete %1"))
        {
            CommandObjectSP undisplay_regex_cmd_sp(undisplay_regex_cmd_ap.release());
            m_command_dict[undisplay_regex_cmd_sp->GetCommandName ()] = undisplay_regex_cmd_sp;
        }
    }

    std::auto_ptr<CommandObjectRegexCommand>
    connect_gdb_remote_cmd_ap(new CommandObjectRegexCommand (*this,
                                                      "gdb-remote",
                                                      "Connect to a remote GDB server.  If no hostname is provided, localhost is assumed.",
                                                      "gdb-remote [<hostname>:]<portnum>", 2));
    if (connect_gdb_remote_cmd_ap.get())
    {
        if (connect_gdb_remote_cmd_ap->AddRegexCommand("^([^:]+:[[:digit:]]+)$", "process connect --plugin gdb-remote connect://%1") &&
            connect_gdb_remote_cmd_ap->AddRegexCommand("^([[:digit:]]+)$", "process connect --plugin gdb-remote connect://localhost:%1"))
        {
            CommandObjectSP command_sp(connect_gdb_remote_cmd_ap.release());
            m_command_dict[command_sp->GetCommandName ()] = command_sp;
        }
    }

    std::auto_ptr<CommandObjectRegexCommand>
    connect_kdp_remote_cmd_ap(new CommandObjectRegexCommand (*this,
                                                             "kdp-remote",
                                                             "Connect to a remote KDP server.  udp port 41139 is the default port number.",
                                                             "kdp-remote <hostname>[:<portnum>]", 2));
    if (connect_kdp_remote_cmd_ap.get())
    {
        if (connect_kdp_remote_cmd_ap->AddRegexCommand("^([^:]+:[[:digit:]]+)$", "process connect --plugin kdp-remote udp://%1") &&
            connect_kdp_remote_cmd_ap->AddRegexCommand("^(.+)$", "process connect --plugin kdp-remote udp://%1:41139"))
        {
            CommandObjectSP command_sp(connect_kdp_remote_cmd_ap.release());
            m_command_dict[command_sp->GetCommandName ()] = command_sp;
        }
    }

    std::auto_ptr<CommandObjectRegexCommand>
    bt_regex_cmd_ap(new CommandObjectRegexCommand (*this,
                                                     "_regexp-bt",
                                                     "Show a backtrace.  An optional argument is accepted; if that argument is a number, it specifies the number of frames to display.  If that argument is 'all', full backtraces of all threads are displayed.",
                                                     "bt [<digit>|all]", 2));
    if (bt_regex_cmd_ap.get())
    {
        // accept but don't document "bt -c <number>" -- before bt was a regex command if you wanted to backtrace
        // three frames you would do "bt -c 3" but the intention is to have this emulate the gdb "bt" command and
        // so now "bt 3" is the preferred form, in line with gdb.
        if (bt_regex_cmd_ap->AddRegexCommand("^([[:digit:]]+)$", "thread backtrace -c %1") &&
            bt_regex_cmd_ap->AddRegexCommand("^-c ([[:digit:]]+)$", "thread backtrace -c %1") &&
            bt_regex_cmd_ap->AddRegexCommand("^all$", "thread backtrace all") &&
            bt_regex_cmd_ap->AddRegexCommand("^$", "thread backtrace"))
        {
            CommandObjectSP command_sp(bt_regex_cmd_ap.release());
            m_command_dict[command_sp->GetCommandName ()] = command_sp;
        }
    }

}

int
CommandInterpreter::GetCommandNamesMatchingPartialString (const char *cmd_str, bool include_aliases,
                                                          StringList &matches)
{
    CommandObject::AddNamesMatchingPartialString (m_command_dict, cmd_str, matches);

    if (include_aliases)
    {
        CommandObject::AddNamesMatchingPartialString (m_alias_dict, cmd_str, matches);
    }

    return matches.GetSize();
}

CommandObjectSP
CommandInterpreter::GetCommandSP (const char *cmd_cstr, bool include_aliases, bool exact, StringList *matches)
{
    CommandObject::CommandMap::iterator pos;
    CommandObjectSP ret_val;

    std::string cmd(cmd_cstr);

    if (HasCommands())
    {
        pos = m_command_dict.find(cmd);
        if (pos != m_command_dict.end())
            ret_val = pos->second;
    }

    if (include_aliases && HasAliases())
    {
        pos = m_alias_dict.find(cmd);
        if (pos != m_alias_dict.end())
            ret_val = pos->second;
    }

    if (HasUserCommands())
    {
        pos = m_user_dict.find(cmd);
        if (pos != m_user_dict.end())
            ret_val = pos->second;
    }

    if (!exact && !ret_val)
    {
        // We will only get into here if we didn't find any exact matches.
        
        CommandObjectSP user_match_sp, alias_match_sp, real_match_sp;

        StringList local_matches;
        if (matches == NULL)
            matches = &local_matches;

        unsigned int num_cmd_matches = 0;
        unsigned int num_alias_matches = 0;
        unsigned int num_user_matches = 0;
        
        // Look through the command dictionaries one by one, and if we get only one match from any of
        // them in toto, then return that, otherwise return an empty CommandObjectSP and the list of matches.
        
        if (HasCommands())
        {
            num_cmd_matches = CommandObject::AddNamesMatchingPartialString (m_command_dict, cmd_cstr, *matches);
        }

        if (num_cmd_matches == 1)
        {
            cmd.assign(matches->GetStringAtIndex(0));
            pos = m_command_dict.find(cmd);
            if (pos != m_command_dict.end())
                real_match_sp = pos->second;
        }

        if (include_aliases && HasAliases())
        {
            num_alias_matches = CommandObject::AddNamesMatchingPartialString (m_alias_dict, cmd_cstr, *matches);

        }

        if (num_alias_matches == 1)
        {
            cmd.assign(matches->GetStringAtIndex (num_cmd_matches));
            pos = m_alias_dict.find(cmd);
            if (pos != m_alias_dict.end())
                alias_match_sp = pos->second;
        }

        if (HasUserCommands())
        {
            num_user_matches = CommandObject::AddNamesMatchingPartialString (m_user_dict, cmd_cstr, *matches);
        }

        if (num_user_matches == 1)
        {
            cmd.assign (matches->GetStringAtIndex (num_cmd_matches + num_alias_matches));

            pos = m_user_dict.find (cmd);
            if (pos != m_user_dict.end())
                user_match_sp = pos->second;
        }
        
        // If we got exactly one match, return that, otherwise return the match list.
        
        if (num_user_matches + num_cmd_matches + num_alias_matches == 1)
        {
            if (num_cmd_matches)
                return real_match_sp;
            else if (num_alias_matches)
                return alias_match_sp;
            else
                return user_match_sp;
        }
    }
    else if (matches && ret_val)
    {
        matches->AppendString (cmd_cstr);
    }


    return ret_val;
}

bool
CommandInterpreter::AddCommand (const char *name, const lldb::CommandObjectSP &cmd_sp, bool can_replace)
{
    if (name && name[0])
    {
        std::string name_sstr(name);
        bool found = (m_command_dict.find (name_sstr) != m_command_dict.end());
        if (found && !can_replace)
            return false;
        if (found && m_command_dict[name_sstr]->IsRemovable() == false)
            return false;
        m_command_dict[name_sstr] = cmd_sp;
        return true;
    }
    return false;
}

bool
CommandInterpreter::AddUserCommand (std::string name, 
                                    const lldb::CommandObjectSP &cmd_sp,
                                    bool can_replace)
{
    if (!name.empty())
    {
        
        const char* name_cstr = name.c_str();
        
        // do not allow replacement of internal commands
        if (CommandExists(name_cstr))
        {
            if (can_replace == false)
                return false;
            if (m_command_dict[name]->IsRemovable() == false)
                return false;
        }
        
        if (UserCommandExists(name_cstr))
        {
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

CommandObjectSP
CommandInterpreter::GetCommandSPExact (const char *cmd_cstr, bool include_aliases)
{
    Args cmd_words (cmd_cstr); // Break up the command string into words, in case it's a multi-word command.
    CommandObjectSP ret_val;   // Possibly empty return value.
    
    if (cmd_cstr == NULL)
        return ret_val;
    
    if (cmd_words.GetArgumentCount() == 1)
        return GetCommandSP(cmd_cstr, include_aliases, true, NULL);
    else
    {
        // We have a multi-word command (seemingly), so we need to do more work.
        // First, get the cmd_obj_sp for the first word in the command.
        CommandObjectSP cmd_obj_sp = GetCommandSP (cmd_words.GetArgumentAtIndex (0), include_aliases, true, NULL);
        if (cmd_obj_sp.get() != NULL)
        {
            // Loop through the rest of the words in the command (everything passed in was supposed to be part of a 
            // command name), and find the appropriate sub-command SP for each command word....
            size_t end = cmd_words.GetArgumentCount();
            for (size_t j= 1; j < end; ++j)
            {
                if (cmd_obj_sp->IsMultiwordObject())
                {
                    cmd_obj_sp = cmd_obj_sp->GetSubcommandSP (cmd_words.GetArgumentAtIndex (j));
                    if (cmd_obj_sp.get() == NULL)
                        // The sub-command name was invalid.  Fail and return the empty 'ret_val'.
                        return ret_val;
                }
                else
                    // We have more words in the command name, but we don't have a multiword object. Fail and return 
                    // empty 'ret_val'.
                    return ret_val;
            }
            // We successfully looped through all the command words and got valid command objects for them.  Assign the 
            // last object retrieved to 'ret_val'.
            ret_val = cmd_obj_sp;
        }
    }
    return ret_val;
}

CommandObject *
CommandInterpreter::GetCommandObjectExact (const char *cmd_cstr, bool include_aliases)
{
    return GetCommandSPExact (cmd_cstr, include_aliases).get();
}

CommandObject *
CommandInterpreter::GetCommandObject (const char *cmd_cstr, StringList *matches)
{
    CommandObject *command_obj = GetCommandSP (cmd_cstr, false, true, matches).get();

    // If we didn't find an exact match to the command string in the commands, look in
    // the aliases.

    if (command_obj == NULL)
    {
        command_obj = GetCommandSP (cmd_cstr, true, true, matches).get();
    }

    // Finally, if there wasn't an exact match among the aliases, look for an inexact match
    // in both the commands and the aliases.

    if (command_obj == NULL)
        command_obj = GetCommandSP(cmd_cstr, true, false, matches).get();

    return command_obj;
}

bool
CommandInterpreter::CommandExists (const char *cmd)
{
    return m_command_dict.find(cmd) != m_command_dict.end();
}

bool
CommandInterpreter::ProcessAliasOptionsArgs (lldb::CommandObjectSP &cmd_obj_sp, 
                                            const char *options_args, 
                                            OptionArgVectorSP &option_arg_vector_sp)
{
    bool success = true;
    OptionArgVector *option_arg_vector = option_arg_vector_sp.get();
    
    if (!options_args || (strlen (options_args) < 1))
        return true;

    std::string options_string (options_args);
    Args args (options_args);
    CommandReturnObject result;
    // Check to see if the command being aliased can take any command options.
    Options *options = cmd_obj_sp->GetOptions ();
    if (options)
    {
        // See if any options were specified as part of the alias;  if so, handle them appropriately.
        options->NotifyOptionParsingStarting ();
        args.Unshift ("dummy_arg");
        args.ParseAliasOptions (*options, result, option_arg_vector, options_string);
        args.Shift ();
        if (result.Succeeded())
            options->VerifyPartialOptions (result);
        if (!result.Succeeded() && result.GetStatus() != lldb::eReturnStatusStarted)
        {
            result.AppendError ("Unable to create requested alias.\n");
            return false;
        }
    }
    
    if (!options_string.empty())
    {
        if (cmd_obj_sp->WantsRawCommandString ())
            option_arg_vector->push_back (OptionArgPair ("<argument>",
                                                          OptionArgValue (-1,
                                                                          options_string)));
        else
        {
            int argc = args.GetArgumentCount();
            for (size_t i = 0; i < argc; ++i)
                if (strcmp (args.GetArgumentAtIndex (i), "") != 0)
                    option_arg_vector->push_back 
                                (OptionArgPair ("<argument>",
                                                OptionArgValue (-1,
                                                                std::string (args.GetArgumentAtIndex (i)))));
        }
    }
        
    return success;
}

bool
CommandInterpreter::AliasExists (const char *cmd)
{
    return m_alias_dict.find(cmd) != m_alias_dict.end();
}

bool
CommandInterpreter::UserCommandExists (const char *cmd)
{
    return m_user_dict.find(cmd) != m_user_dict.end();
}

void
CommandInterpreter::AddAlias (const char *alias_name, CommandObjectSP& command_obj_sp)
{
    command_obj_sp->SetIsAlias (true);
    m_alias_dict[alias_name] = command_obj_sp;
}

bool
CommandInterpreter::RemoveAlias (const char *alias_name)
{
    CommandObject::CommandMap::iterator pos = m_alias_dict.find(alias_name);
    if (pos != m_alias_dict.end())
    {
        m_alias_dict.erase(pos);
        return true;
    }
    return false;
}
bool
CommandInterpreter::RemoveUser (const char *alias_name)
{
    CommandObject::CommandMap::iterator pos = m_user_dict.find(alias_name);
    if (pos != m_user_dict.end())
    {
        m_user_dict.erase(pos);
        return true;
    }
    return false;
}

void
CommandInterpreter::GetAliasHelp (const char *alias_name, const char *command_name, StreamString &help_string)
{
    help_string.Printf ("'%s", command_name);
    OptionArgVectorSP option_arg_vector_sp = GetAliasOptions (alias_name);

    if (option_arg_vector_sp)
    {
        OptionArgVector *options = option_arg_vector_sp.get();
        for (int i = 0; i < options->size(); ++i)
        {
            OptionArgPair cur_option = (*options)[i];
            std::string opt = cur_option.first;
            OptionArgValue value_pair = cur_option.second;
            std::string value = value_pair.second;
            if (opt.compare("<argument>") == 0)
            {
                help_string.Printf (" %s", value.c_str());
            }
            else
            {
                help_string.Printf (" %s", opt.c_str());
                if ((value.compare ("<no-argument>") != 0)
                    && (value.compare ("<need-argument") != 0))
                {
                    help_string.Printf (" %s", value.c_str());
                }
            }
        }
    }

    help_string.Printf ("'");
}

size_t
CommandInterpreter::FindLongestCommandWord (CommandObject::CommandMap &dict)
{
    CommandObject::CommandMap::const_iterator pos;
    CommandObject::CommandMap::const_iterator end = dict.end();
    size_t max_len = 0;

    for (pos = dict.begin(); pos != end; ++pos)
    {
        size_t len = pos->first.size();
        if (max_len < len)
            max_len = len;
    }
    return max_len;
}

void
CommandInterpreter::GetHelp (CommandReturnObject &result,
                             uint32_t cmd_types)
{
    CommandObject::CommandMap::const_iterator pos;
    uint32_t max_len = FindLongestCommandWord (m_command_dict);
    
    if ( (cmd_types & eCommandTypesBuiltin) == eCommandTypesBuiltin )
    {
    
        result.AppendMessage("The following is a list of built-in, permanent debugger commands:");
        result.AppendMessage("");

        for (pos = m_command_dict.begin(); pos != m_command_dict.end(); ++pos)
        {
            OutputFormattedHelpText (result.GetOutputStream(), pos->first.c_str(), "--", pos->second->GetHelp(),
                                     max_len);
        }
        result.AppendMessage("");

    }

    if (!m_alias_dict.empty() && ( (cmd_types & eCommandTypesAliases) == eCommandTypesAliases ))
    {
        result.AppendMessage("The following is a list of your current command abbreviations "
                             "(see 'help command alias' for more info):");
        result.AppendMessage("");
        max_len = FindLongestCommandWord (m_alias_dict);

        for (pos = m_alias_dict.begin(); pos != m_alias_dict.end(); ++pos)
        {
            StreamString sstr;
            StreamString translation_and_help;
            std::string entry_name = pos->first;
            std::string second_entry = pos->second.get()->GetCommandName();
            GetAliasHelp (pos->first.c_str(), pos->second->GetCommandName(), sstr);
            
            translation_and_help.Printf ("(%s)  %s", sstr.GetData(), pos->second->GetHelp());
            OutputFormattedHelpText (result.GetOutputStream(), pos->first.c_str(), "--", 
                                     translation_and_help.GetData(), max_len);
        }
        result.AppendMessage("");
    }

    if (!m_user_dict.empty() && ( (cmd_types & eCommandTypesUserDef) == eCommandTypesUserDef ))
    {
        result.AppendMessage ("The following is a list of your current user-defined commands:");
        result.AppendMessage("");
        max_len = FindLongestCommandWord (m_user_dict);
        for (pos = m_user_dict.begin(); pos != m_user_dict.end(); ++pos)
        {
            OutputFormattedHelpText (result.GetOutputStream(), pos->first.c_str(), "--", pos->second->GetHelp(),
                                     max_len);
        }
        result.AppendMessage("");
    }

    result.AppendMessage("For more information on any particular command, try 'help <command-name>'.");
}

CommandObject *
CommandInterpreter::GetCommandObjectForCommand (std::string &command_string)
{
    // This function finds the final, lowest-level, alias-resolved command object whose 'Execute' function will
    // eventually be invoked by the given command line.
    
    CommandObject *cmd_obj = NULL;
    std::string white_space (" \t\v");
    size_t start = command_string.find_first_not_of (white_space);
    size_t end = 0;
    bool done = false;
    while (!done)
    {
        if (start != std::string::npos)
        {
            // Get the next word from command_string.
            end = command_string.find_first_of (white_space, start);
            if (end == std::string::npos)
                end = command_string.size();
            std::string cmd_word = command_string.substr (start, end - start);
            
            if (cmd_obj == NULL)
                // Since cmd_obj is NULL we are on our first time through this loop. Check to see if cmd_word is a valid 
                // command or alias.
                cmd_obj = GetCommandObject (cmd_word.c_str());
            else if (cmd_obj->IsMultiwordObject ())
            {
                // Our current object is a multi-word object; see if the cmd_word is a valid sub-command for our object.
                CommandObject *sub_cmd_obj = cmd_obj->GetSubcommandObject (cmd_word.c_str());
                if (sub_cmd_obj)
                    cmd_obj = sub_cmd_obj;
                else // cmd_word was not a valid sub-command word, so we are donee
                    done = true;
            }
            else  
                // We have a cmd_obj and it is not a multi-word object, so we are done.
                done = true;

            // If we didn't find a valid command object, or our command object is not a multi-word object, or
            // we are at the end of the command_string, then we are done.  Otherwise, find the start of the
            // next word.
            
            if (!cmd_obj || !cmd_obj->IsMultiwordObject() || end >= command_string.size())
                done = true;
            else
                start = command_string.find_first_not_of (white_space, end);
        }
        else
            // Unable to find any more words.
            done = true;
    }
    
    if (end == command_string.size())
        command_string.clear();
    else
        command_string = command_string.substr(end);
    
    return cmd_obj;
}

static const char *k_white_space = " \t\v";
static const char *k_valid_command_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_";
static void
StripLeadingSpaces (std::string &s)
{
    if (!s.empty())
    {
        size_t pos = s.find_first_not_of (k_white_space);
        if (pos == std::string::npos)
            s.clear();
        else if (pos == 0)
            return;
        s.erase (0, pos);
    }
}

static size_t
FindArgumentTerminator (const std::string &s)
{
    const size_t s_len = s.size();
    size_t offset = 0;
    while (offset < s_len)
    {
        size_t pos = s.find ("--", offset);
        if (pos == std::string::npos)
            break;
        if (pos > 0)
        {
            if (isspace(s[pos-1]))
            {
                // Check if the string ends "\s--" (where \s is a space character)
                // or if we have "\s--\s".
                if ((pos + 2 >= s_len) || isspace(s[pos+2]))
                {
                    return pos;
                }
            }
        }
        offset = pos + 2;
    }
    return std::string::npos;
}

static bool
ExtractCommand (std::string &command_string, std::string &command, std::string &suffix, char &quote_char)
{
    command.clear();
    suffix.clear();
    StripLeadingSpaces (command_string);

    bool result = false;
    quote_char = '\0';
    
    if (!command_string.empty())
    {
        const char first_char = command_string[0];
        if (first_char == '\'' || first_char == '"')
        {
            quote_char = first_char;
            const size_t end_quote_pos = command_string.find (quote_char, 1);
            if (end_quote_pos == std::string::npos)
            {
                command.swap (command_string);
                command_string.erase ();
            }
            else
            {
                command.assign (command_string, 1, end_quote_pos - 1);
                if (end_quote_pos + 1 < command_string.size())
                    command_string.erase (0, command_string.find_first_not_of (k_white_space, end_quote_pos + 1));
                else
                    command_string.erase ();
            }
        }
        else
        {
            const size_t first_space_pos = command_string.find_first_of (k_white_space);
            if (first_space_pos == std::string::npos)
            {
                command.swap (command_string);
                command_string.erase();
            }
            else
            {
                command.assign (command_string, 0, first_space_pos);
                command_string.erase(0, command_string.find_first_not_of (k_white_space, first_space_pos));
            }
        }
        result = true;
    }
    
    
    if (!command.empty())
    {
        // actual commands can't start with '-' or '_'
        if (command[0] != '-' && command[0] != '_')
        {
            size_t pos = command.find_first_not_of(k_valid_command_chars);
            if (pos > 0 && pos != std::string::npos)
            {
                suffix.assign (command.begin() + pos, command.end());
                command.erase (pos);
            }
        }
    }

    return result;
}

CommandObject *
CommandInterpreter::BuildAliasResult (const char *alias_name, 
                                      std::string &raw_input_string, 
                                      std::string &alias_result,
                                      CommandReturnObject &result)
{
    CommandObject *alias_cmd_obj = NULL;
    Args cmd_args (raw_input_string.c_str());
    alias_cmd_obj = GetCommandObject (alias_name);
    StreamString result_str;
    
    if (alias_cmd_obj)
    {
        std::string alias_name_str = alias_name;
        if ((cmd_args.GetArgumentCount() == 0)
            || (alias_name_str.compare (cmd_args.GetArgumentAtIndex(0)) != 0))
            cmd_args.Unshift (alias_name);
            
        result_str.Printf ("%s", alias_cmd_obj->GetCommandName ());
        OptionArgVectorSP option_arg_vector_sp = GetAliasOptions (alias_name);

        if (option_arg_vector_sp.get())
        {
            OptionArgVector *option_arg_vector = option_arg_vector_sp.get();

            for (int i = 0; i < option_arg_vector->size(); ++i)
            {
                OptionArgPair option_pair = (*option_arg_vector)[i];
                OptionArgValue value_pair = option_pair.second;
                int value_type = value_pair.first;
                std::string option = option_pair.first;
                std::string value = value_pair.second;
                if (option.compare ("<argument>") == 0)
                    result_str.Printf (" %s", value.c_str());
                else
                {
                    result_str.Printf (" %s", option.c_str());
                    if (value_type != optional_argument)
                        result_str.Printf (" ");
                    if (value.compare ("<no_argument>") != 0)
                    {
                        int index = GetOptionArgumentPosition (value.c_str());
                        if (index == 0)
                            result_str.Printf ("%s", value.c_str());
                        else if (index >= cmd_args.GetArgumentCount())
                        {
                            
                            result.AppendErrorWithFormat
                            ("Not enough arguments provided; you need at least %d arguments to use this alias.\n",
                             index);
                            result.SetStatus (eReturnStatusFailed);
                            return alias_cmd_obj;
                        }
                        else
                        {
                            size_t strpos = raw_input_string.find (cmd_args.GetArgumentAtIndex (index));
                            if (strpos != std::string::npos)
                                raw_input_string = raw_input_string.erase (strpos, 
                                                                          strlen (cmd_args.GetArgumentAtIndex (index)));
                            result_str.Printf ("%s", cmd_args.GetArgumentAtIndex (index));
                        }
                    }
                }
            }
        }
        
        alias_result = result_str.GetData();
    }
    return alias_cmd_obj;
}

Error
CommandInterpreter::PreprocessCommand (std::string &command)
{
    // The command preprocessor needs to do things to the command 
    // line before any parsing of arguments or anything else is done.
    // The only current stuff that gets proprocessed is anyting enclosed
    // in backtick ('`') characters is evaluated as an expression and
    // the result of the expression must be a scalar that can be substituted
    // into the command. An example would be:
    // (lldb) memory read `$rsp + 20`
    Error error; // Error for any expressions that might not evaluate
    size_t start_backtick;
    size_t pos = 0;
    while ((start_backtick = command.find ('`', pos)) != std::string::npos)
    {
        if (start_backtick > 0 && command[start_backtick-1] == '\\')
        {
            // The backtick was preceeded by a '\' character, remove the slash
            // and don't treat the backtick as the start of an expression
            command.erase(start_backtick-1, 1);
            // No need to add one to start_backtick since we just deleted a char
            pos = start_backtick; 
        }
        else
        {
            const size_t expr_content_start = start_backtick + 1;
            const size_t end_backtick = command.find ('`', expr_content_start);
            if (end_backtick == std::string::npos)
                return error;
            else if (end_backtick == expr_content_start)
            {
                // Empty expression (two backticks in a row)
                command.erase (start_backtick, 2);
            }
            else
            {
                std::string expr_str (command, expr_content_start, end_backtick - expr_content_start);
                
                ExecutionContext exe_ctx(GetExecutionContext());
                Target *target = exe_ctx.GetTargetPtr();
                // Get a dummy target to allow for calculator mode while processing backticks.
                // This also helps break the infinite loop caused when target is null.
                if (!target)
                    target = Host::GetDummyTarget(GetDebugger()).get();
                if (target)
                {
                    ValueObjectSP expr_result_valobj_sp;
                    
                    EvaluateExpressionOptions options;
                    options.SetCoerceToId(false)
                    .SetUnwindOnError(true)
                    .SetKeepInMemory(false)
                    .SetRunOthers(true)
                    .SetTimeoutUsec(0);
                    
                    ExecutionResults expr_result = target->EvaluateExpression (expr_str.c_str(), 
                                                                               exe_ctx.GetFramePtr(),
                                                                               expr_result_valobj_sp,
                                                                               options);
                    
                    if (expr_result == eExecutionCompleted)
                    {
                        Scalar scalar;
                        if (expr_result_valobj_sp->ResolveValue (scalar))
                        {
                            command.erase (start_backtick, end_backtick - start_backtick + 1);
                            StreamString value_strm;
                            const bool show_type = false;
                            scalar.GetValue (&value_strm, show_type);
                            size_t value_string_size = value_strm.GetSize();
                            if (value_string_size)
                            {
                                command.insert (start_backtick, value_strm.GetData(), value_string_size);
                                pos = start_backtick + value_string_size;
                                continue;
                            }
                            else
                            {
                                error.SetErrorStringWithFormat("expression value didn't result in a scalar value for the expression '%s'", expr_str.c_str());
                            }
                        }
                        else
                        {
                            error.SetErrorStringWithFormat("expression value didn't result in a scalar value for the expression '%s'", expr_str.c_str());
                        }
                    }
                    else
                    {
                        if (expr_result_valobj_sp)
                            error = expr_result_valobj_sp->GetError();
                        if (error.Success())
                        {

                            switch (expr_result)
                            {
                                case eExecutionSetupError: 
                                    error.SetErrorStringWithFormat("expression setup error for the expression '%s'", expr_str.c_str());
                                    break;
                                case eExecutionCompleted:
                                    break;
                                case eExecutionDiscarded:
                                    error.SetErrorStringWithFormat("expression discarded for the expression '%s'", expr_str.c_str());
                                    break;
                                case eExecutionInterrupted:
                                    error.SetErrorStringWithFormat("expression interrupted for the expression '%s'", expr_str.c_str());
                                    break;
                                case eExecutionTimedOut:
                                    error.SetErrorStringWithFormat("expression timed out for the expression '%s'", expr_str.c_str());
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


bool
CommandInterpreter::HandleCommand (const char *command_line, 
                                   LazyBool lazy_add_to_history,
                                   CommandReturnObject &result,
                                   ExecutionContext *override_context,
                                   bool repeat_on_empty_command,
                                   bool no_context_switching)

{

    bool done = false;
    CommandObject *cmd_obj = NULL;
    bool wants_raw_input = false;
    std::string command_string (command_line);
    std::string original_command_string (command_line);
    
    LogSP log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_COMMANDS));
    Host::SetCrashDescriptionWithFormat ("HandleCommand(command = \"%s\")", command_line);
    
    // Make a scoped cleanup object that will clear the crash description string 
    // on exit of this function.
    lldb_utility::CleanUp <const char *> crash_description_cleanup(NULL, Host::SetCrashDescription);

    if (log)
        log->Printf ("Processing command: %s", command_line);

    Timer scoped_timer (__PRETTY_FUNCTION__, "Handling command: %s.", command_line);
    
    if (!no_context_switching)
        UpdateExecutionContext (override_context);
    
    // <rdar://problem/11328896>
    bool add_to_history;
    if (lazy_add_to_history == eLazyBoolCalculate)
        add_to_history = (m_command_source_depth == 0);
    else
        add_to_history = (lazy_add_to_history == eLazyBoolYes);
    
    bool empty_command = false;
    bool comment_command = false;
    if (command_string.empty())
        empty_command = true;
    else
    {
        const char *k_space_characters = "\t\n\v\f\r ";

        size_t non_space = command_string.find_first_not_of (k_space_characters);
        // Check for empty line or comment line (lines whose first
        // non-space character is the comment character for this interpreter)
        if (non_space == std::string::npos)
            empty_command = true;
        else if (command_string[non_space] == m_comment_char)
             comment_command = true;
        else if (command_string[non_space] == m_repeat_char)
        {
            const char *history_string = FindHistoryString (command_string.c_str() + non_space);
            if (history_string == NULL)
            {
                result.AppendErrorWithFormat ("Could not find entry: %s in history", command_string.c_str());
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
            add_to_history = false;
            command_string = history_string;
            original_command_string = history_string;
        }
    }
    
    if (empty_command)
    {
        if (repeat_on_empty_command)
        {
            if (m_command_history.empty())
            {
                result.AppendError ("empty command");
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
            else
            {
                command_line = m_repeat_command.c_str();
                command_string = command_line;
                original_command_string = command_line;
                if (m_repeat_command.empty())
                {
                    result.AppendErrorWithFormat("No auto repeat.\n");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            add_to_history = false;
        }
        else
        {
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
            return true;
        }
    }
    else if (comment_command)
    {
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
        return true;
    }
    
    
    Error error (PreprocessCommand (command_string));
    
    if (error.Fail())
    {
        result.AppendError (error.AsCString());
        result.SetStatus(eReturnStatusFailed);
        return false;
    }
    // Phase 1.
    
    // Before we do ANY kind of argument processing, etc. we need to figure out what the real/final command object
    // is for the specified command, and whether or not it wants raw input.  This gets complicated by the fact that
    // the user could have specified an alias, and in translating the alias there may also be command options and/or
    // even data (including raw text strings) that need to be found and inserted into the command line as part of
    // the translation.  So this first step is plain look-up & replacement, resulting in three things:  1). the command
    // object whose Execute method will actually be called; 2). a revised command string, with all substitutions &
    // replacements taken care of; 3). whether or not the Execute function wants raw input or not.

    StreamString revised_command_line;
    size_t actual_cmd_name_len = 0;
    std::string next_word;
    StringList matches;
    while (!done)
    {
        char quote_char = '\0';
        std::string suffix;
        ExtractCommand (command_string, next_word, suffix, quote_char);
        if (cmd_obj == NULL)
        {
            if (AliasExists (next_word.c_str())) 
            {
                std::string alias_result;
                cmd_obj = BuildAliasResult (next_word.c_str(), command_string, alias_result, result);
                revised_command_line.Printf ("%s", alias_result.c_str());
                if (cmd_obj)
                {
                    wants_raw_input = cmd_obj->WantsRawCommandString ();
                    actual_cmd_name_len = strlen (cmd_obj->GetCommandName());
                }
            }
            else
            {
                cmd_obj = GetCommandObject (next_word.c_str(), &matches);
                if (cmd_obj)
                {
                    actual_cmd_name_len += next_word.length();
                    revised_command_line.Printf ("%s", next_word.c_str());
                    wants_raw_input = cmd_obj->WantsRawCommandString ();
                }
                else
                {
                    revised_command_line.Printf ("%s", next_word.c_str());
                }
            }
        }
        else
        {
            if (cmd_obj->IsMultiwordObject ())
            {
                CommandObject *sub_cmd_obj = cmd_obj->GetSubcommandObject (next_word.c_str());
                if (sub_cmd_obj)
                {
                    actual_cmd_name_len += next_word.length() + 1;
                    revised_command_line.Printf (" %s", next_word.c_str());
                    cmd_obj = sub_cmd_obj;
                    wants_raw_input = cmd_obj->WantsRawCommandString ();
                }
                else
                {
                    if (quote_char)
                        revised_command_line.Printf (" %c%s%s%c", quote_char, next_word.c_str(), suffix.c_str(), quote_char);
                    else
                        revised_command_line.Printf (" %s%s", next_word.c_str(), suffix.c_str());
                    done = true;
                }
            }
            else
            {
                if (quote_char)
                    revised_command_line.Printf (" %c%s%s%c", quote_char, next_word.c_str(), suffix.c_str(), quote_char);
                else
                    revised_command_line.Printf (" %s%s", next_word.c_str(), suffix.c_str());
                done = true;
            }
        }

        if (cmd_obj == NULL)
        {
            uint32_t num_matches = matches.GetSize();
            if (matches.GetSize() > 1) {
                std::string error_msg;
                error_msg.assign ("Ambiguous command '");
                error_msg.append(next_word.c_str());
                error_msg.append ("'.");

                error_msg.append (" Possible matches:");

                for (uint32_t i = 0; i < num_matches; ++i) {
                    error_msg.append ("\n\t");
                    error_msg.append (matches.GetStringAtIndex(i));
                }
                error_msg.append ("\n");
                result.AppendRawError (error_msg.c_str(), error_msg.size());
            } else {
                // We didn't have only one match, otherwise we wouldn't get here.
                assert(num_matches == 0);
                result.AppendErrorWithFormat ("'%s' is not a valid command.\n", next_word.c_str());
            }
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        if (cmd_obj->IsMultiwordObject ())
        {
            if (!suffix.empty())
            {

                result.AppendErrorWithFormat ("multi-word commands ('%s') can't have shorthand suffixes: '%s'\n", 
                                              next_word.c_str(),
                                              suffix.c_str());
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
        }
        else
        {
            // If we found a normal command, we are done
            done = true;
            if (!suffix.empty())
            {
                switch (suffix[0])
                {
                case '/':
                    // GDB format suffixes
                    {
                        Options *command_options = cmd_obj->GetOptions();
                        if (command_options && command_options->SupportsLongOption("gdb-format"))
                        {
                            std::string gdb_format_option ("--gdb-format=");
                            gdb_format_option += (suffix.c_str() + 1);
    
                            bool inserted = false;
                            std::string &cmd = revised_command_line.GetString();
                            size_t arg_terminator_idx = FindArgumentTerminator (cmd);
                            if (arg_terminator_idx != std::string::npos)
                            {
                                // Insert the gdb format option before the "--" that terminates options
                                gdb_format_option.append(1,' ');
                                cmd.insert(arg_terminator_idx, gdb_format_option);
                                inserted = true;
                            }

                            if (!inserted)
                                revised_command_line.Printf (" %s", gdb_format_option.c_str());
                        
                            if (wants_raw_input && FindArgumentTerminator(cmd) == std::string::npos)
                                revised_command_line.PutCString (" --");
                        }
                        else
                        {
                            result.AppendErrorWithFormat ("the '%s' command doesn't support the --gdb-format option\n", 
                                                          cmd_obj->GetCommandName());
                            result.SetStatus (eReturnStatusFailed);
                            return false;
                        }
                    }
                    break;

                default:
                    result.AppendErrorWithFormat ("unknown command shorthand suffix: '%s'\n", 
                                                  suffix.c_str());
                    result.SetStatus (eReturnStatusFailed);
                    return false;
        
                }
            }
        }
        if (command_string.length() == 0)
            done = true;
            
    }

    if (!command_string.empty())
        revised_command_line.Printf (" %s", command_string.c_str());

    // End of Phase 1.
    // At this point cmd_obj should contain the CommandObject whose Execute method will be called, if the command
    // specified was valid; revised_command_line contains the complete command line (including command name(s)),
    // fully translated with all substitutions & translations taken care of (still in raw text format); and
    // wants_raw_input specifies whether the Execute method expects raw input or not.

 
    if (log)
    {
        log->Printf ("HandleCommand, cmd_obj : '%s'", cmd_obj ? cmd_obj->GetCommandName() : "<not found>");
        log->Printf ("HandleCommand, revised_command_line: '%s'", revised_command_line.GetData());
        log->Printf ("HandleCommand, wants_raw_input:'%s'", wants_raw_input ? "True" : "False");
    }

    // Phase 2.
    // Take care of things like setting up the history command & calling the appropriate Execute method on the
    // CommandObject, with the appropriate arguments.
    
    if (cmd_obj != NULL)
    {
        if (add_to_history)
        {
            Args command_args (revised_command_line.GetData());
            const char *repeat_command = cmd_obj->GetRepeatCommand(command_args, 0);
            if (repeat_command != NULL)
                m_repeat_command.assign(repeat_command);
            else
                m_repeat_command.assign(original_command_string.c_str());
            
            // Don't keep pushing the same command onto the history...
            if (m_command_history.empty() || m_command_history.back() != original_command_string) 
                m_command_history.push_back (original_command_string);
        }
        
        command_string = revised_command_line.GetData();
        std::string command_name (cmd_obj->GetCommandName());
        std::string remainder;
        if (actual_cmd_name_len < command_string.length()) 
            remainder = command_string.substr (actual_cmd_name_len);  // Note: 'actual_cmd_name_len' may be considerably shorter
                                                           // than cmd_obj->GetCommandName(), because name completion
                                                           // allows users to enter short versions of the names,
                                                           // e.g. 'br s' for 'breakpoint set'.
        
        // Remove any initial spaces
        std::string white_space (" \t\v");
        size_t pos = remainder.find_first_not_of (white_space);
        if (pos != 0 && pos != std::string::npos)
            remainder.erase(0, pos);

        if (log)
            log->Printf ("HandleCommand, command line after removing command name(s): '%s'", remainder.c_str());
    
        cmd_obj->Execute (remainder.c_str(), result);
    }
    else
    {
        // We didn't find the first command object, so complete the first argument.
        Args command_args (revised_command_line.GetData());
        StringList matches;
        int num_matches;
        int cursor_index = 0;
        int cursor_char_position = strlen (command_args.GetArgumentAtIndex(0));
        bool word_complete;
        num_matches = HandleCompletionMatches (command_args, 
                                               cursor_index,
                                               cursor_char_position,
                                               0, 
                                               -1, 
                                               word_complete,
                                               matches);
        
        if (num_matches > 0)
        {
            std::string error_msg;
            error_msg.assign ("ambiguous command '");
            error_msg.append(command_args.GetArgumentAtIndex(0));
            error_msg.append ("'.");
            
            error_msg.append (" Possible completions:");
            for (int i = 0; i < num_matches; i++)
            {
                error_msg.append ("\n\t");
                error_msg.append (matches.GetStringAtIndex (i));
            }
            error_msg.append ("\n");
            result.AppendRawError (error_msg.c_str(), error_msg.size());
        }
        else
            result.AppendErrorWithFormat ("Unrecognized command '%s'.\n", command_args.GetArgumentAtIndex (0));
        
        result.SetStatus (eReturnStatusFailed);
    }
    
    if (log)
      log->Printf ("HandleCommand, command %s", (result.Succeeded() ? "succeeded" : "did not succeed"));

    return result.Succeeded();
}

int
CommandInterpreter::HandleCompletionMatches (Args &parsed_line,
                                             int &cursor_index,
                                             int &cursor_char_position,
                                             int match_start_point,
                                             int max_return_elements,
                                             bool &word_complete,
                                             StringList &matches)
{
    int num_command_matches = 0;
    bool look_for_subcommand = false;
    
    // For any of the command completions a unique match will be a complete word.
    word_complete = true;

    if (cursor_index == -1)
    {
        // We got nothing on the command line, so return the list of commands
        bool include_aliases = true;
        num_command_matches = GetCommandNamesMatchingPartialString ("", include_aliases, matches);
    }
    else if (cursor_index == 0)
    {
        // The cursor is in the first argument, so just do a lookup in the dictionary.
        CommandObject *cmd_obj = GetCommandObject (parsed_line.GetArgumentAtIndex(0), &matches);
        num_command_matches = matches.GetSize();

        if (num_command_matches == 1
            && cmd_obj && cmd_obj->IsMultiwordObject()
            && matches.GetStringAtIndex(0) != NULL
            && strcmp (parsed_line.GetArgumentAtIndex(0), matches.GetStringAtIndex(0)) == 0)
        {
            look_for_subcommand = true;
            num_command_matches = 0;
            matches.DeleteStringAtIndex(0);
            parsed_line.AppendArgument ("");
            cursor_index++;
            cursor_char_position = 0;
        }
    }

    if (cursor_index > 0 || look_for_subcommand)
    {
        // We are completing further on into a commands arguments, so find the command and tell it
        // to complete the command.
        // First see if there is a matching initial command:
        CommandObject *command_object = GetCommandObject (parsed_line.GetArgumentAtIndex(0));
        if (command_object == NULL)
        {
            return 0;
        }
        else
        {
            parsed_line.Shift();
            cursor_index--;
            num_command_matches = command_object->HandleCompletion (parsed_line, 
                                                                    cursor_index, 
                                                                    cursor_char_position,
                                                                    match_start_point, 
                                                                    max_return_elements,
                                                                    word_complete, 
                                                                    matches);
        }
    }

    return num_command_matches;

}

int
CommandInterpreter::HandleCompletion (const char *current_line,
                                      const char *cursor,
                                      const char *last_char,
                                      int match_start_point,
                                      int max_return_elements,
                                      StringList &matches)
{
    // We parse the argument up to the cursor, so the last argument in parsed_line is
    // the one containing the cursor, and the cursor is after the last character.

    Args parsed_line(current_line, last_char - current_line);
    Args partial_parsed_line(current_line, cursor - current_line);

    // Don't complete comments, and if the line we are completing is just the history repeat character, 
    // substitute the appropriate history line.
    const char *first_arg = parsed_line.GetArgumentAtIndex(0);
    if (first_arg)
    {
        if (first_arg[0] == m_comment_char)
            return 0;
        else if (first_arg[0] == m_repeat_char)
        {
            const char *history_string = FindHistoryString (first_arg);
            if (history_string != NULL)
            {
                matches.Clear();
                matches.InsertStringAtIndex(0, history_string);
                return -2;
            }
            else
                return 0;

        }
    }
    
    
    int num_args = partial_parsed_line.GetArgumentCount();
    int cursor_index = partial_parsed_line.GetArgumentCount() - 1;
    int cursor_char_position;

    if (cursor_index == -1)
        cursor_char_position = 0;
    else
        cursor_char_position = strlen (partial_parsed_line.GetArgumentAtIndex(cursor_index));
        
    if (cursor > current_line && cursor[-1] == ' ')
    {
        // We are just after a space.  If we are in an argument, then we will continue
        // parsing, but if we are between arguments, then we have to complete whatever the next
        // element would be.
        // We can distinguish the two cases because if we are in an argument (e.g. because the space is
        // protected by a quote) then the space will also be in the parsed argument...
        
        const char *current_elem = partial_parsed_line.GetArgumentAtIndex(cursor_index);
        if (cursor_char_position == 0 || current_elem[cursor_char_position - 1] != ' ')
        {
            parsed_line.InsertArgumentAtIndex(cursor_index + 1, "", '"');
            cursor_index++;
            cursor_char_position = 0;
        }
    }

    int num_command_matches;

    matches.Clear();

    // Only max_return_elements == -1 is supported at present:
    assert (max_return_elements == -1);
    bool word_complete;
    num_command_matches = HandleCompletionMatches (parsed_line, 
                                                   cursor_index, 
                                                   cursor_char_position, 
                                                   match_start_point,
                                                   max_return_elements,
                                                   word_complete,
                                                   matches);

    if (num_command_matches <= 0)
            return num_command_matches;

    if (num_args == 0)
    {
        // If we got an empty string, insert nothing.
        matches.InsertStringAtIndex(0, "");
    }
    else
    {
        // Now figure out if there is a common substring, and if so put that in element 0, otherwise
        // put an empty string in element 0.
        std::string command_partial_str;
        if (cursor_index >= 0)
            command_partial_str.assign(parsed_line.GetArgumentAtIndex(cursor_index), 
                                       parsed_line.GetArgumentAtIndex(cursor_index) + cursor_char_position);

        std::string common_prefix;
        matches.LongestCommonPrefix (common_prefix);
        int partial_name_len = command_partial_str.size();

        // If we matched a unique single command, add a space...
        // Only do this if the completer told us this was a complete word, however...
        if (num_command_matches == 1 && word_complete)
        {
            char quote_char = parsed_line.GetArgumentQuoteCharAtIndex(cursor_index);
            if (quote_char != '\0')
                common_prefix.push_back(quote_char);

            common_prefix.push_back(' ');
        }
        common_prefix.erase (0, partial_name_len);
        matches.InsertStringAtIndex(0, common_prefix.c_str());
    }
    return num_command_matches;
}


CommandInterpreter::~CommandInterpreter ()
{
}

const char *
CommandInterpreter::GetPrompt ()
{
    return m_debugger.GetPrompt();
}

void
CommandInterpreter::SetPrompt (const char *new_prompt)
{
    m_debugger.SetPrompt (new_prompt);
}

size_t
CommandInterpreter::GetConfirmationInputReaderCallback 
(
    void *baton,
    InputReader &reader,
    lldb::InputReaderAction action,
    const char *bytes,
    size_t bytes_len
)
{
    File &out_file = reader.GetDebugger().GetOutputFile();
    bool *response_ptr = (bool *) baton;
    
    switch (action)
    {
    case eInputReaderActivate:
        if (out_file.IsValid())
        {
            if (reader.GetPrompt())
            {
                out_file.Printf ("%s", reader.GetPrompt());
                out_file.Flush ();
            }
        }
        break;

    case eInputReaderDeactivate:
        break;

    case eInputReaderReactivate:
        if (out_file.IsValid() && reader.GetPrompt())
        {
            out_file.Printf ("%s", reader.GetPrompt());
            out_file.Flush ();
        }
        break;
        
    case eInputReaderAsynchronousOutputWritten:
        break;
        
    case eInputReaderGotToken:
        if (bytes_len == 0)
        {
            reader.SetIsDone(true);
        }
        else if (bytes[0] == 'y' || bytes[0] == 'Y')
        {
            *response_ptr = true;
            reader.SetIsDone(true);
        }
        else if (bytes[0] == 'n' || bytes[0] == 'N')
        {
            *response_ptr = false;
            reader.SetIsDone(true);
        }
        else
        {
            if (out_file.IsValid() && !reader.IsDone() && reader.GetPrompt())
            {
                out_file.Printf ("Please answer \"y\" or \"n\".\n%s", reader.GetPrompt());
                out_file.Flush ();
            }
        }
        break;
        
    case eInputReaderInterrupt:
    case eInputReaderEndOfFile:
        *response_ptr = false;  // Assume ^C or ^D means cancel the proposed action
        reader.SetIsDone (true);
        break;
        
    case eInputReaderDone:
        break;
    }

    return bytes_len;

}

bool 
CommandInterpreter::Confirm (const char *message, bool default_answer)
{
    // Check AutoConfirm first:
    if (m_debugger.GetAutoConfirm())
        return default_answer;
        
    InputReaderSP reader_sp (new InputReader(GetDebugger()));
    bool response = default_answer;
    if (reader_sp)
    {
        std::string prompt(message);
        prompt.append(": [");
        if (default_answer)
            prompt.append ("Y/n] ");
        else
            prompt.append ("y/N] ");
            
        Error err (reader_sp->Initialize (CommandInterpreter::GetConfirmationInputReaderCallback,
                                          &response,                    // baton
                                          eInputReaderGranularityLine,  // token size, to pass to callback function
                                          NULL,                         // end token
                                          prompt.c_str(),               // prompt
                                          true));                       // echo input
        if (err.Success())
        {
            GetDebugger().PushInputReader (reader_sp);
        }
        reader_sp->WaitOnReaderIsDone();
    }
    return response;        
}
    

void
CommandInterpreter::CrossRegisterCommand (const char * dest_cmd, const char * object_type)
{
    CommandObjectSP cmd_obj_sp = GetCommandSPExact (dest_cmd, true);

    if (cmd_obj_sp)
    {
        CommandObject *cmd_obj = cmd_obj_sp.get();
        if (cmd_obj->IsCrossRefObject ())
            cmd_obj->AddObject (object_type);
    }
}

OptionArgVectorSP
CommandInterpreter::GetAliasOptions (const char *alias_name)
{
    OptionArgMap::iterator pos;
    OptionArgVectorSP ret_val;

    std::string alias (alias_name);

    if (HasAliasOptions())
    {
        pos = m_alias_options.find (alias);
        if (pos != m_alias_options.end())
          ret_val = pos->second;
    }

    return ret_val;
}

void
CommandInterpreter::RemoveAliasOptions (const char *alias_name)
{
    OptionArgMap::iterator pos = m_alias_options.find(alias_name);
    if (pos != m_alias_options.end())
    {
        m_alias_options.erase (pos);
    }
}

void
CommandInterpreter::AddOrReplaceAliasOptions (const char *alias_name, OptionArgVectorSP &option_arg_vector_sp)
{
    m_alias_options[alias_name] = option_arg_vector_sp;
}

bool
CommandInterpreter::HasCommands ()
{
    return (!m_command_dict.empty());
}

bool
CommandInterpreter::HasAliases ()
{
    return (!m_alias_dict.empty());
}

bool
CommandInterpreter::HasUserCommands ()
{
    return (!m_user_dict.empty());
}

bool
CommandInterpreter::HasAliasOptions ()
{
    return (!m_alias_options.empty());
}

void
CommandInterpreter::BuildAliasCommandArgs (CommandObject *alias_cmd_obj,
                                           const char *alias_name,
                                           Args &cmd_args,
                                           std::string &raw_input_string,
                                           CommandReturnObject &result)
{
    OptionArgVectorSP option_arg_vector_sp = GetAliasOptions (alias_name);
    
    bool wants_raw_input = alias_cmd_obj->WantsRawCommandString();

    // Make sure that the alias name is the 0th element in cmd_args
    std::string alias_name_str = alias_name;
    if (alias_name_str.compare (cmd_args.GetArgumentAtIndex(0)) != 0)
        cmd_args.Unshift (alias_name);
    
    Args new_args (alias_cmd_obj->GetCommandName());
    if (new_args.GetArgumentCount() == 2)
        new_args.Shift();
    
    if (option_arg_vector_sp.get())
    {
        if (wants_raw_input)
        {
            // We have a command that both has command options and takes raw input.  Make *sure* it has a
            // " -- " in the right place in the raw_input_string.
            size_t pos = raw_input_string.find(" -- ");
            if (pos == std::string::npos)
            {
                // None found; assume it goes at the beginning of the raw input string
                raw_input_string.insert (0, " -- ");
            }
        }

        OptionArgVector *option_arg_vector = option_arg_vector_sp.get();
        int old_size = cmd_args.GetArgumentCount();
        std::vector<bool> used (old_size + 1, false);
        
        used[0] = true;

        for (int i = 0; i < option_arg_vector->size(); ++i)
        {
            OptionArgPair option_pair = (*option_arg_vector)[i];
            OptionArgValue value_pair = option_pair.second;
            int value_type = value_pair.first;
            std::string option = option_pair.first;
            std::string value = value_pair.second;
            if (option.compare ("<argument>") == 0)
            {
                if (!wants_raw_input
                    || (value.compare("--") != 0)) // Since we inserted this above, make sure we don't insert it twice
                    new_args.AppendArgument (value.c_str());
            }
            else
            {
                if (value_type != optional_argument)
                    new_args.AppendArgument (option.c_str());
                if (value.compare ("<no-argument>") != 0)
                {
                    int index = GetOptionArgumentPosition (value.c_str());
                    if (index == 0)
                    {
                        // value was NOT a positional argument; must be a real value
                        if (value_type != optional_argument)
                            new_args.AppendArgument (value.c_str());
                        else
                        {
                            char buffer[255];
                            ::snprintf (buffer, sizeof (buffer), "%s%s", option.c_str(), value.c_str());
                            new_args.AppendArgument (buffer);
                        }

                    }
                    else if (index >= cmd_args.GetArgumentCount())
                    {
                        result.AppendErrorWithFormat
                                    ("Not enough arguments provided; you need at least %d arguments to use this alias.\n",
                                     index);
                        result.SetStatus (eReturnStatusFailed);
                        return;
                    }
                    else
                    {
                        // Find and remove cmd_args.GetArgumentAtIndex(i) from raw_input_string
                        size_t strpos = raw_input_string.find (cmd_args.GetArgumentAtIndex (index));
                        if (strpos != std::string::npos)
                        {
                            raw_input_string = raw_input_string.erase (strpos, strlen (cmd_args.GetArgumentAtIndex (index)));
                        }

                        if (value_type != optional_argument)
                            new_args.AppendArgument (cmd_args.GetArgumentAtIndex (index));
                        else
                        {
                            char buffer[255];
                            ::snprintf (buffer, sizeof(buffer), "%s%s", option.c_str(), 
                                        cmd_args.GetArgumentAtIndex (index));
                            new_args.AppendArgument (buffer);
                        }
                        used[index] = true;
                    }
                }
            }
        }

        for (int j = 0; j < cmd_args.GetArgumentCount(); ++j)
        {
            if (!used[j] && !wants_raw_input)
                new_args.AppendArgument (cmd_args.GetArgumentAtIndex (j));
        }

        cmd_args.Clear();
        cmd_args.SetArguments (new_args.GetArgumentCount(), (const char **) new_args.GetArgumentVector());
    }
    else
    {
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
        // This alias was not created with any options; nothing further needs to be done, unless it is a command that
        // wants raw input, in which case we need to clear the rest of the data from cmd_args, since its in the raw
        // input string.
        if (wants_raw_input)
        {
            cmd_args.Clear();
            cmd_args.SetArguments (new_args.GetArgumentCount(), (const char **) new_args.GetArgumentVector());
        }
        return;
    }

    result.SetStatus (eReturnStatusSuccessFinishNoResult);
    return;
}


int
CommandInterpreter::GetOptionArgumentPosition (const char *in_string)
{
    int position = 0;   // Any string that isn't an argument position, i.e. '%' followed by an integer, gets a position
                        // of zero.

    char *cptr = (char *) in_string;

    // Does it start with '%'
    if (cptr[0] == '%')
    {
        ++cptr;

        // Is the rest of it entirely digits?
        if (isdigit (cptr[0]))
        {
            const char *start = cptr;
            while (isdigit (cptr[0]))
                ++cptr;

            // We've gotten to the end of the digits; are we at the end of the string?
            if (cptr[0] == '\0')
                position = atoi (start);
        }
    }

    return position;
}

void
CommandInterpreter::SourceInitFile (bool in_cwd, CommandReturnObject &result)
{
    FileSpec init_file;
    if (in_cwd)
    {
        // In the current working directory we don't load any program specific
        // .lldbinit files, we only look for a "./.lldbinit" file.
        if (m_skip_lldbinit_files)
            return;

        init_file.SetFile ("./.lldbinit", true);
    }
    else
    {
        // If we aren't looking in the current working directory we are looking
        // in the home directory. We will first see if there is an application
        // specific ".lldbinit" file whose name is "~/.lldbinit" followed by a 
        // "-" and the name of the program. If this file doesn't exist, we fall
        // back to just the "~/.lldbinit" file. We also obey any requests to not
        // load the init files.
        const char *init_file_path = "~/.lldbinit";

        if (m_skip_app_init_files == false)
        {
            FileSpec program_file_spec (Host::GetProgramFileSpec());
            const char *program_name = program_file_spec.GetFilename().AsCString();
    
            if (program_name)
            {
                char program_init_file_name[PATH_MAX];
                ::snprintf (program_init_file_name, sizeof(program_init_file_name), "%s-%s", init_file_path, program_name);
                init_file.SetFile (program_init_file_name, true);
                if (!init_file.Exists())
                    init_file.Clear();
            }
        }
        
        if (!init_file && !m_skip_lldbinit_files)
			init_file.SetFile (init_file_path, true);
    }

    // If the file exists, tell HandleCommand to 'source' it; this will do the actual broadcasting
    // of the commands back to any appropriate listener (see CommandObjectSource::Execute for more details).

    if (init_file.Exists())
    {
        ExecutionContext *exe_ctx = NULL;  // We don't have any context yet.
        bool stop_on_continue = true;
        bool stop_on_error    = false;
        bool echo_commands    = false;
        bool print_results    = false;
        
        HandleCommandsFromFile (init_file, exe_ctx, stop_on_continue, stop_on_error, echo_commands, print_results, eLazyBoolNo, result);
    }
    else
    {
        // nothing to be done if the file doesn't exist
        result.SetStatus(eReturnStatusSuccessFinishNoResult);
    }
}

PlatformSP
CommandInterpreter::GetPlatform (bool prefer_target_platform)
{
    PlatformSP platform_sp;
    if (prefer_target_platform)
    {
        ExecutionContext exe_ctx(GetExecutionContext());
        Target *target = exe_ctx.GetTargetPtr();
        if (target)
            platform_sp = target->GetPlatform();
    }

    if (!platform_sp)
        platform_sp = m_debugger.GetPlatformList().GetSelectedPlatform();
    return platform_sp;
}

void
CommandInterpreter::HandleCommands (const StringList &commands, 
                                    ExecutionContext *override_context, 
                                    bool stop_on_continue,
                                    bool stop_on_error,
                                    bool echo_commands,
                                    bool print_results,
                                    LazyBool add_to_history,
                                    CommandReturnObject &result)
{
    size_t num_lines = commands.GetSize();
    
    // If we are going to continue past a "continue" then we need to run the commands synchronously.
    // Make sure you reset this value anywhere you return from the function.
    
    bool old_async_execution = m_debugger.GetAsyncExecution();
    
    // If we've been given an execution context, set it at the start, but don't keep resetting it or we will
    // cause series of commands that change the context, then do an operation that relies on that context to fail.
    
    if (override_context != NULL)
        UpdateExecutionContext (override_context);
            
    if (!stop_on_continue)
    {
        m_debugger.SetAsyncExecution (false);
    }

    for (int idx = 0; idx < num_lines; idx++)
    {
        const char *cmd = commands.GetStringAtIndex(idx);
        if (cmd[0] == '\0')
            continue;
            
        if (echo_commands)
        {
            result.AppendMessageWithFormat ("%s %s\n", 
                                             GetPrompt(), 
                                             cmd);
        }

        CommandReturnObject tmp_result;
        // If override_context is not NULL, pass no_context_switching = true for
        // HandleCommand() since we updated our context already.
        bool success = HandleCommand(cmd, add_to_history, tmp_result,
                                     NULL, /* override_context */
                                     true, /* repeat_on_empty_command */
                                     override_context != NULL /* no_context_switching */);
        
        if (print_results)
        {
            if (tmp_result.Succeeded())
              result.AppendMessageWithFormat("%s", tmp_result.GetOutputData());
        }
                
        if (!success || !tmp_result.Succeeded())
        {
            const char *error_msg = tmp_result.GetErrorData();
            if (error_msg == NULL || error_msg[0] == '\0')
                error_msg = "<unknown error>.\n";
            if (stop_on_error)
            {
                result.AppendErrorWithFormat("Aborting reading of commands after command #%d: '%s' failed with %s",
                                         idx, cmd, error_msg);
                result.SetStatus (eReturnStatusFailed);
                m_debugger.SetAsyncExecution (old_async_execution);
                return;
            }
            else if (print_results)
            {
                result.AppendMessageWithFormat ("Command #%d '%s' failed with %s",
                                                idx + 1, 
                                                cmd, 
                                                error_msg);
            }
        }
        
        if (result.GetImmediateOutputStream())
            result.GetImmediateOutputStream()->Flush();
        
        if (result.GetImmediateErrorStream())
            result.GetImmediateErrorStream()->Flush();
        
        // N.B. Can't depend on DidChangeProcessState, because the state coming into the command execution
        // could be running (for instance in Breakpoint Commands.
        // So we check the return value to see if it is has running in it.
        if ((tmp_result.GetStatus() == eReturnStatusSuccessContinuingNoResult)
                || (tmp_result.GetStatus() == eReturnStatusSuccessContinuingResult))
        {
            if (stop_on_continue)
            {
                // If we caused the target to proceed, and we're going to stop in that case, set the
                // status in our real result before returning.  This is an error if the continue was not the
                // last command in the set of commands to be run.
                if (idx != num_lines - 1)
                    result.AppendErrorWithFormat("Aborting reading of commands after command #%d: '%s' continued the target.\n", 
                                                 idx + 1, cmd);
                else
                    result.AppendMessageWithFormat ("Command #%d '%s' continued the target.\n", idx + 1, cmd);
                    
                result.SetStatus(tmp_result.GetStatus());
                m_debugger.SetAsyncExecution (old_async_execution);

                return;
            }
        }
        
    }
    
    result.SetStatus (eReturnStatusSuccessFinishResult);
    m_debugger.SetAsyncExecution (old_async_execution);

    return;
}

void
CommandInterpreter::HandleCommandsFromFile (FileSpec &cmd_file, 
                                            ExecutionContext *context, 
                                            bool stop_on_continue,
                                            bool stop_on_error,
                                            bool echo_command,
                                            bool print_result,
                                            LazyBool add_to_history,
                                            CommandReturnObject &result)
{
    if (cmd_file.Exists())
    {
        bool success;
        StringList commands;
        success = commands.ReadFileLines(cmd_file);
        if (!success)
        {
            result.AppendErrorWithFormat ("Error reading commands from file: %s.\n", cmd_file.GetFilename().AsCString());
            result.SetStatus (eReturnStatusFailed);
            return;
        }
        m_command_source_depth++;
        HandleCommands (commands, context, stop_on_continue, stop_on_error, echo_command, print_result, add_to_history, result);
        m_command_source_depth--;
    }
    else
    {
        result.AppendErrorWithFormat ("Error reading commands from file %s - file not found.\n", 
                                      cmd_file.GetFilename().AsCString());
        result.SetStatus (eReturnStatusFailed);
        return;
    }
}

ScriptInterpreter *
CommandInterpreter::GetScriptInterpreter (bool can_create)
{
    if (m_script_interpreter_ap.get() != NULL)
        return m_script_interpreter_ap.get();
    
    if (!can_create)
        return NULL;
 
    // <rdar://problem/11751427>
    // we need to protect the initialization of the script interpreter
    // otherwise we could end up with two threads both trying to create
    // their instance of it, and for some languages (e.g. Python)
    // this is a bulletproof recipe for disaster!
    // this needs to be a function-level static because multiple Debugger instances living in the same process
    // still need to be isolated and not try to initialize Python concurrently
    static Mutex g_interpreter_mutex(Mutex::eMutexTypeRecursive);
    Mutex::Locker interpreter_lock(g_interpreter_mutex);
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
        log->Printf("Initializing the ScriptInterpreter now\n");
    
    lldb::ScriptLanguage script_lang = GetDebugger().GetScriptLanguage();
    switch (script_lang)
    {
        case eScriptLanguagePython:
#ifndef LLDB_DISABLE_PYTHON
            m_script_interpreter_ap.reset (new ScriptInterpreterPython (*this));
            break;
#else
            // Fall through to the None case when python is disabled
#endif
        case eScriptLanguageNone:
            m_script_interpreter_ap.reset (new ScriptInterpreterNone (*this));
            break;
        default:
            break;
    };
    
    return m_script_interpreter_ap.get();
}



bool
CommandInterpreter::GetSynchronous ()
{
    return m_synchronous_execution;
}

void
CommandInterpreter::SetSynchronous (bool value)
{
    m_synchronous_execution  = value;
}

void
CommandInterpreter::OutputFormattedHelpText (Stream &strm,
                                             const char *word_text,
                                             const char *separator,
                                             const char *help_text,
                                             uint32_t max_word_len)
{
    const uint32_t max_columns = m_debugger.GetTerminalWidth();

    int indent_size = max_word_len + strlen (separator) + 2;

    strm.IndentMore (indent_size);
    
    StreamString text_strm;
    text_strm.Printf ("%-*s %s %s",  max_word_len, word_text, separator, help_text);
    
    size_t len = text_strm.GetSize();
    const char *text = text_strm.GetData();
    if (text[len - 1] == '\n')
    {
        text_strm.EOL();
        len = text_strm.GetSize();
    }

    if (len  < max_columns)
    {
        // Output it as a single line.
        strm.Printf ("%s", text);
    }
    else
    {
        // We need to break it up into multiple lines.
        bool first_line = true;
        int text_width;
        int start = 0;
        int end = start;
        int final_end = strlen (text);
        int sub_len;
        
        while (end < final_end)
        {
            if (first_line)
                text_width = max_columns - 1;
            else
                text_width = max_columns - indent_size - 1;

            // Don't start the 'text' on a space, since we're already outputting the indentation.
            if (!first_line)
            {
                while ((start < final_end) && (text[start] == ' '))
                  start++;
            }

            end = start + text_width;
            if (end > final_end)
                end = final_end;
            else
            {
                // If we're not at the end of the text, make sure we break the line on white space.
                while (end > start
                       && text[end] != ' ' && text[end] != '\t' && text[end] != '\n')
                    end--;
                assert (end > 0);
            }

            sub_len = end - start;
            if (start != 0)
              strm.EOL();
            if (!first_line)
                strm.Indent();
            else
                first_line = false;
            assert (start <= final_end);
            assert (start + sub_len <= final_end);
            if (sub_len > 0)
                strm.Write (text + start, sub_len);
            start = end + 1;
        }
    }
    strm.EOL();
    strm.IndentLess(indent_size);
}

void
CommandInterpreter::OutputHelpText (Stream &strm,
                                    const char *word_text,
                                    const char *separator,
                                    const char *help_text,
                                    uint32_t max_word_len)
{
    int indent_size = max_word_len + strlen (separator) + 2;
    
    strm.IndentMore (indent_size);
    
    StreamString text_strm;
    text_strm.Printf ("%-*s %s %s",  max_word_len, word_text, separator, help_text);
    
    const uint32_t max_columns = m_debugger.GetTerminalWidth();
    
    size_t len = text_strm.GetSize();
    const char *text = text_strm.GetData();
        
    uint32_t chars_left = max_columns;

    for (uint32_t i = 0; i < len; i++)
    {
        if ((text[i] == ' ' && ::strchr((text+i+1), ' ') && chars_left < ::strchr((text+i+1), ' ')-(text+i)) || text[i] == '\n')
        {
            chars_left = max_columns - indent_size;
            strm.EOL();
            strm.Indent();
        }
        else
        {
            strm.PutChar(text[i]);
            chars_left--;
        }
        
    }
    
    strm.EOL();
    strm.IndentLess(indent_size);
}

void
CommandInterpreter::FindCommandsForApropos (const char *search_word, StringList &commands_found,
                                            StringList &commands_help)
{
    CommandObject::CommandMap::const_iterator pos;

    for (pos = m_command_dict.begin(); pos != m_command_dict.end(); ++pos)
    {
        const char *command_name = pos->first.c_str();
        CommandObject *cmd_obj = pos->second.get();

        if (cmd_obj->HelpTextContainsWord (search_word))
        {
            commands_found.AppendString (command_name);
            commands_help.AppendString (cmd_obj->GetHelp());
        }

        if (cmd_obj->IsMultiwordObject())
            cmd_obj->AproposAllSubCommands (command_name,
                                            search_word,
                                            commands_found,
                                            commands_help);
      
    }
}


void
CommandInterpreter::UpdateExecutionContext (ExecutionContext *override_context)
{
    if (override_context != NULL)
    {
        m_exe_ctx_ref = *override_context;
    }
    else
    {
        const bool adopt_selected = true;
        m_exe_ctx_ref.SetTargetPtr (m_debugger.GetSelectedTarget().get(), adopt_selected);
    }
}

void
CommandInterpreter::DumpHistory (Stream &stream, uint32_t count) const
{
    DumpHistory (stream, 0, count - 1);
}

void
CommandInterpreter::DumpHistory (Stream &stream, uint32_t start, uint32_t end) const
{
    const size_t last_idx = std::min<size_t>(m_command_history.size(), end + 1);
    for (size_t i = start; i < last_idx; i++)
    {
        if (!m_command_history[i].empty())
        {
            stream.Indent();
            stream.Printf ("%4zu: %s\n", i, m_command_history[i].c_str());
        }
    }
}

const char *
CommandInterpreter::FindHistoryString (const char *input_str) const
{
    if (input_str[0] != m_repeat_char)
        return NULL;
    if (input_str[1] == '-')
    {
        bool success;
        uint32_t idx = Args::StringToUInt32 (input_str+2, 0, 0, &success);
        if (!success)
            return NULL;
        if (idx > m_command_history.size())
            return NULL;
        idx = m_command_history.size() - idx;
        return m_command_history[idx].c_str();
            
    }
    else if (input_str[1] == m_repeat_char)
    {
        if (m_command_history.empty())
            return NULL;
        else
            return m_command_history.back().c_str();
    }
    else
    {
        bool success;
        uint32_t idx = Args::StringToUInt32 (input_str+1, 0, 0, &success);
        if (!success)
            return NULL;
        if (idx >= m_command_history.size())
            return NULL;
        return m_command_history[idx].c_str();
    }
}
