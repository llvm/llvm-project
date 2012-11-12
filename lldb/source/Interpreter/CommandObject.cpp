//===-- CommandObject.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/CommandObject.h"

#include <string>
#include <map>

#include <getopt.h>
#include <stdlib.h>
#include <ctype.h>

#include "lldb/Core/Address.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Interpreter/Options.h"

// These are for the Sourcename completers.
// FIXME: Make a separate file for the completers.
#include "lldb/Host/FileSpec.h"
#include "lldb/Core/FileSpecList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Interpreter/ScriptInterpreterPython.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObject
//-------------------------------------------------------------------------

CommandObject::CommandObject 
(
    CommandInterpreter &interpreter, 
    const char *name, 
    const char *help, 
    const char *syntax, 
    uint32_t flags
) :
    m_interpreter (interpreter),
    m_cmd_name (name),
    m_cmd_help_short (),
    m_cmd_help_long (),
    m_cmd_syntax (),
    m_is_alias (false),
    m_flags (flags),
    m_arguments(),
    m_command_override_callback (NULL),
    m_command_override_baton (NULL)
{
    if (help && help[0])
        m_cmd_help_short = help;
    if (syntax && syntax[0])
        m_cmd_syntax = syntax;
}

CommandObject::~CommandObject ()
{
}

const char *
CommandObject::GetHelp ()
{
    return m_cmd_help_short.c_str();
}

const char *
CommandObject::GetHelpLong ()
{
    return m_cmd_help_long.c_str();
}

const char *
CommandObject::GetSyntax ()
{
    if (m_cmd_syntax.length() == 0)
    {
        StreamString syntax_str;
        syntax_str.Printf ("%s", GetCommandName());
        if (GetOptions() != NULL)
            syntax_str.Printf (" <cmd-options>");
        if (m_arguments.size() > 0)
        {
            syntax_str.Printf (" ");
            if (WantsRawCommandString())
                syntax_str.Printf("-- ");
            GetFormattedCommandArguments (syntax_str);
        }
        m_cmd_syntax = syntax_str.GetData ();
    }

    return m_cmd_syntax.c_str();
}

const char *
CommandObject::Translate ()
{
    //return m_cmd_func_name.c_str();
    return "This function is currently not implemented.";
}

const char *
CommandObject::GetCommandName ()
{
    return m_cmd_name.c_str();
}

void
CommandObject::SetCommandName (const char *name)
{
    m_cmd_name = name;
}

void
CommandObject::SetHelp (const char *cstr)
{
    m_cmd_help_short = cstr;
}

void
CommandObject::SetHelpLong (const char *cstr)
{
    m_cmd_help_long = cstr;
}

void
CommandObject::SetHelpLong (std::string str)
{
    m_cmd_help_long = str;
}

void
CommandObject::SetSyntax (const char *cstr)
{
    m_cmd_syntax = cstr;
}

Options *
CommandObject::GetOptions ()
{
    // By default commands don't have options unless this virtual function
    // is overridden by base classes.
    return NULL;
}

bool
CommandObject::ParseOptions
(
    Args& args,
    CommandReturnObject &result
)
{
    // See if the subclass has options?
    Options *options = GetOptions();
    if (options != NULL)
    {
        Error error;
        options->NotifyOptionParsingStarting();

        // ParseOptions calls getopt_long, which always skips the zero'th item in the array and starts at position 1,
        // so we need to push a dummy value into position zero.
        args.Unshift("dummy_string");
        error = args.ParseOptions (*options);

        // The "dummy_string" will have already been removed by ParseOptions,
        // so no need to remove it.

        if (error.Success())
            error = options->NotifyOptionParsingFinished();

        if (error.Success())
        {
            if (options->VerifyOptions (result))
                return true;
        }
        else
        {
            const char *error_cstr = error.AsCString();
            if (error_cstr)
            {
                // We got an error string, lets use that
                result.AppendError(error_cstr);
            }
            else
            {
                // No error string, output the usage information into result
                options->GenerateOptionUsage (result.GetErrorStream(), this);
            }
        }
        result.SetStatus (eReturnStatusFailed);
        return false;
    }
    return true;
}



bool
CommandObject::CheckFlags (CommandReturnObject &result)
{
    if (GetFlags().AnySet (CommandObject::eFlagProcessMustBeLaunched | CommandObject::eFlagProcessMustBePaused))
    {
        Process *process = m_interpreter.GetExecutionContext().GetProcessPtr();
        if (process == NULL)
        {
            // A process that is not running is considered paused.
            if (GetFlags().Test(CommandObject::eFlagProcessMustBeLaunched))
            {
                result.AppendError ("Process must exist.");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
        }
        else
        {
            StateType state = process->GetState();
            
            switch (state)
            {
            case eStateInvalid:
            case eStateSuspended:
            case eStateCrashed:
            case eStateStopped:
                break;
            
            case eStateConnected:
            case eStateAttaching:
            case eStateLaunching:
            case eStateDetached:
            case eStateExited:
            case eStateUnloaded:
                if (GetFlags().Test(CommandObject::eFlagProcessMustBeLaunched))
                {
                    result.AppendError ("Process must be launched.");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                break;

            case eStateRunning:
            case eStateStepping:
                if (GetFlags().Test(CommandObject::eFlagProcessMustBePaused))
                {
                    result.AppendError ("Process is running.  Use 'process interrupt' to pause execution.");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
        }
    }
    return true;
}

class CommandDictCommandPartialMatch
{
    public:
        CommandDictCommandPartialMatch (const char *match_str)
        {
            m_match_str = match_str;
        }
        bool operator() (const std::pair<std::string, lldb::CommandObjectSP> map_element) const
        {
            // A NULL or empty string matches everything.
            if (m_match_str == NULL || *m_match_str == '\0')
                return 1;

            size_t found = map_element.first.find (m_match_str, 0);
            if (found == std::string::npos)
                return 0;
            else
                return found == 0;
        }

    private:
        const char *m_match_str;
};

int
CommandObject::AddNamesMatchingPartialString (CommandObject::CommandMap &in_map, const char *cmd_str,
                                              StringList &matches)
{
    int number_added = 0;
    CommandDictCommandPartialMatch matcher(cmd_str);

    CommandObject::CommandMap::iterator matching_cmds = std::find_if (in_map.begin(), in_map.end(), matcher);

    while (matching_cmds != in_map.end())
    {
        ++number_added;
        matches.AppendString((*matching_cmds).first.c_str());
        matching_cmds = std::find_if (++matching_cmds, in_map.end(), matcher);;
    }
    return number_added;
}

int
CommandObject::HandleCompletion
(
    Args &input,
    int &cursor_index,
    int &cursor_char_position,
    int match_start_point,
    int max_return_elements,
    bool &word_complete,
    StringList &matches
)
{
    // Default implmentation of WantsCompletion() is !WantsRawCommandString().
    // Subclasses who want raw command string but desire, for example,
    // argument completion should override WantsCompletion() to return true,
    // instead.
    if (WantsRawCommandString() && !WantsCompletion())
    {
        // FIXME: Abstract telling the completion to insert the completion character.
        matches.Clear();
        return -1;
    }
    else
    {
        // Can we do anything generic with the options?
        Options *cur_options = GetOptions();
        CommandReturnObject result;
        OptionElementVector opt_element_vector;

        if (cur_options != NULL)
        {
            // Re-insert the dummy command name string which will have been
            // stripped off:
            input.Unshift ("dummy-string");
            cursor_index++;


            // I stick an element on the end of the input, because if the last element is
            // option that requires an argument, getopt_long will freak out.

            input.AppendArgument ("<FAKE-VALUE>");

            input.ParseArgsForCompletion (*cur_options, opt_element_vector, cursor_index);

            input.DeleteArgumentAtIndex(input.GetArgumentCount() - 1);

            bool handled_by_options;
            handled_by_options = cur_options->HandleOptionCompletion (input,
                                                                      opt_element_vector,
                                                                      cursor_index,
                                                                      cursor_char_position,
                                                                      match_start_point,
                                                                      max_return_elements,
                                                                      word_complete,
                                                                      matches);
            if (handled_by_options)
                return matches.GetSize();
        }

        // If we got here, the last word is not an option or an option argument.
        return HandleArgumentCompletion (input,
                                         cursor_index,
                                         cursor_char_position,
                                         opt_element_vector,
                                         match_start_point,
                                         max_return_elements,
                                         word_complete,
                                         matches);
    }
}

bool
CommandObject::HelpTextContainsWord (const char *search_word)
{
    std::string options_usage_help;

    bool found_word = false;

    const char *short_help = GetHelp();
    const char *long_help = GetHelpLong();
    const char *syntax_help = GetSyntax();
    
    if (short_help && strcasestr (short_help, search_word))
        found_word = true;
    else if (long_help && strcasestr (long_help, search_word))
        found_word = true;
    else if (syntax_help && strcasestr (syntax_help, search_word))
        found_word = true;

    if (!found_word
        && GetOptions() != NULL)
    {
        StreamString usage_help;
        GetOptions()->GenerateOptionUsage (usage_help, this);
        if (usage_help.GetSize() > 0)
        {
            const char *usage_text = usage_help.GetData();
            if (strcasestr (usage_text, search_word))
              found_word = true;
        }
    }

    return found_word;
}

int
CommandObject::GetNumArgumentEntries  ()
{
    return m_arguments.size();
}

CommandObject::CommandArgumentEntry *
CommandObject::GetArgumentEntryAtIndex (int idx)
{
    if (idx < m_arguments.size())
        return &(m_arguments[idx]);

    return NULL;
}

CommandObject::ArgumentTableEntry *
CommandObject::FindArgumentDataByType (CommandArgumentType arg_type)
{
    const ArgumentTableEntry *table = CommandObject::GetArgumentTable();

    for (int i = 0; i < eArgTypeLastArg; ++i)
        if (table[i].arg_type == arg_type)
            return (ArgumentTableEntry *) &(table[i]);

    return NULL;
}

void
CommandObject::GetArgumentHelp (Stream &str, CommandArgumentType arg_type, CommandInterpreter &interpreter)
{
    const ArgumentTableEntry* table = CommandObject::GetArgumentTable();
    ArgumentTableEntry *entry = (ArgumentTableEntry *) &(table[arg_type]);
    
    // The table is *supposed* to be kept in arg_type order, but someone *could* have messed it up...

    if (entry->arg_type != arg_type)
        entry = CommandObject::FindArgumentDataByType (arg_type);

    if (!entry)
        return;

    StreamString name_str;
    name_str.Printf ("<%s>", entry->arg_name);

    if (entry->help_function)
    {
        const char* help_text = entry->help_function();
        if (!entry->help_function.self_formatting)
        {
            interpreter.OutputFormattedHelpText (str, name_str.GetData(), "--", help_text,
                                                 name_str.GetSize());
        }
        else
        {
            interpreter.OutputHelpText(str, name_str.GetData(), "--", help_text,
                                       name_str.GetSize());
        }
    }
    else
        interpreter.OutputFormattedHelpText (str, name_str.GetData(), "--", entry->help_text, name_str.GetSize());
}

const char *
CommandObject::GetArgumentName (CommandArgumentType arg_type)
{
    ArgumentTableEntry *entry = (ArgumentTableEntry *) &(CommandObject::GetArgumentTable()[arg_type]);

    // The table is *supposed* to be kept in arg_type order, but someone *could* have messed it up...

    if (entry->arg_type != arg_type)
        entry = CommandObject::FindArgumentDataByType (arg_type);

    if (entry)
        return entry->arg_name;

    StreamString str;
    str << "Arg name for type (" << arg_type << ") not in arg table!";
    return str.GetData();
}

bool
CommandObject::IsPairType (ArgumentRepetitionType arg_repeat_type)
{
    if ((arg_repeat_type == eArgRepeatPairPlain)
        ||  (arg_repeat_type == eArgRepeatPairOptional)
        ||  (arg_repeat_type == eArgRepeatPairPlus)
        ||  (arg_repeat_type == eArgRepeatPairStar)
        ||  (arg_repeat_type == eArgRepeatPairRange)
        ||  (arg_repeat_type == eArgRepeatPairRangeOptional))
        return true;

    return false;
}

static CommandObject::CommandArgumentEntry
OptSetFiltered(uint32_t opt_set_mask, CommandObject::CommandArgumentEntry &cmd_arg_entry)
{
    CommandObject::CommandArgumentEntry ret_val;
    for (unsigned i = 0; i < cmd_arg_entry.size(); ++i)
        if (opt_set_mask & cmd_arg_entry[i].arg_opt_set_association)
            ret_val.push_back(cmd_arg_entry[i]);
    return ret_val;
}

// Default parameter value of opt_set_mask is LLDB_OPT_SET_ALL, which means take
// all the argument data into account.  On rare cases where some argument sticks
// with certain option sets, this function returns the option set filtered args.
void
CommandObject::GetFormattedCommandArguments (Stream &str, uint32_t opt_set_mask)
{
    int num_args = m_arguments.size();
    for (int i = 0; i < num_args; ++i)
    {
        if (i > 0)
            str.Printf (" ");
        CommandArgumentEntry arg_entry =
            opt_set_mask == LLDB_OPT_SET_ALL ? m_arguments[i]
                                             : OptSetFiltered(opt_set_mask, m_arguments[i]);
        int num_alternatives = arg_entry.size();

        if ((num_alternatives == 2)
            && IsPairType (arg_entry[0].arg_repetition))
        {
            const char *first_name = GetArgumentName (arg_entry[0].arg_type);
            const char *second_name = GetArgumentName (arg_entry[1].arg_type);
            switch (arg_entry[0].arg_repetition)
            {
                case eArgRepeatPairPlain:
                    str.Printf ("<%s> <%s>", first_name, second_name);
                    break;
                case eArgRepeatPairOptional:
                    str.Printf ("[<%s> <%s>]", first_name, second_name);
                    break;
                case eArgRepeatPairPlus:
                    str.Printf ("<%s> <%s> [<%s> <%s> [...]]", first_name, second_name, first_name, second_name);
                    break;
                case eArgRepeatPairStar:
                    str.Printf ("[<%s> <%s> [<%s> <%s> [...]]]", first_name, second_name, first_name, second_name);
                    break;
                case eArgRepeatPairRange:
                    str.Printf ("<%s_1> <%s_1> ... <%s_n> <%s_n>", first_name, second_name, first_name, second_name);
                    break;
                case eArgRepeatPairRangeOptional:
                    str.Printf ("[<%s_1> <%s_1> ... <%s_n> <%s_n>]", first_name, second_name, first_name, second_name);
                    break;
                // Explicitly test for all the rest of the cases, so if new types get added we will notice the
                // missing case statement(s).
                case eArgRepeatPlain:
                case eArgRepeatOptional:
                case eArgRepeatPlus:
                case eArgRepeatStar:
                case eArgRepeatRange:
                    // These should not be reached, as they should fail the IsPairType test above.
                    break;
            }
        }
        else
        {
            StreamString names;
            for (int j = 0; j < num_alternatives; ++j)
            {
                if (j > 0)
                    names.Printf (" | ");
                names.Printf ("%s", GetArgumentName (arg_entry[j].arg_type));
            }
            switch (arg_entry[0].arg_repetition)
            {
                case eArgRepeatPlain:
                    str.Printf ("<%s>", names.GetData());
                    break;
                case eArgRepeatPlus:
                    str.Printf ("<%s> [<%s> [...]]", names.GetData(), names.GetData());
                    break;
                case eArgRepeatStar:
                    str.Printf ("[<%s> [<%s> [...]]]", names.GetData(), names.GetData());
                    break;
                case eArgRepeatOptional:
                    str.Printf ("[<%s>]", names.GetData());
                    break;
                case eArgRepeatRange:
                    str.Printf ("<%s_1> .. <%s_n>", names.GetData(), names.GetData());
                    break;
                // Explicitly test for all the rest of the cases, so if new types get added we will notice the
                // missing case statement(s).
                case eArgRepeatPairPlain:
                case eArgRepeatPairOptional:
                case eArgRepeatPairPlus:
                case eArgRepeatPairStar:
                case eArgRepeatPairRange:
                case eArgRepeatPairRangeOptional:
                    // These should not be hit, as they should pass the IsPairType test above, and control should
                    // have gone into the other branch of the if statement.
                    break;
            }
        }
    }
}

CommandArgumentType
CommandObject::LookupArgumentName (const char *arg_name)
{
    CommandArgumentType return_type = eArgTypeLastArg;

    std::string arg_name_str (arg_name);
    size_t len = arg_name_str.length();
    if (arg_name[0] == '<'
        && arg_name[len-1] == '>')
        arg_name_str = arg_name_str.substr (1, len-2);

    const ArgumentTableEntry *table = GetArgumentTable();
    for (int i = 0; i < eArgTypeLastArg; ++i)
        if (arg_name_str.compare (table[i].arg_name) == 0)
            return_type = g_arguments_data[i].arg_type;

    return return_type;
}

static const char *
RegisterNameHelpTextCallback ()
{
    return "Register names can be specified using the architecture specific names.  "
    "They can also be specified using generic names.  Not all generic entities have "
    "registers backing them on all architectures.  When they don't the generic name "
    "will return an error.\n"
    "The generic names defined in lldb are:\n"
    "\n"
    "pc       - program counter register\n"
    "ra       - return address register\n"
    "fp       - frame pointer register\n"
    "sp       - stack pointer register\n"
    "flags    - the flags register\n"
    "arg{1-6} - integer argument passing registers.\n";
}

static const char *
BreakpointIDHelpTextCallback ()
{
    return "Breakpoint ID's consist major and minor numbers;  the major number "
    "corresponds to the single entity that was created with a 'breakpoint set' "
    "command; the minor numbers correspond to all the locations that were actually "
    "found/set based on the major breakpoint.  A full breakpoint ID might look like "
    "3.14, meaning the 14th location set for the 3rd breakpoint.  You can specify "
    "all the locations of a breakpoint by just indicating the major breakpoint "
    "number. A valid breakpoint id consists either of just the major id number, "
    "or the major number, a dot, and the location number (e.g. 3 or 3.2 could "
    "both be valid breakpoint ids).";
}

static const char *
BreakpointIDRangeHelpTextCallback ()
{
    return "A 'breakpoint id list' is a manner of specifying multiple breakpoints. "
    "This can be done  through several mechanisms.  The easiest way is to just "
    "enter a space-separated list of breakpoint ids.  To specify all the "
    "breakpoint locations under a major breakpoint, you can use the major "
    "breakpoint number followed by '.*', eg. '5.*' means all the locations under "
    "breakpoint 5.  You can also indicate a range of breakpoints by using "
    "<start-bp-id> - <end-bp-id>.  The start-bp-id and end-bp-id for a range can "
    "be any valid breakpoint ids.  It is not legal, however, to specify a range "
    "using specific locations that cross major breakpoint numbers.  I.e. 3.2 - 3.7"
    " is legal; 2 - 5 is legal; but 3.2 - 4.4 is not legal.";
}

static const char *
GDBFormatHelpTextCallback ()
{
    return "A GDB format consists of a repeat count, a format letter and a size letter. "
    "The repeat count is optional and defaults to 1. The format letter is optional "
    "and defaults to the previous format that was used. The size letter is optional "
    "and defaults to the previous size that was used.\n"
    "\n"
    "Format letters include:\n"
    "o - octal\n"
    "x - hexadecimal\n"
    "d - decimal\n"
    "u - unsigned decimal\n"
    "t - binary\n"
    "f - float\n"
    "a - address\n"
    "i - instruction\n"
    "c - char\n"
    "s - string\n"
    "T - OSType\n"
    "A - float as hex\n"
    "\n"
    "Size letters include:\n"
    "b - 1 byte  (byte)\n"
    "h - 2 bytes (halfword)\n"
    "w - 4 bytes (word)\n"
    "g - 8 bytes (giant)\n"
    "\n"
    "Example formats:\n"
    "32xb - show 32 1 byte hexadecimal integer values\n"
    "16xh - show 16 2 byte hexadecimal integer values\n"
    "64   - show 64 2 byte hexadecimal integer values (format and size from the last format)\n"
    "dw   - show 1 4 byte decimal integer value\n"
    ;
} 

static const char *
FormatHelpTextCallback ()
{
    
    static char* help_text_ptr = NULL;
    
    if (help_text_ptr)
        return help_text_ptr;
    
    StreamString sstr;
    sstr << "One of the format names (or one-character names) that can be used to show a variable's value:\n";
    for (Format f = eFormatDefault; f < kNumFormats; f = Format(f+1))
    {
        if (f != eFormatDefault)
            sstr.PutChar('\n');
        
        char format_char = FormatManager::GetFormatAsFormatChar(f);
        if (format_char)
            sstr.Printf("'%c' or ", format_char);
        
        sstr.Printf ("\"%s\"", FormatManager::GetFormatAsCString(f));
    }
    
    sstr.Flush();
    
    std::string data = sstr.GetString();
    
    help_text_ptr = new char[data.length()+1];
    
    data.copy(help_text_ptr, data.length());
    
    return help_text_ptr;
}

static const char *
LanguageTypeHelpTextCallback ()
{
    static char* help_text_ptr = NULL;
    
    if (help_text_ptr)
        return help_text_ptr;
    
    StreamString sstr;
    sstr << "One of the following languages:\n";
    
    for (LanguageType l = eLanguageTypeUnknown; l < eNumLanguageTypes; ++l)
    {
        sstr << "  " << LanguageRuntime::GetNameForLanguageType(l) << "\n";
    }
    
    sstr.Flush();
    
    std::string data = sstr.GetString();
    
    help_text_ptr = new char[data.length()+1];
    
    data.copy(help_text_ptr, data.length());
    
    return help_text_ptr;
}

static const char *
SummaryStringHelpTextCallback()
{
    return
        "A summary string is a way to extract information from variables in order to present them using a summary.\n"
        "Summary strings contain static text, variables, scopes and control sequences:\n"
        "  - Static text can be any sequence of non-special characters, i.e. anything but '{', '}', '$', or '\\'.\n"
        "  - Variables are sequences of characters beginning with ${, ending with } and that contain symbols in the format described below.\n"
        "  - Scopes are any sequence of text between { and }. Anything included in a scope will only appear in the output summary if there were no errors.\n"
        "  - Control sequences are the usual C/C++ '\\a', '\\n', ..., plus '\\$', '\\{' and '\\}'.\n"
        "A summary string works by copying static text verbatim, turning control sequences into their character counterpart, expanding variables and trying to expand scopes.\n"
        "A variable is expanded by giving it a value other than its textual representation, and the way this is done depends on what comes after the ${ marker.\n"
        "The most common sequence if ${var followed by an expression path, which is the text one would type to access a member of an aggregate types, given a variable of that type"
        " (e.g. if type T has a member named x, which has a member named y, and if t is of type T, the expression path would be .x.y and the way to fit that into a summary string would be"
        " ${var.x.y}). You can also use ${*var followed by an expression path and in that case the object referred by the path will be dereferenced before being displayed."
        " If the object is not a pointer, doing so will cause an error. For additional details on expression paths, you can type 'help expr-path'. \n"
        "By default, summary strings attempt to display the summary for any variable they reference, and if that fails the value. If neither can be shown, nothing is displayed."
        "In a summary string, you can also use an array index [n], or a slice-like range [n-m]. This can have two different meanings depending on what kind of object the expression"
        " path refers to:\n"
        "  - if it is a scalar type (any basic type like int, float, ...) the expression is a bitfield, i.e. the bits indicated by the indexing operator are extracted out of the number"
        " and displayed as an individual variable\n"
        "  - if it is an array or pointer the array items indicated by the indexing operator are shown as the result of the variable. if the expression is an array, real array items are"
        " printed; if it is a pointer, the pointer-as-array syntax is used to obtain the values (this means, the latter case can have no range checking)\n"
        "If you are trying to display an array for which the size is known, you can also use [] instead of giving an exact range. This has the effect of showing items 0 thru size - 1.\n"
        "Additionally, a variable can contain an (optional) format code, as in ${var.x.y%code}, where code can be any of the valid formats described in 'help format', or one of the"
        " special symbols only allowed as part of a variable:\n"
        "    %V: show the value of the object by default\n"
        "    %S: show the summary of the object by default\n"
        "    %@: show the runtime-provided object description (for Objective-C, it calls NSPrintForDebugger; for C/C++ it does nothing)\n"
        "    %L: show the location of the object (memory address or a register name)\n"
        "    %#: show the number of children of the object\n"
        "    %T: show the type of the object\n"
        "Another variable that you can use in summary strings is ${svar . This sequence works exactly like ${var, including the fact that ${*svar is an allowed sequence, but uses"
        " the object's synthetic children provider instead of the actual objects. For instance, if you are using STL synthetic children providers, the following summary string would"
        " count the number of actual elements stored in an std::list:\n"
        "type summary add -s \"${svar%#}\" -x \"std::list<\"";
}

static const char *
ExprPathHelpTextCallback()
{
    return
    "An expression path is the sequence of symbols that is used in C/C++ to access a member variable of an aggregate object (class).\n"
    "For instance, given a class:\n"
    "  class foo {\n"
    "      int a;\n"
    "      int b; .\n"
    "      foo* next;\n"
    "  };\n"
    "the expression to read item b in the item pointed to by next for foo aFoo would be aFoo.next->b.\n"
    "Given that aFoo could just be any object of type foo, the string '.next->b' is the expression path, because it can be attached to any foo instance to achieve the effect.\n"
    "Expression paths in LLDB include dot (.) and arrow (->) operators, and most commands using expression paths have ways to also accept the star (*) operator.\n"
    "The meaning of these operators is the same as the usual one given to them by the C/C++ standards.\n"
    "LLDB also has support for indexing ([ ]) in expression paths, and extends the traditional meaning of the square brackets operator to allow bitfield extraction:\n"
    "for objects of native types (int, float, char, ...) saying '[n-m]' as an expression path (where n and m are any positive integers, e.g. [3-5]) causes LLDB to extract"
    " bits n thru m from the value of the variable. If n == m, [n] is also allowed as a shortcut syntax. For arrays and pointers, expression paths can only contain one index"
    " and the meaning of the operation is the same as the one defined by C/C++ (item extraction). Some commands extend bitfield-like syntax for arrays and pointers with the"
    " meaning of array slicing (taking elements n thru m inside the array or pointed-to memory).";
}

void
CommandObject::AddIDsArgumentData(CommandArgumentEntry &arg, CommandArgumentType ID, CommandArgumentType IDRange)
{
    CommandArgumentData id_arg;
    CommandArgumentData id_range_arg;

    // Create the first variant for the first (and only) argument for this command.
    id_arg.arg_type = ID;
    id_arg.arg_repetition = eArgRepeatOptional;

    // Create the second variant for the first (and only) argument for this command.
    id_range_arg.arg_type = IDRange;
    id_range_arg.arg_repetition = eArgRepeatOptional;

    // The first (and only) argument for this command could be either an id or an id_range.
    // Push both variants into the entry for the first argument for this command.
    arg.push_back(id_arg);
    arg.push_back(id_range_arg);
}

const char * 
CommandObject::GetArgumentTypeAsCString (const lldb::CommandArgumentType arg_type)
{
    if (arg_type >=0 && arg_type < eArgTypeLastArg)
        return g_arguments_data[arg_type].arg_name;
    return NULL;

}

const char * 
CommandObject::GetArgumentDescriptionAsCString (const lldb::CommandArgumentType arg_type)
{
    if (arg_type >=0 && arg_type < eArgTypeLastArg)
        return g_arguments_data[arg_type].help_text;
    return NULL;
}

bool
CommandObjectParsed::Execute (const char *args_string, CommandReturnObject &result)
{
    CommandOverrideCallback command_callback = GetOverrideCallback();
    bool handled = false;
    Args cmd_args (args_string);
    if (command_callback)
    {
        Args full_args (GetCommandName ());
        full_args.AppendArguments(cmd_args);
        handled = command_callback (GetOverrideCallbackBaton(), full_args.GetConstArgumentVector());
    }
    if (!handled)
    {
        for (size_t i = 0; i < cmd_args.GetArgumentCount();  ++i)
        {
            const char *tmp_str = cmd_args.GetArgumentAtIndex (i);
            if (tmp_str[0] == '`')  // back-quote
                cmd_args.ReplaceArgumentAtIndex (i, m_interpreter.ProcessEmbeddedScriptCommands (tmp_str));
        }

        if (!CheckFlags(result))
            return false;
            
        if (!ParseOptions (cmd_args, result))
            return false;

        // Call the command-specific version of 'Execute', passing it the already processed arguments.
        handled = DoExecute (cmd_args, result);
    }
    return handled;
}

bool
CommandObjectRaw::Execute (const char *args_string, CommandReturnObject &result)
{
    CommandOverrideCallback command_callback = GetOverrideCallback();
    bool handled = false;
    if (command_callback)
    {
        std::string full_command (GetCommandName ());
        full_command += ' ';
        full_command += args_string;
        const char *argv[2] = { NULL, NULL };
        argv[0] = full_command.c_str();
        handled = command_callback (GetOverrideCallbackBaton(), argv);
    }
    if (!handled)
    {
        if (!CheckFlags(result))
            return false;
        else
            handled = DoExecute (args_string, result);
    }
    return handled;
}

static
const char *arch_helper()
{
    static StreamString g_archs_help;
    if (g_archs_help.Empty())
    {
        StringList archs;
        ArchSpec::AutoComplete(NULL, archs);
        g_archs_help.Printf("These are the supported architecture names:\n");
        archs.Join("\n", g_archs_help);
    }
    return g_archs_help.GetData();
}

CommandObject::ArgumentTableEntry
CommandObject::g_arguments_data[] =
{
    { eArgTypeAddress, "address", CommandCompletions::eNoCompletion, { NULL, false }, "A valid address in the target program's execution space." },
    { eArgTypeAliasName, "alias-name", CommandCompletions::eNoCompletion, { NULL, false }, "The name of an abbreviation (alias) for a debugger command." },
    { eArgTypeAliasOptions, "options-for-aliased-command", CommandCompletions::eNoCompletion, { NULL, false }, "Command options to be used as part of an alias (abbreviation) definition.  (See 'help commands alias' for more information.)" },
    { eArgTypeArchitecture, "arch", CommandCompletions::eArchitectureCompletion, { arch_helper, true }, "The architecture name, e.g. i386 or x86_64." },
    { eArgTypeBoolean, "boolean", CommandCompletions::eNoCompletion, { NULL, false }, "A Boolean value: 'true' or 'false'" },
    { eArgTypeBreakpointID, "breakpt-id", CommandCompletions::eNoCompletion, { BreakpointIDHelpTextCallback, false }, NULL },
    { eArgTypeBreakpointIDRange, "breakpt-id-list", CommandCompletions::eNoCompletion, { BreakpointIDRangeHelpTextCallback, false }, NULL },
    { eArgTypeByteSize, "byte-size", CommandCompletions::eNoCompletion, { NULL, false }, "Number of bytes to use." },
    { eArgTypeClassName, "class-name", CommandCompletions::eNoCompletion, { NULL, false }, "Then name of a class from the debug information in the program." },
    { eArgTypeCommandName, "cmd-name", CommandCompletions::eNoCompletion, { NULL, false }, "A debugger command (may be multiple words), without any options or arguments." },
    { eArgTypeCount, "count", CommandCompletions::eNoCompletion, { NULL, false }, "An unsigned integer." },
    { eArgTypeDirectoryName, "directory", CommandCompletions::eDiskDirectoryCompletion, { NULL, false }, "A directory name." },
    { eArgTypeEndAddress, "end-address", CommandCompletions::eNoCompletion, { NULL, false }, "Help text goes here." },
    { eArgTypeExpression, "expr", CommandCompletions::eNoCompletion, { NULL, false }, "Help text goes here." },
    { eArgTypeExpressionPath, "expr-path", CommandCompletions::eNoCompletion, { ExprPathHelpTextCallback, true }, NULL },
    { eArgTypeExprFormat, "expression-format", CommandCompletions::eNoCompletion, { NULL, false }, "[ [bool|b] | [bin] | [char|c] | [oct|o] | [dec|i|d|u] | [hex|x] | [float|f] | [cstr|s] ]" },
    { eArgTypeFilename, "filename", CommandCompletions::eDiskFileCompletion, { NULL, false }, "The name of a file (can include path)." },
    { eArgTypeFormat, "format", CommandCompletions::eNoCompletion, { FormatHelpTextCallback, true }, NULL },
    { eArgTypeFrameIndex, "frame-index", CommandCompletions::eNoCompletion, { NULL, false }, "Index into a thread's list of frames." },
    { eArgTypeFullName, "fullname", CommandCompletions::eNoCompletion, { NULL, false }, "Help text goes here." },
    { eArgTypeFunctionName, "function-name", CommandCompletions::eNoCompletion, { NULL, false }, "The name of a function." },
    { eArgTypeFunctionOrSymbol, "function-or-symbol", CommandCompletions::eNoCompletion, { NULL, false }, "The name of a function or symbol." },
    { eArgTypeGDBFormat, "gdb-format", CommandCompletions::eNoCompletion, { GDBFormatHelpTextCallback, true }, NULL },
    { eArgTypeIndex, "index", CommandCompletions::eNoCompletion, { NULL, false }, "An index into a list." },
    { eArgTypeLanguage, "language", CommandCompletions::eNoCompletion, { LanguageTypeHelpTextCallback, true }, NULL },
    { eArgTypeLineNum, "linenum", CommandCompletions::eNoCompletion, { NULL, false }, "Line number in a source file." },
    { eArgTypeLogCategory, "log-category", CommandCompletions::eNoCompletion, { NULL, false }, "The name of a category within a log channel, e.g. all (try \"log list\" to see a list of all channels and their categories." },
    { eArgTypeLogChannel, "log-channel", CommandCompletions::eNoCompletion, { NULL, false }, "The name of a log channel, e.g. process.gdb-remote (try \"log list\" to see a list of all channels and their categories)." },
    { eArgTypeMethod, "method", CommandCompletions::eNoCompletion, { NULL, false }, "A C++ method name." },
    { eArgTypeName, "name", CommandCompletions::eNoCompletion, { NULL, false }, "Help text goes here." },
    { eArgTypeNewPathPrefix, "new-path-prefix", CommandCompletions::eNoCompletion, { NULL, false }, "Help text goes here." },
    { eArgTypeNumLines, "num-lines", CommandCompletions::eNoCompletion, { NULL, false }, "The number of lines to use." },
    { eArgTypeNumberPerLine, "number-per-line", CommandCompletions::eNoCompletion, { NULL, false }, "The number of items per line to display." },
    { eArgTypeOffset, "offset", CommandCompletions::eNoCompletion, { NULL, false }, "Help text goes here." },
    { eArgTypeOldPathPrefix, "old-path-prefix", CommandCompletions::eNoCompletion, { NULL, false }, "Help text goes here." },
    { eArgTypeOneLiner, "one-line-command", CommandCompletions::eNoCompletion, { NULL, false }, "A command that is entered as a single line of text." },
    { eArgTypePid, "pid", CommandCompletions::eNoCompletion, { NULL, false }, "The process ID number." },
    { eArgTypePlugin, "plugin", CommandCompletions::eNoCompletion, { NULL, false }, "Help text goes here." },
    { eArgTypeProcessName, "process-name", CommandCompletions::eNoCompletion, { NULL, false }, "The name of the process." },
    { eArgTypePythonClass, "python-class", CommandCompletions::eNoCompletion, { NULL, false }, "The name of a Python class." },
    { eArgTypePythonFunction, "python-function", CommandCompletions::eNoCompletion, { NULL, false }, "The name of a Python function." },
    { eArgTypePythonScript, "python-script", CommandCompletions::eNoCompletion, { NULL, false }, "Source code written in Python." },
    { eArgTypeQueueName, "queue-name", CommandCompletions::eNoCompletion, { NULL, false }, "The name of the thread queue." },
    { eArgTypeRegisterName, "register-name", CommandCompletions::eNoCompletion, { RegisterNameHelpTextCallback, true }, NULL },
    { eArgTypeRegularExpression, "regular-expression", CommandCompletions::eNoCompletion, { NULL, false }, "A regular expression." },
    { eArgTypeRunArgs, "run-args", CommandCompletions::eNoCompletion, { NULL, false }, "Arguments to be passed to the target program when it starts executing." },
    { eArgTypeRunMode, "run-mode", CommandCompletions::eNoCompletion, { NULL, false }, "Help text goes here." },
    { eArgTypeScriptedCommandSynchronicity, "script-cmd-synchronicity", CommandCompletions::eNoCompletion, { NULL, false }, "The synchronicity to use to run scripted commands with regard to LLDB event system." },
    { eArgTypeScriptLang, "script-language", CommandCompletions::eNoCompletion, { NULL, false }, "The scripting language to be used for script-based commands.  Currently only Python is valid." },
    { eArgTypeSearchWord, "search-word", CommandCompletions::eNoCompletion, { NULL, false }, "The word for which you wish to search for information about." },
    { eArgTypeSelector, "selector", CommandCompletions::eNoCompletion, { NULL, false }, "An Objective-C selector name." },
    { eArgTypeSettingIndex, "setting-index", CommandCompletions::eNoCompletion, { NULL, false }, "An index into a settings variable that is an array (try 'settings list' to see all the possible settings variables and their types)." },
    { eArgTypeSettingKey, "setting-key", CommandCompletions::eNoCompletion, { NULL, false }, "A key into a settings variables that is a dictionary (try 'settings list' to see all the possible settings variables and their types)." },
    { eArgTypeSettingPrefix, "setting-prefix", CommandCompletions::eNoCompletion, { NULL, false }, "The name of a settable internal debugger variable up to a dot ('.'), e.g. 'target.process.'" },
    { eArgTypeSettingVariableName, "setting-variable-name", CommandCompletions::eNoCompletion, { NULL, false }, "The name of a settable internal debugger variable.  Type 'settings list' to see a complete list of such variables." }, 
    { eArgTypeShlibName, "shlib-name", CommandCompletions::eNoCompletion, { NULL, false }, "The name of a shared library." },
    { eArgTypeSourceFile, "source-file", CommandCompletions::eSourceFileCompletion, { NULL, false }, "The name of a source file.." },
    { eArgTypeSortOrder, "sort-order", CommandCompletions::eNoCompletion, { NULL, false }, "Specify a sort order when dumping lists." },
    { eArgTypeStartAddress, "start-address", CommandCompletions::eNoCompletion, { NULL, false }, "Help text goes here." },
    { eArgTypeSummaryString, "summary-string", CommandCompletions::eNoCompletion, { SummaryStringHelpTextCallback, true }, NULL },
    { eArgTypeSymbol, "symbol", CommandCompletions::eSymbolCompletion, { NULL, false }, "Any symbol name (function name, variable, argument, etc.)" },
    { eArgTypeThreadID, "thread-id", CommandCompletions::eNoCompletion, { NULL, false }, "Thread ID number." },
    { eArgTypeThreadIndex, "thread-index", CommandCompletions::eNoCompletion, { NULL, false }, "Index into the process' list of threads." },
    { eArgTypeThreadName, "thread-name", CommandCompletions::eNoCompletion, { NULL, false }, "The thread's name." },
    { eArgTypeUnsignedInteger, "unsigned-integer", CommandCompletions::eNoCompletion, { NULL, false }, "An unsigned integer." },
    { eArgTypeUnixSignal, "unix-signal", CommandCompletions::eNoCompletion, { NULL, false }, "A valid Unix signal name or number (e.g. SIGKILL, KILL or 9)." },
    { eArgTypeVarName, "variable-name", CommandCompletions::eNoCompletion, { NULL, false }, "The name of a variable in your program." },
    { eArgTypeValue, "value", CommandCompletions::eNoCompletion, { NULL, false }, "A value could be anything, depending on where and how it is used." },
    { eArgTypeWidth, "width", CommandCompletions::eNoCompletion, { NULL, false }, "Help text goes here." },
    { eArgTypeNone, "none", CommandCompletions::eNoCompletion, { NULL, false }, "No help available for this." },
    { eArgTypePlatform, "platform-name", CommandCompletions::ePlatformPluginCompletion, { NULL, false }, "The name of an installed platform plug-in . Type 'platform list' to see a complete list of installed platforms." },
    { eArgTypeWatchpointID, "watchpt-id", CommandCompletions::eNoCompletion, { NULL, false }, "Watchpoint IDs are positive integers." },
    { eArgTypeWatchpointIDRange, "watchpt-id-list", CommandCompletions::eNoCompletion, { NULL, false }, "For example, '1-3' or '1 to 3'." },
    { eArgTypeWatchType, "watch-type", CommandCompletions::eNoCompletion, { NULL, false }, "Specify the type for a watchpoint." }
};

const CommandObject::ArgumentTableEntry*
CommandObject::GetArgumentTable ()
{
    // If this assertion fires, then the table above is out of date with the CommandArgumentType enumeration
    assert ((sizeof (CommandObject::g_arguments_data) / sizeof (CommandObject::ArgumentTableEntry)) == eArgTypeLastArg);
    return CommandObject::g_arguments_data;
}


