//===-- Driver.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"

#include <getopt.h>
#include <libgen.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <fcntl.h>

#include <string>

#include "IOChannel.h"
#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBCommunication.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBHostOS.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBProcess.h"

using namespace lldb;

static void reset_stdin_termios ();
static bool g_old_stdin_termios_is_valid = false;
static struct termios g_old_stdin_termios;

static char *g_debugger_name =  (char *) "";
static Driver *g_driver = NULL;

// In the Driver::MainLoop, we change the terminal settings.  This function is
// added as an atexit handler to make sure we clean them up.
static void
reset_stdin_termios ()
{
    if (g_old_stdin_termios_is_valid)
    {
        g_old_stdin_termios_is_valid = false;
        ::tcsetattr (STDIN_FILENO, TCSANOW, &g_old_stdin_termios);
    }
}

typedef struct
{
    uint32_t usage_mask;                     // Used to mark options that can be used together.  If (1 << n & usage_mask) != 0
                                             // then this option belongs to option set n.
    bool required;                           // This option is required (in the current usage level)
    const char * long_option;                // Full name for this option.
    char short_option;                       // Single character for this option.
    int option_has_arg;                      // no_argument, required_argument or optional_argument
    uint32_t completion_type;                // Cookie the option class can use to do define the argument completion.
    lldb::CommandArgumentType argument_type; // Type of argument this option takes
    const char *  usage_text;                // Full text explaining what this options does and what (if any) argument to
                                             // pass it.
} OptionDefinition;

#define LLDB_3_TO_5 LLDB_OPT_SET_3|LLDB_OPT_SET_4|LLDB_OPT_SET_5
#define LLDB_4_TO_5 LLDB_OPT_SET_4|LLDB_OPT_SET_5

static OptionDefinition g_options[] =
{
    { LLDB_OPT_SET_1,    true , "help"           , 'h', no_argument      , 0,  eArgTypeNone,
        "Prints out the usage information for the LLDB debugger." },
    { LLDB_OPT_SET_2,    true , "version"        , 'v', no_argument      , 0,  eArgTypeNone,
        "Prints out the current version number of the LLDB debugger." },
    { LLDB_OPT_SET_3,    true , "arch"           , 'a', required_argument, 0,  eArgTypeArchitecture,
        "Tells the debugger to use the specified architecture when starting and running the program.  <architecture> must "
        "be one of the architectures for which the program was compiled." },
    { LLDB_OPT_SET_3,    true , "file"           , 'f', required_argument, 0,  eArgTypeFilename,
        "Tells the debugger to use the file <filename> as the program to be debugged." },
    { LLDB_OPT_SET_3,    false, "core"           , 'c', required_argument, 0,  eArgTypeFilename,
        "Tells the debugger to use the fullpath to <path> as the core file." },
    { LLDB_OPT_SET_4,    true , "attach-name"    , 'n', required_argument, 0,  eArgTypeProcessName,
        "Tells the debugger to attach to a process with the given name." },
    { LLDB_OPT_SET_4,    true , "wait-for"       , 'w', no_argument      , 0,  eArgTypeNone,
        "Tells the debugger to wait for a process with the given pid or name to launch before attaching." },
    { LLDB_OPT_SET_5,    true , "attach-pid"     , 'p', required_argument, 0,  eArgTypePid,
        "Tells the debugger to attach to a process with the given pid." },
    { LLDB_3_TO_5,       false, "script-language", 'l', required_argument, 0,  eArgTypeScriptLang,
        "Tells the debugger to use the specified scripting language for user-defined scripts, rather than the default.  "
        "Valid scripting languages that can be specified include Python, Perl, Ruby and Tcl.  Currently only the Python "
        "extensions have been implemented." },
    { LLDB_3_TO_5,       false, "debug"          , 'd', no_argument      , 0,  eArgTypeNone,
        "Tells the debugger to print out extra information for debugging itself." },
    { LLDB_3_TO_5,       false, "source"         , 's', required_argument, 0,  eArgTypeFilename,
        "Tells the debugger to read in and execute the file <file>, which should contain lldb commands." },
    { LLDB_3_TO_5,       false, "editor"         , 'e', no_argument      , 0,  eArgTypeNone,
        "Tells the debugger to open source files using the host's \"external editor\" mechanism." },
    { LLDB_3_TO_5,       false, "no-lldbinit"    , 'x', no_argument      , 0,  eArgTypeNone,
        "Do not automatically parse any '.lldbinit' files." },
    { 0,                 false, NULL             , 0  , 0                , 0,  eArgTypeNone,         NULL }
};

static const uint32_t last_option_set_with_args = 2;

Driver::Driver () :
    SBBroadcaster ("Driver"),
    m_debugger (SBDebugger::Create(false)),
    m_editline_pty (),
    m_editline_slave_fh (NULL),
    m_editline_reader (),
    m_io_channel_ap (),
    m_option_data (),
    m_waiting_for_command (false)
{
    // We want to be able to handle CTRL+D in the terminal to have it terminate
    // certain input
    m_debugger.SetCloseInputOnEOF (false);
    g_debugger_name = (char *) m_debugger.GetInstanceName();
    if (g_debugger_name == NULL)
        g_debugger_name = (char *) "";
    g_driver = this;
}

Driver::~Driver ()
{
    g_driver = NULL;
    g_debugger_name = NULL;
}

void
Driver::CloseIOChannelFile ()
{
    // Write an End of File sequence to the file descriptor to ensure any
    // read functions can exit.
    char eof_str[] = "\x04";
    ::write (m_editline_pty.GetMasterFileDescriptor(), eof_str, strlen(eof_str));

    m_editline_pty.CloseMasterFileDescriptor();

    if (m_editline_slave_fh)
    {
        ::fclose (m_editline_slave_fh);
        m_editline_slave_fh = NULL;
    }
}

// This function takes INDENT, which tells how many spaces to output at the front
// of each line; TEXT, which is the text that is to be output. It outputs the 
// text, on multiple lines if necessary, to RESULT, with INDENT spaces at the 
// front of each line.  It breaks lines on spaces, tabs or newlines, shortening 
// the line if necessary to not break in the middle of a word. It assumes that 
// each output line should contain a maximum of OUTPUT_MAX_COLUMNS characters.

void
OutputFormattedUsageText (FILE *out, int indent, const char *text, int output_max_columns)
{
    int len = strlen (text);
    std::string text_string (text);

    // Force indentation to be reasonable.
    if (indent >= output_max_columns)
        indent = 0;

    // Will it all fit on one line?

    if (len + indent < output_max_columns)
        // Output as a single line
        fprintf (out, "%*s%s\n", indent, "", text);
    else
    {
        // We need to break it up into multiple lines.
        int text_width = output_max_columns - indent - 1;
        int start = 0;
        int end = start;
        int final_end = len;
        int sub_len;

        while (end < final_end)
        {
              // Dont start the 'text' on a space, since we're already outputting the indentation.
              while ((start < final_end) && (text[start] == ' '))
                  start++;

              end = start + text_width;
              if (end > final_end)
                  end = final_end;
              else
              {
                  // If we're not at the end of the text, make sure we break the line on white space.
                  while (end > start
                         && text[end] != ' ' && text[end] != '\t' && text[end] != '\n')
                      end--;
              }
              sub_len = end - start;
              std::string substring = text_string.substr (start, sub_len);
              fprintf (out, "%*s%s\n", indent, "", substring.c_str());
              start = end + 1;
        }
    }
}

void
ShowUsage (FILE *out, OptionDefinition *option_table, Driver::OptionData data)
{
    uint32_t screen_width = 80;
    uint32_t indent_level = 0;
    const char *name = "lldb";
    
    fprintf (out, "\nUsage:\n\n");

    indent_level += 2;


    // First, show each usage level set of options, e.g. <cmd> [options-for-level-0]
    //                                                   <cmd> [options-for-level-1]
    //                                                   etc.

    uint32_t num_options;
    uint32_t num_option_sets = 0;
    
    for (num_options = 0; option_table[num_options].long_option != NULL; ++num_options)
    {
        uint32_t this_usage_mask = option_table[num_options].usage_mask;
        if (this_usage_mask == LLDB_OPT_SET_ALL)
        {
            if (num_option_sets == 0)
                num_option_sets = 1;
        }
        else
        {
            for (uint32_t j = 0; j < LLDB_MAX_NUM_OPTION_SETS; j++)
            {
                if (this_usage_mask & 1 << j)
                {
                    if (num_option_sets <= j)
                        num_option_sets = j + 1;
                }
            }
        }
    }

    for (uint32_t opt_set = 0; opt_set < num_option_sets; opt_set++)
    {
        uint32_t opt_set_mask;
        
        opt_set_mask = 1 << opt_set;
        
        if (opt_set > 0)
            fprintf (out, "\n");
        fprintf (out, "%*s%s", indent_level, "", name);
        bool is_help_line = false;
        
        for (uint32_t i = 0; i < num_options; ++i)
        {
            if (option_table[i].usage_mask & opt_set_mask)
            {
                CommandArgumentType arg_type = option_table[i].argument_type;
                const char *arg_name = SBCommandInterpreter::GetArgumentTypeAsCString (arg_type);
                // This is a bit of a hack, but there's no way to say certain options don't have arguments yet...
                // so we do it by hand here.
                if (option_table[i].short_option == 'h')
                    is_help_line = true;
                    
                if (option_table[i].required)
                {
                    if (option_table[i].option_has_arg == required_argument)
                        fprintf (out, " -%c <%s>", option_table[i].short_option, arg_name);
                    else if (option_table[i].option_has_arg == optional_argument)
                        fprintf (out, " -%c [<%s>]", option_table[i].short_option, arg_name);
                    else
                        fprintf (out, " -%c", option_table[i].short_option);
                }
                else
                {
                    if (option_table[i].option_has_arg == required_argument)
                        fprintf (out, " [-%c <%s>]", option_table[i].short_option, arg_name);
                    else if (option_table[i].option_has_arg == optional_argument)
                        fprintf (out, " [-%c [<%s>]]", option_table[i].short_option, arg_name);
                    else
                        fprintf (out, " [-%c]", option_table[i].short_option);
                }
            }
        }
        if (!is_help_line && (opt_set <= last_option_set_with_args))
            fprintf (out, " [[--] <PROGRAM-ARG-1> [<PROGRAM_ARG-2> ...]]");
    }

    fprintf (out, "\n\n");

    // Now print out all the detailed information about the various options:  long form, short form and help text:
    //   -- long_name <argument>
    //   - short <argument>
    //   help text

    // This variable is used to keep track of which options' info we've printed out, because some options can be in
    // more than one usage level, but we only want to print the long form of its information once.

    Driver::OptionData::OptionSet options_seen;
    Driver::OptionData::OptionSet::iterator pos;

    indent_level += 5;

    for (uint32_t i = 0; i < num_options; ++i)
    {
        // Only print this option if we haven't already seen it.
        pos = options_seen.find (option_table[i].short_option);
        if (pos == options_seen.end())
        {
            CommandArgumentType arg_type = option_table[i].argument_type;
            const char *arg_name = SBCommandInterpreter::GetArgumentTypeAsCString (arg_type);

            options_seen.insert (option_table[i].short_option);
            fprintf (out, "%*s-%c ", indent_level, "", option_table[i].short_option);
            if (arg_type != eArgTypeNone)
                fprintf (out, "<%s>", arg_name);
            fprintf (out, "\n");
            fprintf (out, "%*s--%s ", indent_level, "", option_table[i].long_option);
            if (arg_type != eArgTypeNone)
                fprintf (out, "<%s>", arg_name);
            fprintf (out, "\n");
            indent_level += 5;
            OutputFormattedUsageText (out, indent_level, option_table[i].usage_text, screen_width);
            indent_level -= 5;
            fprintf (out, "\n");
        }
    }

    indent_level -= 5;

    fprintf (out, "\n%*s(If you don't provide -f then the first argument will be the file to be debugged"
                  "\n%*s so '%s -- <filename> [<ARG1> [<ARG2>]]' also works."
                  "\n%*s Remember to end the options with \"--\" if any of your arguments have a \"-\" in them.)\n\n",
             indent_level, "", 
             indent_level, "",
             name, 
             indent_level, "");
}

void
BuildGetOptTable (OptionDefinition *expanded_option_table, std::vector<struct option> &getopt_table, 
                  uint32_t num_options)
{
    if (num_options == 0)
        return;

    uint32_t i;
    uint32_t j;
    std::bitset<256> option_seen;

    getopt_table.resize (num_options + 1);

    for (i = 0, j = 0; i < num_options; ++i)
    {
        char short_opt = expanded_option_table[i].short_option;
        
        if (option_seen.test(short_opt) == false)
        {
            getopt_table[j].name    = expanded_option_table[i].long_option;
            getopt_table[j].has_arg = expanded_option_table[i].option_has_arg;
            getopt_table[j].flag    = NULL;
            getopt_table[j].val     = expanded_option_table[i].short_option;
            option_seen.set(short_opt);
            ++j;
        }
    }

    getopt_table[j].name    = NULL;
    getopt_table[j].has_arg = 0;
    getopt_table[j].flag    = NULL;
    getopt_table[j].val     = 0;

}

Driver::OptionData::OptionData () :
    m_args(),
    m_script_lang (lldb::eScriptLanguageDefault),
    m_core_file (),
    m_crash_log (),
    m_source_command_files (),
    m_debug_mode (false),
    m_print_version (false),
    m_print_help (false),
    m_wait_for(false),
    m_process_name(),
    m_process_pid(LLDB_INVALID_PROCESS_ID),
    m_use_external_editor(false),
    m_seen_options()
{
}

Driver::OptionData::~OptionData ()
{
}

void
Driver::OptionData::Clear ()
{
    m_args.clear ();
    m_script_lang = lldb::eScriptLanguageDefault;
    m_source_command_files.clear ();
    m_debug_mode = false;
    m_print_help = false;
    m_print_version = false;
    m_use_external_editor = false;
    m_wait_for = false;
    m_process_name.erase();
    m_process_pid = LLDB_INVALID_PROCESS_ID;
}

void
Driver::ResetOptionValues ()
{
    m_option_data.Clear ();
}

const char *
Driver::GetFilename() const
{
    if (m_option_data.m_args.empty())
        return NULL;
    return m_option_data.m_args.front().c_str();
}

const char *
Driver::GetCrashLogFilename() const
{
    if (m_option_data.m_crash_log.empty())
        return NULL;
    return m_option_data.m_crash_log.c_str();
}

lldb::ScriptLanguage
Driver::GetScriptLanguage() const
{
    return m_option_data.m_script_lang;
}

size_t
Driver::GetNumSourceCommandFiles () const
{
    return m_option_data.m_source_command_files.size();
}

const char *
Driver::GetSourceCommandFileAtIndex (uint32_t idx) const
{
    if (idx < m_option_data.m_source_command_files.size())
        return m_option_data.m_source_command_files[idx].c_str();
    return NULL;
}

bool
Driver::GetDebugMode() const
{
    return m_option_data.m_debug_mode;
}


// Check the arguments that were passed to this program to make sure they are valid and to get their
// argument values (if any).  Return a boolean value indicating whether or not to start up the full
// debugger (i.e. the Command Interpreter) or not.  Return FALSE if the arguments were invalid OR
// if the user only wanted help or version information.

SBError
Driver::ParseArgs (int argc, const char *argv[], FILE *out_fh, bool &exit)
{
    ResetOptionValues ();

    SBCommandReturnObject result;

    SBError error;
    std::string option_string;
    struct option *long_options = NULL;
    std::vector<struct option> long_options_vector;
    uint32_t num_options;

    for (num_options = 0; g_options[num_options].long_option != NULL; ++num_options)
        /* Do Nothing. */;

    if (num_options == 0)
    {
        if (argc > 1)
            error.SetErrorStringWithFormat ("invalid number of options");
        return error;
    }

    BuildGetOptTable (g_options, long_options_vector, num_options);

    if (long_options_vector.empty())
        long_options = NULL;
    else
        long_options = &long_options_vector.front();

    if (long_options == NULL)
    {
        error.SetErrorStringWithFormat ("invalid long options");
        return error;
    }

    // Build the option_string argument for call to getopt_long.

    for (int i = 0; long_options[i].name != NULL; ++i)
    {
        if (long_options[i].flag == NULL)
        {
            option_string.push_back ((char) long_options[i].val);
            switch (long_options[i].has_arg)
            {
                default:
                case no_argument:
                    break;
                case required_argument:
                    option_string.push_back (':');
                    break;
                case optional_argument:
                    option_string.append ("::");
                    break;
            }
        }
    }

    // This is kind of a pain, but since we make the debugger in the Driver's constructor, we can't
    // know at that point whether we should read in init files yet.  So we don't read them in in the
    // Driver constructor, then set the flags back to "read them in" here, and then if we see the
    // "-n" flag, we'll turn it off again.  Finally we have to read them in by hand later in the
    // main loop.
    
    m_debugger.SkipLLDBInitFiles (false);
    m_debugger.SkipAppInitFiles (false);

    // Prepare for & make calls to getopt_long.
#if __GLIBC__
    optind = 0;
#else
    optreset = 1;
    optind = 1;
#endif
    int val;
    while (1)
    {
        int long_options_index = -1;
        val = ::getopt_long (argc, const_cast<char **>(argv), option_string.c_str(), long_options, &long_options_index);

        if (val == -1)
            break;
        else if (val == '?')
        {
            m_option_data.m_print_help = true;
            error.SetErrorStringWithFormat ("unknown or ambiguous option");
            break;
        }
        else if (val == 0)
            continue;
        else
        {
            m_option_data.m_seen_options.insert ((char) val);
            if (long_options_index == -1)
            {
                for (int i = 0;
                     long_options[i].name || long_options[i].has_arg || long_options[i].flag || long_options[i].val;
                     ++i)
                {
                    if (long_options[i].val == val)
                    {
                        long_options_index = i;
                        break;
                    }
                }
            }

            if (long_options_index >= 0)
            {
                const char short_option = (char) g_options[long_options_index].short_option;

                switch (short_option)
                {
                    case 'h':
                        m_option_data.m_print_help = true;
                        break;

                    case 'v':
                        m_option_data.m_print_version = true;
                        break;

                    case 'c':
                        {
                            SBFileSpec file(optarg);
                            if (file.Exists())
                            {
                                m_option_data.m_core_file = optarg;
                            }
                            else
                                error.SetErrorStringWithFormat("file specified in --core (-c) option doesn't exist: '%s'", optarg);
                        }
                        break;
                    
                    case 'e':
                        m_option_data.m_use_external_editor = true;
                        break;

                    case 'x':
                        m_debugger.SkipLLDBInitFiles (true);
                        m_debugger.SkipAppInitFiles (true);
                        break;

                    case 'f':
                        {
                            SBFileSpec file(optarg);
                            if (file.Exists())
                            {
                                m_option_data.m_args.push_back (optarg);
                            }
                            else if (file.ResolveExecutableLocation())
                            {
                                char path[PATH_MAX];
                                file.GetPath (path, sizeof(path));
                                m_option_data.m_args.push_back (path);
                            }
                            else
                                error.SetErrorStringWithFormat("file specified in --file (-f) option doesn't exist: '%s'", optarg);
                        }
                        break;

                    case 'a':
                        if (!m_debugger.SetDefaultArchitecture (optarg))
                            error.SetErrorStringWithFormat("invalid architecture in the -a or --arch option: '%s'", optarg);
                        break;

                    case 'l':
                        m_option_data.m_script_lang = m_debugger.GetScriptingLanguage (optarg);
                        break;

                    case 'd':
                        m_option_data.m_debug_mode = true;
                        break;

                    case 'n':
                        m_option_data.m_process_name = optarg;
                        break;
                    
                    case 'w':
                        m_option_data.m_wait_for = true;
                        break;
                        
                    case 'p':
                        {
                            char *remainder;
                            m_option_data.m_process_pid = strtol (optarg, &remainder, 0);
                            if (remainder == optarg || *remainder != '\0')
                                error.SetErrorStringWithFormat ("Could not convert process PID: \"%s\" into a pid.",
                                                                optarg);
                        }
                        break;
                    case 's':
                        {
                            SBFileSpec file(optarg);
                            if (file.Exists())
                                m_option_data.m_source_command_files.push_back (optarg);
                            else if (file.ResolveExecutableLocation())
                            {
                                char final_path[PATH_MAX];
                                file.GetPath (final_path, sizeof(final_path));
                                std::string path_str (final_path);
                                m_option_data.m_source_command_files.push_back (path_str);
                            }
                            else
                                error.SetErrorStringWithFormat("file specified in --source (-s) option doesn't exist: '%s'", optarg);
                        }
                        break;

                    default:
                        m_option_data.m_print_help = true;
                        error.SetErrorStringWithFormat ("unrecognized option %c", short_option);
                        break;
                }
            }
            else
            {
                error.SetErrorStringWithFormat ("invalid option with value %i", val);
            }
            if (error.Fail())
            {
                return error;
            }
        }
    }
    
    if (error.Fail() || m_option_data.m_print_help)
    {
        ShowUsage (out_fh, g_options, m_option_data);
        exit = true;
    }
    else if (m_option_data.m_print_version)
    {
        ::fprintf (out_fh, "%s\n", m_debugger.GetVersionString());
        exit = true;
    }
    else if (m_option_data.m_process_name.empty() && m_option_data.m_process_pid == LLDB_INVALID_PROCESS_ID)
    {
        // Any arguments that are left over after option parsing are for
        // the program. If a file was specified with -f then the filename
        // is already in the m_option_data.m_args array, and any remaining args
        // are arguments for the inferior program. If no file was specified with
        // -f, then what is left is the program name followed by any arguments.

        // Skip any options we consumed with getopt_long
        argc -= optind;
        argv += optind;

        if (argc > 0)
        {
            for (int arg_idx=0; arg_idx<argc; ++arg_idx)
            {
                const char *arg = argv[arg_idx];
                if (arg)
                    m_option_data.m_args.push_back (arg);
            }
        }
        
    }
    else
    {
        // Skip any options we consumed with getopt_long
        argc -= optind;
        //argv += optind; // Commented out to keep static analyzer happy

        if (argc > 0)
            ::fprintf (out_fh, "Warning: program arguments are ignored when attaching.\n");
    }

    return error;
}

size_t
Driver::GetProcessSTDOUT ()
{
    //  The process has stuff waiting for stdout; get it and write it out to the appropriate place.
    char stdio_buffer[1024];
    size_t len;
    size_t total_bytes = 0;
    while ((len = m_debugger.GetSelectedTarget().GetProcess().GetSTDOUT (stdio_buffer, sizeof (stdio_buffer))) > 0)
    {
        m_io_channel_ap->OutWrite (stdio_buffer, len, ASYNC);
        total_bytes += len;
    }
    return total_bytes;
}

size_t
Driver::GetProcessSTDERR ()
{
    //  The process has stuff waiting for stderr; get it and write it out to the appropriate place.
    char stdio_buffer[1024];
    size_t len;
    size_t total_bytes = 0;
    while ((len = m_debugger.GetSelectedTarget().GetProcess().GetSTDERR (stdio_buffer, sizeof (stdio_buffer))) > 0)
    {
        m_io_channel_ap->ErrWrite (stdio_buffer, len, ASYNC);
        total_bytes += len;
    }
    return total_bytes;
}

void
Driver::UpdateSelectedThread ()
{
    using namespace lldb;
    SBProcess process(m_debugger.GetSelectedTarget().GetProcess());
    if (process.IsValid())
    {
        SBThread curr_thread (process.GetSelectedThread());
        SBThread thread;
        StopReason curr_thread_stop_reason = eStopReasonInvalid;
        curr_thread_stop_reason = curr_thread.GetStopReason();

        if (!curr_thread.IsValid() ||
            curr_thread_stop_reason == eStopReasonInvalid ||
            curr_thread_stop_reason == eStopReasonNone)
        {
            // Prefer a thread that has just completed its plan over another thread as current thread.
            SBThread plan_thread;
            SBThread other_thread;
            const size_t num_threads = process.GetNumThreads();
            size_t i;
            for (i = 0; i < num_threads; ++i)
            {
                thread = process.GetThreadAtIndex(i);
                StopReason thread_stop_reason = thread.GetStopReason();
                switch (thread_stop_reason)
                {
                default:
                case eStopReasonInvalid:
                case eStopReasonNone:
                    break;

                case eStopReasonTrace:
                case eStopReasonBreakpoint:
                case eStopReasonWatchpoint:
                case eStopReasonSignal:
                case eStopReasonException:
                    if (!other_thread.IsValid())
                        other_thread = thread;
                    break;
                case eStopReasonPlanComplete:
                    if (!plan_thread.IsValid())
                        plan_thread = thread;
                    break;
                }
            }
            if (plan_thread.IsValid())
                process.SetSelectedThread (plan_thread);
            else if (other_thread.IsValid())
                process.SetSelectedThread (other_thread);
            else
            {
                if (curr_thread.IsValid())
                    thread = curr_thread;
                else
                    thread = process.GetThreadAtIndex(0);

                if (thread.IsValid())
                    process.SetSelectedThread (thread);
            }
        }
    }
}

// This function handles events that were broadcast by the process.
void
Driver::HandleBreakpointEvent (const SBEvent &event)
{
    using namespace lldb;
    const uint32_t event_type = SBBreakpoint::GetBreakpointEventTypeFromEvent (event);
    
    if (event_type & eBreakpointEventTypeAdded
        || event_type & eBreakpointEventTypeRemoved
        || event_type & eBreakpointEventTypeEnabled
        || event_type & eBreakpointEventTypeDisabled
        || event_type & eBreakpointEventTypeCommandChanged
        || event_type & eBreakpointEventTypeConditionChanged
        || event_type & eBreakpointEventTypeIgnoreChanged
        || event_type & eBreakpointEventTypeLocationsResolved)
    {
        // Don't do anything about these events, since the breakpoint commands already echo these actions.
    }              
    else if (event_type & eBreakpointEventTypeLocationsAdded)
    {
        char message[256];
        uint32_t num_new_locations = SBBreakpoint::GetNumBreakpointLocationsFromEvent(event);
        if (num_new_locations > 0)
        {
            SBBreakpoint breakpoint = SBBreakpoint::GetBreakpointFromEvent(event);
            int message_len = ::snprintf (message, sizeof(message), "%d location%s added to breakpoint %d\n", 
                                          num_new_locations,
                                          num_new_locations == 1 ? "" : "s",
                                          breakpoint.GetID());
            m_io_channel_ap->OutWrite(message, message_len, ASYNC);
        }
    }
    else if (event_type & eBreakpointEventTypeLocationsRemoved)
    {
       // These locations just get disabled, not sure it is worth spamming folks about this on the command line.
    }
    else if (event_type & eBreakpointEventTypeLocationsResolved)
    {
       // This might be an interesting thing to note, but I'm going to leave it quiet for now, it just looked noisy.
    }
}

// This function handles events that were broadcast by the process.
void
Driver::HandleProcessEvent (const SBEvent &event)
{
    using namespace lldb;
    const uint32_t event_type = event.GetType();

    if (event_type & SBProcess::eBroadcastBitSTDOUT)
    {
        // The process has stdout available, get it and write it out to the
        // appropriate place.
        GetProcessSTDOUT ();
    }
    else if (event_type & SBProcess::eBroadcastBitSTDERR)
    {
        // The process has stderr available, get it and write it out to the
        // appropriate place.
        GetProcessSTDERR ();
    }
    else if (event_type & SBProcess::eBroadcastBitStateChanged)
    {
        // Drain all stout and stderr so we don't see any output come after
        // we print our prompts
        GetProcessSTDOUT ();
        GetProcessSTDERR ();
        // Something changed in the process;  get the event and report the process's current status and location to
        // the user.
        StateType event_state = SBProcess::GetStateFromEvent (event);
        if (event_state == eStateInvalid)
            return;

        SBProcess process (SBProcess::GetProcessFromEvent (event));
        assert (process.IsValid());

        switch (event_state)
        {
        case eStateInvalid:
        case eStateUnloaded:
        case eStateConnected:
        case eStateAttaching:
        case eStateLaunching:
        case eStateStepping:
        case eStateDetached:
            {
                char message[1024];
                int message_len = ::snprintf (message, sizeof(message), "Process %llu %s\n", process.GetProcessID(),
                                              m_debugger.StateAsCString (event_state));
                m_io_channel_ap->OutWrite(message, message_len, ASYNC);
            }
            break;

        case eStateRunning:
            // Don't be chatty when we run...
            break;

        case eStateExited:
            {
                SBCommandReturnObject result;
                m_debugger.GetCommandInterpreter().HandleCommand("process status", result, false);
                m_io_channel_ap->ErrWrite (result.GetError(), result.GetErrorSize(), ASYNC);
                m_io_channel_ap->OutWrite (result.GetOutput(), result.GetOutputSize(), ASYNC);
            }
            break;

        case eStateStopped:
        case eStateCrashed:
        case eStateSuspended:
            // Make sure the program hasn't been auto-restarted:
            if (SBProcess::GetRestartedFromEvent (event))
            {
                // FIXME: Do we want to report this, or would that just be annoyingly chatty?
                char message[1024];
                int message_len = ::snprintf (message, sizeof(message), "Process %llu stopped and was programmatically restarted.\n",
                                              process.GetProcessID());
                m_io_channel_ap->OutWrite(message, message_len, ASYNC);
            }
            else
            {
                if (GetDebugger().GetSelectedTarget() == process.GetTarget())
                {
                    SBCommandReturnObject result;
                    UpdateSelectedThread ();
                    m_debugger.GetCommandInterpreter().HandleCommand("process status", result, false);
                    m_io_channel_ap->ErrWrite (result.GetError(), result.GetErrorSize(), ASYNC);
                    m_io_channel_ap->OutWrite (result.GetOutput(), result.GetOutputSize(), ASYNC);
                }
                else
                {
                    SBStream out_stream;
                    uint32_t target_idx = GetDebugger().GetIndexOfTarget(process.GetTarget());
                    if (target_idx != UINT32_MAX)
                        out_stream.Printf ("Target %d: (", target_idx);
                    else
                        out_stream.Printf ("Target <unknown index>: (");
                    process.GetTarget().GetDescription (out_stream, eDescriptionLevelBrief);
                    out_stream.Printf (") stopped.\n");
                    m_io_channel_ap->OutWrite (out_stream.GetData(), out_stream.GetSize(), ASYNC);
                }
            }
            break;
        }
    }
}

void
Driver::HandleThreadEvent (const SBEvent &event)
{
    // At present the only thread event we handle is the Frame Changed event, and all we do for that is just
    // reprint the thread status for that thread.
    using namespace lldb;
    const uint32_t event_type = event.GetType();
    if (event_type == SBThread::eBroadcastBitStackChanged)
    {
        SBThread thread = SBThread::GetThreadFromEvent (event);
        if (thread.IsValid())
        {
            SBStream out_stream;
            thread.GetStatus(out_stream);
            m_io_channel_ap->OutWrite (out_stream.GetData (), out_stream.GetSize (), ASYNC);
        }
    }
}

//  This function handles events broadcast by the IOChannel (HasInput, UserInterrupt, or ThreadShouldExit).

bool
Driver::HandleIOEvent (const SBEvent &event)
{
    bool quit = false;

    const uint32_t event_type = event.GetType();

    if (event_type & IOChannel::eBroadcastBitHasUserInput)
    {
        // We got some input (i.e. a command string) from the user; pass it off to the command interpreter for
        // handling.

        const char *command_string = SBEvent::GetCStringFromEvent(event);
        if (command_string == NULL)
            command_string = "";
        SBCommandReturnObject result;
        
        // We don't want the result to bypass the OutWrite function in IOChannel, as this can result in odd
        // output orderings and problems with the prompt.
        m_debugger.GetCommandInterpreter().HandleCommand (command_string, result, true);

        const bool only_if_no_immediate = true;

        const size_t output_size = result.GetOutputSize();
        
        if (output_size > 0)
            m_io_channel_ap->OutWrite (result.GetOutput(only_if_no_immediate), output_size, NO_ASYNC);

        const size_t error_size = result.GetErrorSize();

        if (error_size > 0)
            m_io_channel_ap->OutWrite (result.GetError(only_if_no_immediate), error_size, NO_ASYNC);

        // We are done getting and running our command, we can now clear the
        // m_waiting_for_command so we can get another one.
        m_waiting_for_command = false;

        // If our editline input reader is active, it means another input reader
        // got pushed onto the input reader and caused us to become deactivated.
        // When the input reader above us gets popped, we will get re-activated
        // and our prompt will refresh in our callback
        if (m_editline_reader.IsActive())
        {
            ReadyForCommand ();
        }
    }
    else if (event_type & IOChannel::eBroadcastBitUserInterrupt)
    {
        // This is here to handle control-c interrupts from the user.  It has not yet really been implemented.
        // TO BE DONE:  PROPERLY HANDLE CONTROL-C FROM USER
        //m_io_channel_ap->CancelInput();
        // Anything else?  Send Interrupt to process?
    }
    else if ((event_type & IOChannel::eBroadcastBitThreadShouldExit) ||
             (event_type & IOChannel::eBroadcastBitThreadDidExit))
    {
        // If the IOChannel thread is trying to go away, then it is definitely
        // time to end the debugging session.
        quit = true;
    }

    return quit;
}

void
Driver::MasterThreadBytesReceived (void *baton, const void *src, size_t src_len)
{
    Driver *driver = (Driver*)baton;
    driver->GetFromMaster ((const char *)src, src_len);
}

void
Driver::GetFromMaster (const char *src, size_t src_len)
{
    // Echo the characters back to the Debugger's stdout, that way if you
    // type characters while a command is running, you'll see what you've typed.
    FILE *out_fh = m_debugger.GetOutputFileHandle();
    if (out_fh)
        ::fwrite (src, 1, src_len, out_fh);
}

size_t
Driver::EditLineInputReaderCallback 
(
    void *baton, 
    SBInputReader *reader, 
    InputReaderAction notification,
    const char *bytes, 
    size_t bytes_len
)
{
    Driver *driver = (Driver *)baton;

    switch (notification)
    {
    case eInputReaderActivate:
        break;

    case eInputReaderReactivate:
        driver->ReadyForCommand();
        break;

    case eInputReaderDeactivate:
        break;
        
    case eInputReaderAsynchronousOutputWritten:
        if (driver->m_io_channel_ap.get() != NULL)
            driver->m_io_channel_ap->RefreshPrompt();
        break;

    case eInputReaderInterrupt:
        if (driver->m_io_channel_ap.get() != NULL)
        {
            SBProcess process(driver->GetDebugger().GetSelectedTarget().GetProcess());
            if (!driver->m_io_channel_ap->EditLineHasCharacters()
                &&  process.IsValid()
                && (process.GetState() == lldb::eStateRunning || process.GetState() == lldb::eStateAttaching))
            {
                process.SendAsyncInterrupt ();
            }
            else
            {
                driver->m_io_channel_ap->OutWrite ("^C\n", 3, NO_ASYNC);
                // I wish I could erase the entire input line, but there's no public API for that.
                driver->m_io_channel_ap->EraseCharsBeforeCursor();
                driver->m_io_channel_ap->RefreshPrompt();
            }
        }
        break;
        
    case eInputReaderEndOfFile:
        if (driver->m_io_channel_ap.get() != NULL)
        {
            driver->m_io_channel_ap->OutWrite ("^D\n", 3, NO_ASYNC);
            driver->m_io_channel_ap->RefreshPrompt ();
        }
        write (driver->m_editline_pty.GetMasterFileDescriptor(), "quit\n", 5);
        break;

    case eInputReaderGotToken:
        write (driver->m_editline_pty.GetMasterFileDescriptor(), bytes, bytes_len);
        break;
        
    case eInputReaderDone:
        break;
    }
    return bytes_len;
}

void
Driver::MainLoop ()
{
    char error_str[1024];
    if (m_editline_pty.OpenFirstAvailableMaster(O_RDWR|O_NOCTTY, error_str, sizeof(error_str)) == false)
    {
        ::fprintf (stderr, "error: failed to open driver pseudo terminal : %s", error_str);
        exit(1);
    }
    else
    {
        const char *driver_slave_name = m_editline_pty.GetSlaveName (error_str, sizeof(error_str));
        if (driver_slave_name == NULL)
        {
            ::fprintf (stderr, "error: failed to get slave name for driver pseudo terminal : %s", error_str);
            exit(2);
        }
        else
        {
            m_editline_slave_fh = ::fopen (driver_slave_name, "r+");
            if (m_editline_slave_fh == NULL)
            {
                SBError error;
                error.SetErrorToErrno();
                ::fprintf (stderr, "error: failed to get open slave for driver pseudo terminal : %s",
                           error.GetCString());
                exit(3);
            }

            ::setbuf (m_editline_slave_fh, NULL);
        }
    }

    lldb_utility::PseudoTerminal editline_output_pty;
    FILE *editline_output_slave_fh = NULL;
    
    if (editline_output_pty.OpenFirstAvailableMaster (O_RDWR|O_NOCTTY, error_str, sizeof (error_str)) == false)
    {
        ::fprintf (stderr, "error: failed to open output pseudo terminal : %s", error_str);
        exit(1);
    }
    else
    {
        const char *output_slave_name = editline_output_pty.GetSlaveName (error_str, sizeof(error_str));
        if (output_slave_name == NULL)
        {
            ::fprintf (stderr, "error: failed to get slave name for output pseudo terminal : %s", error_str);
            exit(2);
        }
        else
        {
            editline_output_slave_fh = ::fopen (output_slave_name, "r+");
            if (editline_output_slave_fh == NULL)
            {
                SBError error;
                error.SetErrorToErrno();
                ::fprintf (stderr, "error: failed to get open slave for output pseudo terminal : %s",
                           error.GetCString());
                exit(3);
            }
            ::setbuf (editline_output_slave_fh, NULL);
        }
    }

   // struct termios stdin_termios;

    if (::tcgetattr(STDIN_FILENO, &g_old_stdin_termios) == 0)
    {
        g_old_stdin_termios_is_valid = true;
        atexit (reset_stdin_termios);
    }

    ::setbuf (stdin, NULL);
    ::setbuf (stdout, NULL);

    m_debugger.SetErrorFileHandle (stderr, false);
    m_debugger.SetOutputFileHandle (stdout, false);
    m_debugger.SetInputFileHandle (stdin, true);
    
    m_debugger.SetUseExternalEditor(m_option_data.m_use_external_editor);

    // You have to drain anything that comes to the master side of the PTY.  master_out_comm is
    // for that purpose.  The reason you need to do this is a curious reason...  editline will echo
    // characters to the PTY when it gets characters while el_gets is not running, and then when
    // you call el_gets (or el_getc) it will try to reset the terminal back to raw mode which blocks
    // if there are unconsumed characters in the out buffer.
    // However, you don't need to do anything with the characters, since editline will dump these
    // unconsumed characters after printing the prompt again in el_gets.

    SBCommunication master_out_comm("driver.editline");
    master_out_comm.SetCloseOnEOF (false);
    master_out_comm.AdoptFileDesriptor(m_editline_pty.GetMasterFileDescriptor(), false);
    master_out_comm.SetReadThreadBytesReceivedCallback(Driver::MasterThreadBytesReceived, this);

    if (master_out_comm.ReadThreadStart () == false)
    {
        ::fprintf (stderr, "error: failed to start master out read thread");
        exit(5);
    }

    SBCommandInterpreter sb_interpreter = m_debugger.GetCommandInterpreter();

    m_io_channel_ap.reset (new IOChannel(m_editline_slave_fh, editline_output_slave_fh, stdout, stderr, this));

    SBCommunication out_comm_2("driver.editline_output");
    out_comm_2.SetCloseOnEOF (false);
    out_comm_2.AdoptFileDesriptor (editline_output_pty.GetMasterFileDescriptor(), false);
    out_comm_2.SetReadThreadBytesReceivedCallback (IOChannel::LibeditOutputBytesReceived, m_io_channel_ap.get());

    if (out_comm_2.ReadThreadStart () == false)
    {
        ::fprintf (stderr, "error: failed to start libedit output read thread");
        exit (5);
    }


    struct winsize window_size;
    if (isatty (STDIN_FILENO)
        && ::ioctl (STDIN_FILENO, TIOCGWINSZ, &window_size) == 0)
    {
        if (window_size.ws_col > 0)
            m_debugger.SetTerminalWidth (window_size.ws_col);
    }

    // Since input can be redirected by the debugger, we must insert our editline
    // input reader in the queue so we know when our reader should be active
    // and so we can receive bytes only when we are supposed to.
    SBError err (m_editline_reader.Initialize (m_debugger, 
                                               Driver::EditLineInputReaderCallback, // callback
                                               this,                              // baton
                                               eInputReaderGranularityByte,       // token_size
                                               NULL,                              // end token - NULL means never done
                                               NULL,                              // prompt - taken care of elsewhere
                                               false));                           // echo input - don't need Debugger 
                                                                                  // to do this, we handle it elsewhere
    
    if (err.Fail())
    {
        ::fprintf (stderr, "error: %s", err.GetCString());
        exit (6);
    }
    
    m_debugger.PushInputReader (m_editline_reader);

    SBListener listener(m_debugger.GetListener());
    if (listener.IsValid())
    {

        listener.StartListeningForEventClass(m_debugger, 
                                         SBTarget::GetBroadcasterClassName(), 
                                         SBTarget::eBroadcastBitBreakpointChanged);
        listener.StartListeningForEventClass(m_debugger, 
                                         SBThread::GetBroadcasterClassName(),
                                         SBThread::eBroadcastBitStackChanged);
        listener.StartListeningForEvents (*m_io_channel_ap,
                                          IOChannel::eBroadcastBitHasUserInput |
                                          IOChannel::eBroadcastBitUserInterrupt |
                                          IOChannel::eBroadcastBitThreadShouldExit |
                                          IOChannel::eBroadcastBitThreadDidStart |
                                          IOChannel::eBroadcastBitThreadDidExit);

        if (m_io_channel_ap->Start ())
        {
            bool iochannel_thread_exited = false;

            listener.StartListeningForEvents (sb_interpreter.GetBroadcaster(),
                                              SBCommandInterpreter::eBroadcastBitQuitCommandReceived |
                                              SBCommandInterpreter::eBroadcastBitAsynchronousOutputData |
                                              SBCommandInterpreter::eBroadcastBitAsynchronousErrorData);

            // Before we handle any options from the command line, we parse the
            // .lldbinit file in the user's home directory.
            SBCommandReturnObject result;
            sb_interpreter.SourceInitFileInHomeDirectory(result);
            if (GetDebugMode())
            {
                result.PutError (m_debugger.GetErrorFileHandle());
                result.PutOutput (m_debugger.GetOutputFileHandle());
            }

            // Now we handle options we got from the command line
            char command_string[PATH_MAX * 2];
            const size_t num_source_command_files = GetNumSourceCommandFiles();
            if (num_source_command_files > 0)
            {
                for (size_t i=0; i < num_source_command_files; ++i)
                {
                    const char *command_file = GetSourceCommandFileAtIndex(i);
                    ::snprintf (command_string, sizeof(command_string), "command source '%s'", command_file);
                    m_debugger.GetCommandInterpreter().HandleCommand (command_string, result, false);
                    if (GetDebugMode())
                    {
                        result.PutError (m_debugger.GetErrorFileHandle());
                        result.PutOutput (m_debugger.GetOutputFileHandle());
                    }
                }
            }

            // Was there a core file specified?
            std::string core_file_spec("");
            if (!m_option_data.m_core_file.empty())
                core_file_spec.append("--core ").append(m_option_data.m_core_file);

            const size_t num_args = m_option_data.m_args.size();
            if (num_args > 0)
            {
                char arch_name[64];
                if (m_debugger.GetDefaultArchitecture (arch_name, sizeof (arch_name)))
                    ::snprintf (command_string, 
                                sizeof (command_string), 
                                "target create --arch=%s %s \"%s\"", 
                                arch_name,
                                core_file_spec.c_str(),
                                m_option_data.m_args[0].c_str());
                else
                    ::snprintf (command_string, 
                                sizeof(command_string), 
                                "target create %s \"%s\"", 
                                core_file_spec.c_str(),
                                m_option_data.m_args[0].c_str());

                m_debugger.HandleCommand (command_string);
                
                if (num_args > 1)
                {
                    m_debugger.HandleCommand ("settings clear target.run-args");
                    char arg_cstr[1024];
                    for (size_t arg_idx = 1; arg_idx < num_args; ++arg_idx)
                    {
                        ::snprintf (arg_cstr, 
                                    sizeof(arg_cstr), 
                                    "settings append target.run-args \"%s\"", 
                                    m_option_data.m_args[arg_idx].c_str());
                        m_debugger.HandleCommand (arg_cstr);
                    }
                }
            }
            else if (!core_file_spec.empty())
            {
                ::snprintf (command_string, 
                            sizeof(command_string), 
                            "target create %s", 
                            core_file_spec.c_str());
                m_debugger.HandleCommand (command_string);;
            }

            // Now that all option parsing is done, we try and parse the .lldbinit
            // file in the current working directory
            sb_interpreter.SourceInitFileInCurrentWorkingDirectory (result);
            if (GetDebugMode())
            {
                result.PutError(m_debugger.GetErrorFileHandle());
                result.PutOutput(m_debugger.GetOutputFileHandle());
            }

            SBEvent event;

            // Make sure the IO channel is started up before we try to tell it we
            // are ready for input
            listener.WaitForEventForBroadcasterWithType (UINT32_MAX, 
                                                         *m_io_channel_ap,
                                                         IOChannel::eBroadcastBitThreadDidStart, 
                                                         event);
            // If we were asked to attach, then do that here:
            // I'm going to use the command string rather than directly
            // calling the API's because then I don't have to recode the
            // event handling here.
            if (!m_option_data.m_process_name.empty()
                || m_option_data.m_process_pid != LLDB_INVALID_PROCESS_ID)
            {
                std::string command_str("process attach ");
                if (m_option_data.m_process_pid != LLDB_INVALID_PROCESS_ID)
                {
                    command_str.append("-p ");
                    char pid_buffer[32];
                    ::snprintf (pid_buffer, sizeof(pid_buffer), "%llu", m_option_data.m_process_pid);
                    command_str.append(pid_buffer);
                }
                else 
                {
                    command_str.append("-n \"");
                    command_str.append(m_option_data.m_process_name);
                    command_str.push_back('\"');
                    if (m_option_data.m_wait_for)
                        command_str.append(" -w");
                }
                
                if (m_debugger.GetOutputFileHandle())
                    ::fprintf (m_debugger.GetOutputFileHandle(), 
                               "Attaching to process with:\n    %s\n", 
                               command_str.c_str());
                                               
                // Force the attach to be synchronous:
                bool orig_async = m_debugger.GetAsync();
                m_debugger.SetAsync(true);
                m_debugger.HandleCommand(command_str.c_str());
                m_debugger.SetAsync(orig_async);                
            }
                        
            ReadyForCommand ();

            while (!GetIsDone())
            {
                listener.WaitForEvent (UINT32_MAX, event);
                if (event.IsValid())
                {
                    if (event.GetBroadcaster().IsValid())
                    {
                        uint32_t event_type = event.GetType();
                        if (event.BroadcasterMatchesRef (*m_io_channel_ap))
                        {
                            if ((event_type & IOChannel::eBroadcastBitThreadShouldExit) ||
                                (event_type & IOChannel::eBroadcastBitThreadDidExit))
                            {
                                SetIsDone();
                                if (event_type & IOChannel::eBroadcastBitThreadDidExit)
                                    iochannel_thread_exited = true;
                            }
                            else
                            {
                                if (HandleIOEvent (event))
                                    SetIsDone();
                            }
                        }
                        else if (SBProcess::EventIsProcessEvent (event))
                        {
                            HandleProcessEvent (event);
                        }
                        else if (SBBreakpoint::EventIsBreakpointEvent (event))
                        {
                            HandleBreakpointEvent (event);
                        }
                        else if (SBThread::EventIsThreadEvent (event))
                        {
                            HandleThreadEvent (event);
                        }
                        else if (event.BroadcasterMatchesRef (sb_interpreter.GetBroadcaster()))
                        {
                            // TODO: deprecate the eBroadcastBitQuitCommandReceived event
                            // now that we have SBCommandInterpreter::SetCommandOverrideCallback()
                            // that can take over a command
                            if (event_type & SBCommandInterpreter::eBroadcastBitQuitCommandReceived)
                            {
                                SetIsDone();
                            }
                            else if (event_type & SBCommandInterpreter::eBroadcastBitAsynchronousErrorData)
                            {
                                const char *data = SBEvent::GetCStringFromEvent (event);
                                m_io_channel_ap->ErrWrite (data, strlen(data), ASYNC);
                            }
                            else if (event_type & SBCommandInterpreter::eBroadcastBitAsynchronousOutputData)
                            {
                                const char *data = SBEvent::GetCStringFromEvent (event);
                                m_io_channel_ap->OutWrite (data, strlen(data), ASYNC);
                            }
                        }
                    }
                }
            }

            editline_output_pty.CloseMasterFileDescriptor();
            master_out_comm.Disconnect();
            out_comm_2.Disconnect();
            reset_stdin_termios();
            fclose (stdin);

            CloseIOChannelFile ();

            if (!iochannel_thread_exited)
            {
                event.Clear();
                listener.GetNextEventForBroadcasterWithType (*m_io_channel_ap,
                                                             IOChannel::eBroadcastBitThreadDidExit,
                                                             event);
                if (!event.IsValid())
                {
                    // Send end EOF to the driver file descriptor
                    m_io_channel_ap->Stop();
                }
            }

            SBDebugger::Destroy (m_debugger);
        }
    }
}


void
Driver::ReadyForCommand ()
{
    if (m_waiting_for_command == false)
    {
        m_waiting_for_command = true;
        BroadcastEventByType (Driver::eBroadcastBitReadyForInput, true);
    }
}


void
sigwinch_handler (int signo)
{
    struct winsize window_size;
    if (isatty (STDIN_FILENO)
        && ::ioctl (STDIN_FILENO, TIOCGWINSZ, &window_size) == 0)
    {
        if ((window_size.ws_col > 0) && g_driver != NULL)
        {
            g_driver->GetDebugger().SetTerminalWidth (window_size.ws_col);
        }
    }
}

void
sigint_handler (int signo)
{
	static bool g_interrupt_sent = false;
    if (g_driver)
	{
		if (!g_interrupt_sent)
		{
			g_interrupt_sent = true;
        	g_driver->GetDebugger().DispatchInputInterrupt();
			g_interrupt_sent = false;
			return;
		}
	}
    
	exit (signo);
}

int
main (int argc, char const *argv[], const char *envp[])
{
    SBDebugger::Initialize();
    
    SBHostOS::ThreadCreated ("<lldb.driver.main-thread>");

    signal (SIGPIPE, SIG_IGN);
    signal (SIGWINCH, sigwinch_handler);
    signal (SIGINT, sigint_handler);

    // Create a scope for driver so that the driver object will destroy itself
    // before SBDebugger::Terminate() is called.
    {
        Driver driver;

        bool exit = false;
        SBError error (driver.ParseArgs (argc, argv, stdout, exit));
        if (error.Fail())
        {
            const char *error_cstr = error.GetCString ();
            if (error_cstr)
                ::fprintf (stderr, "error: %s\n", error_cstr);
        }
        else if (!exit)
        {
            driver.MainLoop ();
        }
    }

    SBDebugger::Terminate();
    return 0;
}
