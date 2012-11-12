//===-- Process.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Process.h"

#include "lldb/lldb-private-log.h"

#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Event.h"
#include "lldb/Core/ConnectionFileDescriptor.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/InputReader.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/State.h"
#include "lldb/Expression/ClangUserExpression.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/OperatingSystem.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/CPPLanguageRuntime.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/TargetList.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanBase.h"

using namespace lldb;
using namespace lldb_private;


// Comment out line below to disable memory caching, overriding the process setting
// target.process.disable-memory-cache
#define ENABLE_MEMORY_CACHING

#ifdef ENABLE_MEMORY_CACHING
#define DISABLE_MEM_CACHE_DEFAULT false
#else
#define DISABLE_MEM_CACHE_DEFAULT true
#endif

class ProcessOptionValueProperties : public OptionValueProperties
{
public:
    ProcessOptionValueProperties (const ConstString &name) :
        OptionValueProperties (name)
    {
    }
    
    // This constructor is used when creating ProcessOptionValueProperties when it
    // is part of a new lldb_private::Process instance. It will copy all current
    // global property values as needed
    ProcessOptionValueProperties (ProcessProperties *global_properties) :
        OptionValueProperties(*global_properties->GetValueProperties())
    {
    }
    
    virtual const Property *
    GetPropertyAtIndex (const ExecutionContext *exe_ctx, bool will_modify, uint32_t idx) const
    {
        // When gettings the value for a key from the process options, we will always
        // try and grab the setting from the current process if there is one. Else we just
        // use the one from this instance.
        if (exe_ctx)
        {
            Process *process = exe_ctx->GetProcessPtr();
            if (process)
            {
                ProcessOptionValueProperties *instance_properties = static_cast<ProcessOptionValueProperties *>(process->GetValueProperties().get());
                if (this != instance_properties)
                    return instance_properties->ProtectedGetPropertyAtIndex (idx);
            }
        }
        return ProtectedGetPropertyAtIndex (idx);
    }
};

static PropertyDefinition
g_properties[] =
{
    { "disable-memory-cache" , OptionValue::eTypeBoolean, false, DISABLE_MEM_CACHE_DEFAULT, NULL, NULL, "Disable reading and caching of memory in fixed-size units." },
    { "extra-startup-command", OptionValue::eTypeArray  , false, OptionValue::eTypeString, NULL, NULL, "A list containing extra commands understood by the particular process plugin used." },
    { "python-os-plugin-path", OptionValue::eTypeFileSpec, false, 0, NULL, NULL, "A path to a python OS plug-in module file that contains a OperatingSystemPlugIn class." },
    {  NULL                  , OptionValue::eTypeInvalid, false, 0, NULL, NULL, NULL  }
};

enum {
    ePropertyDisableMemCache,
    ePropertyExtraStartCommand,
    ePropertyPythonOSPluginPath
};

ProcessProperties::ProcessProperties (bool is_global) :
    Properties ()
{
    if (is_global)
    {
        m_collection_sp.reset (new ProcessOptionValueProperties(ConstString("process")));
        m_collection_sp->Initialize(g_properties);
        m_collection_sp->AppendProperty(ConstString("thread"),
                                        ConstString("Settings specify to threads."),
                                        true,
                                        Thread::GetGlobalProperties()->GetValueProperties());
    }
    else
        m_collection_sp.reset (new ProcessOptionValueProperties(Process::GetGlobalProperties().get()));
}

ProcessProperties::~ProcessProperties()
{
}

bool
ProcessProperties::GetDisableMemoryCache() const
{
    const uint32_t idx = ePropertyDisableMemCache;
    return m_collection_sp->GetPropertyAtIndexAsBoolean (NULL, idx, g_properties[idx].default_uint_value != 0);
}

Args
ProcessProperties::GetExtraStartupCommands () const
{
    Args args;
    const uint32_t idx = ePropertyExtraStartCommand;
    m_collection_sp->GetPropertyAtIndexAsArgs(NULL, idx, args);
    return args;
}

void
ProcessProperties::SetExtraStartupCommands (const Args &args)
{
    const uint32_t idx = ePropertyExtraStartCommand;
    m_collection_sp->SetPropertyAtIndexFromArgs(NULL, idx, args);
}

FileSpec
ProcessProperties::GetPythonOSPluginPath () const
{
    const uint32_t idx = ePropertyPythonOSPluginPath;
    return m_collection_sp->GetPropertyAtIndexAsFileSpec(NULL, idx);
}

void
ProcessProperties::SetPythonOSPluginPath (const FileSpec &file)
{
    const uint32_t idx = ePropertyPythonOSPluginPath;
    m_collection_sp->SetPropertyAtIndexAsFileSpec(NULL, idx, file);
}

void
ProcessInstanceInfo::Dump (Stream &s, Platform *platform) const
{
    const char *cstr;
    if (m_pid != LLDB_INVALID_PROCESS_ID)       
        s.Printf ("    pid = %llu\n", m_pid);

    if (m_parent_pid != LLDB_INVALID_PROCESS_ID)
        s.Printf (" parent = %llu\n", m_parent_pid);

    if (m_executable)
    {
        s.Printf ("   name = %s\n", m_executable.GetFilename().GetCString());
        s.PutCString ("   file = ");
        m_executable.Dump(&s);
        s.EOL();
    }
    const uint32_t argc = m_arguments.GetArgumentCount();
    if (argc > 0)
    {
        for (uint32_t i=0; i<argc; i++)
        {
            const char *arg = m_arguments.GetArgumentAtIndex(i);
            if (i < 10)
                s.Printf (" arg[%u] = %s\n", i, arg);
            else
                s.Printf ("arg[%u] = %s\n", i, arg);
        }
    }

    const uint32_t envc = m_environment.GetArgumentCount();
    if (envc > 0)
    {
        for (uint32_t i=0; i<envc; i++)
        {
            const char *env = m_environment.GetArgumentAtIndex(i);
            if (i < 10)
                s.Printf (" env[%u] = %s\n", i, env);
            else
                s.Printf ("env[%u] = %s\n", i, env);
        }
    }

    if (m_arch.IsValid())                       
        s.Printf ("   arch = %s\n", m_arch.GetTriple().str().c_str());

    if (m_uid != UINT32_MAX)
    {
        cstr = platform->GetUserName (m_uid);
        s.Printf ("    uid = %-5u (%s)\n", m_uid, cstr ? cstr : "");
    }
    if (m_gid != UINT32_MAX)
    {
        cstr = platform->GetGroupName (m_gid);
        s.Printf ("    gid = %-5u (%s)\n", m_gid, cstr ? cstr : "");
    }
    if (m_euid != UINT32_MAX)
    {
        cstr = platform->GetUserName (m_euid);
        s.Printf ("   euid = %-5u (%s)\n", m_euid, cstr ? cstr : "");
    }
    if (m_egid != UINT32_MAX)
    {
        cstr = platform->GetGroupName (m_egid);
        s.Printf ("   egid = %-5u (%s)\n", m_egid, cstr ? cstr : "");
    }
}

void
ProcessInstanceInfo::DumpTableHeader (Stream &s, Platform *platform, bool show_args, bool verbose)
{
    const char *label;
    if (show_args || verbose)
        label = "ARGUMENTS";
    else
        label = "NAME";

    if (verbose)
    {
        s.Printf     ("PID    PARENT USER       GROUP      EFF USER   EFF GROUP  TRIPLE                   %s\n", label);
        s.PutCString ("====== ====== ========== ========== ========== ========== ======================== ============================\n");
    }
    else
    {
        s.Printf     ("PID    PARENT USER       ARCH    %s\n", label);
        s.PutCString ("====== ====== ========== ======= ============================\n");
    }
}

void
ProcessInstanceInfo::DumpAsTableRow (Stream &s, Platform *platform, bool show_args, bool verbose) const
{
    if (m_pid != LLDB_INVALID_PROCESS_ID)
    {
        const char *cstr;
        s.Printf ("%-6llu %-6llu ", m_pid, m_parent_pid);

    
        if (verbose)
        {
            cstr = platform->GetUserName (m_uid);
            if (cstr && cstr[0]) // Watch for empty string that indicates lookup failed
                s.Printf ("%-10s ", cstr);
            else
                s.Printf ("%-10u ", m_uid);

            cstr = platform->GetGroupName (m_gid);
            if (cstr && cstr[0]) // Watch for empty string that indicates lookup failed
                s.Printf ("%-10s ", cstr);
            else
                s.Printf ("%-10u ", m_gid);

            cstr = platform->GetUserName (m_euid);
            if (cstr && cstr[0]) // Watch for empty string that indicates lookup failed
                s.Printf ("%-10s ", cstr);
            else
                s.Printf ("%-10u ", m_euid);
            
            cstr = platform->GetGroupName (m_egid);
            if (cstr && cstr[0]) // Watch for empty string that indicates lookup failed
                s.Printf ("%-10s ", cstr);
            else
                s.Printf ("%-10u ", m_egid);
            s.Printf ("%-24s ", m_arch.IsValid() ? m_arch.GetTriple().str().c_str() : "");
        }
        else
        {
            s.Printf ("%-10s %-7d %s ", 
                      platform->GetUserName (m_euid),
                      (int)m_arch.GetTriple().getArchName().size(),
                      m_arch.GetTriple().getArchName().data());
        }

        if (verbose || show_args)
        {
            const uint32_t argc = m_arguments.GetArgumentCount();
            if (argc > 0)
            {
                for (uint32_t i=0; i<argc; i++)
                {
                    if (i > 0)
                        s.PutChar (' ');
                    s.PutCString (m_arguments.GetArgumentAtIndex(i));
                }
            }
        }
        else
        {
            s.PutCString (GetName());
        }

        s.EOL();
    }
}


void
ProcessInfo::SetArguments (char const **argv, bool first_arg_is_executable)
{
    m_arguments.SetArguments (argv);
        
    // Is the first argument the executable?
    if (first_arg_is_executable)
    {
        const char *first_arg = m_arguments.GetArgumentAtIndex (0);
        if (first_arg)
        {
            // Yes the first argument is an executable, set it as the executable
            // in the launch options. Don't resolve the file path as the path
            // could be a remote platform path
            const bool resolve = false;
            m_executable.SetFile(first_arg, resolve); 
        }
    }
}
void
ProcessInfo::SetArguments (const Args& args, bool first_arg_is_executable)
{
    // Copy all arguments
    m_arguments = args;

    // Is the first argument the executable?
    if (first_arg_is_executable)
    {
        const char *first_arg = m_arguments.GetArgumentAtIndex (0);
        if (first_arg)
        {
            // Yes the first argument is an executable, set it as the executable
            // in the launch options. Don't resolve the file path as the path
            // could be a remote platform path
            const bool resolve = false;
            m_executable.SetFile(first_arg, resolve); 
        }
    }
}

void
ProcessLaunchInfo::FinalizeFileActions (Target *target, bool default_to_use_pty)
{
    // If notthing was specified, then check the process for any default 
    // settings that were set with "settings set"
    if (m_file_actions.empty())
    {
        if (m_flags.Test(eLaunchFlagDisableSTDIO))
        {
            AppendSuppressFileAction (STDIN_FILENO , true, false);
            AppendSuppressFileAction (STDOUT_FILENO, false, true);
            AppendSuppressFileAction (STDERR_FILENO, false, true);
        }
        else
        {
            // Check for any values that might have gotten set with any of:
            // (lldb) settings set target.input-path
            // (lldb) settings set target.output-path
            // (lldb) settings set target.error-path
            FileSpec in_path;
            FileSpec out_path;
            FileSpec err_path;
            if (target)
            {
                in_path = target->GetStandardInputPath();
                out_path = target->GetStandardOutputPath();
                err_path = target->GetStandardErrorPath();
            }
            
            if (in_path || out_path || err_path)
            {
                char path[PATH_MAX];
                if (in_path && in_path.GetPath(path, sizeof(path)))
                    AppendOpenFileAction(STDIN_FILENO, path, true, false);
                
                if (out_path && out_path.GetPath(path, sizeof(path)))
                    AppendOpenFileAction(STDOUT_FILENO, path, false, true);
                
                if (err_path && err_path.GetPath(path, sizeof(path)))
                    AppendOpenFileAction(STDERR_FILENO, path, false, true);
            }
            else if (default_to_use_pty)
            {
                if (m_pty.OpenFirstAvailableMaster (O_RDWR|O_NOCTTY, NULL, 0))
                {
                    const char *slave_path = m_pty.GetSlaveName (NULL, 0);
                    AppendOpenFileAction(STDIN_FILENO, slave_path, true, false);
                    AppendOpenFileAction(STDOUT_FILENO, slave_path, false, true);
                    AppendOpenFileAction(STDERR_FILENO, slave_path, false, true);
                }
            }
        }
    }
}


bool
ProcessLaunchInfo::ConvertArgumentsForLaunchingInShell (Error &error,
                                                        bool localhost,
                                                        bool will_debug,
                                                        bool first_arg_is_full_shell_command)
{
    error.Clear();

    if (GetFlags().Test (eLaunchFlagLaunchInShell))
    {
        const char *shell_executable = GetShell();
        if (shell_executable)
        {
            char shell_resolved_path[PATH_MAX];

            if (localhost)
            {
                FileSpec shell_filespec (shell_executable, true);
                
                if (!shell_filespec.Exists())
                {
                    // Resolve the path in case we just got "bash", "sh" or "tcsh"
                    if (!shell_filespec.ResolveExecutableLocation ())
                    {
                        error.SetErrorStringWithFormat("invalid shell path '%s'", shell_executable);
                        return false;
                    }
                }
                shell_filespec.GetPath (shell_resolved_path, sizeof(shell_resolved_path));
                shell_executable = shell_resolved_path;
            }
            
            const char **argv = GetArguments().GetConstArgumentVector ();
            if (argv == NULL || argv[0] == NULL)
                return false;
            Args shell_arguments;
            std::string safe_arg;
            shell_arguments.AppendArgument (shell_executable);
            shell_arguments.AppendArgument ("-c");
            StreamString shell_command;
            if (will_debug)
            {
                // Add a modified PATH environment variable in case argv[0]
                // is a relative path
                const char *argv0 = argv[0];
                if (argv0 && (argv0[0] != '/' && argv0[0] != '~'))
                {
                    // We have a relative path to our executable which may not work if
                    // we just try to run "a.out" (without it being converted to "./a.out")
                    const char *working_dir = GetWorkingDirectory();
                    std::string new_path("PATH=");
                    const size_t empty_path_len = new_path.size();
                    
                    if (working_dir && working_dir[0])
                    {
                        new_path += working_dir;
                    }
                    else
                    {
                        char current_working_dir[PATH_MAX];
                        const char *cwd = getcwd(current_working_dir, sizeof(current_working_dir));
                        if (cwd && cwd[0])
                            new_path += cwd;
                    }
                    const char *curr_path = getenv("PATH");
                    if (curr_path)
                    {
                        if (new_path.size() > empty_path_len)
                            new_path += ':';
                        new_path += curr_path;
                    }
                    new_path += ' ';
                    shell_command.PutCString(new_path.c_str());
                }

                shell_command.PutCString ("exec");

#if defined(__APPLE__)
                // Only Apple supports /usr/bin/arch being able to specify the architecture
                if (GetArchitecture().IsValid())
                {
                    shell_command.Printf(" /usr/bin/arch -arch %s", GetArchitecture().GetArchitectureName());
                    // Set the resume count to 2:
                    // 1 - stop in shell
                    // 2 - stop in /usr/bin/arch
                    // 3 - then we will stop in our program
                    SetResumeCount(2);
                }
                else
                {
                    // Set the resume count to 1:
                    // 1 - stop in shell
                    // 2 - then we will stop in our program
                    SetResumeCount(1);
                }
#else
                // Set the resume count to 1:
                // 1 - stop in shell
                // 2 - then we will stop in our program
                SetResumeCount(1);
#endif
            }
        
            if (first_arg_is_full_shell_command)
            {
                // There should only be one argument that is the shell command itself to be used as is
                if (argv[0] && !argv[1])
                    shell_command.Printf("%s", argv[0]);
                else
                    return false;
            }
            else
            {
                for (size_t i=0; argv[i] != NULL; ++i)
                {
                    const char *arg = Args::GetShellSafeArgument (argv[i], safe_arg);
                    shell_command.Printf(" %s", arg);
                }
            }
            shell_arguments.AppendArgument (shell_command.GetString().c_str());
            m_executable.SetFile(shell_executable, false);
            m_arguments = shell_arguments;
            return true;
        }
        else
        {
            error.SetErrorString ("invalid shell path");
        }
    }
    else
    {
        error.SetErrorString ("not launching in shell");
    }
    return false;
}


bool
ProcessLaunchInfo::FileAction::Open (int fd, const char *path, bool read, bool write)
{
    if ((read || write) && fd >= 0 && path && path[0])
    {
        m_action = eFileActionOpen;
        m_fd = fd;
        if (read && write)
            m_arg = O_NOCTTY | O_CREAT | O_RDWR;
        else if (read)
            m_arg = O_NOCTTY | O_RDONLY;
        else
            m_arg = O_NOCTTY | O_CREAT | O_WRONLY;
        m_path.assign (path);
        return true;
    }
    else
    {
        Clear();
    }
    return false;
}

bool
ProcessLaunchInfo::FileAction::Close (int fd)
{
    Clear();
    if (fd >= 0)
    {
        m_action = eFileActionClose;
        m_fd = fd;
    }
    return m_fd >= 0;
}


bool
ProcessLaunchInfo::FileAction::Duplicate (int fd, int dup_fd)
{
    Clear();
    if (fd >= 0 && dup_fd >= 0)
    {
        m_action = eFileActionDuplicate;
        m_fd = fd;
        m_arg = dup_fd;
    }
    return m_fd >= 0;
}



bool
ProcessLaunchInfo::FileAction::AddPosixSpawnFileAction (posix_spawn_file_actions_t *file_actions,
                                                        const FileAction *info,
                                                        Log *log, 
                                                        Error& error)
{
    if (info == NULL)
        return false;

    switch (info->m_action)
    {
        case eFileActionNone:
            error.Clear();
            break;

        case eFileActionClose:
            if (info->m_fd == -1)
                error.SetErrorString ("invalid fd for posix_spawn_file_actions_addclose(...)");
            else
            {
                error.SetError (::posix_spawn_file_actions_addclose (file_actions, info->m_fd), 
                                eErrorTypePOSIX);
                if (log && (error.Fail() || log))
                    error.PutToLog(log, "posix_spawn_file_actions_addclose (action=%p, fd=%i)", 
                                   file_actions, info->m_fd);
            }
            break;

        case eFileActionDuplicate:
            if (info->m_fd == -1)
                error.SetErrorString ("invalid fd for posix_spawn_file_actions_adddup2(...)");
            else if (info->m_arg == -1)
                error.SetErrorString ("invalid duplicate fd for posix_spawn_file_actions_adddup2(...)");
            else
            {
                error.SetError (::posix_spawn_file_actions_adddup2 (file_actions, info->m_fd, info->m_arg),
                                eErrorTypePOSIX);
                if (log && (error.Fail() || log))
                    error.PutToLog(log, "posix_spawn_file_actions_adddup2 (action=%p, fd=%i, dup_fd=%i)", 
                                   file_actions, info->m_fd, info->m_arg);
            }
            break;

        case eFileActionOpen:
            if (info->m_fd == -1)
                error.SetErrorString ("invalid fd in posix_spawn_file_actions_addopen(...)");
            else
            {
                int oflag = info->m_arg;
                
                mode_t mode = 0;

                if (oflag & O_CREAT)
                    mode = 0640;

                error.SetError (::posix_spawn_file_actions_addopen (file_actions, 
                                                                    info->m_fd,
                                                                    info->m_path.c_str(), 
                                                                    oflag,
                                                                    mode), 
                                eErrorTypePOSIX);
                if (error.Fail() || log)
                    error.PutToLog(log, 
                                   "posix_spawn_file_actions_addopen (action=%p, fd=%i, path='%s', oflag=%i, mode=%i)", 
                                   file_actions, info->m_fd, info->m_path.c_str(), oflag, mode);
            }
            break;
        
        default:
            error.SetErrorStringWithFormat ("invalid file action: %i", info->m_action);
            break;
    }
    return error.Success();
}

Error
ProcessLaunchCommandOptions::SetOptionValue (uint32_t option_idx, const char *option_arg)
{
    Error error;
    char short_option = (char) m_getopt_table[option_idx].val;
    
    switch (short_option)
    {
        case 's':   // Stop at program entry point
            launch_info.GetFlags().Set (eLaunchFlagStopAtEntry); 
            break;
            
        case 'i':   // STDIN for read only
            {   
                ProcessLaunchInfo::FileAction action;
                if (action.Open (STDIN_FILENO, option_arg, true, false))
                    launch_info.AppendFileAction (action);
            }
            break;
            
        case 'o':   // Open STDOUT for write only
            {   
                ProcessLaunchInfo::FileAction action;
                if (action.Open (STDOUT_FILENO, option_arg, false, true))
                    launch_info.AppendFileAction (action);
            }
            break;

        case 'e':   // STDERR for write only
            {   
                ProcessLaunchInfo::FileAction action;
                if (action.Open (STDERR_FILENO, option_arg, false, true))
                    launch_info.AppendFileAction (action);
            }
            break;
            

        case 'p':   // Process plug-in name
            launch_info.SetProcessPluginName (option_arg);    
            break;
            
        case 'n':   // Disable STDIO
            {
                ProcessLaunchInfo::FileAction action;
                if (action.Open (STDIN_FILENO, "/dev/null", true, false))
                    launch_info.AppendFileAction (action);
                if (action.Open (STDOUT_FILENO, "/dev/null", false, true))
                    launch_info.AppendFileAction (action);
                if (action.Open (STDERR_FILENO, "/dev/null", false, true))
                    launch_info.AppendFileAction (action);
            }
            break;
            
        case 'w': 
            launch_info.SetWorkingDirectory (option_arg);    
            break;
            
        case 't':   // Open process in new terminal window
            launch_info.GetFlags().Set (eLaunchFlagLaunchInTTY); 
            break;
            
        case 'a':
            if (!launch_info.GetArchitecture().SetTriple (option_arg, m_interpreter.GetPlatform(true).get()))
                launch_info.GetArchitecture().SetTriple (option_arg);
            break;
            
        case 'A':   
            launch_info.GetFlags().Set (eLaunchFlagDisableASLR); 
            break;
            
        case 'c':   
            if (option_arg && option_arg[0])
                launch_info.SetShell (option_arg);
            else
                launch_info.SetShell ("/bin/bash");
            break;
            
        case 'v':
            launch_info.GetEnvironmentEntries().AppendArgument(option_arg);
            break;

        default:
            error.SetErrorStringWithFormat("unrecognized short option character '%c'", short_option);
            break;
            
    }
    return error;
}

OptionDefinition
ProcessLaunchCommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_ALL, false, "stop-at-entry", 's', no_argument,       NULL, 0, eArgTypeNone,          "Stop at the entry point of the program when launching a process."},
{ LLDB_OPT_SET_ALL, false, "disable-aslr",  'A', no_argument,       NULL, 0, eArgTypeNone,          "Disable address space layout randomization when launching a process."},
{ LLDB_OPT_SET_ALL, false, "plugin",        'p', required_argument, NULL, 0, eArgTypePlugin,        "Name of the process plugin you want to use."},
{ LLDB_OPT_SET_ALL, false, "working-dir",   'w', required_argument, NULL, 0, eArgTypeDirectoryName,          "Set the current working directory to <path> when running the inferior."},
{ LLDB_OPT_SET_ALL, false, "arch",          'a', required_argument, NULL, 0, eArgTypeArchitecture,  "Set the architecture for the process to launch when ambiguous."},
{ LLDB_OPT_SET_ALL, false, "environment",   'v', required_argument, NULL, 0, eArgTypeNone,          "Specify an environment variable name/value stirng (--environement NAME=VALUE). Can be specified multiple times for subsequent environment entries."},
{ LLDB_OPT_SET_ALL, false, "shell",         'c', optional_argument, NULL, 0, eArgTypeFilename,          "Run the process in a shell (not supported on all platforms)."},

{ LLDB_OPT_SET_1  , false, "stdin",         'i', required_argument, NULL, 0, eArgTypeFilename,    "Redirect stdin for the process to <filename>."},
{ LLDB_OPT_SET_1  , false, "stdout",        'o', required_argument, NULL, 0, eArgTypeFilename,    "Redirect stdout for the process to <filename>."},
{ LLDB_OPT_SET_1  , false, "stderr",        'e', required_argument, NULL, 0, eArgTypeFilename,    "Redirect stderr for the process to <filename>."},

{ LLDB_OPT_SET_2  , false, "tty",           't', no_argument,       NULL, 0, eArgTypeNone,    "Start the process in a terminal (not supported on all platforms)."},

{ LLDB_OPT_SET_3  , false, "no-stdio",      'n', no_argument,       NULL, 0, eArgTypeNone,    "Do not set up for terminal I/O to go to running process."},

{ 0               , false, NULL,             0,  0,                 NULL, 0, eArgTypeNone,    NULL }
};



bool
ProcessInstanceInfoMatch::NameMatches (const char *process_name) const
{
    if (m_name_match_type == eNameMatchIgnore || process_name == NULL)
        return true;
    const char *match_name = m_match_info.GetName();
    if (!match_name)
        return true;
    
    return lldb_private::NameMatches (process_name, m_name_match_type, match_name);
}

bool
ProcessInstanceInfoMatch::Matches (const ProcessInstanceInfo &proc_info) const
{
    if (!NameMatches (proc_info.GetName()))
        return false;

    if (m_match_info.ProcessIDIsValid() &&
        m_match_info.GetProcessID() != proc_info.GetProcessID())
        return false;

    if (m_match_info.ParentProcessIDIsValid() &&
        m_match_info.GetParentProcessID() != proc_info.GetParentProcessID())
        return false;

    if (m_match_info.UserIDIsValid () && 
        m_match_info.GetUserID() != proc_info.GetUserID())
        return false;
    
    if (m_match_info.GroupIDIsValid () && 
        m_match_info.GetGroupID() != proc_info.GetGroupID())
        return false;
    
    if (m_match_info.EffectiveUserIDIsValid () && 
        m_match_info.GetEffectiveUserID() != proc_info.GetEffectiveUserID())
        return false;
    
    if (m_match_info.EffectiveGroupIDIsValid () && 
        m_match_info.GetEffectiveGroupID() != proc_info.GetEffectiveGroupID())
        return false;
    
    if (m_match_info.GetArchitecture().IsValid() && 
        m_match_info.GetArchitecture() != proc_info.GetArchitecture())
        return false;
    return true;
}

bool
ProcessInstanceInfoMatch::MatchAllProcesses () const
{
    if (m_name_match_type != eNameMatchIgnore)
        return false;
    
    if (m_match_info.ProcessIDIsValid())
        return false;
    
    if (m_match_info.ParentProcessIDIsValid())
        return false;
    
    if (m_match_info.UserIDIsValid ())
        return false;
    
    if (m_match_info.GroupIDIsValid ())
        return false;
    
    if (m_match_info.EffectiveUserIDIsValid ())
        return false;
    
    if (m_match_info.EffectiveGroupIDIsValid ())
        return false;
    
    if (m_match_info.GetArchitecture().IsValid())
        return false;

    if (m_match_all_users)
        return false;

    return true;

}

void
ProcessInstanceInfoMatch::Clear()
{
    m_match_info.Clear();
    m_name_match_type = eNameMatchIgnore;
    m_match_all_users = false;
}

ProcessSP
Process::FindPlugin (Target &target, const char *plugin_name, Listener &listener, const FileSpec *crash_file_path)
{
    ProcessSP process_sp;
    ProcessCreateInstance create_callback = NULL;
    if (plugin_name)
    {
        create_callback  = PluginManager::GetProcessCreateCallbackForPluginName (plugin_name);
        if (create_callback)
        {
            process_sp = create_callback(target, listener, crash_file_path);
            if (process_sp)
            {
                if (!process_sp->CanDebug(target, true))
                    process_sp.reset();
            }
        }
    }
    else
    {
        for (uint32_t idx = 0; (create_callback = PluginManager::GetProcessCreateCallbackAtIndex(idx)) != NULL; ++idx)
        {
            process_sp = create_callback(target, listener, crash_file_path);
            if (process_sp)
            {
                if (!process_sp->CanDebug(target, false))
                    process_sp.reset();
                else
                    break;
            }
        }
    }
    return process_sp;
}

ConstString &
Process::GetStaticBroadcasterClass ()
{
    static ConstString class_name ("lldb.process");
    return class_name;
}

//----------------------------------------------------------------------
// Process constructor
//----------------------------------------------------------------------
Process::Process(Target &target, Listener &listener) :
    ProcessProperties (false),
    UserID (LLDB_INVALID_PROCESS_ID),
    Broadcaster (&(target.GetDebugger()), "lldb.process"),
    m_target (target),
    m_public_state (eStateUnloaded),
    m_private_state (eStateUnloaded),
    m_private_state_broadcaster (NULL, "lldb.process.internal_state_broadcaster"),
    m_private_state_control_broadcaster (NULL, "lldb.process.internal_state_control_broadcaster"),
    m_private_state_listener ("lldb.process.internal_state_listener"),
    m_private_state_control_wait(),
    m_private_state_thread (LLDB_INVALID_HOST_THREAD),
    m_mod_id (),
    m_thread_index_id (0),
    m_exit_status (-1),
    m_exit_string (),
    m_thread_list (this),
    m_notifications (),
    m_image_tokens (),
    m_listener (listener),
    m_breakpoint_site_list (),
    m_dynamic_checkers_ap (),
    m_unix_signals (),
    m_abi_sp (),
    m_process_input_reader (),
    m_stdio_communication ("process.stdio"),
    m_stdio_communication_mutex (Mutex::eMutexTypeRecursive),
    m_stdout_data (),
    m_stderr_data (),
    m_memory_cache (*this),
    m_allocated_memory_cache (*this),
    m_should_detach (false),
    m_next_event_action_ap(),
    m_run_lock (),
    m_currently_handling_event(false),
    m_finalize_called(false),
    m_can_jit(eCanJITDontKnow)
{
    CheckInWithManager ();

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
        log->Printf ("%p Process::Process()", this);

    SetEventName (eBroadcastBitStateChanged, "state-changed");
    SetEventName (eBroadcastBitInterrupt, "interrupt");
    SetEventName (eBroadcastBitSTDOUT, "stdout-available");
    SetEventName (eBroadcastBitSTDERR, "stderr-available");
    
    m_private_state_control_broadcaster.SetEventName (eBroadcastInternalStateControlStop  , "control-stop"  );
    m_private_state_control_broadcaster.SetEventName (eBroadcastInternalStateControlPause , "control-pause" );
    m_private_state_control_broadcaster.SetEventName (eBroadcastInternalStateControlResume, "control-resume");

    listener.StartListeningForEvents (this,
                                      eBroadcastBitStateChanged |
                                      eBroadcastBitInterrupt |
                                      eBroadcastBitSTDOUT |
                                      eBroadcastBitSTDERR);

    m_private_state_listener.StartListeningForEvents(&m_private_state_broadcaster,
                                                     eBroadcastBitStateChanged |
                                                     eBroadcastBitInterrupt);

    m_private_state_listener.StartListeningForEvents(&m_private_state_control_broadcaster,
                                                     eBroadcastInternalStateControlStop |
                                                     eBroadcastInternalStateControlPause |
                                                     eBroadcastInternalStateControlResume);
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
Process::~Process()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
        log->Printf ("%p Process::~Process()", this);
    StopPrivateStateThread();
}

const ProcessPropertiesSP &
Process::GetGlobalProperties()
{
    static ProcessPropertiesSP g_settings_sp;
    if (!g_settings_sp)
        g_settings_sp.reset (new ProcessProperties (true));
    return g_settings_sp;
}

void
Process::Finalize()
{
    switch (GetPrivateState())
    {
        case eStateConnected:
        case eStateAttaching:
        case eStateLaunching:
        case eStateStopped:
        case eStateRunning:
        case eStateStepping:
        case eStateCrashed:
        case eStateSuspended:
            if (GetShouldDetach())
                Detach();
            else
                Destroy();
            break;
            
        case eStateInvalid:
        case eStateUnloaded:
        case eStateDetached:
        case eStateExited:
            break;
    }

    // Clear our broadcaster before we proceed with destroying
    Broadcaster::Clear();

    // Do any cleanup needed prior to being destructed... Subclasses
    // that override this method should call this superclass method as well.
    
    // We need to destroy the loader before the derived Process class gets destroyed
    // since it is very likely that undoing the loader will require access to the real process.
    m_dynamic_checkers_ap.reset();
    m_abi_sp.reset();
    m_os_ap.reset();
    m_dyld_ap.reset();
    m_thread_list.Destroy();
    std::vector<Notifications> empty_notifications;
    m_notifications.swap(empty_notifications);
    m_image_tokens.clear();
    m_memory_cache.Clear();
    m_allocated_memory_cache.Clear();
    m_language_runtimes.clear();
    m_next_event_action_ap.reset();
//#ifdef LLDB_CONFIGURATION_DEBUG
//    StreamFile s(stdout, false);
//    EventSP event_sp;
//    while (m_private_state_listener.GetNextEvent(event_sp))
//    {
//        event_sp->Dump (&s);
//        s.EOL();
//    }
//#endif
    // We have to be very careful here as the m_private_state_listener might
    // contain events that have ProcessSP values in them which can keep this
    // process around forever. These events need to be cleared out.
    m_private_state_listener.Clear();
    m_finalize_called = true;
}

void
Process::RegisterNotificationCallbacks (const Notifications& callbacks)
{
    m_notifications.push_back(callbacks);
    if (callbacks.initialize != NULL)
        callbacks.initialize (callbacks.baton, this);
}

bool
Process::UnregisterNotificationCallbacks(const Notifications& callbacks)
{
    std::vector<Notifications>::iterator pos, end = m_notifications.end();
    for (pos = m_notifications.begin(); pos != end; ++pos)
    {
        if (pos->baton == callbacks.baton &&
            pos->initialize == callbacks.initialize &&
            pos->process_state_changed == callbacks.process_state_changed)
        {
            m_notifications.erase(pos);
            return true;
        }
    }
    return false;
}

void
Process::SynchronouslyNotifyStateChanged (StateType state)
{
    std::vector<Notifications>::iterator notification_pos, notification_end = m_notifications.end();
    for (notification_pos = m_notifications.begin(); notification_pos != notification_end; ++notification_pos)
    {
        if (notification_pos->process_state_changed)
            notification_pos->process_state_changed (notification_pos->baton, this, state);
    }
}

// FIXME: We need to do some work on events before the general Listener sees them.
// For instance if we are continuing from a breakpoint, we need to ensure that we do
// the little "insert real insn, step & stop" trick.  But we can't do that when the
// event is delivered by the broadcaster - since that is done on the thread that is
// waiting for new events, so if we needed more than one event for our handling, we would
// stall.  So instead we do it when we fetch the event off of the queue.
//

StateType
Process::GetNextEvent (EventSP &event_sp)
{
    StateType state = eStateInvalid;

    if (m_listener.GetNextEventForBroadcaster (this, event_sp) && event_sp)
        state = Process::ProcessEventData::GetStateFromEvent (event_sp.get());

    return state;
}


StateType
Process::WaitForProcessToStop (const TimeValue *timeout, lldb::EventSP *event_sp_ptr)
{
    // We can't just wait for a "stopped" event, because the stopped event may have restarted the target.
    // We have to actually check each event, and in the case of a stopped event check the restarted flag
    // on the event.
    if (event_sp_ptr)
        event_sp_ptr->reset();
    StateType state = GetState();
    // If we are exited or detached, we won't ever get back to any
    // other valid state...
    if (state == eStateDetached || state == eStateExited)
        return state;

    while (state != eStateInvalid)
    {
        EventSP event_sp;
        state = WaitForStateChangedEvents (timeout, event_sp);
        if (event_sp_ptr && event_sp)
            *event_sp_ptr = event_sp;

        switch (state)
        {
        case eStateCrashed:
        case eStateDetached:
        case eStateExited:
        case eStateUnloaded:
            return state;
        case eStateStopped:
            if (Process::ProcessEventData::GetRestartedFromEvent(event_sp.get()))
                continue;
            else
                return state;
        default:
            continue;
        }
    }
    return state;
}


StateType
Process::WaitForState
(
    const TimeValue *timeout,
    const StateType *match_states, const uint32_t num_match_states
)
{
    EventSP event_sp;
    uint32_t i;
    StateType state = GetState();
    while (state != eStateInvalid)
    {
        // If we are exited or detached, we won't ever get back to any
        // other valid state...
        if (state == eStateDetached || state == eStateExited)
            return state;

        state = WaitForStateChangedEvents (timeout, event_sp);

        for (i=0; i<num_match_states; ++i)
        {
            if (match_states[i] == state)
                return state;
        }
    }
    return state;
}

bool
Process::HijackProcessEvents (Listener *listener)
{
    if (listener != NULL)
    {
        return HijackBroadcaster(listener, eBroadcastBitStateChanged | eBroadcastBitInterrupt);
    }
    else
        return false;
}

void
Process::RestoreProcessEvents ()
{
    RestoreBroadcaster();
}

bool
Process::HijackPrivateProcessEvents (Listener *listener)
{
    if (listener != NULL)
    {
        return m_private_state_broadcaster.HijackBroadcaster(listener, eBroadcastBitStateChanged | eBroadcastBitInterrupt);
    }
    else
        return false;
}

void
Process::RestorePrivateProcessEvents ()
{
    m_private_state_broadcaster.RestoreBroadcaster();
}

StateType
Process::WaitForStateChangedEvents (const TimeValue *timeout, EventSP &event_sp)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    if (log)
        log->Printf ("Process::%s (timeout = %p, event_sp)...", __FUNCTION__, timeout);

    StateType state = eStateInvalid;
    if (m_listener.WaitForEventForBroadcasterWithType (timeout,
                                                       this,
                                                       eBroadcastBitStateChanged | eBroadcastBitInterrupt,
                                                       event_sp))
    {
        if (event_sp && event_sp->GetType() == eBroadcastBitStateChanged)
            state = Process::ProcessEventData::GetStateFromEvent(event_sp.get());
        else if (log)
            log->Printf ("Process::%s got no event or was interrupted.", __FUNCTION__);
    }

    if (log)
        log->Printf ("Process::%s (timeout = %p, event_sp) => %s",
                     __FUNCTION__,
                     timeout,
                     StateAsCString(state));
    return state;
}

Event *
Process::PeekAtStateChangedEvents ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    if (log)
        log->Printf ("Process::%s...", __FUNCTION__);

    Event *event_ptr;
    event_ptr = m_listener.PeekAtNextEventForBroadcasterWithType (this,
                                                                  eBroadcastBitStateChanged);
    if (log)
    {
        if (event_ptr)
        {
            log->Printf ("Process::%s (event_ptr) => %s",
                         __FUNCTION__,
                         StateAsCString(ProcessEventData::GetStateFromEvent (event_ptr)));
        }
        else 
        {
            log->Printf ("Process::%s no events found",
                         __FUNCTION__);
        }
    }
    return event_ptr;
}

StateType
Process::WaitForStateChangedEventsPrivate (const TimeValue *timeout, EventSP &event_sp)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    if (log)
        log->Printf ("Process::%s (timeout = %p, event_sp)...", __FUNCTION__, timeout);

    StateType state = eStateInvalid;
    if (m_private_state_listener.WaitForEventForBroadcasterWithType (timeout,
                                                                     &m_private_state_broadcaster,
                                                                     eBroadcastBitStateChanged | eBroadcastBitInterrupt,
                                                                     event_sp))
        if (event_sp && event_sp->GetType() == eBroadcastBitStateChanged)
            state = Process::ProcessEventData::GetStateFromEvent(event_sp.get());

    // This is a bit of a hack, but when we wait here we could very well return
    // to the command-line, and that could disable the log, which would render the
    // log we got above invalid.
    if (log)
    {
        if (state == eStateInvalid)
            log->Printf ("Process::%s (timeout = %p, event_sp) => TIMEOUT", __FUNCTION__, timeout);
        else
            log->Printf ("Process::%s (timeout = %p, event_sp) => %s", __FUNCTION__, timeout, StateAsCString(state));
    }
    return state;
}

bool
Process::WaitForEventsPrivate (const TimeValue *timeout, EventSP &event_sp, bool control_only)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    if (log)
        log->Printf ("Process::%s (timeout = %p, event_sp)...", __FUNCTION__, timeout);

    if (control_only)
        return m_private_state_listener.WaitForEventForBroadcaster(timeout, &m_private_state_control_broadcaster, event_sp);
    else
        return m_private_state_listener.WaitForEvent(timeout, event_sp);
}

bool
Process::IsRunning () const
{
    return StateIsRunningState (m_public_state.GetValue());
}

int
Process::GetExitStatus ()
{
    if (m_public_state.GetValue() == eStateExited)
        return m_exit_status;
    return -1;
}


const char *
Process::GetExitDescription ()
{
    if (m_public_state.GetValue() == eStateExited && !m_exit_string.empty())
        return m_exit_string.c_str();
    return NULL;
}

bool
Process::SetExitStatus (int status, const char *cstr)
{
    LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_STATE | LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf("Process::SetExitStatus (status=%i (0x%8.8x), description=%s%s%s)", 
                    status, status,
                    cstr ? "\"" : "",
                    cstr ? cstr : "NULL",
                    cstr ? "\"" : "");

    // We were already in the exited state
    if (m_private_state.GetValue() == eStateExited)
    {
        if (log)
            log->Printf("Process::SetExitStatus () ignoring exit status because state was already set to eStateExited");
        return false;
    }
    
    m_exit_status = status;
    if (cstr)
        m_exit_string = cstr;
    else
        m_exit_string.clear();

    DidExit ();

    SetPrivateState (eStateExited);
    return true;
}

// This static callback can be used to watch for local child processes on
// the current host. The the child process exits, the process will be
// found in the global target list (we want to be completely sure that the
// lldb_private::Process doesn't go away before we can deliver the signal.
bool
Process::SetProcessExitStatus (void *callback_baton,
                               lldb::pid_t pid,
                               bool exited,
                               int signo,          // Zero for no signal
                               int exit_status     // Exit value of process if signal is zero
)
{
    LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf ("Process::SetProcessExitStatus (baton=%p, pid=%llu, exited=%i, signal=%i, exit_status=%i)\n", 
                     callback_baton,
                     pid,
                     exited,
                     signo,
                     exit_status);

    if (exited)
    {
        TargetSP target_sp(Debugger::FindTargetWithProcessID (pid));
        if (target_sp)
        {
            ProcessSP process_sp (target_sp->GetProcessSP());
            if (process_sp)
            {
                const char *signal_cstr = NULL;
                if (signo)
                    signal_cstr = process_sp->GetUnixSignals().GetSignalAsCString (signo);

                process_sp->SetExitStatus (exit_status, signal_cstr);
            }
        }
        return true;
    }
    return false;
}


void
Process::UpdateThreadListIfNeeded ()
{
    const uint32_t stop_id = GetStopID();
    if (m_thread_list.GetSize(false) == 0 || stop_id != m_thread_list.GetStopID())
    {
        const StateType state = GetPrivateState();
        if (StateIsStoppedState (state, true))
        {
            Mutex::Locker locker (m_thread_list.GetMutex ());
            // m_thread_list does have its own mutex, but we need to
            // hold onto the mutex between the call to UpdateThreadList(...)
            // and the os->UpdateThreadList(...) so it doesn't change on us
            ThreadList new_thread_list(this);
            // Always update the thread list with the protocol specific
            // thread list, but only update if "true" is returned
            if (UpdateThreadList (m_thread_list, new_thread_list))
            {
                OperatingSystem *os = GetOperatingSystem ();
                if (os)
                    os->UpdateThreadList (m_thread_list, new_thread_list);
                m_thread_list.Update (new_thread_list);
                m_thread_list.SetStopID (stop_id);
            }
        }
    }
}

uint32_t
Process::GetNextThreadIndexID ()
{
    return ++m_thread_index_id;
}

StateType
Process::GetState()
{
    // If any other threads access this we will need a mutex for it
    return m_public_state.GetValue ();
}

void
Process::SetPublicState (StateType new_state)
{
    LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_STATE | LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf("Process::SetPublicState (%s)", StateAsCString(new_state));
    const StateType old_state = m_public_state.GetValue();
    m_public_state.SetValue (new_state);
    
    // On the transition from Run to Stopped, we unlock the writer end of the
    // run lock.  The lock gets locked in Resume, which is the public API
    // to tell the program to run.
    if (!IsHijackedForEvent(eBroadcastBitStateChanged))
    {
        if (new_state == eStateDetached)
        {
            if (log)
                log->Printf("Process::SetPublicState (%s) -- unlocking run lock for detach", StateAsCString(new_state));
            m_run_lock.WriteUnlock();
        }
        else
        {
            const bool old_state_is_stopped = StateIsStoppedState(old_state, false);
            const bool new_state_is_stopped = StateIsStoppedState(new_state, false);
            if (old_state_is_stopped != new_state_is_stopped)
            {
                if (new_state_is_stopped)
                {
                    if (log)
                        log->Printf("Process::SetPublicState (%s) -- unlocking run lock", StateAsCString(new_state));
                    m_run_lock.WriteUnlock();
                }
            }
        }
    }
}

Error
Process::Resume ()
{
    LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_STATE | LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf("Process::Resume -- locking run lock");
    if (!m_run_lock.WriteTryLock())
    {
        Error error("Resume request failed - process still running.");
        if (log)
            log->Printf ("Process::Resume: -- WriteTryLock failed, not resuming.");
        return error;
    }
    return PrivateResume();
}

StateType
Process::GetPrivateState ()
{
    return m_private_state.GetValue();
}

void
Process::SetPrivateState (StateType new_state)
{
    LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_STATE | LIBLLDB_LOG_PROCESS));
    bool state_changed = false;

    if (log)
        log->Printf("Process::SetPrivateState (%s)", StateAsCString(new_state));

    Mutex::Locker locker(m_private_state.GetMutex());

    const StateType old_state = m_private_state.GetValueNoLock ();
    state_changed = old_state != new_state;
    // This code is left commented out in case we ever need to control
    // the private process state with another run lock. Right now it doesn't
    // seem like we need to do this, but if we ever do, we can uncomment and
    // use this code.
//    const bool old_state_is_stopped = StateIsStoppedState(old_state, false);
//    const bool new_state_is_stopped = StateIsStoppedState(new_state, false);
//    if (old_state_is_stopped != new_state_is_stopped)
//    {
//        if (new_state_is_stopped)
//            m_private_run_lock.WriteUnlock();
//        else
//            m_private_run_lock.WriteLock();
//    }

    if (state_changed)
    {
        m_private_state.SetValueNoLock (new_state);
        if (StateIsStoppedState(new_state, false))
        {
            m_mod_id.BumpStopID();
            m_memory_cache.Clear();
            if (log)
                log->Printf("Process::SetPrivateState (%s) stop_id = %u", StateAsCString(new_state), m_mod_id.GetStopID());
        }
        // Use our target to get a shared pointer to ourselves...
        if (m_finalize_called && PrivateStateThreadIsValid() == false)
            BroadcastEvent (eBroadcastBitStateChanged, new ProcessEventData (shared_from_this(), new_state));
        else
            m_private_state_broadcaster.BroadcastEvent (eBroadcastBitStateChanged, new ProcessEventData (shared_from_this(), new_state));
    }
    else
    {
        if (log)
            log->Printf("Process::SetPrivateState (%s) state didn't change. Ignoring...", StateAsCString(new_state));
    }
}

void
Process::SetRunningUserExpression (bool on)
{
    m_mod_id.SetRunningUserExpression (on);
}

addr_t
Process::GetImageInfoAddress()
{
    return LLDB_INVALID_ADDRESS;
}

//----------------------------------------------------------------------
// LoadImage
//
// This function provides a default implementation that works for most
// unix variants. Any Process subclasses that need to do shared library
// loading differently should override LoadImage and UnloadImage and
// do what is needed.
//----------------------------------------------------------------------
uint32_t
Process::LoadImage (const FileSpec &image_spec, Error &error)
{
    char path[PATH_MAX];
    image_spec.GetPath(path, sizeof(path));

    DynamicLoader *loader = GetDynamicLoader();
    if (loader)
    {
        error = loader->CanLoadImage();
        if (error.Fail())
            return LLDB_INVALID_IMAGE_TOKEN;
    }
    
    if (error.Success())
    {
        ThreadSP thread_sp(GetThreadList ().GetSelectedThread());
        
        if (thread_sp)
        {
            StackFrameSP frame_sp (thread_sp->GetStackFrameAtIndex (0));
            
            if (frame_sp)
            {
                ExecutionContext exe_ctx;
                frame_sp->CalculateExecutionContext (exe_ctx);
                bool unwind_on_error = true;
                StreamString expr;
                expr.Printf("dlopen (\"%s\", 2)", path);
                const char *prefix = "extern \"C\" void* dlopen (const char *path, int mode);\n";
                lldb::ValueObjectSP result_valobj_sp;
                ClangUserExpression::Evaluate (exe_ctx,
                                               eExecutionPolicyAlways,
                                               lldb::eLanguageTypeUnknown,
                                               ClangUserExpression::eResultTypeAny,
                                               unwind_on_error,
                                               expr.GetData(),
                                               prefix,
                                               result_valobj_sp,
                                               true,
                                               ClangUserExpression::kDefaultTimeout);
                error = result_valobj_sp->GetError();
                if (error.Success())
                {
                    Scalar scalar;
                    if (result_valobj_sp->ResolveValue (scalar))
                    {
                        addr_t image_ptr = scalar.ULongLong(LLDB_INVALID_ADDRESS);
                        if (image_ptr != 0 && image_ptr != LLDB_INVALID_ADDRESS)
                        {
                            uint32_t image_token = m_image_tokens.size();
                            m_image_tokens.push_back (image_ptr);
                            return image_token;
                        }
                    }
                }
            }
        }
    }
    if (!error.AsCString())
        error.SetErrorStringWithFormat("unable to load '%s'", path);
    return LLDB_INVALID_IMAGE_TOKEN;
}

//----------------------------------------------------------------------
// UnloadImage
//
// This function provides a default implementation that works for most
// unix variants. Any Process subclasses that need to do shared library
// loading differently should override LoadImage and UnloadImage and
// do what is needed.
//----------------------------------------------------------------------
Error
Process::UnloadImage (uint32_t image_token)
{
    Error error;
    if (image_token < m_image_tokens.size())
    {
        const addr_t image_addr = m_image_tokens[image_token];
        if (image_addr == LLDB_INVALID_ADDRESS)
        {
            error.SetErrorString("image already unloaded");
        }
        else
        {
            DynamicLoader *loader = GetDynamicLoader();
            if (loader)
                error = loader->CanLoadImage();
            
            if (error.Success())
            {
                ThreadSP thread_sp(GetThreadList ().GetSelectedThread());
                
                if (thread_sp)
                {
                    StackFrameSP frame_sp (thread_sp->GetStackFrameAtIndex (0));
                    
                    if (frame_sp)
                    {
                        ExecutionContext exe_ctx;
                        frame_sp->CalculateExecutionContext (exe_ctx);
                        bool unwind_on_error = true;
                        StreamString expr;
                        expr.Printf("dlclose ((void *)0x%llx)", image_addr);
                        const char *prefix = "extern \"C\" int dlclose(void* handle);\n";
                        lldb::ValueObjectSP result_valobj_sp;
                        ClangUserExpression::Evaluate (exe_ctx,
                                                       eExecutionPolicyAlways,
                                                       lldb::eLanguageTypeUnknown,
                                                       ClangUserExpression::eResultTypeAny,
                                                       unwind_on_error,
                                                       expr.GetData(),
                                                       prefix,
                                                       result_valobj_sp,
                                                       true,
                                                       ClangUserExpression::kDefaultTimeout);
                        if (result_valobj_sp->GetError().Success())
                        {
                            Scalar scalar;
                            if (result_valobj_sp->ResolveValue (scalar))
                            {
                                if (scalar.UInt(1))
                                {
                                    error.SetErrorStringWithFormat("expression failed: \"%s\"", expr.GetData());
                                }
                                else
                                {
                                    m_image_tokens[image_token] = LLDB_INVALID_ADDRESS;
                                }
                            }
                        }
                        else
                        {
                            error = result_valobj_sp->GetError();
                        }
                    }
                }
            }
        }
    }
    else
    {
        error.SetErrorString("invalid image token");
    }
    return error;
}

const lldb::ABISP &
Process::GetABI()
{
    if (!m_abi_sp)
        m_abi_sp = ABI::FindPlugin(m_target.GetArchitecture());
    return m_abi_sp;
}

LanguageRuntime *
Process::GetLanguageRuntime(lldb::LanguageType language, bool retry_if_null)
{
    LanguageRuntimeCollection::iterator pos;
    pos = m_language_runtimes.find (language);
    if (pos == m_language_runtimes.end() || (retry_if_null && !(*pos).second))
    {
        lldb::LanguageRuntimeSP runtime_sp(LanguageRuntime::FindPlugin(this, language));
        
        m_language_runtimes[language] = runtime_sp;
        return runtime_sp.get();
    }
    else
        return (*pos).second.get();
}

CPPLanguageRuntime *
Process::GetCPPLanguageRuntime (bool retry_if_null)
{
    LanguageRuntime *runtime = GetLanguageRuntime(eLanguageTypeC_plus_plus, retry_if_null);
    if (runtime != NULL && runtime->GetLanguageType() == eLanguageTypeC_plus_plus)
        return static_cast<CPPLanguageRuntime *> (runtime);
    return NULL;
}

ObjCLanguageRuntime *
Process::GetObjCLanguageRuntime (bool retry_if_null)
{
    LanguageRuntime *runtime = GetLanguageRuntime(eLanguageTypeObjC, retry_if_null);
    if (runtime != NULL && runtime->GetLanguageType() == eLanguageTypeObjC)
        return static_cast<ObjCLanguageRuntime *> (runtime);
    return NULL;
}

bool
Process::IsPossibleDynamicValue (ValueObject& in_value)
{
    if (in_value.IsDynamic())
        return false;
    LanguageType known_type = in_value.GetObjectRuntimeLanguage();

    if (known_type != eLanguageTypeUnknown && known_type != eLanguageTypeC)
    {
        LanguageRuntime *runtime = GetLanguageRuntime (known_type);
        return runtime ? runtime->CouldHaveDynamicValue(in_value) : false;
    }

    LanguageRuntime *cpp_runtime = GetLanguageRuntime (eLanguageTypeC_plus_plus);
    if (cpp_runtime && cpp_runtime->CouldHaveDynamicValue(in_value))
        return true;
    
    LanguageRuntime *objc_runtime = GetLanguageRuntime (eLanguageTypeObjC);
    return objc_runtime ? objc_runtime->CouldHaveDynamicValue(in_value) : false;
}

BreakpointSiteList &
Process::GetBreakpointSiteList()
{
    return m_breakpoint_site_list;
}

const BreakpointSiteList &
Process::GetBreakpointSiteList() const
{
    return m_breakpoint_site_list;
}


void
Process::DisableAllBreakpointSites ()
{
    m_breakpoint_site_list.SetEnabledForAll (false);
    size_t num_sites = m_breakpoint_site_list.GetSize();
    for (size_t i = 0; i < num_sites; i++)
    {
        DisableBreakpoint (m_breakpoint_site_list.GetByIndex(i).get());
    }
}

Error
Process::ClearBreakpointSiteByID (lldb::user_id_t break_id)
{
    Error error (DisableBreakpointSiteByID (break_id));
    
    if (error.Success())
        m_breakpoint_site_list.Remove(break_id);

    return error;
}

Error
Process::DisableBreakpointSiteByID (lldb::user_id_t break_id)
{
    Error error;
    BreakpointSiteSP bp_site_sp = m_breakpoint_site_list.FindByID (break_id);
    if (bp_site_sp)
    {
        if (bp_site_sp->IsEnabled())
            error = DisableBreakpoint (bp_site_sp.get());
    }
    else
    {
        error.SetErrorStringWithFormat("invalid breakpoint site ID: %llu", break_id);
    }

    return error;
}

Error
Process::EnableBreakpointSiteByID (lldb::user_id_t break_id)
{
    Error error;
    BreakpointSiteSP bp_site_sp = m_breakpoint_site_list.FindByID (break_id);
    if (bp_site_sp)
    {
        if (!bp_site_sp->IsEnabled())
            error = EnableBreakpoint (bp_site_sp.get());
    }
    else
    {
        error.SetErrorStringWithFormat("invalid breakpoint site ID: %llu", break_id);
    }
    return error;
}

lldb::break_id_t
Process::CreateBreakpointSite (const BreakpointLocationSP &owner, bool use_hardware)
{
    const addr_t load_addr = owner->GetAddress().GetOpcodeLoadAddress (&m_target);
    if (load_addr != LLDB_INVALID_ADDRESS)
    {
        BreakpointSiteSP bp_site_sp;

        // Look up this breakpoint site.  If it exists, then add this new owner, otherwise
        // create a new breakpoint site and add it.

        bp_site_sp = m_breakpoint_site_list.FindByAddress (load_addr);

        if (bp_site_sp)
        {
            bp_site_sp->AddOwner (owner);
            owner->SetBreakpointSite (bp_site_sp);
            return bp_site_sp->GetID();
        }
        else
        {
            bp_site_sp.reset (new BreakpointSite (&m_breakpoint_site_list, owner, load_addr, LLDB_INVALID_THREAD_ID, use_hardware));
            if (bp_site_sp)
            {
                if (EnableBreakpoint (bp_site_sp.get()).Success())
                {
                    owner->SetBreakpointSite (bp_site_sp);
                    return m_breakpoint_site_list.Add (bp_site_sp);
                }
            }
        }
    }
    // We failed to enable the breakpoint
    return LLDB_INVALID_BREAK_ID;

}

void
Process::RemoveOwnerFromBreakpointSite (lldb::user_id_t owner_id, lldb::user_id_t owner_loc_id, BreakpointSiteSP &bp_site_sp)
{
    uint32_t num_owners = bp_site_sp->RemoveOwner (owner_id, owner_loc_id);
    if (num_owners == 0)
    {
        DisableBreakpoint(bp_site_sp.get());
        m_breakpoint_site_list.RemoveByAddress(bp_site_sp->GetLoadAddress());
    }
}


size_t
Process::RemoveBreakpointOpcodesFromBuffer (addr_t bp_addr, size_t size, uint8_t *buf) const
{
    size_t bytes_removed = 0;
    addr_t intersect_addr;
    size_t intersect_size;
    size_t opcode_offset;
    size_t idx;
    BreakpointSiteSP bp_sp;
    BreakpointSiteList bp_sites_in_range;

    if (m_breakpoint_site_list.FindInRange (bp_addr, bp_addr + size, bp_sites_in_range))
    {
        for (idx = 0; (bp_sp = bp_sites_in_range.GetByIndex(idx)); ++idx)
        {
            if (bp_sp->GetType() == BreakpointSite::eSoftware)
            {
                if (bp_sp->IntersectsRange(bp_addr, size, &intersect_addr, &intersect_size, &opcode_offset))
                {
                    assert(bp_addr <= intersect_addr && intersect_addr < bp_addr + size);
                    assert(bp_addr < intersect_addr + intersect_size && intersect_addr + intersect_size <= bp_addr + size);
                    assert(opcode_offset + intersect_size <= bp_sp->GetByteSize());
                    size_t buf_offset = intersect_addr - bp_addr;
                    ::memcpy(buf + buf_offset, bp_sp->GetSavedOpcodeBytes() + opcode_offset, intersect_size);
                }
            }
        }
    }
    return bytes_removed;
}



size_t
Process::GetSoftwareBreakpointTrapOpcode (BreakpointSite* bp_site)
{
    PlatformSP platform_sp (m_target.GetPlatform());
    if (platform_sp)
        return platform_sp->GetSoftwareBreakpointTrapOpcode (m_target, bp_site);
    return 0;
}

Error
Process::EnableSoftwareBreakpoint (BreakpointSite *bp_site)
{
    Error error;
    assert (bp_site != NULL);
    LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));
    const addr_t bp_addr = bp_site->GetLoadAddress();
    if (log)
        log->Printf ("Process::EnableSoftwareBreakpoint (site_id = %d) addr = 0x%llx", bp_site->GetID(), (uint64_t)bp_addr);
    if (bp_site->IsEnabled())
    {
        if (log)
            log->Printf ("Process::EnableSoftwareBreakpoint (site_id = %d) addr = 0x%llx -- already enabled", bp_site->GetID(), (uint64_t)bp_addr);
        return error;
    }

    if (bp_addr == LLDB_INVALID_ADDRESS)
    {
        error.SetErrorString("BreakpointSite contains an invalid load address.");
        return error;
    }
    // Ask the lldb::Process subclass to fill in the correct software breakpoint
    // trap for the breakpoint site
    const size_t bp_opcode_size = GetSoftwareBreakpointTrapOpcode(bp_site);

    if (bp_opcode_size == 0)
    {
        error.SetErrorStringWithFormat ("Process::GetSoftwareBreakpointTrapOpcode() returned zero, unable to get breakpoint trap for address 0x%llx", bp_addr);
    }
    else
    {
        const uint8_t * const bp_opcode_bytes = bp_site->GetTrapOpcodeBytes();

        if (bp_opcode_bytes == NULL)
        {
            error.SetErrorString ("BreakpointSite doesn't contain a valid breakpoint trap opcode.");
            return error;
        }

        // Save the original opcode by reading it
        if (DoReadMemory(bp_addr, bp_site->GetSavedOpcodeBytes(), bp_opcode_size, error) == bp_opcode_size)
        {
            // Write a software breakpoint in place of the original opcode
            if (DoWriteMemory(bp_addr, bp_opcode_bytes, bp_opcode_size, error) == bp_opcode_size)
            {
                uint8_t verify_bp_opcode_bytes[64];
                if (DoReadMemory(bp_addr, verify_bp_opcode_bytes, bp_opcode_size, error) == bp_opcode_size)
                {
                    if (::memcmp(bp_opcode_bytes, verify_bp_opcode_bytes, bp_opcode_size) == 0)
                    {
                        bp_site->SetEnabled(true);
                        bp_site->SetType (BreakpointSite::eSoftware);
                        if (log)
                            log->Printf ("Process::EnableSoftwareBreakpoint (site_id = %d) addr = 0x%llx -- SUCCESS",
                                         bp_site->GetID(),
                                         (uint64_t)bp_addr);
                    }
                    else
                        error.SetErrorString("failed to verify the breakpoint trap in memory.");
                }
                else
                    error.SetErrorString("Unable to read memory to verify breakpoint trap.");
            }
            else
                error.SetErrorString("Unable to write breakpoint trap to memory.");
        }
        else
            error.SetErrorString("Unable to read memory at breakpoint address.");
    }
    if (log && error.Fail())
        log->Printf ("Process::EnableSoftwareBreakpoint (site_id = %d) addr = 0x%llx -- FAILED: %s",
                     bp_site->GetID(),
                     (uint64_t)bp_addr,
                     error.AsCString());
    return error;
}

Error
Process::DisableSoftwareBreakpoint (BreakpointSite *bp_site)
{
    Error error;
    assert (bp_site != NULL);
    LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));
    addr_t bp_addr = bp_site->GetLoadAddress();
    lldb::user_id_t breakID = bp_site->GetID();
    if (log)
        log->Printf ("Process::DisableBreakpoint (breakID = %llu) addr = 0x%llx", breakID, (uint64_t)bp_addr);

    if (bp_site->IsHardware())
    {
        error.SetErrorString("Breakpoint site is a hardware breakpoint.");
    }
    else if (bp_site->IsEnabled())
    {
        const size_t break_op_size = bp_site->GetByteSize();
        const uint8_t * const break_op = bp_site->GetTrapOpcodeBytes();
        if (break_op_size > 0)
        {
            // Clear a software breakoint instruction
            uint8_t curr_break_op[8];
            assert (break_op_size <= sizeof(curr_break_op));
            bool break_op_found = false;

            // Read the breakpoint opcode
            if (DoReadMemory (bp_addr, curr_break_op, break_op_size, error) == break_op_size)
            {
                bool verify = false;
                // Make sure we have the a breakpoint opcode exists at this address
                if (::memcmp (curr_break_op, break_op, break_op_size) == 0)
                {
                    break_op_found = true;
                    // We found a valid breakpoint opcode at this address, now restore
                    // the saved opcode.
                    if (DoWriteMemory (bp_addr, bp_site->GetSavedOpcodeBytes(), break_op_size, error) == break_op_size)
                    {
                        verify = true;
                    }
                    else
                        error.SetErrorString("Memory write failed when restoring original opcode.");
                }
                else
                {
                    error.SetErrorString("Original breakpoint trap is no longer in memory.");
                    // Set verify to true and so we can check if the original opcode has already been restored
                    verify = true;
                }

                if (verify)
                {
                    uint8_t verify_opcode[8];
                    assert (break_op_size < sizeof(verify_opcode));
                    // Verify that our original opcode made it back to the inferior
                    if (DoReadMemory (bp_addr, verify_opcode, break_op_size, error) == break_op_size)
                    {
                        // compare the memory we just read with the original opcode
                        if (::memcmp (bp_site->GetSavedOpcodeBytes(), verify_opcode, break_op_size) == 0)
                        {
                            // SUCCESS
                            bp_site->SetEnabled(false);
                            if (log)
                                log->Printf ("Process::DisableSoftwareBreakpoint (site_id = %d) addr = 0x%llx -- SUCCESS", bp_site->GetID(), (uint64_t)bp_addr);
                            return error;
                        }
                        else
                        {
                            if (break_op_found)
                                error.SetErrorString("Failed to restore original opcode.");
                        }
                    }
                    else
                        error.SetErrorString("Failed to read memory to verify that breakpoint trap was restored.");
                }
            }
            else
                error.SetErrorString("Unable to read memory that should contain the breakpoint trap.");
        }
    }
    else
    {
        if (log)
            log->Printf ("Process::DisableSoftwareBreakpoint (site_id = %d) addr = 0x%llx -- already disabled", bp_site->GetID(), (uint64_t)bp_addr);
        return error;
    }

    if (log)
        log->Printf ("Process::DisableSoftwareBreakpoint (site_id = %d) addr = 0x%llx -- FAILED: %s",
                     bp_site->GetID(),
                     (uint64_t)bp_addr,
                     error.AsCString());
    return error;

}

// Uncomment to verify memory caching works after making changes to caching code
//#define VERIFY_MEMORY_READS

size_t
Process::ReadMemory (addr_t addr, void *buf, size_t size, Error &error)
{
    if (!GetDisableMemoryCache())
    {        
#if defined (VERIFY_MEMORY_READS)
        // Memory caching is enabled, with debug verification
        
        if (buf && size)
        {
            // Uncomment the line below to make sure memory caching is working.
            // I ran this through the test suite and got no assertions, so I am 
            // pretty confident this is working well. If any changes are made to
            // memory caching, uncomment the line below and test your changes!
            
            // Verify all memory reads by using the cache first, then redundantly
            // reading the same memory from the inferior and comparing to make sure
            // everything is exactly the same.
            std::string verify_buf (size, '\0');
            assert (verify_buf.size() == size);
            const size_t cache_bytes_read = m_memory_cache.Read (this, addr, buf, size, error);
            Error verify_error;
            const size_t verify_bytes_read = ReadMemoryFromInferior (addr, const_cast<char *>(verify_buf.data()), verify_buf.size(), verify_error);
            assert (cache_bytes_read == verify_bytes_read);
            assert (memcmp(buf, verify_buf.data(), verify_buf.size()) == 0);
            assert (verify_error.Success() == error.Success());
            return cache_bytes_read;
        }
        return 0;
#else // !defined(VERIFY_MEMORY_READS)
        // Memory caching is enabled, without debug verification
        
        return m_memory_cache.Read (addr, buf, size, error);
#endif // defined (VERIFY_MEMORY_READS)
    }
    else
    {
        // Memory caching is disabled
        
        return ReadMemoryFromInferior (addr, buf, size, error);
    }
}
    
size_t
Process::ReadCStringFromMemory (addr_t addr, std::string &out_str, Error &error)
{
    char buf[256];
    out_str.clear();
    addr_t curr_addr = addr;
    while (1)
    {
        size_t length = ReadCStringFromMemory (curr_addr, buf, sizeof(buf), error);
        if (length == 0)
            break;
        out_str.append(buf, length);
        // If we got "length - 1" bytes, we didn't get the whole C string, we
        // need to read some more characters
        if (length == sizeof(buf) - 1)
            curr_addr += length;
        else
            break;
    }
    return out_str.size();
}


size_t
Process::ReadCStringFromMemory (addr_t addr, char *dst, size_t dst_max_len, Error &result_error)
{
    size_t total_cstr_len = 0;
    if (dst && dst_max_len)
    {
        result_error.Clear();
        // NULL out everything just to be safe
        memset (dst, 0, dst_max_len);
        Error error;
        addr_t curr_addr = addr;
        const size_t cache_line_size = m_memory_cache.GetMemoryCacheLineSize();
        size_t bytes_left = dst_max_len - 1;
        char *curr_dst = dst;
        
        while (bytes_left > 0)
        {
            addr_t cache_line_bytes_left = cache_line_size - (curr_addr % cache_line_size);
            addr_t bytes_to_read = std::min<addr_t>(bytes_left, cache_line_bytes_left);
            size_t bytes_read = ReadMemory (curr_addr, curr_dst, bytes_to_read, error);
            
            if (bytes_read == 0)
            {
                result_error = error;
                dst[total_cstr_len] = '\0';
                break;
            }
            const size_t len = strlen(curr_dst);

            total_cstr_len += len;

            if (len < bytes_to_read)
                break;

            curr_dst += bytes_read;
            curr_addr += bytes_read;
            bytes_left -= bytes_read;
        }
    }
    else
    {
        if (dst == NULL)
            result_error.SetErrorString("invalid arguments");
        else
            result_error.Clear();
    }
    return total_cstr_len;
}

size_t
Process::ReadMemoryFromInferior (addr_t addr, void *buf, size_t size, Error &error)
{
    if (buf == NULL || size == 0)
        return 0;

    size_t bytes_read = 0;
    uint8_t *bytes = (uint8_t *)buf;
    
    while (bytes_read < size)
    {
        const size_t curr_size = size - bytes_read;
        const size_t curr_bytes_read = DoReadMemory (addr + bytes_read, 
                                                     bytes + bytes_read, 
                                                     curr_size,
                                                     error);
        bytes_read += curr_bytes_read;
        if (curr_bytes_read == curr_size || curr_bytes_read == 0)
            break;
    }

    // Replace any software breakpoint opcodes that fall into this range back
    // into "buf" before we return
    if (bytes_read > 0)
        RemoveBreakpointOpcodesFromBuffer (addr, bytes_read, (uint8_t *)buf);
    return bytes_read;
}

uint64_t
Process::ReadUnsignedIntegerFromMemory (lldb::addr_t vm_addr, size_t integer_byte_size, uint64_t fail_value, Error &error)
{
    Scalar scalar;
    if (ReadScalarIntegerFromMemory(vm_addr, integer_byte_size, false, scalar, error))
        return scalar.ULongLong(fail_value);
    return fail_value;
}

addr_t
Process::ReadPointerFromMemory (lldb::addr_t vm_addr, Error &error)
{
    Scalar scalar;
    if (ReadScalarIntegerFromMemory(vm_addr, GetAddressByteSize(), false, scalar, error))
        return scalar.ULongLong(LLDB_INVALID_ADDRESS);
    return LLDB_INVALID_ADDRESS;
}


bool
Process::WritePointerToMemory (lldb::addr_t vm_addr, 
                               lldb::addr_t ptr_value, 
                               Error &error)
{
    Scalar scalar;
    const uint32_t addr_byte_size = GetAddressByteSize();
    if (addr_byte_size <= 4)
        scalar = (uint32_t)ptr_value;
    else
        scalar = ptr_value;
    return WriteScalarToMemory(vm_addr, scalar, addr_byte_size, error) == addr_byte_size;
}

size_t
Process::WriteMemoryPrivate (addr_t addr, const void *buf, size_t size, Error &error)
{
    size_t bytes_written = 0;
    const uint8_t *bytes = (const uint8_t *)buf;
    
    while (bytes_written < size)
    {
        const size_t curr_size = size - bytes_written;
        const size_t curr_bytes_written = DoWriteMemory (addr + bytes_written, 
                                                         bytes + bytes_written, 
                                                         curr_size,
                                                         error);
        bytes_written += curr_bytes_written;
        if (curr_bytes_written == curr_size || curr_bytes_written == 0)
            break;
    }
    return bytes_written;
}

size_t
Process::WriteMemory (addr_t addr, const void *buf, size_t size, Error &error)
{
#if defined (ENABLE_MEMORY_CACHING)
    m_memory_cache.Flush (addr, size);
#endif

    if (buf == NULL || size == 0)
        return 0;

    m_mod_id.BumpMemoryID();

    // We need to write any data that would go where any current software traps
    // (enabled software breakpoints) any software traps (breakpoints) that we
    // may have placed in our tasks memory.

    BreakpointSiteList::collection::const_iterator iter = m_breakpoint_site_list.GetMap()->lower_bound (addr);
    BreakpointSiteList::collection::const_iterator end =  m_breakpoint_site_list.GetMap()->end();

    if (iter == end || iter->second->GetLoadAddress() > addr + size)
        return WriteMemoryPrivate (addr, buf, size, error);

    BreakpointSiteList::collection::const_iterator pos;
    size_t bytes_written = 0;
    addr_t intersect_addr = 0;
    size_t intersect_size = 0;
    size_t opcode_offset = 0;
    const uint8_t *ubuf = (const uint8_t *)buf;

    for (pos = iter; pos != end; ++pos)
    {
        BreakpointSiteSP bp;
        bp = pos->second;

        assert(bp->IntersectsRange(addr, size, &intersect_addr, &intersect_size, &opcode_offset));
        assert(addr <= intersect_addr && intersect_addr < addr + size);
        assert(addr < intersect_addr + intersect_size && intersect_addr + intersect_size <= addr + size);
        assert(opcode_offset + intersect_size <= bp->GetByteSize());

        // Check for bytes before this breakpoint
        const addr_t curr_addr = addr + bytes_written;
        if (intersect_addr > curr_addr)
        {
            // There are some bytes before this breakpoint that we need to
            // just write to memory
            size_t curr_size = intersect_addr - curr_addr;
            size_t curr_bytes_written = WriteMemoryPrivate (curr_addr, 
                                                            ubuf + bytes_written, 
                                                            curr_size, 
                                                            error);
            bytes_written += curr_bytes_written;
            if (curr_bytes_written != curr_size)
            {
                // We weren't able to write all of the requested bytes, we
                // are done looping and will return the number of bytes that
                // we have written so far.
                break;
            }
        }

        // Now write any bytes that would cover up any software breakpoints
        // directly into the breakpoint opcode buffer
        ::memcpy(bp->GetSavedOpcodeBytes() + opcode_offset, ubuf + bytes_written, intersect_size);
        bytes_written += intersect_size;
    }

    // Write any remaining bytes after the last breakpoint if we have any left
    if (bytes_written < size)
        bytes_written += WriteMemoryPrivate (addr + bytes_written, 
                                             ubuf + bytes_written, 
                                             size - bytes_written, 
                                             error);
                                             
    return bytes_written;
}

size_t
Process::WriteScalarToMemory (addr_t addr, const Scalar &scalar, uint32_t byte_size, Error &error)
{
    if (byte_size == UINT32_MAX)
        byte_size = scalar.GetByteSize();
    if (byte_size > 0)
    {
        uint8_t buf[32];
        const size_t mem_size = scalar.GetAsMemoryData (buf, byte_size, GetByteOrder(), error);
        if (mem_size > 0)
            return WriteMemory(addr, buf, mem_size, error);
        else
            error.SetErrorString ("failed to get scalar as memory data");
    }
    else
    {
        error.SetErrorString ("invalid scalar value");
    }
    return 0;
}

size_t
Process::ReadScalarIntegerFromMemory (addr_t addr, 
                                      uint32_t byte_size, 
                                      bool is_signed, 
                                      Scalar &scalar, 
                                      Error &error)
{
    uint64_t uval;

    if (byte_size <= sizeof(uval))
    {
        size_t bytes_read = ReadMemory (addr, &uval, byte_size, error);
        if (bytes_read == byte_size)
        {
            DataExtractor data (&uval, sizeof(uval), GetByteOrder(), GetAddressByteSize());
            uint32_t offset = 0;
            if (byte_size <= 4)
                scalar = data.GetMaxU32 (&offset, byte_size);
            else
                scalar = data.GetMaxU64 (&offset, byte_size);

            if (is_signed)
                scalar.SignExtend(byte_size * 8);
            return bytes_read;
        }
    }
    else
    {
        error.SetErrorStringWithFormat ("byte size of %u is too large for integer scalar type", byte_size);
    }
    return 0;
}

#define USE_ALLOCATE_MEMORY_CACHE 1
addr_t
Process::AllocateMemory(size_t size, uint32_t permissions, Error &error)
{
    if (GetPrivateState() != eStateStopped)
        return LLDB_INVALID_ADDRESS;
        
#if defined (USE_ALLOCATE_MEMORY_CACHE)
    return m_allocated_memory_cache.AllocateMemory(size, permissions, error);
#else
    addr_t allocated_addr = DoAllocateMemory (size, permissions, error);
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf("Process::AllocateMemory(size=%4zu, permissions=%s) => 0x%16.16llx (m_stop_id = %u m_memory_id = %u)", 
                    size, 
                    GetPermissionsAsCString (permissions),
                    (uint64_t)allocated_addr,
                    m_mod_id.GetStopID(),
                    m_mod_id.GetMemoryID());
    return allocated_addr;
#endif
}

bool
Process::CanJIT ()
{
    if (m_can_jit == eCanJITDontKnow)
    {
        Error err;
        
        uint64_t allocated_memory = AllocateMemory(8, 
                                                   ePermissionsReadable | ePermissionsWritable | ePermissionsExecutable, 
                                                   err);
        
        if (err.Success())
            m_can_jit = eCanJITYes;
        else
            m_can_jit = eCanJITNo;
        
        DeallocateMemory (allocated_memory);
    }
    
    return m_can_jit == eCanJITYes;
}

void
Process::SetCanJIT (bool can_jit)
{
    m_can_jit = (can_jit ? eCanJITYes : eCanJITNo);
}

Error
Process::DeallocateMemory (addr_t ptr)
{
    Error error;
#if defined (USE_ALLOCATE_MEMORY_CACHE)
    if (!m_allocated_memory_cache.DeallocateMemory(ptr))
    {
        error.SetErrorStringWithFormat ("deallocation of memory at 0x%llx failed.", (uint64_t)ptr);
    }
#else
    error = DoDeallocateMemory (ptr);
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf("Process::DeallocateMemory(addr=0x%16.16llx) => err = %s (m_stop_id = %u, m_memory_id = %u)", 
                    ptr, 
                    error.AsCString("SUCCESS"),
                    m_mod_id.GetStopID(),
                    m_mod_id.GetMemoryID());
#endif
    return error;
}

ModuleSP
Process::ReadModuleFromMemory (const FileSpec& file_spec, 
                               lldb::addr_t header_addr, 
                               bool add_image_to_target,
                               bool load_sections_in_target)
{
    ModuleSP module_sp (new Module (file_spec, ArchSpec()));
    if (module_sp)
    {
        Error error;
        ObjectFile *objfile = module_sp->GetMemoryObjectFile (shared_from_this(), header_addr, error);
        if (objfile)
        {
            if (add_image_to_target)
            {
                m_target.GetImages().Append(module_sp);
                if (load_sections_in_target)
                {
                    bool changed = false;
                    module_sp->SetLoadAddress (m_target, 0, changed);
                }
            }
            return module_sp;
        }
    }
    return ModuleSP();
}

Error
Process::EnableWatchpoint (Watchpoint *watchpoint)
{
    Error error;
    error.SetErrorString("watchpoints are not supported");
    return error;
}

Error
Process::DisableWatchpoint (Watchpoint *watchpoint)
{
    Error error;
    error.SetErrorString("watchpoints are not supported");
    return error;
}

StateType
Process::WaitForProcessStopPrivate (const TimeValue *timeout, EventSP &event_sp)
{
    StateType state;
    // Now wait for the process to launch and return control to us, and then
    // call DidLaunch:
    while (1)
    {
        event_sp.reset();
        state = WaitForStateChangedEventsPrivate (timeout, event_sp);

        if (StateIsStoppedState(state, false))
            break;

        // If state is invalid, then we timed out
        if (state == eStateInvalid)
            break;

        if (event_sp)
            HandlePrivateEvent (event_sp);
    }
    return state;
}

Error
Process::Launch (const ProcessLaunchInfo &launch_info)
{
    Error error;
    m_abi_sp.reset();
    m_dyld_ap.reset();
    m_os_ap.reset();
    m_process_input_reader.reset();

    Module *exe_module = m_target.GetExecutableModulePointer();
    if (exe_module)
    {
        char local_exec_file_path[PATH_MAX];
        char platform_exec_file_path[PATH_MAX];
        exe_module->GetFileSpec().GetPath(local_exec_file_path, sizeof(local_exec_file_path));
        exe_module->GetPlatformFileSpec().GetPath(platform_exec_file_path, sizeof(platform_exec_file_path));
        if (exe_module->GetFileSpec().Exists())
        {
            if (PrivateStateThreadIsValid ())
                PausePrivateStateThread ();
    
            error = WillLaunch (exe_module);
            if (error.Success())
            {
                SetPublicState (eStateLaunching);
                m_should_detach = false;

                if (m_run_lock.WriteTryLock())
                {
                    // Now launch using these arguments.
                    error = DoLaunch (exe_module, launch_info);
                }
                else
                {
                    // This shouldn't happen
                    error.SetErrorString("failed to acquire process run lock");
                }

                if (error.Fail())
                {
                    if (GetID() != LLDB_INVALID_PROCESS_ID)
                    {
                        SetID (LLDB_INVALID_PROCESS_ID);
                        const char *error_string = error.AsCString();
                        if (error_string == NULL)
                            error_string = "launch failed";
                        SetExitStatus (-1, error_string);
                    }
                }
                else
                {
                    EventSP event_sp;
                    TimeValue timeout_time;
                    timeout_time = TimeValue::Now();
                    timeout_time.OffsetWithSeconds(10);
                    StateType state = WaitForProcessStopPrivate(&timeout_time, event_sp);

                    if (state == eStateInvalid || event_sp.get() == NULL)
                    {
                        // We were able to launch the process, but we failed to
                        // catch the initial stop.
                        SetExitStatus (0, "failed to catch stop after launch");
                        Destroy();
                    }
                    else if (state == eStateStopped || state == eStateCrashed)
                    {

                        DidLaunch ();

                        DynamicLoader *dyld = GetDynamicLoader ();
                        if (dyld)
                            dyld->DidLaunch();

                        m_os_ap.reset (OperatingSystem::FindPlugin (this, NULL));
                        // This delays passing the stopped event to listeners till DidLaunch gets
                        // a chance to complete...
                        HandlePrivateEvent (event_sp);

                        if (PrivateStateThreadIsValid ())
                            ResumePrivateStateThread ();
                        else
                            StartPrivateStateThread ();
                    }
                    else if (state == eStateExited)
                    {
                        // We exited while trying to launch somehow.  Don't call DidLaunch as that's
                        // not likely to work, and return an invalid pid.
                        HandlePrivateEvent (event_sp);
                    }
                }
            }
        }
        else
        {
            error.SetErrorStringWithFormat("file doesn't exist: '%s'", local_exec_file_path);
        }
    }
    return error;
}


Error
Process::LoadCore ()
{
    Error error = DoLoadCore();
    if (error.Success())
    {
        if (PrivateStateThreadIsValid ())
            ResumePrivateStateThread ();
        else
            StartPrivateStateThread ();

        DynamicLoader *dyld = GetDynamicLoader ();
        if (dyld)
            dyld->DidAttach();
        
        m_os_ap.reset (OperatingSystem::FindPlugin (this, NULL));
        // We successfully loaded a core file, now pretend we stopped so we can
        // show all of the threads in the core file and explore the crashed
        // state.
        SetPrivateState (eStateStopped);
        
    }
    return error;
}

DynamicLoader *
Process::GetDynamicLoader ()
{
    if (m_dyld_ap.get() == NULL)
        m_dyld_ap.reset (DynamicLoader::FindPlugin(this, NULL));
    return m_dyld_ap.get();
}


Process::NextEventAction::EventActionResult
Process::AttachCompletionHandler::PerformAction (lldb::EventSP &event_sp)
{
    StateType state = ProcessEventData::GetStateFromEvent (event_sp.get());
    switch (state) 
    {
        case eStateRunning:
        case eStateConnected:
            return eEventActionRetry;
        
        case eStateStopped:
        case eStateCrashed:
            {
                // During attach, prior to sending the eStateStopped event, 
                // lldb_private::Process subclasses must set the new process ID.
                assert (m_process->GetID() != LLDB_INVALID_PROCESS_ID);
                if (m_exec_count > 0)
                {
                    --m_exec_count;
                    m_process->PrivateResume ();
                    Process::ProcessEventData::SetRestartedInEvent (event_sp.get(), true);
                    return eEventActionRetry;
                }
                else
                {
                    m_process->CompleteAttach ();
                    return eEventActionSuccess;
                }
            }
            break;

        default:
        case eStateExited:   
        case eStateInvalid:
            break;
    }

    m_exit_string.assign ("No valid Process");
    return eEventActionExit;
}

Process::NextEventAction::EventActionResult
Process::AttachCompletionHandler::HandleBeingInterrupted()
{
    return eEventActionSuccess;
}

const char *
Process::AttachCompletionHandler::GetExitString ()
{
    return m_exit_string.c_str();
}

Error
Process::Attach (ProcessAttachInfo &attach_info)
{
    m_abi_sp.reset();
    m_process_input_reader.reset();
    m_dyld_ap.reset();
    m_os_ap.reset();
    
    lldb::pid_t attach_pid = attach_info.GetProcessID();
    Error error;
    if (attach_pid == LLDB_INVALID_PROCESS_ID)
    {
        char process_name[PATH_MAX];
        
        if (attach_info.GetExecutableFile().GetPath (process_name, sizeof(process_name)))
        {
            const bool wait_for_launch = attach_info.GetWaitForLaunch();
            
            if (wait_for_launch)
            {
                error = WillAttachToProcessWithName(process_name, wait_for_launch);
                if (error.Success())
                {
                    if (m_run_lock.WriteTryLock())
                    {
                        m_should_detach = true;
                        SetPublicState (eStateAttaching);
                        // Now attach using these arguments.
                        error = DoAttachToProcessWithName (process_name, wait_for_launch, attach_info);
                    }
                    else
                    {
                        // This shouldn't happen
                        error.SetErrorString("failed to acquire process run lock");
                    }

                    if (error.Fail())
                    {
                        if (GetID() != LLDB_INVALID_PROCESS_ID)
                        {
                            SetID (LLDB_INVALID_PROCESS_ID);
                            if (error.AsCString() == NULL)
                                error.SetErrorString("attach failed");
                            
                            SetExitStatus(-1, error.AsCString());
                        }
                    }
                    else
                    {
                        SetNextEventAction(new Process::AttachCompletionHandler(this, attach_info.GetResumeCount()));
                        StartPrivateStateThread();
                    }
                    return error;
                }
            }
            else
            {
                ProcessInstanceInfoList process_infos;
                PlatformSP platform_sp (m_target.GetPlatform ());
                
                if (platform_sp)
                {
                    ProcessInstanceInfoMatch match_info;
                    match_info.GetProcessInfo() = attach_info;
                    match_info.SetNameMatchType (eNameMatchEquals);
                    platform_sp->FindProcesses (match_info, process_infos);
                    const uint32_t num_matches = process_infos.GetSize();
                    if (num_matches == 1)
                    {
                        attach_pid = process_infos.GetProcessIDAtIndex(0);
                        // Fall through and attach using the above process ID
                    }
                    else
                    {
                        match_info.GetProcessInfo().GetExecutableFile().GetPath (process_name, sizeof(process_name));    
                        if (num_matches > 1)
                            error.SetErrorStringWithFormat ("more than one process named %s", process_name);
                        else
                            error.SetErrorStringWithFormat ("could not find a process named %s", process_name);
                    }
                }
                else
                {        
                    error.SetErrorString ("invalid platform, can't find processes by name");
                    return error;
                }
            }
        }
        else
        {
            error.SetErrorString ("invalid process name");
        }
    }
    
    if (attach_pid != LLDB_INVALID_PROCESS_ID)
    {
        error = WillAttachToProcessWithID(attach_pid);
        if (error.Success())
        {

            if (m_run_lock.WriteTryLock())
            {
                // Now attach using these arguments.
                m_should_detach = true;
                SetPublicState (eStateAttaching);
                error = DoAttachToProcessWithID (attach_pid, attach_info);
            }
            else
            {
                // This shouldn't happen
                error.SetErrorString("failed to acquire process run lock");
            }

            if (error.Success())
            {
                
                SetNextEventAction(new Process::AttachCompletionHandler(this, attach_info.GetResumeCount()));
                StartPrivateStateThread();
            }
            else
            {
                if (GetID() != LLDB_INVALID_PROCESS_ID)
                {
                    SetID (LLDB_INVALID_PROCESS_ID);
                    const char *error_string = error.AsCString();
                    if (error_string == NULL)
                        error_string = "attach failed";

                    SetExitStatus(-1, error_string);
                }
            }
        }
    }
    return error;
}

void
Process::CompleteAttach ()
{
    // Let the process subclass figure out at much as it can about the process
    // before we go looking for a dynamic loader plug-in.
    DidAttach();

    // We just attached.  If we have a platform, ask it for the process architecture, and if it isn't
    // the same as the one we've already set, switch architectures.
    PlatformSP platform_sp (m_target.GetPlatform ());
    assert (platform_sp.get());
    if (platform_sp)
    {
        const ArchSpec &target_arch = m_target.GetArchitecture();
        if (target_arch.IsValid() && !platform_sp->IsCompatibleArchitecture (target_arch))
        {
            ArchSpec platform_arch;
            platform_sp = platform_sp->GetPlatformForArchitecture (target_arch, &platform_arch);
            if (platform_sp)
            {
                m_target.SetPlatform (platform_sp);
                m_target.SetArchitecture(platform_arch);
            }
        }
        else
        {
            ProcessInstanceInfo process_info;
            platform_sp->GetProcessInfo (GetID(), process_info);
            const ArchSpec &process_arch = process_info.GetArchitecture();
            if (process_arch.IsValid() && m_target.GetArchitecture() != process_arch)
                m_target.SetArchitecture (process_arch);
        }
    }

    // We have completed the attach, now it is time to find the dynamic loader
    // plug-in
    DynamicLoader *dyld = GetDynamicLoader ();
    if (dyld)
        dyld->DidAttach();

    m_os_ap.reset (OperatingSystem::FindPlugin (this, NULL));
    // Figure out which one is the executable, and set that in our target:
    const ModuleList &target_modules = m_target.GetImages();
    Mutex::Locker modules_locker(target_modules.GetMutex());
    size_t num_modules = target_modules.GetSize();
    ModuleSP new_executable_module_sp;
    
    for (int i = 0; i < num_modules; i++)
    {
        ModuleSP module_sp (target_modules.GetModuleAtIndexUnlocked (i));
        if (module_sp && module_sp->IsExecutable())
        {
            if (m_target.GetExecutableModulePointer() != module_sp.get())
                new_executable_module_sp = module_sp;
            break;
        }
    }
    if (new_executable_module_sp)
        m_target.SetExecutableModule (new_executable_module_sp, false);
}

Error
Process::ConnectRemote (Stream *strm, const char *remote_url)
{
    m_abi_sp.reset();
    m_process_input_reader.reset();
    
    // Find the process and its architecture.  Make sure it matches the architecture
    // of the current Target, and if not adjust it.
    
    Error error (DoConnectRemote (strm, remote_url));
    if (error.Success())
    {
        if (GetID() != LLDB_INVALID_PROCESS_ID)
        {
            EventSP event_sp;
            StateType state = WaitForProcessStopPrivate(NULL, event_sp);
        
            if (state == eStateStopped || state == eStateCrashed)
            {
                // If we attached and actually have a process on the other end, then 
                // this ended up being the equivalent of an attach.
                CompleteAttach ();
                
                // This delays passing the stopped event to listeners till 
                // CompleteAttach gets a chance to complete...
                HandlePrivateEvent (event_sp);
                
            }
        }

        if (PrivateStateThreadIsValid ())
            ResumePrivateStateThread ();
        else
            StartPrivateStateThread ();
    }
    return error;
}


Error
Process::PrivateResume ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS|LIBLLDB_LOG_STEP));
    if (log)
        log->Printf("Process::Resume() m_stop_id = %u, public state: %s private state: %s", 
                    m_mod_id.GetStopID(),
                    StateAsCString(m_public_state.GetValue()),
                    StateAsCString(m_private_state.GetValue()));

    Error error (WillResume());
    // Tell the process it is about to resume before the thread list
    if (error.Success())
    {
        // Now let the thread list know we are about to resume so it
        // can let all of our threads know that they are about to be
        // resumed. Threads will each be called with
        // Thread::WillResume(StateType) where StateType contains the state
        // that they are supposed to have when the process is resumed
        // (suspended/running/stepping). Threads should also check
        // their resume signal in lldb::Thread::GetResumeSignal()
        // to see if they are suppoed to start back up with a signal.
        if (m_thread_list.WillResume())
        {
            // Last thing, do the PreResumeActions.
            if (!RunPreResumeActions())
            {
                error.SetErrorStringWithFormat ("Process::Resume PreResumeActions failed, not resuming.");
            }
            else
            {
                m_mod_id.BumpResumeID();
                error = DoResume();
                if (error.Success())
                {
                    DidResume();
                    m_thread_list.DidResume();
                    if (log)
                        log->Printf ("Process thinks the process has resumed.");
                }
            }
        }
        else
        {
            // Somebody wanted to run without running.  So generate a continue & a stopped event,
            // and let the world handle them.
            if (log)
                log->Printf ("Process::PrivateResume() asked to simulate a start & stop.");
            
            SetPrivateState(eStateRunning);
            SetPrivateState(eStateStopped);
        }
    }
    else if (log)
        log->Printf ("Process::WillResume() got an error \"%s\".", error.AsCString("<unknown error>"));
    return error;
}

Error
Process::Halt ()
{
    // First make sure we aren't in the middle of handling an event, or we might restart.  This is pretty weak, since
    // we could just straightaway get another event.  It just narrows the window...
    m_currently_handling_event.WaitForValueEqualTo(false);

    
    // Pause our private state thread so we can ensure no one else eats
    // the stop event out from under us.
    Listener halt_listener ("lldb.process.halt_listener");
    HijackPrivateProcessEvents(&halt_listener);

    EventSP event_sp;
    Error error (WillHalt());
    
    if (error.Success())
    {
        
        bool caused_stop = false;
        
        // Ask the process subclass to actually halt our process
        error = DoHalt(caused_stop);
        if (error.Success())
        {
            if (m_public_state.GetValue() == eStateAttaching)
            {
                SetExitStatus(SIGKILL, "Cancelled async attach.");
                Destroy ();
            }
            else
            {
                // If "caused_stop" is true, then DoHalt stopped the process. If
                // "caused_stop" is false, the process was already stopped.
                // If the DoHalt caused the process to stop, then we want to catch
                // this event and set the interrupted bool to true before we pass
                // this along so clients know that the process was interrupted by
                // a halt command.
                if (caused_stop)
                {
                    // Wait for 1 second for the process to stop.
                    TimeValue timeout_time;
                    timeout_time = TimeValue::Now();
                    timeout_time.OffsetWithSeconds(1);
                    bool got_event = halt_listener.WaitForEvent (&timeout_time, event_sp);
                    StateType state = ProcessEventData::GetStateFromEvent(event_sp.get());
                    
                    if (!got_event || state == eStateInvalid)
                    {
                        // We timeout out and didn't get a stop event...
                        error.SetErrorStringWithFormat ("Halt timed out. State = %s", StateAsCString(GetState()));
                    }
                    else
                    {
                        if (StateIsStoppedState (state, false))
                        {
                            // We caused the process to interrupt itself, so mark this
                            // as such in the stop event so clients can tell an interrupted
                            // process from a natural stop
                            ProcessEventData::SetInterruptedInEvent (event_sp.get(), true);
                        }
                        else
                        {
                            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
                            if (log)
                                log->Printf("Process::Halt() failed to stop, state is: %s", StateAsCString(state));
                            error.SetErrorString ("Did not get stopped event after halt.");
                        }
                    }
                }
                DidHalt();
            }
        }
    }
    // Resume our private state thread before we post the event (if any)
    RestorePrivateProcessEvents();

    // Post any event we might have consumed. If all goes well, we will have
    // stopped the process, intercepted the event and set the interrupted
    // bool in the event.  Post it to the private event queue and that will end up
    // correctly setting the state.
    if (event_sp)
        m_private_state_broadcaster.BroadcastEvent(event_sp);

    return error;
}

Error
Process::Detach ()
{
    Error error (WillDetach());

    if (error.Success())
    {
        DisableAllBreakpointSites();
        error = DoDetach(); 
        if (error.Success())
        {
            DidDetach();
            StopPrivateStateThread();
        }
    }
    return error;
}

Error
Process::Destroy ()
{
    Error error (WillDestroy());
    if (error.Success())
    {
        EventSP exit_event_sp;
        if (m_public_state.GetValue() == eStateRunning)
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
            if (log)
                log->Printf("Process::Destroy() About to halt.");
            error = Halt();
            if (error.Success())
            {
                // Consume the halt event.
                TimeValue timeout (TimeValue::Now());
                timeout.OffsetWithSeconds(1);
                StateType state = WaitForProcessToStop (&timeout, &exit_event_sp);
                if (state != eStateExited)
                    exit_event_sp.reset(); // It is ok to consume any non-exit stop events
        
                if (state != eStateStopped)
                {
                    if (log)
                        log->Printf("Process::Destroy() Halt failed to stop, state is: %s", StateAsCString(state));
                    // If we really couldn't stop the process then we should just error out here, but if the
                    // lower levels just bobbled sending the event and we really are stopped, then continue on.
                    StateType private_state = m_private_state.GetValue();
                    if (private_state != eStateStopped && private_state != eStateExited)
                    {
                        // If we exited when we were waiting for a process to stop, then
                        // forward the event here so we don't lose the event
                        return error;
                    }
                }
            }
            else
            {
                if (log)
                    log->Printf("Process::Destroy() Halt got error: %s", error.AsCString());
                return error;
            }
        }

        if (m_public_state.GetValue() != eStateRunning)
        {
            // Ditch all thread plans, and remove all our breakpoints: in case we have to restart the target to
            // kill it, we don't want it hitting a breakpoint...
            // Only do this if we've stopped, however, since if we didn't manage to halt it above, then
            // we're not going to have much luck doing this now.
            m_thread_list.DiscardThreadPlans();
            DisableAllBreakpointSites();
        }
        
        error = DoDestroy();
        if (error.Success())
        {
            DidDestroy();
            StopPrivateStateThread();
        }
        m_stdio_communication.StopReadThread();
        m_stdio_communication.Disconnect();
        if (m_process_input_reader && m_process_input_reader->IsActive())
            m_target.GetDebugger().PopInputReader (m_process_input_reader);
        if (m_process_input_reader)
            m_process_input_reader.reset();
        
        // If we exited when we were waiting for a process to stop, then
        // forward the event here so we don't lose the event
        if (exit_event_sp)
        {
            // Directly broadcast our exited event because we shut down our
            // private state thread above
            BroadcastEvent(exit_event_sp);
        }

        // If we have been interrupted (to kill us) in the middle of running, we may not end up propagating
        // the last events through the event system, in which case we might strand the write lock.  Unlock
        // it here so when we do to tear down the process we don't get an error destroying the lock.
        m_run_lock.WriteUnlock();
    }
    return error;
}

Error
Process::Signal (int signal)
{
    Error error (WillSignal());
    if (error.Success())
    {
        error = DoSignal(signal);
        if (error.Success())
            DidSignal();
    }
    return error;
}

lldb::ByteOrder
Process::GetByteOrder () const
{
    return m_target.GetArchitecture().GetByteOrder();
}

uint32_t
Process::GetAddressByteSize () const
{
    return m_target.GetArchitecture().GetAddressByteSize();
}


bool
Process::ShouldBroadcastEvent (Event *event_ptr)
{
    const StateType state = Process::ProcessEventData::GetStateFromEvent (event_ptr);
    bool return_value = true;
    LogSP log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EVENTS));

    switch (state)
    {
        case eStateConnected:
        case eStateAttaching:
        case eStateLaunching:
        case eStateDetached:
        case eStateExited:
        case eStateUnloaded:
            // These events indicate changes in the state of the debugging session, always report them.
            return_value = true;
            break;
        case eStateInvalid:
            // We stopped for no apparent reason, don't report it.
            return_value = false;
            break;
        case eStateRunning:
        case eStateStepping:
            // If we've started the target running, we handle the cases where we
            // are already running and where there is a transition from stopped to
            // running differently.
            // running -> running: Automatically suppress extra running events
            // stopped -> running: Report except when there is one or more no votes
            //     and no yes votes.
            SynchronouslyNotifyStateChanged (state);
            switch (m_public_state.GetValue())
            {
                case eStateRunning:
                case eStateStepping:
                    // We always suppress multiple runnings with no PUBLIC stop in between.
                    return_value = false;
                    break;
                default:
                    // TODO: make this work correctly. For now always report
                    // run if we aren't running so we don't miss any runnning
                    // events. If I run the lldb/test/thread/a.out file and
                    // break at main.cpp:58, run and hit the breakpoints on
                    // multiple threads, then somehow during the stepping over
                    // of all breakpoints no run gets reported.

                    // This is a transition from stop to run.
                    switch (m_thread_list.ShouldReportRun (event_ptr))
                    {
                        default:
                        case eVoteYes:
                        case eVoteNoOpinion:
                            return_value = true;
                            break;
                        case eVoteNo:
                            return_value = false;
                            break;
                    }
                    break;
            }
            break;
        case eStateStopped:
        case eStateCrashed:
        case eStateSuspended:
        {
            // We've stopped.  First see if we're going to restart the target.
            // If we are going to stop, then we always broadcast the event.
            // If we aren't going to stop, let the thread plans decide if we're going to report this event.
            // If no thread has an opinion, we don't report it.
            
            RefreshStateAfterStop ();
            if (ProcessEventData::GetInterruptedFromEvent (event_ptr))
            {
                if (log)
                    log->Printf ("Process::ShouldBroadcastEvent (%p) stopped due to an interrupt, state: %s", event_ptr, StateAsCString(state));
                return true;
            }
            else
            {

                if (m_thread_list.ShouldStop (event_ptr) == false)
                {
                    // ShouldStop may have restarted the target already.  If so, don't
                    // resume it twice.
                    bool was_restarted = ProcessEventData::GetRestartedFromEvent (event_ptr);
                    switch (m_thread_list.ShouldReportStop (event_ptr))
                    {
                        case eVoteYes:
                            Process::ProcessEventData::SetRestartedInEvent (event_ptr, true);
                            // Intentional fall-through here.
                        case eVoteNoOpinion:
                        case eVoteNo:
                            return_value = false;
                            break;
                    }

                    if (log)
                        log->Printf ("Process::ShouldBroadcastEvent (%p) Restarting process from state: %s", event_ptr, StateAsCString(state));
                    if (!was_restarted)
                        PrivateResume ();
                }
                else
                {
                    return_value = true;
                    SynchronouslyNotifyStateChanged (state);
                }
            }
        }
    }

    if (log)
        log->Printf ("Process::ShouldBroadcastEvent (%p) => %s - %s", event_ptr, StateAsCString(state), return_value ? "YES" : "NO");
    return return_value;
}


bool
Process::StartPrivateStateThread (bool force)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EVENTS));

    bool already_running = PrivateStateThreadIsValid ();
    if (log)
        log->Printf ("Process::%s()%s ", __FUNCTION__, already_running ? " already running" : " starting private state thread");

    if (!force && already_running)
        return true;

    // Create a thread that watches our internal state and controls which
    // events make it to clients (into the DCProcess event queue).
    char thread_name[1024];
    if (already_running)
        snprintf(thread_name, sizeof(thread_name), "<lldb.process.internal-state-override(pid=%llu)>", GetID());
    else
        snprintf(thread_name, sizeof(thread_name), "<lldb.process.internal-state(pid=%llu)>", GetID());
        
    // Create the private state thread, and start it running.
    m_private_state_thread = Host::ThreadCreate (thread_name, Process::PrivateStateThread, this, NULL);
    bool success = IS_VALID_LLDB_HOST_THREAD(m_private_state_thread);
    if (success)
    {
        ResumePrivateStateThread();
        return true;
    }
    else
        return false;
}

void
Process::PausePrivateStateThread ()
{
    ControlPrivateStateThread (eBroadcastInternalStateControlPause);
}

void
Process::ResumePrivateStateThread ()
{
    ControlPrivateStateThread (eBroadcastInternalStateControlResume);
}

void
Process::StopPrivateStateThread ()
{
    if (PrivateStateThreadIsValid ())
        ControlPrivateStateThread (eBroadcastInternalStateControlStop);
    else
    {
        LogSP log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
        if (log)
            printf ("Went to stop the private state thread, but it was already invalid.");
    }
}

void
Process::ControlPrivateStateThread (uint32_t signal)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));

    assert (signal == eBroadcastInternalStateControlStop ||
            signal == eBroadcastInternalStateControlPause ||
            signal == eBroadcastInternalStateControlResume);

    if (log)
        log->Printf ("Process::%s (signal = %d)", __FUNCTION__, signal);

    // Signal the private state thread. First we should copy this is case the
    // thread starts exiting since the private state thread will NULL this out
    // when it exits
    const lldb::thread_t private_state_thread = m_private_state_thread;
    if (IS_VALID_LLDB_HOST_THREAD(private_state_thread))
    {
        TimeValue timeout_time;
        bool timed_out;

        m_private_state_control_broadcaster.BroadcastEvent (signal, NULL);

        timeout_time = TimeValue::Now();
        timeout_time.OffsetWithSeconds(2);
        if (log)
            log->Printf ("Sending control event of type: %d.", signal);
        m_private_state_control_wait.WaitForValueEqualTo (true, &timeout_time, &timed_out);
        m_private_state_control_wait.SetValue (false, eBroadcastNever);

        if (signal == eBroadcastInternalStateControlStop)
        {
            if (timed_out)
            {
                Error error;
                Host::ThreadCancel (private_state_thread, &error);
                if (log)
                    log->Printf ("Timed out responding to the control event, cancel got error: \"%s\".", error.AsCString());
            }
            else
            {
                if (log)
                    log->Printf ("The control event killed the private state thread without having to cancel.");
            }

            thread_result_t result = NULL;
            Host::ThreadJoin (private_state_thread, &result, NULL);
            m_private_state_thread = LLDB_INVALID_HOST_THREAD;
        }
    }
    else
    {
        if (log)
            log->Printf ("Private state thread already dead, no need to signal it to stop.");
    }
}

void
Process::SendAsyncInterrupt ()
{
    if (PrivateStateThreadIsValid())
        m_private_state_broadcaster.BroadcastEvent (Process::eBroadcastBitInterrupt, NULL);
    else
        BroadcastEvent (Process::eBroadcastBitInterrupt, NULL);
}

void
Process::HandlePrivateEvent (EventSP &event_sp)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    m_currently_handling_event.SetValue(true, eBroadcastNever);
    
    const StateType new_state = Process::ProcessEventData::GetStateFromEvent(event_sp.get());
    
    // First check to see if anybody wants a shot at this event:
    if (m_next_event_action_ap.get() != NULL)
    {
        NextEventAction::EventActionResult action_result = m_next_event_action_ap->PerformAction(event_sp);
        switch (action_result)
        {
            case NextEventAction::eEventActionSuccess:
                SetNextEventAction(NULL);
                break;

            case NextEventAction::eEventActionRetry:
                break;

            case NextEventAction::eEventActionExit:
                // Handle Exiting Here.  If we already got an exited event,
                // we should just propagate it.  Otherwise, swallow this event,
                // and set our state to exit so the next event will kill us.
                if (new_state != eStateExited)
                {
                    // FIXME: should cons up an exited event, and discard this one.
                    SetExitStatus(0, m_next_event_action_ap->GetExitString());
                    SetNextEventAction(NULL);
                    return;
                }
                SetNextEventAction(NULL);
                break;
        }
    }
    
    // See if we should broadcast this state to external clients?
    const bool should_broadcast = ShouldBroadcastEvent (event_sp.get());

    if (should_broadcast)
    {
        if (log)
        {
            log->Printf ("Process::%s (pid = %llu) broadcasting new state %s (old state %s) to %s", 
                         __FUNCTION__, 
                         GetID(), 
                         StateAsCString(new_state), 
                         StateAsCString (GetState ()),
                         IsHijackedForEvent(eBroadcastBitStateChanged) ? "hijacked" : "public");
        }
        Process::ProcessEventData::SetUpdateStateOnRemoval(event_sp.get());
        if (StateIsRunningState (new_state))
            PushProcessInputReader ();
        else 
            PopProcessInputReader ();

        BroadcastEvent (event_sp);
    }
    else
    {
        if (log)
        {
            log->Printf ("Process::%s (pid = %llu) suppressing state %s (old state %s): should_broadcast == false", 
                         __FUNCTION__, 
                         GetID(), 
                         StateAsCString(new_state), 
                         StateAsCString (GetState ()));
        }
    }
    m_currently_handling_event.SetValue(false, eBroadcastAlways);
}

void *
Process::PrivateStateThread (void *arg)
{
    Process *proc = static_cast<Process*> (arg);
    void *result = proc->RunPrivateStateThread ();
    return result;
}

void *
Process::RunPrivateStateThread ()
{
    bool control_only = true;
    m_private_state_control_wait.SetValue (false, eBroadcastNever);

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf ("Process::%s (arg = %p, pid = %llu) thread starting...", __FUNCTION__, this, GetID());

    bool exit_now = false;
    while (!exit_now)
    {
        EventSP event_sp;
        WaitForEventsPrivate (NULL, event_sp, control_only);
        if (event_sp->BroadcasterIs(&m_private_state_control_broadcaster))
        {
            if (log)
                log->Printf ("Process::%s (arg = %p, pid = %llu) got a control event: %d", __FUNCTION__, this, GetID(), event_sp->GetType());

            switch (event_sp->GetType())
            {
            case eBroadcastInternalStateControlStop:
                exit_now = true;
                break;      // doing any internal state managment below

            case eBroadcastInternalStateControlPause:
                control_only = true;
                break;

            case eBroadcastInternalStateControlResume:
                control_only = false;
                break;
            }
            
            m_private_state_control_wait.SetValue (true, eBroadcastAlways);
            continue;
        }
        else if (event_sp->GetType() == eBroadcastBitInterrupt)
        {
            if (m_public_state.GetValue() == eStateAttaching)
            {
                if (log)
                    log->Printf ("Process::%s (arg = %p, pid = %llu) woke up with an interrupt while attaching - forwarding interrupt.", __FUNCTION__, this, GetID());
                BroadcastEvent (eBroadcastBitInterrupt, NULL);
            }
            else
            {
                if (log)
                    log->Printf ("Process::%s (arg = %p, pid = %llu) woke up with an interrupt - Halting.", __FUNCTION__, this, GetID());
                Halt();
            }
            continue;
        }

        const StateType internal_state = Process::ProcessEventData::GetStateFromEvent(event_sp.get());

        if (internal_state != eStateInvalid)
        {
            HandlePrivateEvent (event_sp);
        }

        if (internal_state == eStateInvalid || 
            internal_state == eStateExited  ||
            internal_state == eStateDetached )
        {
            if (log)
                log->Printf ("Process::%s (arg = %p, pid = %llu) about to exit with internal state %s...", __FUNCTION__, this, GetID(), StateAsCString(internal_state));

            break;
        }
    }

    // Verify log is still enabled before attempting to write to it...
    if (log)
        log->Printf ("Process::%s (arg = %p, pid = %llu) thread exiting...", __FUNCTION__, this, GetID());

    m_private_state_control_wait.SetValue (true, eBroadcastAlways);
    m_private_state_thread = LLDB_INVALID_HOST_THREAD;
    return NULL;
}

//------------------------------------------------------------------
// Process Event Data
//------------------------------------------------------------------

Process::ProcessEventData::ProcessEventData () :
    EventData (),
    m_process_sp (),
    m_state (eStateInvalid),
    m_restarted (false),
    m_update_state (0),
    m_interrupted (false)
{
}

Process::ProcessEventData::ProcessEventData (const ProcessSP &process_sp, StateType state) :
    EventData (),
    m_process_sp (process_sp),
    m_state (state),
    m_restarted (false),
    m_update_state (0),
    m_interrupted (false)
{
}

Process::ProcessEventData::~ProcessEventData()
{
}

const ConstString &
Process::ProcessEventData::GetFlavorString ()
{
    static ConstString g_flavor ("Process::ProcessEventData");
    return g_flavor;
}

const ConstString &
Process::ProcessEventData::GetFlavor () const
{
    return ProcessEventData::GetFlavorString ();
}

void
Process::ProcessEventData::DoOnRemoval (Event *event_ptr)
{
    // This function gets called twice for each event, once when the event gets pulled 
    // off of the private process event queue, and then any number of times, first when it gets pulled off of
    // the public event queue, then other times when we're pretending that this is where we stopped at the
    // end of expression evaluation.  m_update_state is used to distinguish these
    // three cases; it is 0 when we're just pulling it off for private handling, 
    // and > 1 for expression evaluation, and we don't want to do the breakpoint command handling then.
    
    if (m_update_state != 1)
        return;
        
    m_process_sp->SetPublicState (m_state);
        
    // If we're stopped and haven't restarted, then do the breakpoint commands here:
    if (m_state == eStateStopped && ! m_restarted)
    {        
        ThreadList &curr_thread_list = m_process_sp->GetThreadList();
        uint32_t num_threads = curr_thread_list.GetSize();
        uint32_t idx;

        // The actions might change one of the thread's stop_info's opinions about whether we should
        // stop the process, so we need to query that as we go.
        
        // One other complication here, is that we try to catch any case where the target has run (except for expressions)
        // and immediately exit, but if we get that wrong (which is possible) then the thread list might have changed, and
        // that would cause our iteration here to crash.  We could make a copy of the thread list, but we'd really like
        // to also know if it has changed at all, so we make up a vector of the thread ID's and check what we get back 
        // against this list & bag out if anything differs.
        std::vector<uint32_t> thread_index_array(num_threads);
        for (idx = 0; idx < num_threads; ++idx)
            thread_index_array[idx] = curr_thread_list.GetThreadAtIndex(idx)->GetIndexID();
        
        bool still_should_stop = true;
        
        for (idx = 0; idx < num_threads; ++idx)
        {
            curr_thread_list = m_process_sp->GetThreadList();
            if (curr_thread_list.GetSize() != num_threads)
            {
                lldb::LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_STEP | LIBLLDB_LOG_PROCESS));
                if (log)
                    log->Printf("Number of threads changed from %u to %u while processing event.", num_threads, curr_thread_list.GetSize());
                break;
            }
            
            lldb::ThreadSP thread_sp = curr_thread_list.GetThreadAtIndex(idx);
            
            if (thread_sp->GetIndexID() != thread_index_array[idx])
            {
                lldb::LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_STEP | LIBLLDB_LOG_PROCESS));
                if (log)
                    log->Printf("The thread at position %u changed from %u to %u while processing event.", 
                                idx, 
                                thread_index_array[idx],
                                thread_sp->GetIndexID());
                break;
            }
            
            StopInfoSP stop_info_sp = thread_sp->GetStopInfo ();
            if (stop_info_sp && stop_info_sp->IsValid())
            {
                stop_info_sp->PerformAction(event_ptr);
                // The stop action might restart the target.  If it does, then we want to mark that in the
                // event so that whoever is receiving it will know to wait for the running event and reflect
                // that state appropriately.
                // We also need to stop processing actions, since they aren't expecting the target to be running.
                
                // FIXME: we might have run.
                if (stop_info_sp->HasTargetRunSinceMe())
                {
                    SetRestarted (true);
                    break;
                }
                else if (!stop_info_sp->ShouldStop(event_ptr))
                {
                    still_should_stop = false;
                }
            }
        }

        
        if (m_process_sp->GetPrivateState() != eStateRunning)
        {
            if (!still_should_stop)
            {
                // We've been asked to continue, so do that here.
                SetRestarted(true);
                // Use the public resume method here, since this is just
                // extending a public resume.
                m_process_sp->Resume();
            }
            else
            {
                // If we didn't restart, run the Stop Hooks here:
                // They might also restart the target, so watch for that.
                m_process_sp->GetTarget().RunStopHooks();
                if (m_process_sp->GetPrivateState() == eStateRunning)
                    SetRestarted(true);
            }
        }
        
    }
}

void
Process::ProcessEventData::Dump (Stream *s) const
{
    if (m_process_sp)
        s->Printf(" process = %p (pid = %llu), ", m_process_sp.get(), m_process_sp->GetID());

    s->Printf("state = %s", StateAsCString(GetState()));
}

const Process::ProcessEventData *
Process::ProcessEventData::GetEventDataFromEvent (const Event *event_ptr)
{
    if (event_ptr)
    {
        const EventData *event_data = event_ptr->GetData();
        if (event_data && event_data->GetFlavor() == ProcessEventData::GetFlavorString())
            return static_cast <const ProcessEventData *> (event_ptr->GetData());
    }
    return NULL;
}

ProcessSP
Process::ProcessEventData::GetProcessFromEvent (const Event *event_ptr)
{
    ProcessSP process_sp;
    const ProcessEventData *data = GetEventDataFromEvent (event_ptr);
    if (data)
        process_sp = data->GetProcessSP();
    return process_sp;
}

StateType
Process::ProcessEventData::GetStateFromEvent (const Event *event_ptr)
{
    const ProcessEventData *data = GetEventDataFromEvent (event_ptr);
    if (data == NULL)
        return eStateInvalid;
    else
        return data->GetState();
}

bool
Process::ProcessEventData::GetRestartedFromEvent (const Event *event_ptr)
{
    const ProcessEventData *data = GetEventDataFromEvent (event_ptr);
    if (data == NULL)
        return false;
    else
        return data->GetRestarted();
}

void
Process::ProcessEventData::SetRestartedInEvent (Event *event_ptr, bool new_value)
{
    ProcessEventData *data = const_cast<ProcessEventData *>(GetEventDataFromEvent (event_ptr));
    if (data != NULL)
        data->SetRestarted(new_value);
}

bool
Process::ProcessEventData::GetInterruptedFromEvent (const Event *event_ptr)
{
    const ProcessEventData *data = GetEventDataFromEvent (event_ptr);
    if (data == NULL)
        return false;
    else
        return data->GetInterrupted ();
}

void
Process::ProcessEventData::SetInterruptedInEvent (Event *event_ptr, bool new_value)
{
    ProcessEventData *data = const_cast<ProcessEventData *>(GetEventDataFromEvent (event_ptr));
    if (data != NULL)
        data->SetInterrupted(new_value);
}

bool
Process::ProcessEventData::SetUpdateStateOnRemoval (Event *event_ptr)
{
    ProcessEventData *data = const_cast<ProcessEventData *>(GetEventDataFromEvent (event_ptr));
    if (data)
    {
        data->SetUpdateStateOnRemoval();
        return true;
    }
    return false;
}

lldb::TargetSP
Process::CalculateTarget ()
{
    return m_target.shared_from_this();
}

void
Process::CalculateExecutionContext (ExecutionContext &exe_ctx)
{
    exe_ctx.SetTargetPtr (&m_target);
    exe_ctx.SetProcessPtr (this);
    exe_ctx.SetThreadPtr(NULL);
    exe_ctx.SetFramePtr (NULL);
}

//uint32_t
//Process::ListProcessesMatchingName (const char *name, StringList &matches, std::vector<lldb::pid_t> &pids)
//{
//    return 0;
//}
//    
//ArchSpec
//Process::GetArchSpecForExistingProcess (lldb::pid_t pid)
//{
//    return Host::GetArchSpecForExistingProcess (pid);
//}
//
//ArchSpec
//Process::GetArchSpecForExistingProcess (const char *process_name)
//{
//    return Host::GetArchSpecForExistingProcess (process_name);
//}
//
void
Process::AppendSTDOUT (const char * s, size_t len)
{
    Mutex::Locker locker (m_stdio_communication_mutex);
    m_stdout_data.append (s, len);
    BroadcastEventIfUnique (eBroadcastBitSTDOUT, new ProcessEventData (shared_from_this(), GetState()));
}

void
Process::AppendSTDERR (const char * s, size_t len)
{
    Mutex::Locker locker (m_stdio_communication_mutex);
    m_stderr_data.append (s, len);
    BroadcastEventIfUnique (eBroadcastBitSTDERR, new ProcessEventData (shared_from_this(), GetState()));
}

//------------------------------------------------------------------
// Process STDIO
//------------------------------------------------------------------

size_t
Process::GetSTDOUT (char *buf, size_t buf_size, Error &error)
{
    Mutex::Locker locker(m_stdio_communication_mutex);
    size_t bytes_available = m_stdout_data.size();
    if (bytes_available > 0)
    {
        LogSP log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
        if (log)
            log->Printf ("Process::GetSTDOUT (buf = %p, size = %llu)", buf, (uint64_t)buf_size);
        if (bytes_available > buf_size)
        {
            memcpy(buf, m_stdout_data.c_str(), buf_size);
            m_stdout_data.erase(0, buf_size);
            bytes_available = buf_size;
        }
        else
        {
            memcpy(buf, m_stdout_data.c_str(), bytes_available);
            m_stdout_data.clear();
        }
    }
    return bytes_available;
}


size_t
Process::GetSTDERR (char *buf, size_t buf_size, Error &error)
{
    Mutex::Locker locker(m_stdio_communication_mutex);
    size_t bytes_available = m_stderr_data.size();
    if (bytes_available > 0)
    {
        LogSP log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
        if (log)
            log->Printf ("Process::GetSTDERR (buf = %p, size = %llu)", buf, (uint64_t)buf_size);
        if (bytes_available > buf_size)
        {
            memcpy(buf, m_stderr_data.c_str(), buf_size);
            m_stderr_data.erase(0, buf_size);
            bytes_available = buf_size;
        }
        else
        {
            memcpy(buf, m_stderr_data.c_str(), bytes_available);
            m_stderr_data.clear();
        }
    }
    return bytes_available;
}

void
Process::STDIOReadThreadBytesReceived (void *baton, const void *src, size_t src_len)
{
    Process *process = (Process *) baton;
    process->AppendSTDOUT (static_cast<const char *>(src), src_len);
}

size_t
Process::ProcessInputReaderCallback (void *baton,
                                     InputReader &reader,
                                     lldb::InputReaderAction notification,
                                     const char *bytes,
                                     size_t bytes_len)
{
    Process *process = (Process *) baton;
    
    switch (notification)
    {
    case eInputReaderActivate:
        break;
        
    case eInputReaderDeactivate:
        break;
        
    case eInputReaderReactivate:
        break;
        
    case eInputReaderAsynchronousOutputWritten:
        break;
        
    case eInputReaderGotToken:
        {
            Error error;
            process->PutSTDIN (bytes, bytes_len, error);
        }
        break;
        
    case eInputReaderInterrupt:
        process->Halt ();
        break;
            
    case eInputReaderEndOfFile:
        process->AppendSTDOUT ("^D", 2);
        break;
        
    case eInputReaderDone:
        break;
        
    }
    
    return bytes_len;
}

void
Process::ResetProcessInputReader ()
{   
    m_process_input_reader.reset();
}

void
Process::SetSTDIOFileDescriptor (int file_descriptor)
{
    // First set up the Read Thread for reading/handling process I/O
    
    std::auto_ptr<ConnectionFileDescriptor> conn_ap (new ConnectionFileDescriptor (file_descriptor, true));
    
    if (conn_ap.get())
    {
        m_stdio_communication.SetConnection (conn_ap.release());
        if (m_stdio_communication.IsConnected())
        {
            m_stdio_communication.SetReadThreadBytesReceivedCallback (STDIOReadThreadBytesReceived, this);
            m_stdio_communication.StartReadThread();
            
            // Now read thread is set up, set up input reader.
            
            if (!m_process_input_reader.get())
            {
                m_process_input_reader.reset (new InputReader(m_target.GetDebugger()));
                Error err (m_process_input_reader->Initialize (Process::ProcessInputReaderCallback,
                                                               this,
                                                               eInputReaderGranularityByte,
                                                               NULL,
                                                               NULL,
                                                               false));
                
                if  (err.Fail())
                    m_process_input_reader.reset();
            }
        }
    }
}

void
Process::PushProcessInputReader ()
{
    if (m_process_input_reader && !m_process_input_reader->IsActive())
        m_target.GetDebugger().PushInputReader (m_process_input_reader);
}

void
Process::PopProcessInputReader ()
{
    if (m_process_input_reader && m_process_input_reader->IsActive())
        m_target.GetDebugger().PopInputReader (m_process_input_reader);
}

// The process needs to know about installed plug-ins
void
Process::SettingsInitialize ()
{
//    static std::vector<OptionEnumValueElement> g_plugins;
//    
//    int i=0; 
//    const char *name;
//    OptionEnumValueElement option_enum;
//    while ((name = PluginManager::GetProcessPluginNameAtIndex (i)) != NULL)
//    {
//        if (name)
//        {
//            option_enum.value = i;
//            option_enum.string_value = name;
//            option_enum.usage = PluginManager::GetProcessPluginDescriptionAtIndex (i);
//            g_plugins.push_back (option_enum);
//        }
//        ++i;
//    }
//    option_enum.value = 0;
//    option_enum.string_value = NULL;
//    option_enum.usage = NULL;
//    g_plugins.push_back (option_enum);
//    
//    for (i=0; (name = SettingsController::instance_settings_table[i].var_name); ++i)
//    {
//        if (::strcmp (name, "plugin") == 0)
//        {
//            SettingsController::instance_settings_table[i].enum_values = &g_plugins[0];
//            break;
//        }
//    }
//                                                          
    Thread::SettingsInitialize ();
}

void
Process::SettingsTerminate ()
{
    Thread::SettingsTerminate ();
}

ExecutionResults
Process::RunThreadPlan (ExecutionContext &exe_ctx,
                        lldb::ThreadPlanSP &thread_plan_sp,
                        bool stop_others,
                        bool run_others,
                        bool discard_on_error,
                        uint32_t timeout_usec,
                        Stream &errors)
{
    ExecutionResults return_value = eExecutionSetupError;
    
    if (thread_plan_sp.get() == NULL)
    {
        errors.Printf("RunThreadPlan called with empty thread plan.");
        return eExecutionSetupError;
    }

    if (exe_ctx.GetProcessPtr() != this)
    {
        errors.Printf("RunThreadPlan called on wrong process.");
        return eExecutionSetupError;
    }

    Thread *thread = exe_ctx.GetThreadPtr();
    if (thread == NULL)
    {
        errors.Printf("RunThreadPlan called with invalid thread.");
        return eExecutionSetupError;
    }
    
    // We rely on the thread plan we are running returning "PlanCompleted" if when it successfully completes.
    // For that to be true the plan can't be private - since private plans suppress themselves in the
    // GetCompletedPlan call. 
    
    bool orig_plan_private = thread_plan_sp->GetPrivate();
    thread_plan_sp->SetPrivate(false);
    
    if (m_private_state.GetValue() != eStateStopped)
    {
        errors.Printf ("RunThreadPlan called while the private state was not stopped.");
        return eExecutionSetupError;
    }
    
    // Save the thread & frame from the exe_ctx for restoration after we run
    const uint32_t thread_idx_id = thread->GetIndexID();
    StackID ctx_frame_id = thread->GetSelectedFrame()->GetStackID();

    // N.B. Running the target may unset the currently selected thread and frame.  We don't want to do that either, 
    // so we should arrange to reset them as well.
    
    lldb::ThreadSP selected_thread_sp = GetThreadList().GetSelectedThread();
    
    uint32_t selected_tid;
    StackID selected_stack_id;
    if (selected_thread_sp)
    {
        selected_tid = selected_thread_sp->GetIndexID();
        selected_stack_id = selected_thread_sp->GetSelectedFrame()->GetStackID();
    }
    else
    {
        selected_tid = LLDB_INVALID_THREAD_ID;
    }

    lldb::thread_t backup_private_state_thread = LLDB_INVALID_HOST_THREAD;
    lldb::StateType old_state;
    lldb::ThreadPlanSP stopper_base_plan_sp;
    
    lldb::LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_STEP | LIBLLDB_LOG_PROCESS));
    if (Host::GetCurrentThread() == m_private_state_thread)
    {
        // Yikes, we are running on the private state thread!  So we can't wait for public events on this thread, since
        // we are the thread that is generating public events.
        // The simplest thing to do is to spin up a temporary thread to handle private state thread events while
        // we are fielding public events here.
        if (log)
			log->Printf ("Running thread plan on private state thread, spinning up another state thread to handle the events.");
            

        backup_private_state_thread = m_private_state_thread;

        // One other bit of business: we want to run just this thread plan and anything it pushes, and then stop,
        // returning control here.
        // But in the normal course of things, the plan above us on the stack would be given a shot at the stop
        // event before deciding to stop, and we don't want that.  So we insert a "stopper" base plan on the stack
        // before the plan we want to run.  Since base plans always stop and return control to the user, that will
        // do just what we want.
        stopper_base_plan_sp.reset(new ThreadPlanBase (*thread));
        thread->QueueThreadPlan (stopper_base_plan_sp, false);
        // Have to make sure our public state is stopped, since otherwise the reporting logic below doesn't work correctly.
        old_state = m_public_state.GetValue();
        m_public_state.SetValueNoLock(eStateStopped);
        
        // Now spin up the private state thread:
        StartPrivateStateThread(true);
    }
    
    thread->QueueThreadPlan(thread_plan_sp, false); // This used to pass "true" does that make sense?
    
    Listener listener("lldb.process.listener.run-thread-plan");
    
    lldb::EventSP event_to_broadcast_sp;
    
    {
        // This process event hijacker Hijacks the Public events and its destructor makes sure that the process events get
        // restored on exit to the function.
        //
        // If the event needs to propagate beyond the hijacker (e.g., the process exits during execution), then the event
        // is put into event_to_broadcast_sp for rebroadcasting.
        
        ProcessEventHijacker run_thread_plan_hijacker (*this, &listener);
            
        if (log)
        {
            StreamString s;
            thread_plan_sp->GetDescription(&s, lldb::eDescriptionLevelVerbose);
            log->Printf ("Process::RunThreadPlan(): Resuming thread %u - 0x%4.4llx to run thread plan \"%s\".",  
                         thread->GetIndexID(), 
                         thread->GetID(), 
                         s.GetData());
        }
        
        bool got_event;
        lldb::EventSP event_sp;
        lldb::StateType stop_state = lldb::eStateInvalid;
        
        TimeValue* timeout_ptr = NULL;
        TimeValue real_timeout;
        
        bool first_timeout = true;
        bool do_resume = true;
        const uint64_t default_one_thread_timeout_usec = 250000;
        uint64_t computed_timeout = 0;
        
        while (1)
        {
            // We usually want to resume the process if we get to the top of the loop.
            // The only exception is if we get two running events with no intervening
            // stop, which can happen, we will just wait for then next stop event.
            
            if (do_resume)
            {
                // Do the initial resume and wait for the running event before going further.
        
                Error resume_error = PrivateResume ();
                if (!resume_error.Success())
                {
                    errors.Printf("Error resuming inferior: \"%s\".\n", resume_error.AsCString());
                    return_value = eExecutionSetupError;
                    break;
                }
        
                real_timeout = TimeValue::Now();
                real_timeout.OffsetWithMicroSeconds(500000);
                timeout_ptr = &real_timeout;
                
                got_event = listener.WaitForEvent(timeout_ptr, event_sp);
                if (!got_event)
                {
                    if (log)
                        log->PutCString("Process::RunThreadPlan(): didn't get any event after initial resume, exiting.");

                    errors.Printf("Didn't get any event after initial resume, exiting.");
                    return_value = eExecutionSetupError;
                    break;
                }
                
                stop_state = Process::ProcessEventData::GetStateFromEvent(event_sp.get());
                if (stop_state != eStateRunning)
                {
                    if (log)
                        log->Printf("Process::RunThreadPlan(): didn't get running event after "
                                    "initial resume, got %s instead.",
                                    StateAsCString(stop_state));

                    errors.Printf("Didn't get running event after initial resume, got %s instead.",
                                  StateAsCString(stop_state));
                    return_value = eExecutionSetupError;
                    break;
                }
            
                if (log)
                    log->PutCString ("Process::RunThreadPlan(): resuming succeeded.");
                // We need to call the function synchronously, so spin waiting for it to return.
                // If we get interrupted while executing, we're going to lose our context, and
                // won't be able to gather the result at this point.
                // We set the timeout AFTER the resume, since the resume takes some time and we
                // don't want to charge that to the timeout.
                
                if (first_timeout)
                {
                    if (run_others)
                    {
                        // If we are running all threads then we take half the time to run all threads, bounded by
                        // .25 sec.
                        if (timeout_usec == 0)
                            computed_timeout = default_one_thread_timeout_usec;
                        else
                        {
                            computed_timeout = timeout_usec / 2;
                            if (computed_timeout > default_one_thread_timeout_usec)
                            {
                                computed_timeout = default_one_thread_timeout_usec;
                            }
                            timeout_usec -= computed_timeout;
                        }
                    }
                    else
                    {
                        computed_timeout = timeout_usec;
                    }
                }
                else
                {
                    computed_timeout = timeout_usec;
                }
                
                if (computed_timeout != 0)
                {
                    // we have a > 0 timeout, let us set it so that we stop after the deadline
                    real_timeout = TimeValue::Now();
                    real_timeout.OffsetWithMicroSeconds(computed_timeout);
                        
                    timeout_ptr = &real_timeout;
                }
                else
                {
                    timeout_ptr = NULL;
                }
            }
            else
            {
                if (log)
                    log->PutCString ("Process::RunThreadPlan(): handled an extra running event.");
                do_resume = true;
            }
            
            // Now wait for the process to stop again:
            event_sp.reset();

            if (log)
            {
                if (timeout_ptr)
                {
                    StreamString s;
                    s.Printf ("about to wait - timeout is:\n   ");
                    timeout_ptr->Dump (&s, 120);
                    s.Printf ("\nNow is:\n    ");
                    TimeValue::Now().Dump (&s, 120);
                    log->Printf ("Process::RunThreadPlan(): %s", s.GetData());
                }
                else
                {
                    log->Printf ("Process::RunThreadPlan(): about to wait forever.");
                }
            }
            
            got_event = listener.WaitForEvent (timeout_ptr, event_sp);
            
            if (got_event)
            {
                if (event_sp.get())
                {
                    bool keep_going = false;
                    if (event_sp->GetType() == eBroadcastBitInterrupt)
                    {
                        Halt();
                        keep_going = false;
                        return_value = eExecutionInterrupted;
                        errors.Printf ("Execution halted by user interrupt.");
                        if (log)
                            log->Printf ("Process::RunThreadPlan(): Got  interrupted by eBroadcastBitInterrupted, exiting.");
                    }
                    else
                    {
                        stop_state = Process::ProcessEventData::GetStateFromEvent(event_sp.get());
                        if (log)
                            log->Printf("Process::RunThreadPlan(): in while loop, got event: %s.", StateAsCString(stop_state));
                            
                        switch (stop_state)
                        {
                        case lldb::eStateStopped:
                            {
                                // Yay, we're done.  Now make sure that our thread plan actually completed.
                                ThreadSP thread_sp = GetThreadList().FindThreadByIndexID (thread_idx_id);
                                if (!thread_sp)
                                {
                                    // Ooh, our thread has vanished.  Unlikely that this was successful execution...
                                    if (log)
                                        log->Printf ("Process::RunThreadPlan(): execution completed but our thread (index-id=%u) has vanished.", thread_idx_id);
                                    return_value = eExecutionInterrupted;
                                }
                                else
                                {
                                    StopInfoSP stop_info_sp (thread_sp->GetStopInfo ());
                                    StopReason stop_reason = eStopReasonInvalid;
                                    if (stop_info_sp)
                                         stop_reason = stop_info_sp->GetStopReason();
                                    if (stop_reason == eStopReasonPlanComplete)
                                    {
                                        if (log)
                                            log->PutCString ("Process::RunThreadPlan(): execution completed successfully.");
                                        // Now mark this plan as private so it doesn't get reported as the stop reason
                                        // after this point.  
                                        if (thread_plan_sp)
                                            thread_plan_sp->SetPrivate (orig_plan_private);
                                        return_value = eExecutionCompleted;
                                    }
                                    else
                                    {
                                        if (log)
                                            log->PutCString ("Process::RunThreadPlan(): thread plan didn't successfully complete.");

                                        return_value = eExecutionInterrupted;
                                    }
                                }
                            }        
                            break;

                        case lldb::eStateCrashed:
                            if (log)
                                log->PutCString ("Process::RunThreadPlan(): execution crashed.");
                            return_value = eExecutionInterrupted;
                            break;

                        case lldb::eStateRunning:
                            do_resume = false;
                            keep_going = true;
                            break;

                        default:
                            if (log)
                                log->Printf("Process::RunThreadPlan(): execution stopped with unexpected state: %s.", StateAsCString(stop_state));
                                
                            if (stop_state == eStateExited)
                                event_to_broadcast_sp = event_sp;
                                
                            errors.Printf ("Execution stopped with unexpected state.\n");
                            return_value = eExecutionInterrupted;
                            break;
                        }
                    }
                    
                    if (keep_going)
                        continue;
                    else
                        break;
                } 
                else
                {
                    if (log)
                        log->PutCString ("Process::RunThreadPlan(): got_event was true, but the event pointer was null.  How odd...");
                    return_value = eExecutionInterrupted;
                    break;
                }
            }
            else
            {
                // If we didn't get an event that means we've timed out...
                // We will interrupt the process here.  Depending on what we were asked to do we will
                // either exit, or try with all threads running for the same timeout.
                // Not really sure what to do if Halt fails here...
                
                if (log) {
                    if (run_others)
                    {
                        if (first_timeout)
                            log->Printf ("Process::RunThreadPlan(): Running function with timeout: %lld timed out, "
                                         "trying  for %d usec with all threads enabled.",
                                         computed_timeout, timeout_usec);
                        else
                            log->Printf ("Process::RunThreadPlan(): Restarting function with all threads enabled "
                                         "and timeout: %d timed out, abandoning execution.",
                                         timeout_usec);
                    }
                    else
                        log->Printf ("Process::RunThreadPlan(): Running function with timeout: %d timed out, "
                                     "abandoning execution.", 
                                     timeout_usec);
                }
                
                Error halt_error = Halt();
                if (halt_error.Success())
                {
                    if (log)
                        log->PutCString ("Process::RunThreadPlan(): Halt succeeded.");
                        
                    // If halt succeeds, it always produces a stopped event.  Wait for that:
                    
                    real_timeout = TimeValue::Now();
                    real_timeout.OffsetWithMicroSeconds(500000);

                    got_event = listener.WaitForEvent(&real_timeout, event_sp);
                    
                    if (got_event)
                    {
                        stop_state = Process::ProcessEventData::GetStateFromEvent(event_sp.get());
                        if (log)
                        {
                            log->Printf ("Process::RunThreadPlan(): Stopped with event: %s", StateAsCString(stop_state));
                            if (stop_state == lldb::eStateStopped 
                                && Process::ProcessEventData::GetInterruptedFromEvent(event_sp.get()))
                                log->PutCString ("    Event was the Halt interruption event.");
                        }
                        
                        if (stop_state == lldb::eStateStopped)
                        {
                            // Between the time we initiated the Halt and the time we delivered it, the process could have
                            // already finished its job.  Check that here:
                            
                            if (thread->IsThreadPlanDone (thread_plan_sp.get()))
                            {
                                if (log)
                                    log->PutCString ("Process::RunThreadPlan(): Even though we timed out, the call plan was done.  "
                                                 "Exiting wait loop.");
                                return_value = eExecutionCompleted;
                                break;
                            }

                            if (!run_others)
                            {
                                if (log)
                                    log->PutCString ("Process::RunThreadPlan(): try_all_threads was false, we stopped so now we're quitting.");
                                return_value = eExecutionInterrupted;
                                break;
                            }
                            
                            if (first_timeout)
                            {
                                // Set all the other threads to run, and return to the top of the loop, which will continue;
                                first_timeout = false;
                                thread_plan_sp->SetStopOthers (false);
                                if (log)
                                    log->PutCString ("Process::RunThreadPlan(): about to resume.");

                                continue;
                            }
                            else
                            {
                                // Running all threads failed, so return Interrupted.
                                if (log)
                                    log->PutCString("Process::RunThreadPlan(): running all threads timed out.");
                                return_value = eExecutionInterrupted;
                                break;
                            }
                        }
                    }
                    else
                    {   if (log)
                            log->PutCString("Process::RunThreadPlan(): halt said it succeeded, but I got no event.  "
                                    "I'm getting out of here passing Interrupted.");
                        return_value = eExecutionInterrupted;
                        break;
                    }
                }
                else
                {
                    // This branch is to work around some problems with gdb-remote's Halt.  It is a little racy, and can return 
                    // an error from halt, but if you wait a bit you'll get a stopped event anyway.
                    if (log)
                        log->Printf ("Process::RunThreadPlan(): halt failed: error = \"%s\", I'm just going to wait a little longer and see if I get a stopped event.", 
                                     halt_error.AsCString());                
                    real_timeout = TimeValue::Now();
                    real_timeout.OffsetWithMicroSeconds(500000);
                    timeout_ptr = &real_timeout;
                    got_event = listener.WaitForEvent(&real_timeout, event_sp);
                    if (!got_event || event_sp.get() == NULL)
                    {
                        // This is not going anywhere, bag out.
                        if (log)
                            log->PutCString ("Process::RunThreadPlan(): halt failed: and waiting for the stopped event failed.");
                        return_value = eExecutionInterrupted;
                        break;                
                    }
                    else
                    {
                        stop_state = Process::ProcessEventData::GetStateFromEvent(event_sp.get());
                        if (log)
                            log->PutCString ("Process::RunThreadPlan(): halt failed: but then I got a stopped event.  Whatever...");
                        if (stop_state == lldb::eStateStopped)
                        {
                            // Between the time we initiated the Halt and the time we delivered it, the process could have
                            // already finished its job.  Check that here:
                            
                            if (thread->IsThreadPlanDone (thread_plan_sp.get()))
                            {
                                if (log)
                                    log->PutCString ("Process::RunThreadPlan(): Even though we timed out, the call plan was done.  "
                                                 "Exiting wait loop.");
                                return_value = eExecutionCompleted;
                                break;
                            }

                            if (first_timeout)
                            {
                                // Set all the other threads to run, and return to the top of the loop, which will continue;
                                first_timeout = false;
                                thread_plan_sp->SetStopOthers (false);
                                if (log)
                                    log->PutCString ("Process::RunThreadPlan(): About to resume.");

                                continue;
                            }
                            else
                            {
                                // Running all threads failed, so return Interrupted.
                                if (log)
                                    log->PutCString ("Process::RunThreadPlan(): running all threads timed out.");
                                return_value = eExecutionInterrupted;
                                break;
                            }
                        }
                        else
                        {
                            if (log)
                                log->Printf ("Process::RunThreadPlan(): halt failed, I waited and didn't get"
                                             " a stopped event, instead got %s.", StateAsCString(stop_state));
                            return_value = eExecutionInterrupted;
                            break;                
                        }
                    }
                }

            }
            
        }  // END WAIT LOOP
        
        // If we had to start up a temporary private state thread to run this thread plan, shut it down now.
        if (IS_VALID_LLDB_HOST_THREAD(backup_private_state_thread))
        {
            StopPrivateStateThread();
            Error error;
            m_private_state_thread = backup_private_state_thread;
            if (stopper_base_plan_sp)
            {
                thread->DiscardThreadPlansUpToPlan(stopper_base_plan_sp);
            }
            m_public_state.SetValueNoLock(old_state);

        }
        
        
        // Now do some processing on the results of the run:
        if (return_value == eExecutionInterrupted)
        {
            if (log)
            {
                StreamString s;
                if (event_sp)
                    event_sp->Dump (&s);
                else
                {
                    log->PutCString ("Process::RunThreadPlan(): Stop event that interrupted us is NULL.");
                }

                StreamString ts;

                const char *event_explanation = NULL;                
                
                do 
                {
                    if (!event_sp)
                    {
                        event_explanation = "<no event>";
                        break;
                    }
                    else if (event_sp->GetType() == eBroadcastBitInterrupt)
                    {
                        event_explanation = "<user interrupt>";
                        break;
                    }
                    else
                    {
                        const Process::ProcessEventData *event_data = Process::ProcessEventData::GetEventDataFromEvent (event_sp.get());

                        if (!event_data)
                        {
                            event_explanation = "<no event data>";
                            break;
                        }
                        
                        Process *process = event_data->GetProcessSP().get();

                        if (!process)
                        {
                            event_explanation = "<no process>";
                            break;
                        }
                        
                        ThreadList &thread_list = process->GetThreadList();
                        
                        uint32_t num_threads = thread_list.GetSize();
                        uint32_t thread_index;
                        
                        ts.Printf("<%u threads> ", num_threads);
                        
                        for (thread_index = 0;
                             thread_index < num_threads;
                             ++thread_index)
                        {
                            Thread *thread = thread_list.GetThreadAtIndex(thread_index).get();
                            
                            if (!thread)
                            {
                                ts.Printf("<?> ");
                                continue;
                            }
                            
                            ts.Printf("<0x%4.4llx ", thread->GetID());
                            RegisterContext *register_context = thread->GetRegisterContext().get();
                            
                            if (register_context)
                                ts.Printf("[ip 0x%llx] ", register_context->GetPC());
                            else
                                ts.Printf("[ip unknown] ");
                            
                            lldb::StopInfoSP stop_info_sp = thread->GetStopInfo();
                            if (stop_info_sp)
                            {
                                const char *stop_desc = stop_info_sp->GetDescription();
                                if (stop_desc)
                                    ts.PutCString (stop_desc);
                            }
                            ts.Printf(">");
                        }
                        
                        event_explanation = ts.GetData();
                    }
                } while (0);
                
                if (event_explanation)
                    log->Printf("Process::RunThreadPlan(): execution interrupted: %s %s", s.GetData(), event_explanation);
                else
                    log->Printf("Process::RunThreadPlan(): execution interrupted: %s", s.GetData());
            }
            
            if (discard_on_error && thread_plan_sp)
            {
                if (log)
                    log->Printf ("Process::RunThreadPlan: ExecutionInterrupted - discarding thread plans up to %p.", thread_plan_sp.get());
                thread->DiscardThreadPlansUpToPlan (thread_plan_sp);
                thread_plan_sp->SetPrivate (orig_plan_private);
            }
            else
            {
                if (log)
                    log->Printf ("Process::RunThreadPlan: ExecutionInterrupted - for plan: %p not discarding.", thread_plan_sp.get());
            }
        }
        else if (return_value == eExecutionSetupError)
        {
            if (log)
                log->PutCString("Process::RunThreadPlan(): execution set up error.");
                
            if (discard_on_error && thread_plan_sp)
            {
                thread->DiscardThreadPlansUpToPlan (thread_plan_sp);
                thread_plan_sp->SetPrivate (orig_plan_private);
            }
        }
        else
        {
            if (thread->IsThreadPlanDone (thread_plan_sp.get()))
            {
                if (log)
                    log->PutCString("Process::RunThreadPlan(): thread plan is done");
                return_value = eExecutionCompleted;
            }
            else if (thread->WasThreadPlanDiscarded (thread_plan_sp.get()))
            {
                if (log)
                    log->PutCString("Process::RunThreadPlan(): thread plan was discarded");
                return_value = eExecutionDiscarded;
            }
            else
            {
                if (log)
                    log->PutCString("Process::RunThreadPlan(): thread plan stopped in mid course");
                if (discard_on_error && thread_plan_sp)
                {
                    if (log)
                        log->PutCString("Process::RunThreadPlan(): discarding thread plan 'cause discard_on_error is set.");
                    thread->DiscardThreadPlansUpToPlan (thread_plan_sp);
                    thread_plan_sp->SetPrivate (orig_plan_private);
                }
            }
        }
                    
        // Thread we ran the function in may have gone away because we ran the target
        // Check that it's still there, and if it is put it back in the context.  Also restore the
        // frame in the context if it is still present.
        thread = GetThreadList().FindThreadByIndexID(thread_idx_id, true).get();
        if (thread)
        {
            exe_ctx.SetFrameSP (thread->GetFrameWithStackID (ctx_frame_id));
        }
        
        // Also restore the current process'es selected frame & thread, since this function calling may
        // be done behind the user's back.
        
        if (selected_tid != LLDB_INVALID_THREAD_ID)
        {
            if (GetThreadList().SetSelectedThreadByIndexID (selected_tid) && selected_stack_id.IsValid())
            {
                // We were able to restore the selected thread, now restore the frame:
                StackFrameSP old_frame_sp = GetThreadList().GetSelectedThread()->GetFrameWithStackID(selected_stack_id);
                if (old_frame_sp)
                    GetThreadList().GetSelectedThread()->SetSelectedFrame(old_frame_sp.get());
            }
        }
    }
    
    // If the process exited during the run of the thread plan, notify everyone.
    
    if (event_to_broadcast_sp)
    {
        if (log)
            log->PutCString("Process::RunThreadPlan(): rebroadcasting event.");
        BroadcastEvent(event_to_broadcast_sp);
    }
    
    return return_value;
}

const char *
Process::ExecutionResultAsCString (ExecutionResults result)
{
    const char *result_name;
    
    switch (result)
    {
        case eExecutionCompleted:
            result_name = "eExecutionCompleted";
            break;
        case eExecutionDiscarded:
            result_name = "eExecutionDiscarded";
            break;
        case eExecutionInterrupted:
            result_name = "eExecutionInterrupted";
            break;
        case eExecutionSetupError:
            result_name = "eExecutionSetupError";
            break;
        case eExecutionTimedOut:
            result_name = "eExecutionTimedOut";
            break;
    }
    return result_name;
}

void
Process::GetStatus (Stream &strm)
{
    const StateType state = GetState();
    if (StateIsStoppedState(state, false))
    {
        if (state == eStateExited)
        {
            int exit_status = GetExitStatus();
            const char *exit_description = GetExitDescription();
            strm.Printf ("Process %llu exited with status = %i (0x%8.8x) %s\n",
                          GetID(),
                          exit_status,
                          exit_status,
                          exit_description ? exit_description : "");
        }
        else
        {
            if (state == eStateConnected)
                strm.Printf ("Connected to remote target.\n");
            else
                strm.Printf ("Process %llu %s\n", GetID(), StateAsCString (state));
        }
    }
    else
    {
        strm.Printf ("Process %llu is running.\n", GetID());
    }
}

size_t
Process::GetThreadStatus (Stream &strm, 
                          bool only_threads_with_stop_reason,
                          uint32_t start_frame, 
                          uint32_t num_frames, 
                          uint32_t num_frames_with_source)
{
    size_t num_thread_infos_dumped = 0;
    
    Mutex::Locker locker (GetThreadList().GetMutex());
    const size_t num_threads = GetThreadList().GetSize();
    for (uint32_t i = 0; i < num_threads; i++)
    {
        Thread *thread = GetThreadList().GetThreadAtIndex(i).get();
        if (thread)
        {
            if (only_threads_with_stop_reason)
            {
                StopInfoSP stop_info_sp = thread->GetStopInfo();
                if (stop_info_sp.get() == NULL || !stop_info_sp->IsValid())
                    continue;
            }
            thread->GetStatus (strm, 
                               start_frame, 
                               num_frames, 
                               num_frames_with_source);
            ++num_thread_infos_dumped;
        }
    }
    return num_thread_infos_dumped;
}

void
Process::AddInvalidMemoryRegion (const LoadRange &region)
{
    m_memory_cache.AddInvalidRange(region.GetRangeBase(), region.GetByteSize());
}

bool
Process::RemoveInvalidMemoryRange (const LoadRange &region)
{
    return m_memory_cache.RemoveInvalidRange(region.GetRangeBase(), region.GetByteSize());
}

void
Process::AddPreResumeAction (PreResumeActionCallback callback, void *baton)
{
    m_pre_resume_actions.push_back(PreResumeCallbackAndBaton (callback, baton));
}

bool
Process::RunPreResumeActions ()
{
    bool result = true;
    while (!m_pre_resume_actions.empty())
    {
        struct PreResumeCallbackAndBaton action = m_pre_resume_actions.back();
        m_pre_resume_actions.pop_back();
        bool this_result = action.callback (action.baton);
        if (result == true) result = this_result;
    }
    return result;
}

void
Process::ClearPreResumeActions ()
{
    m_pre_resume_actions.clear();
}

void
Process::Flush ()
{
    m_thread_list.Flush();
}
