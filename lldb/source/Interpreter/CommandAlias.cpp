//===-- CommandAlias.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/CommandAlias.h"

#include "lldb/Core/StreamString.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/Options.h"

using namespace lldb;
using namespace lldb_private;

static bool
ProcessAliasOptionsArgs (lldb::CommandObjectSP &cmd_obj_sp,
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
            const size_t argc = args.GetArgumentCount();
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

CommandAlias::UniquePointer
CommandAlias::GetCommandAlias (lldb::CommandObjectSP cmd_sp,
                               const char *options_args)
{
    CommandAlias::UniquePointer ret_val(nullptr);
    OptionArgVectorSP opt_args_sp(new OptionArgVector);
    if (ProcessAliasOptionsArgs(cmd_sp, options_args, opt_args_sp))
        ret_val.reset(new CommandAlias(cmd_sp, opt_args_sp));
    return ret_val;
}

CommandAlias::CommandAlias (lldb::CommandObjectSP cmd_sp,
                            OptionArgVectorSP args_sp) :
m_underlying_command_sp(cmd_sp),
m_option_args_sp(args_sp)
{
}

void
CommandAlias::GetAliasExpansion (StreamString &help_string)
{
    const char* command_name = m_underlying_command_sp->GetCommandName();
    help_string.Printf ("'%s", command_name);
    
    if (m_option_args_sp)
    {
        OptionArgVector *options = m_option_args_sp.get();
        for (size_t i = 0; i < options->size(); ++i)
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
