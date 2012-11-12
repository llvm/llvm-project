//===-- fooplugin.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

/*
An example plugin for LLDB that provides a new foo command with a child subcommand
Compile this into a dylib foo.dylib and load by placing in appropriate locations on disk or
by typing plugin load foo.dylib at the LLDB command line
*/

#include <LLDB/SBCommandInterpreter.h>
#include <LLDB/SBCommandReturnObject.h>
#include <LLDB/SBDebugger.h>

namespace lldb {
    bool
    PluginInitialize (lldb::SBDebugger debugger);
}

class ChildCommand : public lldb::SBCommandPluginInterface
{
public:
    virtual bool
    DoExecute (lldb::SBDebugger debugger,
               char** command,
               lldb::SBCommandReturnObject &result)
    {
        if (command)
        {
            const char* arg = *command;
            while (arg)
            {
                result.Printf("%s ",arg);
                arg = *(++command);
            }
            result.Printf("\n");
            return true;
        }
        return false;
    }
    
};

bool
lldb::PluginInitialize (lldb::SBDebugger debugger)
{
    lldb::SBCommandInterpreter interpreter = debugger.GetCommandInterpreter();
    lldb::SBCommand foo = interpreter.AddMultiwordCommand("plugin_loaded_command",NULL);
    foo.AddCommand("child",new ChildCommand(),"a child of plugin_loaded_command");
    return true;
}
