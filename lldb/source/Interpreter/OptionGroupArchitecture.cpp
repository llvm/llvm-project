//===-- OptionGroupArchitecture.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupArchitecture.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Utility/Utils.h"

using namespace lldb;
using namespace lldb_private;

OptionGroupArchitecture::OptionGroupArchitecture() :
    m_arch_str ()
{
}

OptionGroupArchitecture::~OptionGroupArchitecture ()
{
}

static OptionDefinition
g_option_table[] =
{
    { LLDB_OPT_SET_1 , false, "arch"    , 'a', OptionParser::eRequiredArgument, nullptr, nullptr, 0, eArgTypeArchitecture , "Specify the architecture for the target."},
};

uint32_t
OptionGroupArchitecture::GetNumDefinitions ()
{
    return llvm::array_lengthof(g_option_table);
}

const OptionDefinition *
OptionGroupArchitecture::GetDefinitions ()
{
    return g_option_table;
}

bool
OptionGroupArchitecture::GetArchitecture (Platform *platform, ArchSpec &arch)
{
    if (m_arch_str.empty())
        arch.Clear();
    else
        arch.SetTriple(m_arch_str.c_str(), platform);
    return arch.IsValid();
}


Error
OptionGroupArchitecture::SetOptionValue(uint32_t option_idx,
                                        const char *option_arg,
                                        ExecutionContext *execution_context)
{
    Error error;
    const int short_option = g_option_table[option_idx].short_option;

    switch (short_option)
    {
        case 'a':
            m_arch_str.assign (option_arg);
            break;

        default:
            error.SetErrorStringWithFormat ("unrecognized option '%c'", short_option);
            break;
    }

    return error;
}

void
OptionGroupArchitecture::OptionParsingStarting(
                                            ExecutionContext *execution_context)
{
    m_arch_str.clear();
}

