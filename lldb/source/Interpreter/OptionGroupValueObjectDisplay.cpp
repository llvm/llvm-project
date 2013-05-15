//===-- OptionGroupValueObjectDisplay.cpp -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/Interpreter/OptionGroupValueObjectDisplay.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Target.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Utility/Utils.h"

using namespace lldb;
using namespace lldb_private;

OptionGroupValueObjectDisplay::OptionGroupValueObjectDisplay()
{
}

OptionGroupValueObjectDisplay::~OptionGroupValueObjectDisplay ()
{
}

static OptionDefinition
g_option_table[] =
{
    { LLDB_OPT_SET_1, false, "dynamic-type",       'd', required_argument, g_dynamic_value_types, 0, eArgTypeNone,      "Show the object as its full dynamic type, not its static type, if available."},
    { LLDB_OPT_SET_1, false, "synthetic-type",     'S', required_argument, NULL, 0, eArgTypeBoolean,   "Show the object obeying its synthetic provider, if available."},
    { LLDB_OPT_SET_1, false, "depth",              'D', required_argument, NULL, 0, eArgTypeCount,     "Set the max recurse depth when dumping aggregate types (default is infinity)."},
    { LLDB_OPT_SET_1, false, "flat",               'F', no_argument,       NULL, 0, eArgTypeNone,      "Display results in a flat format that uses expression paths for each variable or member."},
    { LLDB_OPT_SET_1, false, "location",           'L', no_argument,       NULL, 0, eArgTypeNone,      "Show variable location information."},
    { LLDB_OPT_SET_1, false, "object-description", 'O', no_argument,       NULL, 0, eArgTypeNone,      "Print as an Objective-C object."},
    { LLDB_OPT_SET_1, false, "ptr-depth",          'P', required_argument, NULL, 0, eArgTypeCount,     "The number of pointers to be traversed when dumping values (default is zero)."},
    { LLDB_OPT_SET_1, false, "show-types",         'T', no_argument,       NULL, 0, eArgTypeNone,      "Show variable types when dumping values."},
    { LLDB_OPT_SET_1, false, "no-summary-depth",   'Y', optional_argument, NULL, 0, eArgTypeCount,     "Set the depth at which omitting summary information stops (default is 1)."},
    { LLDB_OPT_SET_1, false, "raw-output",         'R', no_argument,       NULL, 0, eArgTypeNone,      "Don't use formatting options."},
    { LLDB_OPT_SET_1, false, "show-all-children",  'A', no_argument,       NULL, 0, eArgTypeNone,      "Ignore the upper bound on the number of children to show."},
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

uint32_t
OptionGroupValueObjectDisplay::GetNumDefinitions ()
{
    return llvm::array_lengthof(g_option_table);
}

const OptionDefinition *
OptionGroupValueObjectDisplay::GetDefinitions ()
{
    return g_option_table;
}


Error
OptionGroupValueObjectDisplay::SetOptionValue (CommandInterpreter &interpreter,
                                               uint32_t option_idx,
                                               const char *option_arg)
{
    Error error;
    const int short_option = g_option_table[option_idx].short_option;
    bool success = false;

    switch (short_option)
    {
        case 'd':
            {
                int32_t result;
                result = Args::StringToOptionEnum (option_arg, g_dynamic_value_types, 2, error);
                if (error.Success())
                    use_dynamic = (lldb::DynamicValueType) result;
            }
            break;
        case 'T':   show_types   = true;  break;
        case 'L':   show_location= true;  break;
        case 'F':   flat_output  = true;  break;
        case 'O':   use_objc     = true;  break;
        case 'R':   be_raw       = true;  break;
        case 'A':   ignore_cap   = true;  break;
            
        case 'D':
            max_depth = Args::StringToUInt32 (option_arg, UINT32_MAX, 0, &success);
            if (!success)
                error.SetErrorStringWithFormat("invalid max depth '%s'", option_arg);
            break;
            
        case 'P':
            ptr_depth = Args::StringToUInt32 (option_arg, 0, 0, &success);
            if (!success)
                error.SetErrorStringWithFormat("invalid pointer depth '%s'", option_arg);
            break;
            
        case 'Y':
            if (option_arg)
            {
                no_summary_depth = Args::StringToUInt32 (option_arg, 0, 0, &success);
                if (!success)
                    error.SetErrorStringWithFormat("invalid pointer depth '%s'", option_arg);
            }
            else
                no_summary_depth = 1;
            break;
            
        case 'S':
            use_synth = Args::StringToBoolean(option_arg, true, &success);
            if (!success)
                error.SetErrorStringWithFormat("invalid synthetic-type '%s'", option_arg);
            break;
        default:
            error.SetErrorStringWithFormat ("unrecognized option '%c'", short_option);
            break;
    }

    return error;
}

void
OptionGroupValueObjectDisplay::OptionParsingStarting (CommandInterpreter &interpreter)
{
    // If these defaults change, be sure to modify AnyOptionWasSet().
    show_types        = false;
    no_summary_depth  = 0;
    show_location     = false;
    flat_output       = false;
    use_objc          = false;
    max_depth         = UINT32_MAX;
    ptr_depth         = 0;
    use_synth         = true;
    be_raw            = false;
    ignore_cap        = false;
    
    Target *target = interpreter.GetExecutionContext().GetTargetPtr();
    if (target != NULL)
        use_dynamic = target->GetPreferDynamicValue();
    else
    {
        // If we don't have any targets, then dynamic values won't do us much good.
        use_dynamic = lldb::eNoDynamicValues;
    }
}

ValueObject::DumpValueObjectOptions
OptionGroupValueObjectDisplay::GetAsDumpOptions (bool objc_is_compact,
                                                 lldb::Format format,
                                                 lldb::TypeSummaryImplSP summary_sp)
{
    ValueObject::DumpValueObjectOptions options;
    options.SetMaximumPointerDepth(ptr_depth);
    if (use_objc)
        options.SetShowSummary(false);
    else
        options.SetOmitSummaryDepth(no_summary_depth);
    options.SetMaximumDepth(max_depth)
    .SetShowTypes(show_types)
    .SetShowLocation(show_location)
    .SetUseObjectiveC(use_objc)
    .SetUseDynamicType(use_dynamic)
    .SetUseSyntheticValue(use_synth)
    .SetFlatOutput(flat_output)
    .SetIgnoreCap(ignore_cap)
    .SetFormat(format)
    .SetSummary(summary_sp);
    
    if (objc_is_compact)
        options.SetHideRootType(use_objc)
        .SetHideName(use_objc)
        .SetHideValue(use_objc);
    
    if (be_raw)
        options.SetRawDisplay(true);

    return options;
}
