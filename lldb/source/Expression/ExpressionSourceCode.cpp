//===-- ExpressionSourceCode.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/ExpressionSourceCode.h"

#include "lldb/Core/StreamString.h"

using namespace lldb_private;

static const char *global_defines = "#undef NULL                       \n"
                                    "#undef Nil                        \n"
                                    "#undef nil                        \n"
                                    "#undef YES                        \n"
                                    "#undef NO                         \n"
                                    "#define NULL ((int)0)             \n"
                                    "#define Nil ((Class)0)            \n"
                                    "#define nil ((id)0)               \n"
                                    "#define YES ((BOOL)1)             \n"
                                    "#define NO ((BOOL)0)              \n"
                                    "typedef int BOOL;                 \n"
                                    "typedef unsigned short unichar;   \n";


bool ExpressionSourceCode::GetText (std::string &text, lldb::LanguageType wrapping_language, bool const_object, bool static_method) const
{
    if (m_wrap)
    {
        switch (wrapping_language) 
        {
        default:
            return false;
        case lldb::eLanguageTypeC:
        case lldb::eLanguageTypeC_plus_plus:
        case lldb::eLanguageTypeObjC:
            break;
        }
        
        StreamString wrap_stream;
        
        switch (wrapping_language) 
        {
        default:
            break;
        case lldb::eLanguageTypeC:
            wrap_stream.Printf("%s                             \n"
                               "%s                             \n"
                               "void                           \n"
                               "%s(void *$__lldb_arg)          \n"
                               "{                              \n"
                               "    %s;                        \n" 
                               "}                              \n",
                               m_prefix.c_str(),
                               global_defines,
                               m_name.c_str(),
                               m_body.c_str());
            break;
        case lldb::eLanguageTypeC_plus_plus:
            wrap_stream.Printf("%s                                     \n"
                               "%s                                     \n"
                               "void                                   \n"
                               "$__lldb_class::%s(void *$__lldb_arg) %s\n"
                               "{                                      \n"
                               "    %s;                                \n" 
                               "}                                      \n",
                               m_prefix.c_str(),
                               global_defines,
                               m_name.c_str(),
                               (const_object ? "const" : ""),
                               m_body.c_str());
            break;
        case lldb::eLanguageTypeObjC:
            if (static_method)
            {
                wrap_stream.Printf("%s                                                      \n"
                                   "%s                                                      \n"
                                   "@interface $__lldb_objc_class ($__lldb_category)        \n"
                                   "+(void)%s:(void *)$__lldb_arg;                          \n"
                                   "@end                                                    \n"
                                   "@implementation $__lldb_objc_class ($__lldb_category)   \n"
                                   "+(void)%s:(void *)$__lldb_arg                           \n"
                                   "{                                                       \n"
                                   "    %s;                                                 \n"
                                   "}                                                       \n"
                                   "@end                                                    \n",
                                   m_prefix.c_str(),
                                   global_defines,
                                   m_name.c_str(),
                                   m_name.c_str(),
                                   m_body.c_str());
            }
            else
            {
                wrap_stream.Printf("%s                                                     \n"
                                   "%s                                                     \n"
                                   "@interface $__lldb_objc_class ($__lldb_category)       \n"
                                   "-(void)%s:(void *)$__lldb_arg;                         \n"
                                   "@end                                                   \n"
                                   "@implementation $__lldb_objc_class ($__lldb_category)  \n"
                                   "-(void)%s:(void *)$__lldb_arg                          \n"
                                   "{                                                      \n"
                                   "    %s;                                                \n"
                                   "}                                                      \n"
                                   "@end                                                   \n",
                                   m_prefix.c_str(),
                                   global_defines,
                                   m_name.c_str(),
                                   m_name.c_str(),
                                   m_body.c_str());
            }
            break;
        }
        
        text = wrap_stream.GetString();
    }
    else
    {
        text.append(m_body);
    }
    
    return true;
}
