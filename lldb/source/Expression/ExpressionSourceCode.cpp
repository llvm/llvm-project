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

const char *
ExpressionSourceCode::g_expression_prefix = R"(
#undef NULL
#undef Nil
#undef nil
#undef YES
#undef NO
#define NULL (__null)
#define Nil (__null)
#define nil (__null)
#define YES ((BOOL)1)
#define NO ((BOOL)0)
typedef signed char BOOL;
typedef signed __INT8_TYPE__ int8_t;
typedef unsigned __INT8_TYPE__ uint8_t;
typedef signed __INT16_TYPE__ int16_t;
typedef unsigned __INT16_TYPE__ uint16_t;
typedef signed __INT32_TYPE__ int32_t;
typedef unsigned __INT32_TYPE__ uint32_t;
typedef signed __INT64_TYPE__ int64_t;
typedef unsigned __INT64_TYPE__ uint64_t;
typedef signed __INTPTR_TYPE__ intptr_t;
typedef unsigned __INTPTR_TYPE__ uintptr_t;
typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef unsigned short unichar;
)";


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
                               g_expression_prefix,
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
                               g_expression_prefix,
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
                                   g_expression_prefix,
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
                                   g_expression_prefix,
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
