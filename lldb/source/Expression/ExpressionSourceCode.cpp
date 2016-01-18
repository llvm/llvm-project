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
#include "Plugins/ExpressionParser/Clang/ClangModulesDeclVendor.h"
#include "Plugins/ExpressionParser/Clang/ClangPersistentVariables.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"

using namespace lldb_private;

const char *
ExpressionSourceCode::g_expression_prefix = R"(
#ifndef NULL
#define NULL (__null)
#endif
#ifndef Nil
#define Nil (__null)
#endif
#ifndef nil
#define nil (__null)
#endif
#ifndef YES
#define YES ((BOOL)1)
#endif
#ifndef NO
#define NO ((BOOL)0)
#endif
typedef __INT8_TYPE__ int8_t;
typedef __UINT8_TYPE__ uint8_t;
typedef __INT16_TYPE__ int16_t;
typedef __UINT16_TYPE__ uint16_t;
typedef __INT32_TYPE__ int32_t;
typedef __UINT32_TYPE__ uint32_t;
typedef __INT64_TYPE__ int64_t;
typedef __UINT64_TYPE__ uint64_t;
typedef __INTPTR_TYPE__ intptr_t;
typedef __UINTPTR_TYPE__ uintptr_t;
typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef unsigned short unichar;
extern "C"
{
    int printf(const char * __restrict, ...);
}
)";


bool ExpressionSourceCode::GetText (std::string &text, lldb::LanguageType wrapping_language, bool const_object, bool static_method, ExecutionContext &exe_ctx) const
{
    const char *target_specific_defines = "typedef signed char BOOL;\n";
    std::string module_macros;
    
    if (Target *target = exe_ctx.GetTargetPtr())
    {
        if (target->GetArchitecture().GetMachine() == llvm::Triple::aarch64)
        {
            target_specific_defines = "typedef bool BOOL;\n";
        }
        if (target->GetArchitecture().GetMachine() == llvm::Triple::x86_64)
        {
            if (lldb::PlatformSP platform_sp = target->GetPlatform())
            {
                static ConstString g_platform_ios_simulator ("ios-simulator");
                if (platform_sp->GetPluginName() == g_platform_ios_simulator)
                {
                    target_specific_defines = "typedef bool BOOL;\n";
                }
            }
        }
        
        if (ClangModulesDeclVendor *decl_vendor = target->GetClangModulesDeclVendor())
        {
            ClangPersistentVariables *persistent_vars = llvm::cast<ClangPersistentVariables>(target->GetPersistentExpressionStateForLanguage(lldb::eLanguageTypeC));
            const ClangModulesDeclVendor::ModuleVector &hand_imported_modules = persistent_vars->GetHandLoadedClangModules();
            ClangModulesDeclVendor::ModuleVector modules_for_macros;
            
            for (ClangModulesDeclVendor::ModuleID module : hand_imported_modules)
            {
                modules_for_macros.push_back(module);
            }
            
            if (target->GetEnableAutoImportClangModules())
            {
                if (StackFrame *frame = exe_ctx.GetFramePtr())
                {
                    if (Block *block = frame->GetFrameBlock())
                    {
                        SymbolContext sc;
                        
                        block->CalculateSymbolContext(&sc);
                        
                        if (sc.comp_unit)
                        {
                            StreamString error_stream;
                            
                            decl_vendor->AddModulesForCompileUnit(*sc.comp_unit, modules_for_macros, error_stream);
                        }
                    }
                }
            }
            
            decl_vendor->ForEachMacro(modules_for_macros, [&module_macros] (const std::string &expansion) -> bool {
                module_macros.append(expansion);
                module_macros.append("\n");
                return false;
            });
        }

    }
    
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
        
        wrap_stream.Printf("%s\n%s\n%s\n%s\n",
                           module_macros.c_str(),
                           g_expression_prefix,
                           target_specific_defines,
                           m_prefix.c_str());
        
        switch (wrapping_language) 
        {
        default:
            break;
        case lldb::eLanguageTypeC:
            wrap_stream.Printf("void                           \n"
                               "%s(void *$__lldb_arg)          \n"
                               "{                              \n"
                               "    %s;                        \n" 
                               "}                              \n",
                               m_name.c_str(),
                               m_body.c_str());
            break;
        case lldb::eLanguageTypeC_plus_plus:
            wrap_stream.Printf("void                                   \n"
                               "$__lldb_class::%s(void *$__lldb_arg) %s\n"
                               "{                                      \n"
                               "    %s;                                \n" 
                               "}                                      \n",
                               m_name.c_str(),
                               (const_object ? "const" : ""),
                               m_body.c_str());
            break;
        case lldb::eLanguageTypeObjC:
            if (static_method)
            {
                wrap_stream.Printf("@interface $__lldb_objc_class ($__lldb_category)        \n"
                                   "+(void)%s:(void *)$__lldb_arg;                          \n"
                                   "@end                                                    \n"
                                   "@implementation $__lldb_objc_class ($__lldb_category)   \n"
                                   "+(void)%s:(void *)$__lldb_arg                           \n"
                                   "{                                                       \n"
                                   "    %s;                                                 \n"
                                   "}                                                       \n"
                                   "@end                                                    \n",
                                   m_name.c_str(),
                                   m_name.c_str(),
                                   m_body.c_str());
            }
            else
            {
                wrap_stream.Printf("@interface $__lldb_objc_class ($__lldb_category)       \n"
                                   "-(void)%s:(void *)$__lldb_arg;                         \n"
                                   "@end                                                   \n"
                                   "@implementation $__lldb_objc_class ($__lldb_category)  \n"
                                   "-(void)%s:(void *)$__lldb_arg                          \n"
                                   "{                                                      \n"
                                   "    %s;                                                \n"
                                   "}                                                      \n"
                                   "@end                                                   \n",
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
