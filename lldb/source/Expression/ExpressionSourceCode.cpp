//===-- ExpressionSourceCode.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/ExpressionSourceCode.h"

#include <algorithm>

#include "lldb/Core/StreamString.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "Plugins/ExpressionParser/Clang/ClangModulesDeclVendor.h"
#include "Plugins/ExpressionParser/Clang/ClangPersistentVariables.h"
#include "Plugins/ExpressionParser/Swift/SwiftASTManipulator.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"

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

uint32_t
ExpressionSourceCode::GetNumBodyLines ()
{
    if (m_num_body_lines == 0)
        m_num_body_lines = 1 + std::count(m_body.begin(), m_body.end(), '\n');
    return m_num_body_lines;
}

bool
ExpressionSourceCode::SaveExpressionTextToTempFile (const char *text, const EvaluateExpressionOptions &options, std::string &expr_source_path)
{
    bool success = false;

    const uint32_t expr_number = options.GetExpressionNumber();
    FileSpec tmpdir_file_spec;
    
    const bool playground = options.GetPlaygroundTransformEnabled();
    const bool repl = options.GetREPLEnabled();

    const char *file_prefix = NULL;
    if (playground)
        file_prefix = "playground";
    else if (repl)
        file_prefix = "repl";
    else
        file_prefix = "expr";
    
    StreamString strm;
    if (HostInfo::GetLLDBPath (lldb::ePathTypeLLDBTempSystemDir, tmpdir_file_spec))
    {
        strm.Printf("%s%u", file_prefix, expr_number);
        tmpdir_file_spec.GetFilename().SetCStringWithLength(strm.GetString().c_str(), strm.GetString().size());
        expr_source_path = std::move(tmpdir_file_spec.GetPath());
    }
    else
    {
        strm.Printf("/tmp/%s%u", file_prefix, expr_number);
        expr_source_path = std::move(strm.GetString());
    }
    
    switch (options.GetLanguage())
    {
        default:
            expr_source_path.append(".cpp");
            break;
            
        case lldb::eLanguageTypeSwift:
            expr_source_path.append(".swift");
            break;
    }
    
    int temp_fd = mkstemp(&expr_source_path[0]);
    if (temp_fd != -1)
    {
        lldb_private::File file (temp_fd, true);
        const size_t text_len = strlen(text);
        size_t bytes_written = text_len;
        if (file.Write(text, bytes_written).Success())
        {
            if (bytes_written == text_len)
            {
                // Make sure we have a newline in the file at the end
                bytes_written = 1;
                file.Write("\n", bytes_written);
                if (bytes_written == 1)
                    success = true;
            }
        }
        if (!success)
            FileSystem::Unlink(FileSpec(expr_source_path.c_str(), true));
    }
    if (!success)
        expr_source_path.clear();
    return success;
}

bool
ExpressionSourceCode::GetText (std::string &text,
                               lldb::LanguageType wrapping_language,
                               bool const_object,
                               bool swift_instance_method,
                               bool static_method,
                               bool is_swift_class,
                               const EvaluateExpressionOptions &options,
                               const Expression::SwiftGenericInfo &generic_info,
                               ExecutionContext &exe_ctx,
                               uint32_t &first_body_line) const
{
    first_body_line = 0;

    const char *target_specific_defines = "typedef signed char BOOL;\n";
    std::string module_macros;
    
    if (ClangModulesDeclVendor::LanguageSupportsClangModules(wrapping_language))
    {
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
            
            ClangPersistentVariables *persistent_vars = llvm::dyn_cast_or_null<ClangPersistentVariables>(target->GetPersistentExpressionStateForLanguage(lldb::eLanguageTypeC));
            ClangModulesDeclVendor *decl_vendor = target->GetClangModulesDeclVendor();
            
            if (persistent_vars && decl_vendor)
            {
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
    }
    
    if (m_wrap)
    {
        const char *body = m_body.c_str();
        const char *pound_file = options.GetPoundLineFilePath();
        const uint32_t pound_line = options.GetPoundLineLine();
        StreamString pound_body;
        if (pound_file && pound_line)
        {
            pound_body.Printf("#line %u \"%s\"\n%s", pound_line, pound_file, body);
            body = pound_body.GetString().c_str();
        }

        switch (wrapping_language)
        {
        default:
            return false;
        case lldb::eLanguageTypeC:
        case lldb::eLanguageTypeC_plus_plus:
        case lldb::eLanguageTypeObjC:
        case lldb::eLanguageTypeSwift:
            break;
        }
        
        StreamString wrap_stream;

        if (ClangModulesDeclVendor::LanguageSupportsClangModules(wrapping_language))
        {
            wrap_stream.Printf("%s\n", module_macros.c_str());
            wrap_stream.Printf("%s\n%s\n",
                               g_expression_prefix,
                               target_specific_defines);
            wrap_stream.Printf("%s\n", m_prefix.c_str());
        }
        
        switch (wrapping_language) 
        {
        default:
            break;
        case lldb::eLanguageTypeC:
            wrap_stream.Printf("void                           \n"
                               "%s(void *$__lldb_arg)          \n"
                               "{                              \n"
                               "%s;                            \n"
                               "}                              \n",
                               m_name.c_str(),
                               body);
            break;
        case lldb::eLanguageTypeC_plus_plus:
            wrap_stream.Printf("void                                   \n"
                               "$__lldb_class::%s(void *$__lldb_arg) %s\n"
                               "{                                      \n"
                               "%s;                                    \n"
                               "}                                      \n",
                               m_name.c_str(),
                               (const_object ? "const" : ""),
                               body);
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
                                   "%s;                                                     \n"
                                   "}                                                       \n"
                                   "@end                                                    \n",
                                   m_name.c_str(),
                                   m_name.c_str(),
                                   body);
            }
            else
            {
                wrap_stream.Printf("@interface $__lldb_objc_class ($__lldb_category)       \n"
                                   "-(void)%s:(void *)$__lldb_arg;                         \n"
                                   "@end                                                   \n"
                                   "@implementation $__lldb_objc_class ($__lldb_category)  \n"
                                   "-(void)%s:(void *)$__lldb_arg                          \n"
                                   "{                                                      \n"
                                   "%s;                                                    \n"
                                   "}                                                      \n"
                                   "@end                                                   \n",
                                   m_name.c_str(),
                                   m_name.c_str(),
                                   body);
            }
            break;
        case lldb::eLanguageTypeSwift:
            {
                SwiftASTManipulator::WrapExpression (wrap_stream,
                                                     m_body.c_str(),
                                                     swift_instance_method,
                                                     static_method,
                                                     is_swift_class,
                                                     options,
                                                     generic_info,
                                                     first_body_line);
            }
        }
        
        text = wrap_stream.GetString();
    }
    else
    {
        text.append(m_body);
    }
    
    return true;
}
