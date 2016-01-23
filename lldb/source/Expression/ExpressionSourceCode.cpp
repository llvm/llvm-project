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
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/DebugMacros.h"
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

namespace {

class AddMacroState
{
    enum State
    {
        CURRENT_FILE_NOT_YET_PUSHED,
        CURRENT_FILE_PUSHED,
        CURRENT_FILE_POPPED
    };

public:
    AddMacroState(const FileSpec &current_file, const uint32_t current_file_line)
        : m_state(CURRENT_FILE_NOT_YET_PUSHED),
          m_current_file(current_file),
          m_current_file_line(current_file_line)
    { }

    void
    StartFile(const FileSpec &file)
    {
        m_file_stack.push_back(file);
        if (file == m_current_file)
            m_state = CURRENT_FILE_PUSHED;
    }

    void
    EndFile()
    {
        if (m_file_stack.size() == 0)
            return;

        FileSpec old_top = m_file_stack.back();
        m_file_stack.pop_back();
        if (old_top == m_current_file)
            m_state = CURRENT_FILE_POPPED;
    }

    // An entry is valid if it occurs before the current line in
    // the current file.
    bool
    IsValidEntry(uint32_t line)
    {
        switch (m_state)
        {
            case CURRENT_FILE_NOT_YET_PUSHED:
                return true;
            case CURRENT_FILE_PUSHED:
                // If we are in file included in the current file,
                // the entry should be added.
                if (m_file_stack.back() != m_current_file)
                    return true;

                if (line >= m_current_file_line)
                    return false;
                else
                    return true;
            default:
                return false;
        }
        llvm_unreachable("unhandled state");
    }

private:
    std::vector<FileSpec> m_file_stack;
    State m_state;
    FileSpec m_current_file;
    uint32_t m_current_file_line;
};

} // anonymous namespace

static void
AddMacros(const DebugMacros *dm, CompileUnit *comp_unit, AddMacroState &state, StreamString &stream)
{
    if (dm == nullptr)
        return;

    for (size_t i = 0; i < dm->GetNumMacroEntries(); i++)
    {
        const DebugMacroEntry &entry = dm->GetMacroEntryAtIndex(i);
        uint32_t line;

        switch (entry.GetType())
        {
            case DebugMacroEntry::DEFINE:
                if (state.IsValidEntry(entry.GetLineNumber()))
                    stream.Printf("#define %s\n", entry.GetMacroString().AsCString());
                else
                    return;
                break;
            case DebugMacroEntry::UNDEF:
                if (state.IsValidEntry(entry.GetLineNumber()))
                    stream.Printf("#undef %s\n", entry.GetMacroString().AsCString());
                else
                    return;
                break;
            case DebugMacroEntry::START_FILE:
                line = entry.GetLineNumber();
                if (state.IsValidEntry(line))
                    state.StartFile(entry.GetFileSpec(comp_unit));
                else
                    return;
                break;
            case DebugMacroEntry::END_FILE:
                state.EndFile();
                break;
            case DebugMacroEntry::INDIRECT:
                AddMacros(entry.GetIndirectDebugMacros(), comp_unit, state, stream);
                break;
            default:
                // This is an unknown/invalid entry. Ignore.
                break;
        }
    }
}

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

    StreamString debug_macros_stream;
    if (StackFrame *frame = exe_ctx.GetFramePtr())
    {
        const SymbolContext &sc = frame->GetSymbolContext(
           lldb:: eSymbolContextCompUnit | lldb::eSymbolContextLineEntry);

        if (sc.comp_unit && sc.line_entry.IsValid())
        {
            DebugMacros *dm = sc.comp_unit->GetDebugMacros();
            if (dm)
            {
                AddMacroState state(sc.line_entry.file, sc.line_entry.line);
                AddMacros(dm, sc.comp_unit, state, debug_macros_stream);
            }
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
        
        wrap_stream.Printf("%s\n%s\n%s\n%s\n%s\n",
                           module_macros.c_str(),
                           debug_macros_stream.GetData(),
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
