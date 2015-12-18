//===-- LanguageRuntime.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/LanguageRuntime.h"
#include "Plugins/Language/CPlusPlus/CPlusPlusLanguage.h"
#include "Plugins/Language/ObjC/ObjCLanguage.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Target.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/SearchFilter.h"
#include "lldb/Interpreter/CommandInterpreter.h"

using namespace lldb;
using namespace lldb_private;

class ExceptionSearchFilter : public SearchFilter
{
public:
    ExceptionSearchFilter (const lldb::TargetSP &target_sp,
                           lldb::LanguageType language,
                           bool update_module_list = true) :
        SearchFilter (target_sp),
        m_language (language),
        m_language_runtime (NULL),
        m_filter_sp ()
    {
        if (update_module_list)
            UpdateModuleListIfNeeded ();
    }

    ~ExceptionSearchFilter() override = default;

    bool
    ModulePasses (const lldb::ModuleSP &module_sp) override
    {
        UpdateModuleListIfNeeded ();
        if (m_filter_sp)
            return m_filter_sp->ModulePasses (module_sp);
        return false;
    }
    
    bool
    ModulePasses (const FileSpec &spec) override
    {
        UpdateModuleListIfNeeded ();
        if (m_filter_sp)
            return m_filter_sp->ModulePasses (spec);
        return false;
    }
    
    void
    Search (Searcher &searcher) override
    {
        UpdateModuleListIfNeeded ();
        if (m_filter_sp)
            m_filter_sp->Search (searcher);
    }

    void
    GetDescription (Stream *s) override
    {
        UpdateModuleListIfNeeded ();
        if (m_filter_sp)
            m_filter_sp->GetDescription (s);
    }
    
protected:
    LanguageType m_language;
    LanguageRuntime *m_language_runtime;
    SearchFilterSP m_filter_sp;

    SearchFilterSP
    DoCopyForBreakpoint(Breakpoint &breakpoint) override
    {
        return SearchFilterSP(new ExceptionSearchFilter(TargetSP(), m_language, false));
    }

    void
    UpdateModuleListIfNeeded ()
    {
        ProcessSP process_sp (m_target_sp->GetProcessSP());
        if (process_sp)
        {
            bool refreash_filter = !m_filter_sp;
            if (m_language_runtime == NULL)
            {
                m_language_runtime = process_sp->GetLanguageRuntime(m_language);
                refreash_filter = true;
            }
            else
            {
                LanguageRuntime *language_runtime = process_sp->GetLanguageRuntime(m_language);
                if (m_language_runtime != language_runtime)
                {
                    m_language_runtime = language_runtime;
                    refreash_filter = true;
                }
            }
            
            if (refreash_filter && m_language_runtime)
            {
                m_filter_sp = m_language_runtime->CreateExceptionSearchFilter ();
            }
        }
        else
        {
            m_filter_sp.reset();
            m_language_runtime = NULL;
        }
    }
};

// The Target is the one that knows how to create breakpoints, so this function
// is meant to be used either by the target or internally in Set/ClearExceptionBreakpoints.
class ExceptionBreakpointResolver : public BreakpointResolver
{
public:
    ExceptionBreakpointResolver (lldb::LanguageType language,
                                 bool catch_bp,
                                 bool throw_bp) :
        BreakpointResolver (NULL, BreakpointResolver::ExceptionResolver),
        m_language (language),
        m_language_runtime (NULL),
        m_catch_bp (catch_bp),
        m_throw_bp (throw_bp)
    {
    }

    ~ExceptionBreakpointResolver() override = default;

    Searcher::CallbackReturn
    SearchCallback (SearchFilter &filter,
                    SymbolContext &context,
                    Address *addr,
                    bool containing) override
    {
        
        if (SetActualResolver())
            return m_actual_resolver_sp->SearchCallback (filter, context, addr, containing);
        else
            return eCallbackReturnStop;
    }
    
    Searcher::Depth
    GetDepth () override
    {
        if (SetActualResolver())
            return m_actual_resolver_sp->GetDepth();
        else
            return eDepthTarget;
    }
    
    void
    GetDescription (Stream *s) override
    {
       Language *language_plugin = Language::FindPlugin(m_language);
       if (language_plugin)
           language_plugin->GetExceptionResolverDescription(m_catch_bp, m_throw_bp, *s);
       else
           Language::GetDefaultExceptionResolverDescription(m_catch_bp, m_throw_bp, *s);
           
        SetActualResolver();
        if (m_actual_resolver_sp)
        {
            s->Printf (" using: ");
            m_actual_resolver_sp->GetDescription (s);
        }
        else
            s->Printf (" the correct runtime exception handler will be determined when you run");
    }

    void
    Dump (Stream *s) const override
    {
    }
    
    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const BreakpointResolverName *) { return true; }
    static inline bool classof(const BreakpointResolver *V) {
        return V->getResolverID() == BreakpointResolver::ExceptionResolver;
    }

protected:
    BreakpointResolverSP
    CopyForBreakpoint (Breakpoint &breakpoint) override
    {
        return BreakpointResolverSP(new ExceptionBreakpointResolver(m_language, m_catch_bp, m_throw_bp));
    }

    bool
    SetActualResolver()
    {
        ProcessSP process_sp;
        if (m_breakpoint)
        {
            process_sp = m_breakpoint->GetTarget().GetProcessSP();
            if (process_sp)
            {
                bool refreash_resolver = !m_actual_resolver_sp;
                if (m_language_runtime == NULL)
                {
                    m_language_runtime = process_sp->GetLanguageRuntime(m_language);
                    refreash_resolver = true;
                }
                else
                {
                    LanguageRuntime *language_runtime = process_sp->GetLanguageRuntime(m_language);
                    if (m_language_runtime != language_runtime)
                    {
                        m_language_runtime = language_runtime;
                        refreash_resolver = true;
                    }
                }
                
                if (refreash_resolver && m_language_runtime)
                {
                    m_actual_resolver_sp = m_language_runtime->CreateExceptionResolver (m_breakpoint, m_catch_bp, m_throw_bp);
                }
            }
            else
            {
                m_actual_resolver_sp.reset();
                m_language_runtime = NULL;
            }
        }
        else
        {
            m_actual_resolver_sp.reset();
            m_language_runtime = NULL;
        }
        return (bool)m_actual_resolver_sp;
    }
    lldb::BreakpointResolverSP m_actual_resolver_sp;
    lldb::LanguageType m_language;
    LanguageRuntime *m_language_runtime;
    bool m_catch_bp;
    bool m_throw_bp;
};

LanguageRuntime*
LanguageRuntime::FindPlugin (Process *process, lldb::LanguageType language)
{
    std::unique_ptr<LanguageRuntime> language_runtime_ap;
    LanguageRuntimeCreateInstance create_callback;

    for (uint32_t idx = 0;
         (create_callback = PluginManager::GetLanguageRuntimeCreateCallbackAtIndex(idx)) != NULL;
         ++idx)
    {
        language_runtime_ap.reset (create_callback(process, language));

        if (language_runtime_ap.get())
            return language_runtime_ap.release();
    }

    return NULL;
}

LanguageRuntime::LanguageRuntime(Process *process) :
    m_process (process)
{
}

LanguageRuntime::~LanguageRuntime() = default;

Breakpoint::BreakpointPreconditionSP
LanguageRuntime::CreateExceptionPrecondition (lldb::LanguageType language,
                                              bool catch_bp,
                                              bool throw_bp)
{
    switch (language)
    {
    case eLanguageTypeObjC:
        if (throw_bp)
            return Breakpoint::BreakpointPreconditionSP(new ObjCLanguageRuntime::ObjCExceptionPrecondition ());
        break;
    default:
        break;
    }
    return Breakpoint::BreakpointPreconditionSP();
}

BreakpointSP
LanguageRuntime::CreateExceptionBreakpoint (Target &target,
                                            lldb::LanguageType language,
                                            bool catch_bp,
                                            bool throw_bp,
                                            bool is_internal)
{
    BreakpointResolverSP resolver_sp(new ExceptionBreakpointResolver(language, catch_bp, throw_bp));
    SearchFilterSP filter_sp(new ExceptionSearchFilter(target.shared_from_this(), language));
    bool hardware = false;
    bool resolve_indirect_functions = false;
    BreakpointSP exc_breakpt_sp (target.CreateBreakpoint (filter_sp, resolver_sp, is_internal, hardware, resolve_indirect_functions));
    if (exc_breakpt_sp)
    {
        Breakpoint::BreakpointPreconditionSP precondition_sp = CreateExceptionPrecondition(language, catch_bp, throw_bp);
        if (precondition_sp)
            exc_breakpt_sp->SetPrecondition(precondition_sp);

        if (is_internal)
            exc_breakpt_sp->SetBreakpointKind("exception");
    }
    
    return exc_breakpt_sp;
}

void
LanguageRuntime::InitializeCommands (CommandObject* parent)
{
    if (!parent)
        return;

    if (!parent->IsMultiwordObject())
        return;

    LanguageRuntimeCreateInstance create_callback;

    for (uint32_t idx = 0;
         (create_callback = PluginManager::GetLanguageRuntimeCreateCallbackAtIndex(idx)) != nullptr;
         ++idx)
    {
        if (LanguageRuntimeGetCommandObject command_callback = 
                PluginManager::GetLanguageRuntimeGetCommandObjectAtIndex(idx))
        {
            CommandObjectSP command = command_callback(parent->GetCommandInterpreter());
            if (command)
            {
                parent->LoadSubCommand(command->GetCommandName(), command);
            }
        }
    }
}

lldb::SearchFilterSP
LanguageRuntime::CreateExceptionSearchFilter ()
{
    return m_process->GetTarget().GetSearchFilterForModule(NULL);
}
