//===-- LanguageRuntime.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/Target.h"
#include "lldb/Core/PluginManager.h"

using namespace lldb;
using namespace lldb_private;

LanguageRuntime*
LanguageRuntime::FindPlugin (Process *process, lldb::LanguageType language)
{
    std::auto_ptr<LanguageRuntime> language_runtime_ap;
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

//----------------------------------------------------------------------
// Constructor
//----------------------------------------------------------------------
LanguageRuntime::LanguageRuntime(Process *process) :
    m_process (process)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
LanguageRuntime::~LanguageRuntime()
{
}

BreakpointSP
LanguageRuntime::CreateExceptionBreakpoint(
    Target &target, 
    lldb::LanguageType language, 
    bool catch_bp, 
    bool throw_bp, 
    bool is_internal)
{
    BreakpointSP exc_breakpt_sp;
    BreakpointResolverSP resolver_sp(new ExceptionBreakpointResolver(NULL, language, catch_bp, throw_bp));
    SearchFilterSP filter_sp(target.GetSearchFilterForModule(NULL));
    
    exc_breakpt_sp = target.CreateBreakpoint (filter_sp, resolver_sp, is_internal);
    
    return exc_breakpt_sp;
}

LanguageRuntime::ExceptionBreakpointResolver::ExceptionBreakpointResolver (Breakpoint *bkpt,
                        LanguageType language,
                        bool catch_bp,
                        bool throw_bp) :
    BreakpointResolver (bkpt, BreakpointResolver::ExceptionResolver),
    m_language (language),
    m_catch_bp (catch_bp),
    m_throw_bp (throw_bp)

{
}
                        
void
LanguageRuntime::ExceptionBreakpointResolver::GetDescription (Stream *s)
{
    s->Printf ("Exception breakpoint (catch: %s throw: %s)", 
           m_catch_bp ? "on" : "off",
           m_throw_bp ? "on" : "off");
       
    SetActualResolver();
    if (m_actual_resolver_sp)
    {
        s->Printf (" using: ");
        m_actual_resolver_sp->GetDescription (s);
    }
    else
        s->Printf (" the correct runtime exception handler will be determined when you run");
}

bool
LanguageRuntime::ExceptionBreakpointResolver::SetActualResolver()
{
    ProcessSP process_sp = m_process_wp.lock();
    
    // See if our process weak pointer is still good:
    if (!process_sp)
    {
        // If not, our resolver is no good, so chuck that.  Then see if we can get the 
        // target's new process.
        m_actual_resolver_sp.reset();
        if (m_breakpoint)
        {
            Target &target = m_breakpoint->GetTarget();
            process_sp = target.GetProcessSP();
            if (process_sp)
            {
                m_process_wp = process_sp;
                process_sp = m_process_wp.lock();
            }
        }
    }
    
    if (process_sp)
    {
        if (m_actual_resolver_sp)
            return true;
        else
        {
            // If we have a process but not a resolver, set one now.
            LanguageRuntime *runtime = process_sp->GetLanguageRuntime(m_language);
            if (runtime)
            {
                m_actual_resolver_sp = runtime->CreateExceptionResolver (m_breakpoint, m_catch_bp, m_throw_bp);
                return (bool) m_actual_resolver_sp;
            }
            else
                return false;
        }
    }
    else
        return false;
}

Searcher::CallbackReturn
LanguageRuntime::ExceptionBreakpointResolver::SearchCallback (SearchFilter &filter,
                SymbolContext &context,
                Address *addr,
                bool containing)
{
    
    if (!SetActualResolver())
    {
        return eCallbackReturnStop;
    }
    else
        return m_actual_resolver_sp->SearchCallback (filter, context, addr, containing);
}

Searcher::Depth
LanguageRuntime::ExceptionBreakpointResolver::GetDepth ()
{
    if (!SetActualResolver())
        return eDepthTarget;
    else
        return m_actual_resolver_sp->GetDepth();
}

/*
typedef enum LanguageType
{
    eLanguageTypeUnknown         = 0x0000,   ///< Unknown or invalid language value.
    eLanguageTypeC89             = 0x0001,   ///< ISO C:1989.
    eLanguageTypeC               = 0x0002,   ///< Non-standardized C, such as K&R.
    eLanguageTypeAda83           = 0x0003,   ///< ISO Ada:1983.
    eLanguageTypeC_plus_plus     = 0x0004,   ///< ISO C++:1998.
    eLanguageTypeCobol74         = 0x0005,   ///< ISO Cobol:1974.
    eLanguageTypeCobol85         = 0x0006,   ///< ISO Cobol:1985.
    eLanguageTypeFortran77       = 0x0007,   ///< ISO Fortran 77.
    eLanguageTypeFortran90       = 0x0008,   ///< ISO Fortran 90.
    eLanguageTypePascal83        = 0x0009,   ///< ISO Pascal:1983.
    eLanguageTypeModula2         = 0x000a,   ///< ISO Modula-2:1996.
    eLanguageTypeJava            = 0x000b,   ///< Java.
    eLanguageTypeC99             = 0x000c,   ///< ISO C:1999.
    eLanguageTypeAda95           = 0x000d,   ///< ISO Ada:1995.
    eLanguageTypeFortran95       = 0x000e,   ///< ISO Fortran 95.
    eLanguageTypePLI             = 0x000f,   ///< ANSI PL/I:1976.
    eLanguageTypeObjC            = 0x0010,   ///< Objective-C.
    eLanguageTypeObjC_plus_plus  = 0x0011,   ///< Objective-C++.
    eLanguageTypeUPC             = 0x0012,   ///< Unified Parallel C.
    eLanguageTypeD               = 0x0013,   ///< D.
    eLanguageTypePython          = 0x0014    ///< Python.
} LanguageType;
 */

struct language_name_pair {
    const char *name;
    LanguageType type;
};

struct language_name_pair language_names[] =
{
    // To allow GetNameForLanguageType to be a simple array lookup, the first
    // part of this array must follow enum LanguageType exactly.
    {   "unknown",          eLanguageTypeUnknown        },
    {   "c89",              eLanguageTypeC89            },
    {   "c",                eLanguageTypeC              },
    {   "ada83",            eLanguageTypeAda83          },
    {   "c++",              eLanguageTypeC_plus_plus    },
    {   "cobol74",          eLanguageTypeCobol74        },
    {   "cobol85",          eLanguageTypeCobol85        },
    {   "fortran77",        eLanguageTypeFortran77      },
    {   "fortran90",        eLanguageTypeFortran90      },
    {   "pascal83",         eLanguageTypePascal83       },
    {   "modula2",          eLanguageTypeModula2        },
    {   "java",             eLanguageTypeJava           },
    {   "c99",              eLanguageTypeC99            },
    {   "ada95",            eLanguageTypeAda95          },
    {   "fortran95",        eLanguageTypeFortran95      },
    {   "pli",              eLanguageTypePLI            },
    {   "objective-c",      eLanguageTypeObjC           },
    {   "objective-c++",    eLanguageTypeObjC_plus_plus },
    {   "upc",              eLanguageTypeUPC            },
    {   "d",                eLanguageTypeD              },
    {   "python",           eLanguageTypePython         },
    // Now synonyms, in arbitrary order
    {   "objc",             eLanguageTypeObjC           },
    {   "objc++",           eLanguageTypeObjC_plus_plus }
};

static uint32_t num_languages = sizeof(language_names) / sizeof (struct language_name_pair);

LanguageType
LanguageRuntime::GetLanguageTypeFromString (const char *string)
{
    for (uint32_t i = 0; i < num_languages; i++)
    {
        if (strcasecmp (language_names[i].name, string) == 0)
            return (LanguageType) language_names[i].type;
    }
    return eLanguageTypeUnknown;
}

const char *
LanguageRuntime::GetNameForLanguageType (LanguageType language)
{
    if (language < num_languages)
        return language_names[language].name;
    else
        return language_names[eLanguageTypeUnknown].name;
}
        
