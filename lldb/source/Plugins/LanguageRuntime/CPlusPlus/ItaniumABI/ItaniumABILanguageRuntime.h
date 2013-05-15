//===-- ItaniumABILanguageRuntime.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ItaniumABILanguageRuntime_h_
#define liblldb_ItaniumABILanguageRuntime_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Breakpoint/BreakpointResolver.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/CPPLanguageRuntime.h"
#include "lldb/Core/Value.h"

namespace lldb_private {
    
    class ItaniumABILanguageRuntime :
    public lldb_private::CPPLanguageRuntime
    {
    public:
        ~ItaniumABILanguageRuntime() { }
        
        virtual bool
        IsVTableName (const char *name);
        
        virtual bool
        GetDynamicTypeAndAddress (ValueObject &in_value, 
                                  lldb::DynamicValueType use_dynamic, 
                                  TypeAndOrName &class_type_or_name, 
                                  Address &address);
        
        virtual bool
        CouldHaveDynamicValue (ValueObject &in_value);
        
        //------------------------------------------------------------------
        // Static Functions
        //------------------------------------------------------------------
        static void
        Initialize();
        
        static void
        Terminate();
        
        static lldb_private::LanguageRuntime *
        CreateInstance (Process *process, lldb::LanguageType language);
        
        static lldb_private::ConstString
        GetPluginNameStatic();

        //------------------------------------------------------------------
        // PluginInterface protocol
        //------------------------------------------------------------------
        virtual lldb_private::ConstString
        GetPluginName();
        
        virtual uint32_t
        GetPluginVersion();
        
        virtual void
        SetExceptionBreakpoints ();
        
        virtual void
        ClearExceptionBreakpoints ();
        
        virtual bool
        ExceptionBreakpointsExplainStop (lldb::StopInfoSP stop_reason);

        virtual lldb::BreakpointResolverSP
        CreateExceptionResolver (Breakpoint *bkpt, bool catch_bp, bool throw_bp);
        
        virtual lldb::SearchFilterSP
        CreateExceptionSearchFilter ();

    protected:

        lldb::BreakpointResolverSP
        CreateExceptionResolver (Breakpoint *bkpt, bool catch_bp, bool throw_bp, bool for_expressions);

        lldb::BreakpointSP
        CreateExceptionBreakpoint(bool catch_bp,
                                  bool throw_bp,
                                  bool for_expressions,
                                  bool is_internal);
        
    private:
        ItaniumABILanguageRuntime(Process *process) : lldb_private::CPPLanguageRuntime(process) { } // Call CreateInstance instead.
        
        lldb::BreakpointSP                              m_cxx_exception_bp_sp;
    };
    
} // namespace lldb_private

#endif  // liblldb_ItaniumABILanguageRuntime_h_
