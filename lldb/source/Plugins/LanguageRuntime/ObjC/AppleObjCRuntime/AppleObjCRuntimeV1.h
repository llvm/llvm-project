//===-- AppleObjCRuntimeV1.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_AppleObjCRuntimeV1_h_
#define liblldb_AppleObjCRuntimeV1_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "AppleObjCRuntime.h"

namespace lldb_private {
    
class AppleObjCRuntimeV1 :
        public AppleObjCRuntime
{
public:
    
    class ClassDescriptorV1 : public ObjCLanguageRuntime::ClassDescriptor
    {
    public:
        ClassDescriptorV1 (ValueObject &isa_pointer);
        ClassDescriptorV1 (ObjCISA isa, lldb::ProcessSP process_sp);
        
        virtual ConstString
        GetClassName ()
        {
            return m_name;
        }
        
        virtual ClassDescriptorSP
        GetSuperclass ();
        
        virtual bool
        IsValid ()
        {
            return m_valid;
        }
        
        virtual bool
        IsTagged ()
        {
            return false;   // v1 runtime does not support tagged pointers
        }
        
        virtual uint64_t
        GetInstanceSize ()
        {
            return m_instance_size;
        }
        
        virtual ObjCISA
        GetISA ()
        {
            return m_isa;
        }
        
        virtual
        ~ClassDescriptorV1 ()
        {}
        
    protected:
        void
        Initialize (ObjCISA isa, lldb::ProcessSP process_sp);
        
    private:
        ConstString m_name;
        ObjCISA m_isa;
        ObjCISA m_parent_isa;
        bool m_valid;
        lldb::ProcessWP m_process_wp;
        uint64_t m_instance_size;
    };
    
    virtual ~AppleObjCRuntimeV1() { }
    
    // These are generic runtime functions:
    virtual bool
    GetDynamicTypeAndAddress (ValueObject &in_value, 
                              lldb::DynamicValueType use_dynamic, 
                              TypeAndOrName &class_type_or_name, 
                              Address &address);

    virtual ClangUtilityFunction *
    CreateObjectChecker (const char *);

    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();
    
    static void
    Terminate();
    
    static lldb_private::LanguageRuntime *
    CreateInstance (Process *process, lldb::LanguageType language);
    
    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual const char *
    GetPluginName();
    
    virtual const char *
    GetShortPluginName();
    
    virtual uint32_t
    GetPluginVersion();
    
    virtual ObjCRuntimeVersions
    GetRuntimeVersion ()
    {
        return eAppleObjC_V1;
    }
    
    virtual void
    UpdateISAToDescriptorMapIfNeeded();

protected:
    virtual lldb::BreakpointResolverSP
    CreateExceptionResolver (Breakpoint *bkpt, bool catch_bp, bool throw_bp);
    
    
    class HashTableSignature
    {
    public:
        HashTableSignature () :
            m_count (0),
            m_num_buckets (0),
            m_buckets_ptr (LLDB_INVALID_ADDRESS)
        {
        }
        
        bool
        NeedsUpdate (uint32_t count,
                     uint32_t num_buckets,
                     lldb::addr_t buckets_ptr)
        {
            return m_count       != count       ||
                   m_num_buckets != num_buckets ||
                   m_buckets_ptr != buckets_ptr ;
        }
        
        void
        UpdateSignature (uint32_t count,
                         uint32_t num_buckets,
                         lldb::addr_t buckets_ptr)
        {
            m_count = count;
            m_num_buckets = num_buckets;
            m_buckets_ptr = buckets_ptr;
        }

    protected:
        uint32_t m_count;
        uint32_t m_num_buckets;
        lldb::addr_t m_buckets_ptr;
    };
    

    lldb::addr_t
    GetISAHashTablePointer ();
    
    HashTableSignature m_hash_signature;
    lldb::addr_t m_isa_hash_table_ptr;
private:
    AppleObjCRuntimeV1(Process *process);
};
    
} // namespace lldb_private

#endif  // liblldb_AppleObjCRuntimeV1_h_
