//===-- AppleObjCRuntimeV2.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include <string>
#include <vector>
#include <stdint.h>

#include "lldb/lldb-enumerations.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Symbol/ClangASTType.h"

#include "lldb/Core/ClangForward.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Expression/ClangUtilityFunction.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include "AppleObjCRuntimeV2.h"
#include "AppleObjCTypeVendor.h"
#include "AppleObjCTrampolineHandler.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;

// 2 second timeout when running utility functions
#define UTILITY_FUNCTION_TIMEOUT_USEC 2*1000*1000

static const char *g_get_dynamic_class_info_name = "__lldb_apple_objc_v2_get_dynamic_class_info";
// Testing using the new C++11 raw string literals. If this breaks GCC then we will
// need to revert to the code above...
static const char *g_get_dynamic_class_info_body = R"(

extern "C"
{
    size_t strlen(const char *);
    char *strncpy (char * s1, const char * s2, size_t n);
    int printf(const char * format, ...);
}
//#define ENABLE_DEBUG_PRINTF // COMMENT THIS LINE OUT PRIOR TO CHECKIN
#ifdef ENABLE_DEBUG_PRINTF
#define DEBUG_PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#else
#define DEBUG_PRINTF(fmt, ...)
#endif

typedef struct _NXMapTable {
    void *prototype;
    unsigned num_classes;
    unsigned num_buckets_minus_one;
    void *buckets;
} NXMapTable;

#define NX_MAPNOTAKEY   ((void *)(-1))

typedef struct BucketInfo
{
    const char *name_ptr;
    Class isa;
} BucketInfo;

struct ClassInfo
{
    Class isa;
    uint32_t hash;
} __attribute__((__packed__));

uint32_t
__lldb_apple_objc_v2_get_dynamic_class_info (void *gdb_objc_realized_classes_ptr,
                                             void *class_infos_ptr,
                                             uint32_t class_infos_byte_size)
{
    DEBUG_PRINTF ("gdb_objc_realized_classes_ptr = %p\n", gdb_objc_realized_classes_ptr);
    DEBUG_PRINTF ("class_infos_ptr = %p\n", class_infos_ptr);
    DEBUG_PRINTF ("class_infos_byte_size = %u\n", class_infos_byte_size);
    const NXMapTable *grc = (const NXMapTable *)gdb_objc_realized_classes_ptr;
    if (grc)
    {
        const unsigned num_classes = grc->num_classes;
        if (class_infos_ptr)
        {
            const size_t max_class_infos = class_infos_byte_size/sizeof(ClassInfo);
            ClassInfo *class_infos = (ClassInfo *)class_infos_ptr;
            BucketInfo *buckets = (BucketInfo *)grc->buckets;
            
            uint32_t idx = 0;
            for (unsigned i=0; i<=grc->num_buckets_minus_one; ++i)
            {
                if (buckets[i].name_ptr != NX_MAPNOTAKEY)
                {
                    if (idx < max_class_infos)
                    {
                        const char *s = buckets[i].name_ptr;
                        uint32_t h = 5381;
                        for (unsigned char c = *s; c; c = *++s)
                            h = ((h << 5) + h) + c;
                        class_infos[idx].hash = h;
                        class_infos[idx].isa = buckets[i].isa;
                    }
                    ++idx;
                }
            }
            if (idx < max_class_infos)
            {
                class_infos[idx].isa = NULL;
                class_infos[idx].hash = 0;
            }
        }
        return num_classes;
    }
    return 0;
}

)";

static const char *g_get_shared_cache_class_info_name = "__lldb_apple_objc_v2_get_shared_cache_class_info";
// Testing using the new C++11 raw string literals. If this breaks GCC then we will
// need to revert to the code above...
static const char *g_get_shared_cache_class_info_body = R"(

extern "C"
{
    const char *class_getName(void *objc_class);
    size_t strlen(const char *);
    char *strncpy (char * s1, const char * s2, size_t n);
    int printf(const char * format, ...);
}

//#define ENABLE_DEBUG_PRINTF // COMMENT THIS LINE OUT PRIOR TO CHECKIN
#ifdef ENABLE_DEBUG_PRINTF
#define DEBUG_PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#else
#define DEBUG_PRINTF(fmt, ...)
#endif


struct objc_classheader_t {
    int32_t clsOffset;
    int32_t hiOffset;
};

struct objc_clsopt_t {
    uint32_t capacity;
    uint32_t occupied;
    uint32_t shift;
    uint32_t mask;
    uint32_t zero;
    uint32_t unused;
    uint64_t salt;
    uint32_t scramble[256];
    uint8_t tab[0]; // tab[mask+1]
    //  uint8_t checkbytes[capacity];
    //  int32_t offset[capacity];
    //  objc_classheader_t clsOffsets[capacity];
    //  uint32_t duplicateCount;
    //  objc_classheader_t duplicateOffsets[duplicateCount];
};

struct objc_opt_t {
    uint32_t version;
    int32_t selopt_offset;
    int32_t headeropt_offset;
    int32_t clsopt_offset;
};

struct ClassInfo
{
    Class isa;
    uint32_t hash;
}  __attribute__((__packed__));

uint32_t
__lldb_apple_objc_v2_get_shared_cache_class_info (void *objc_opt_ro_ptr,
                                                  void *class_infos_ptr,
                                                  uint32_t class_infos_byte_size)
{
    uint32_t idx = 0;
    DEBUG_PRINTF ("objc_opt_ro_ptr = %p\n", objc_opt_ro_ptr);
    DEBUG_PRINTF ("class_infos_ptr = %p\n", class_infos_ptr);
    DEBUG_PRINTF ("class_infos_byte_size = %u (%zu class infos)\n", class_infos_byte_size, (size_t)(class_infos_byte_size/sizeof(ClassInfo)));
    if (objc_opt_ro_ptr)
    {
        const objc_opt_t *objc_opt = (objc_opt_t *)objc_opt_ro_ptr;
        DEBUG_PRINTF ("objc_opt->version = %u\n", objc_opt->version);
        DEBUG_PRINTF ("objc_opt->selopt_offset = %d\n", objc_opt->selopt_offset);
        DEBUG_PRINTF ("objc_opt->headeropt_offset = %d\n", objc_opt->headeropt_offset);
        DEBUG_PRINTF ("objc_opt->clsopt_offset = %d\n", objc_opt->clsopt_offset);
        if (objc_opt->version == 12)
        {
            const objc_clsopt_t* clsopt = (const objc_clsopt_t*)((uint8_t *)objc_opt + objc_opt->clsopt_offset);
            const size_t max_class_infos = class_infos_byte_size/sizeof(ClassInfo);
            ClassInfo *class_infos = (ClassInfo *)class_infos_ptr;
            int32_t zeroOffset = 16;
            const uint8_t *checkbytes = &clsopt->tab[clsopt->mask+1];
            const int32_t *offsets = (const int32_t *)(checkbytes + clsopt->capacity);
            const objc_classheader_t *classOffsets = (const objc_classheader_t *)(offsets + clsopt->capacity);
            DEBUG_PRINTF ("clsopt->capacity = %u\n", clsopt->capacity);
            DEBUG_PRINTF ("clsopt->mask = 0x%8.8x\n", clsopt->mask);
            DEBUG_PRINTF ("classOffsets = %p\n", classOffsets);
            for (uint32_t i=0; i<clsopt->capacity; ++i)
            {
                const int32_t clsOffset = classOffsets[i].clsOffset;
                if (clsOffset & 1)
                    continue; // duplicate
                else if (clsOffset == zeroOffset)
                    continue; // zero offset
                
                if (class_infos && idx < max_class_infos)
                {
                    class_infos[idx].isa = (Class)((uint8_t *)clsopt + clsOffset);
                    const char *name = class_getName (class_infos[idx].isa);
                    DEBUG_PRINTF ("[%u] isa = %8p %s\n", idx, class_infos[idx].isa, name);
                    // Hash the class name so we don't have to read it
                    const char *s = name;
                    uint32_t h = 5381;
                    for (unsigned char c = *s; c; c = *++s)
                        h = ((h << 5) + h) + c;
                    class_infos[idx].hash = h;
                }
                ++idx;
            }
            
            const uint32_t *duplicate_count_ptr = (uint32_t *)&classOffsets[clsopt->capacity];
            const uint32_t duplicate_count = *duplicate_count_ptr;
            const objc_classheader_t *duplicateClassOffsets = (const objc_classheader_t *)(&duplicate_count_ptr[1]);
            DEBUG_PRINTF ("duplicate_count = %u\n", duplicate_count);
            DEBUG_PRINTF ("duplicateClassOffsets = %p\n", duplicateClassOffsets);
            for (uint32_t i=0; i<duplicate_count; ++i)
            {
                const int32_t clsOffset = duplicateClassOffsets[i].clsOffset;
                if (clsOffset & 1)
                    continue; // duplicate
                else if (clsOffset == zeroOffset)
                    continue; // zero offset
                
                if (class_infos && idx < max_class_infos)
                {
                    class_infos[idx].isa = (Class)((uint8_t *)clsopt + clsOffset);
                    const char *name = class_getName (class_infos[idx].isa);
                    DEBUG_PRINTF ("[%u] isa = %8p %s\n", idx, class_infos[idx].isa, name);
                    // Hash the class name so we don't have to read it
                    const char *s = name;
                    uint32_t h = 5381;
                    for (unsigned char c = *s; c; c = *++s)
                        h = ((h << 5) + h) + c;
                    class_infos[idx].hash = h;
                }
                ++idx;
            }
        }
        DEBUG_PRINTF ("%u class_infos\n", idx);
        DEBUG_PRINTF ("done\n");
    }
    return idx;
}


)";

static uint64_t
ExtractRuntimeGlobalSymbol (Process* process,
                            ConstString name,
                            const ModuleSP &module_sp,
                            Error& error,
                            bool read_value = true,
                            uint8_t byte_size = 0,
                            uint64_t default_value = LLDB_INVALID_ADDRESS,
                            SymbolType sym_type = lldb::eSymbolTypeData)
{
    if (!process)
    {
        error.SetErrorString("no process");
        return default_value;
    }
    if (!module_sp)
    {
        error.SetErrorString("no module");
        return default_value;
    }
    if (!byte_size)
        byte_size = process->GetAddressByteSize();
    const Symbol *symbol = module_sp->FindFirstSymbolWithNameAndType(name, lldb::eSymbolTypeData);
    if (symbol)
    {
        lldb::addr_t symbol_load_addr = symbol->GetAddress().GetLoadAddress(&process->GetTarget());
        if (symbol_load_addr != LLDB_INVALID_ADDRESS)
        {
            if (read_value)
                return process->ReadUnsignedIntegerFromMemory(symbol_load_addr, byte_size, default_value, error);
            else
                return symbol_load_addr;
        }
        else
        {
            error.SetErrorString("symbol address invalid");
            return default_value;
        }
    }
    else
    {
        error.SetErrorString("no symbol");
        return default_value;
    }

}

AppleObjCRuntimeV2::AppleObjCRuntimeV2 (Process *process,
                                        const ModuleSP &objc_module_sp) :
    AppleObjCRuntime (process),
    m_get_class_info_function(),
    m_get_class_info_code(),
    m_get_class_info_args (LLDB_INVALID_ADDRESS),
    m_get_class_info_args_mutex (Mutex::eMutexTypeNormal),
    m_get_shared_cache_class_info_function(),
    m_get_shared_cache_class_info_code(),
    m_get_shared_cache_class_info_args (LLDB_INVALID_ADDRESS),
    m_get_shared_cache_class_info_args_mutex (Mutex::eMutexTypeNormal),
    m_type_vendor_ap (),
    m_isa_hash_table_ptr (LLDB_INVALID_ADDRESS),
    m_hash_signature (),
    m_has_object_getClass (false),
    m_loaded_objc_opt (false),
    m_non_pointer_isa_cache_ap(NonPointerISACache::CreateInstance(*this,objc_module_sp)),
    m_tagged_pointer_vendor_ap(TaggedPointerVendor::CreateInstance(*this,objc_module_sp))
{
    static const ConstString g_gdb_object_getClass("gdb_object_getClass");
    m_has_object_getClass = (objc_module_sp->FindFirstSymbolWithNameAndType(g_gdb_object_getClass, eSymbolTypeCode) != NULL);
}

AppleObjCRuntimeV2::~AppleObjCRuntimeV2()
{
}

bool
AppleObjCRuntimeV2::GetDynamicTypeAndAddress (ValueObject &in_value,
                                              DynamicValueType use_dynamic, 
                                              TypeAndOrName &class_type_or_name, 
                                              Address &address)
{
    // The Runtime is attached to a particular process, you shouldn't pass in a value from another process.
    assert (in_value.GetProcessSP().get() == m_process);
    assert (m_process != NULL);
    
    class_type_or_name.Clear();

    // Make sure we can have a dynamic value before starting...
    if (CouldHaveDynamicValue (in_value))
    {
        // First job, pull out the address at 0 offset from the object  That will be the ISA pointer.
        ClassDescriptorSP objc_class_sp (GetNonKVOClassDescriptor (in_value));
        if (objc_class_sp)
        {
            const addr_t object_ptr = in_value.GetPointerValue();
            address.SetRawAddress(object_ptr);

            ConstString class_name (objc_class_sp->GetClassName());
            class_type_or_name.SetName(class_name);
            TypeSP type_sp (objc_class_sp->GetType());
            if (type_sp)
                class_type_or_name.SetTypeSP (type_sp);
            else
            {
                type_sp = LookupInCompleteClassCache (class_name);
                if (type_sp)
                {
                    objc_class_sp->SetType (type_sp);
                    class_type_or_name.SetTypeSP (type_sp);
                }
            }
        }
    }    
    return class_type_or_name.IsEmpty() == false;
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
LanguageRuntime *
AppleObjCRuntimeV2::CreateInstance (Process *process, LanguageType language)
{
    // FIXME: This should be a MacOS or iOS process, and we need to look for the OBJC section to make
    // sure we aren't using the V1 runtime.
    if (language == eLanguageTypeObjC)
    {
        ModuleSP objc_module_sp;
        
        if (AppleObjCRuntime::GetObjCVersion (process, objc_module_sp) == eAppleObjC_V2)
            return new AppleObjCRuntimeV2 (process, objc_module_sp);
        else
            return NULL;
    }
    else
        return NULL;
}

void
AppleObjCRuntimeV2::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   "Apple Objective C Language Runtime - Version 2",
                                   CreateInstance);    
}

void
AppleObjCRuntimeV2::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

lldb_private::ConstString
AppleObjCRuntimeV2::GetPluginNameStatic()
{
    static ConstString g_name("apple-objc-v2");
    return g_name;
}


//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
lldb_private::ConstString
AppleObjCRuntimeV2::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
AppleObjCRuntimeV2::GetPluginVersion()
{
    return 1;
}

BreakpointResolverSP
AppleObjCRuntimeV2::CreateExceptionResolver (Breakpoint *bkpt, bool catch_bp, bool throw_bp)
{
    BreakpointResolverSP resolver_sp;
    
    if (throw_bp)
        resolver_sp.reset (new BreakpointResolverName (bkpt,
                                                       "objc_exception_throw",
                                                       eFunctionNameTypeBase,
                                                       Breakpoint::Exact,
                                                       eLazyBoolNo));
    // FIXME: We don't do catch breakpoints for ObjC yet.
    // Should there be some way for the runtime to specify what it can do in this regard?
    return resolver_sp;
}

ClangUtilityFunction *
AppleObjCRuntimeV2::CreateObjectChecker(const char *name)
{
    char check_function_code[2048];
    
    int len = 0;
    if (m_has_object_getClass)
    {
        len = ::snprintf (check_function_code, 
                          sizeof(check_function_code),
                          "extern \"C\" void *gdb_object_getClass(void *);                                          \n"
                          "extern \"C\"  int printf(const char *format, ...);                                       \n"
                          "extern \"C\" void                                                                        \n"
                          "%s(void *$__lldb_arg_obj, void *$__lldb_arg_selector)                                    \n"
                          "{                                                                                        \n"
                          "   if ($__lldb_arg_obj == (void *)0)                                                     \n"
                          "       return; // nil is ok                                                              \n" 
                          "   if (!gdb_object_getClass($__lldb_arg_obj))                                            \n"
                          "       *((volatile int *)0) = 'ocgc';                                                    \n"
                          "   else if ($__lldb_arg_selector != (void *)0)                                           \n"
                          "   {                                                                                     \n"
                          "        signed char responds = (signed char) [(id) $__lldb_arg_obj                       \n"
                          "                                                respondsToSelector:                      \n"
                          "                                       (struct objc_selector *) $__lldb_arg_selector];   \n"
                          "       if (responds == (signed char) 0)                                                  \n"
                          "           *((volatile int *)0) = 'ocgc';                                                \n"
                          "   }                                                                                     \n"
                          "}                                                                                        \n",
                          name);
    }
    else
    {
        len = ::snprintf (check_function_code, 
                          sizeof(check_function_code), 
                          "extern \"C\" void *gdb_class_getClass(void *);                                           \n"
                          "extern \"C\"  int printf(const char *format, ...);                                       \n"
                          "extern \"C\"  void                                                                       \n"
                          "%s(void *$__lldb_arg_obj, void *$__lldb_arg_selector)                                    \n"
                          "{                                                                                        \n"
                          "   if ($__lldb_arg_obj == (void *)0)                                                     \n"
                          "       return; // nil is ok                                                              \n" 
                          "    void **$isa_ptr = (void **)$__lldb_arg_obj;                                          \n"
                          "    if (*$isa_ptr == (void *)0 || !gdb_class_getClass(*$isa_ptr))                        \n"
                          "       *((volatile int *)0) = 'ocgc';                                                    \n"
                          "   else if ($__lldb_arg_selector != (void *)0)                                           \n"
                          "   {                                                                                     \n"
                          "        signed char responds = (signed char) [(id) $__lldb_arg_obj                       \n"
                          "                                                respondsToSelector:                      \n"
                          "                                        (struct objc_selector *) $__lldb_arg_selector];  \n"
                          "       if (responds == (signed char) 0)                                                  \n"
                          "           *((volatile int *)0) = 'ocgc';                                                \n"
                          "   }                                                                                     \n"
                          "}                                                                                        \n", 
                          name);
    }
    
    assert (len < sizeof(check_function_code));

    return new ClangUtilityFunction(check_function_code, name);
}

size_t
AppleObjCRuntimeV2::GetByteOffsetForIvar (ClangASTType &parent_ast_type, const char *ivar_name)
{
    const char *class_name = parent_ast_type.GetConstTypeName().AsCString();

    if (!class_name || *class_name == '\0' || !ivar_name || *ivar_name == '\0')
        return LLDB_INVALID_IVAR_OFFSET;
    
    std::string buffer("OBJC_IVAR_$_");
    buffer.append (class_name);
    buffer.push_back ('.');
    buffer.append (ivar_name);
    ConstString ivar_const_str (buffer.c_str());
    
    SymbolContextList sc_list;
    Target &target = m_process->GetTarget();
    
    target.GetImages().FindSymbolsWithNameAndType(ivar_const_str, eSymbolTypeObjCIVar, sc_list);

    SymbolContext ivar_offset_symbol;
    if (sc_list.GetSize() != 1 
        || !sc_list.GetContextAtIndex(0, ivar_offset_symbol) 
        || ivar_offset_symbol.symbol == NULL)
        return LLDB_INVALID_IVAR_OFFSET;
    
    addr_t ivar_offset_address = ivar_offset_symbol.symbol->GetAddress().GetLoadAddress (&target);
    
    Error error;
    
    uint32_t ivar_offset = m_process->ReadUnsignedIntegerFromMemory (ivar_offset_address, 
                                                                     4, 
                                                                     LLDB_INVALID_IVAR_OFFSET, 
                                                                     error);
    return ivar_offset;
}


// tagged pointers are special not-a-real-pointer values that contain both type and value information
// this routine attempts to check with as little computational effort as possible whether something
// could possibly be a tagged pointer - false positives are possible but false negatives shouldn't
bool
AppleObjCRuntimeV2::IsTaggedPointer(addr_t ptr)
{
    if (!m_tagged_pointer_vendor_ap)
        return false;
    return m_tagged_pointer_vendor_ap->IsPossibleTaggedPointer(ptr);
}

class RemoteNXMapTable
{
public:
    
    RemoteNXMapTable () :
        m_count (0),
        m_num_buckets_minus_one (0),
        m_buckets_ptr (LLDB_INVALID_ADDRESS),
        m_process (NULL),
        m_end_iterator (*this, -1),
        m_load_addr (LLDB_INVALID_ADDRESS),
        m_map_pair_size (0),
        m_invalid_key (0)
    {
    }
    
    void
    Dump ()
    {
        printf ("RemoteNXMapTable.m_load_addr = 0x%" PRIx64 "\n", m_load_addr);
        printf ("RemoteNXMapTable.m_count = %u\n", m_count);
        printf ("RemoteNXMapTable.m_num_buckets_minus_one = %u\n", m_num_buckets_minus_one);
        printf ("RemoteNXMapTable.m_buckets_ptr = 0x%" PRIX64 "\n", m_buckets_ptr);
    }
    
    bool
    ParseHeader (Process* process, lldb::addr_t load_addr)
    {
        m_process = process;
        m_load_addr = load_addr;
        m_map_pair_size = m_process->GetAddressByteSize() * 2;
        m_invalid_key = m_process->GetAddressByteSize() == 8 ? UINT64_MAX : UINT32_MAX;
        Error err;
        
        // This currently holds true for all platforms we support, but we might
        // need to change this to use get the actualy byte size of "unsigned"
        // from the target AST...
        const uint32_t unsigned_byte_size = sizeof(uint32_t);
        // Skip the prototype as we don't need it (const struct +NXMapTablePrototype *prototype)
        
        bool success = true;
        if (load_addr == LLDB_INVALID_ADDRESS)
            success = false;
        else
        {
            lldb::addr_t cursor = load_addr + m_process->GetAddressByteSize();
                    
            // unsigned count;
            m_count = m_process->ReadUnsignedIntegerFromMemory(cursor, unsigned_byte_size, 0, err);
            if (m_count)
            {
                cursor += unsigned_byte_size;
            
                // unsigned nbBucketsMinusOne;
                m_num_buckets_minus_one = m_process->ReadUnsignedIntegerFromMemory(cursor, unsigned_byte_size, 0, err);
                cursor += unsigned_byte_size;
            
                // void *buckets;
                m_buckets_ptr = m_process->ReadPointerFromMemory(cursor, err);
                
                success = m_count > 0 && m_buckets_ptr != LLDB_INVALID_ADDRESS;
            }
        }
        
        if (!success)
        {
            m_count = 0;
            m_num_buckets_minus_one = 0;
            m_buckets_ptr = LLDB_INVALID_ADDRESS;
        }
        return success;
    }
    
    // const_iterator mimics NXMapState and its code comes from NXInitMapState and NXNextMapState.
    typedef std::pair<ConstString, ObjCLanguageRuntime::ObjCISA> element;

    friend class const_iterator;
    class const_iterator
    {
    public:
        const_iterator (RemoteNXMapTable &parent, int index) : m_parent(parent), m_index(index)
        {
            AdvanceToValidIndex();
        }
        
        const_iterator (const const_iterator &rhs) : m_parent(rhs.m_parent), m_index(rhs.m_index)
        {
            // AdvanceToValidIndex() has been called by rhs already.
        }
        
        const_iterator &operator=(const const_iterator &rhs)
        {
            // AdvanceToValidIndex() has been called by rhs already.
            assert (&m_parent == &rhs.m_parent);
            m_index = rhs.m_index;
            return *this;
        }
        
        bool operator==(const const_iterator &rhs) const
        {
            if (&m_parent != &rhs.m_parent)
                return false;
            if (m_index != rhs.m_index)
                return false;
            
            return true;
        }
        
        bool operator!=(const const_iterator &rhs) const
        {
            return !(operator==(rhs));
        }
        
        const_iterator &operator++()
        {
            AdvanceToValidIndex();
            return *this;
        }
        
        const element operator*() const
        {
            if (m_index == -1)
            {
                // TODO find a way to make this an error, but not an assert
                return element();
            }
         
            lldb::addr_t pairs_ptr = m_parent.m_buckets_ptr;
            size_t map_pair_size = m_parent.m_map_pair_size;
            lldb::addr_t pair_ptr = pairs_ptr + (m_index * map_pair_size);
            
            Error err;
            
            lldb::addr_t key = m_parent.m_process->ReadPointerFromMemory(pair_ptr, err);
            if (!err.Success())
                return element();
            lldb::addr_t value = m_parent.m_process->ReadPointerFromMemory(pair_ptr + m_parent.m_process->GetAddressByteSize(), err);
            if (!err.Success())
                return element();
            
            std::string key_string;
            
            m_parent.m_process->ReadCStringFromMemory(key, key_string, err);
            if (!err.Success())
                return element();
            
            return element(ConstString(key_string.c_str()), (ObjCLanguageRuntime::ObjCISA)value);
        }
    private:
        void AdvanceToValidIndex ()
        {
            if (m_index == -1)
                return;
            
            const lldb::addr_t pairs_ptr = m_parent.m_buckets_ptr;
            const size_t map_pair_size = m_parent.m_map_pair_size;
            const lldb::addr_t invalid_key = m_parent.m_invalid_key;
            Error err;

            while (m_index--)
            {
                lldb::addr_t pair_ptr = pairs_ptr + (m_index * map_pair_size);
                lldb::addr_t key = m_parent.m_process->ReadPointerFromMemory(pair_ptr, err);
                
                if (!err.Success())
                {
                    m_index = -1;
                    return;
                }
                
                if (key != invalid_key)
                    return;
            }
        }
        RemoteNXMapTable   &m_parent;
        int                 m_index;
    };
    
    const_iterator begin ()
    {
        return const_iterator(*this, m_num_buckets_minus_one + 1);
    }
    
    const_iterator end ()
    {
        return m_end_iterator;
    }
    
    uint32_t
    GetCount () const
    {
        return m_count;
    }
    
    uint32_t
    GetBucketCount () const
    {
        return m_num_buckets_minus_one;
    }
    
    lldb::addr_t
    GetBucketDataPointer () const
    {
        return m_buckets_ptr;
    }
    
    lldb::addr_t
    GetTableLoadAddress() const
    {
        return m_load_addr;
    }

private:
    // contents of _NXMapTable struct
    uint32_t m_count;
    uint32_t m_num_buckets_minus_one;
    lldb::addr_t m_buckets_ptr;
    lldb_private::Process *m_process;
    const_iterator m_end_iterator;
    lldb::addr_t m_load_addr;
    size_t m_map_pair_size;
    lldb::addr_t m_invalid_key;
};



AppleObjCRuntimeV2::HashTableSignature::HashTableSignature() :
    m_count (0),
    m_num_buckets (0),
    m_buckets_ptr (0)
{
}

void
AppleObjCRuntimeV2::HashTableSignature::UpdateSignature (const RemoteNXMapTable &hash_table)
{
    m_count = hash_table.GetCount();
    m_num_buckets = hash_table.GetBucketCount();
    m_buckets_ptr = hash_table.GetBucketDataPointer();
}

bool
AppleObjCRuntimeV2::HashTableSignature::NeedsUpdate (Process *process, AppleObjCRuntimeV2 *runtime, RemoteNXMapTable &hash_table)
{
    if (!hash_table.ParseHeader(process, runtime->GetISAHashTablePointer ()))
    {
        return false; // Failed to parse the header, no need to update anything
    }

    // Check with out current signature and return true if the count,
    // number of buckets or the hash table address changes.
    if (m_count == hash_table.GetCount() &&
        m_num_buckets == hash_table.GetBucketCount() &&
        m_buckets_ptr == hash_table.GetBucketDataPointer())
    {
        // Hash table hasn't changed
        return false;
    }
    // Hash table data has changed, we need to update
    return true;
}

class ClassDescriptorV2 : public ObjCLanguageRuntime::ClassDescriptor
{
public:
    friend class lldb_private::AppleObjCRuntimeV2;
    
private:
    // The constructor should only be invoked by the runtime as it builds its caches
    // or populates them.  A ClassDescriptorV2 should only ever exist in a cache.
    ClassDescriptorV2 (AppleObjCRuntimeV2 &runtime, ObjCLanguageRuntime::ObjCISA isa, const char *name) :
        m_runtime (runtime),
        m_objc_class_ptr (isa),
        m_name (name)
    {
    }

public:
    virtual ConstString
    GetClassName ()
    {
        if (!m_name)
        {
            lldb_private::Process *process = m_runtime.GetProcess();

            if (process)
            {
                std::unique_ptr<objc_class_t> objc_class;
                std::unique_ptr<class_ro_t> class_ro;
                std::unique_ptr<class_rw_t> class_rw;
                
                if (!Read_objc_class(process, objc_class))
                    return m_name;
                if (!Read_class_row(process, *objc_class, class_ro, class_rw))
                    return m_name;
                
                m_name = ConstString(class_ro->m_name.c_str());
            }
        }
        return m_name;
    }
    
    virtual ObjCLanguageRuntime::ClassDescriptorSP
    GetSuperclass ()
    {
        lldb_private::Process *process = m_runtime.GetProcess();

        if (!process)
            return ObjCLanguageRuntime::ClassDescriptorSP();
        
        std::unique_ptr<objc_class_t> objc_class;

        if (!Read_objc_class(process, objc_class))
            return ObjCLanguageRuntime::ClassDescriptorSP();

        return m_runtime.ObjCLanguageRuntime::GetClassDescriptorFromISA(objc_class->m_superclass);
    }
    
    virtual bool
    IsValid ()
    {
        return true;    // any Objective-C v2 runtime class descriptor we vend is valid
    }
    
    // a custom descriptor is used for tagged pointers
    virtual bool
    GetTaggedPointerInfo (uint64_t* info_bits = NULL,
                          uint64_t* value_bits = NULL,
                          uint64_t* payload = NULL)
    {
        return false;
    }
    
    virtual uint64_t
    GetInstanceSize ()
    {
        lldb_private::Process *process = m_runtime.GetProcess();
        
        if (process)
        {
            std::unique_ptr<objc_class_t> objc_class;
            std::unique_ptr<class_ro_t> class_ro;
            std::unique_ptr<class_rw_t> class_rw;
            
            if (!Read_objc_class(process, objc_class))
                return 0;
            if (!Read_class_row(process, *objc_class, class_ro, class_rw))
                return 0;
            
            return class_ro->m_instanceSize;
        }
        
        return 0;
    }
    
    virtual ObjCLanguageRuntime::ObjCISA
    GetISA ()
    {        
        return m_objc_class_ptr;
    }
    
    virtual bool
    Describe (std::function <void (ObjCLanguageRuntime::ObjCISA)> const &superclass_func,
              std::function <bool (const char *, const char *)> const &instance_method_func,
              std::function <bool (const char *, const char *)> const &class_method_func,
              std::function <bool (const char *, const char *, lldb::addr_t, uint64_t)> const &ivar_func)
    {
        lldb_private::Process *process = m_runtime.GetProcess();

        std::unique_ptr<objc_class_t> objc_class;
        std::unique_ptr<class_ro_t> class_ro;
        std::unique_ptr<class_rw_t> class_rw;
        
        if (!Read_objc_class(process, objc_class))
            return 0;
        if (!Read_class_row(process, *objc_class, class_ro, class_rw))
            return 0;
    
        static ConstString NSObject_name("NSObject");
        
        if (m_name != NSObject_name && superclass_func)
            superclass_func(objc_class->m_superclass);
        
        if (instance_method_func)
        {
            std::unique_ptr<method_list_t> base_method_list;
            
            base_method_list.reset(new method_list_t);
            if (!base_method_list->Read(process, class_ro->m_baseMethods_ptr))
                return false;
            
            if (base_method_list->m_entsize != method_t::GetSize(process))
                return false;
            
            std::unique_ptr<method_t> method;
            method.reset(new method_t);
            
            for (uint32_t i = 0, e = base_method_list->m_count; i < e; ++i)
            {
                method->Read(process, base_method_list->m_first_ptr + (i * base_method_list->m_entsize));
                
                if (instance_method_func(method->m_name.c_str(), method->m_types.c_str()))
                    break;
            }
        }
        
        if (class_method_func)
        {
            ClassDescriptorV2 metaclass(m_runtime, objc_class->m_isa, NULL); // The metaclass is not in the cache
            
            // We don't care about the metaclass's superclass, or its class methods.  Its instance methods are
            // our class methods.
            
            metaclass.Describe(std::function <void (ObjCLanguageRuntime::ObjCISA)> (nullptr),
                               class_method_func,
                               std::function <bool (const char *, const char *)> (nullptr),
                               std::function <bool (const char *, const char *, lldb::addr_t, uint64_t)> (nullptr));
        }
        
        if (ivar_func)
        {
            ivar_list_t ivar_list;
            if (!ivar_list.Read(process, class_ro->m_ivars_ptr))
                return false;
            
            if (ivar_list.m_entsize != ivar_t::GetSize(process))
                return false;
            
            ivar_t ivar;
            
            for (uint32_t i = 0, e = ivar_list.m_count; i < e; ++i)
            {
                ivar.Read(process, ivar_list.m_first_ptr + (i * ivar_list.m_entsize));
                
                if (ivar_func(ivar.m_name.c_str(), ivar.m_type.c_str(), ivar.m_offset_ptr, ivar.m_size))
                    break;
            }
        }
            
        return true;
    }
    
    virtual
    ~ClassDescriptorV2 ()
    {
    }
        
private:
    static const uint32_t RW_REALIZED = (1 << 31);
    
    struct objc_class_t {
        ObjCLanguageRuntime::ObjCISA    m_isa;              // The class's metaclass.
        ObjCLanguageRuntime::ObjCISA    m_superclass;
        lldb::addr_t                    m_cache_ptr;
        lldb::addr_t                    m_vtable_ptr;
        lldb::addr_t                    m_data_ptr;
        uint8_t                         m_flags;
        
        objc_class_t () :
            m_isa (0),
            m_superclass (0),
            m_cache_ptr (0),
            m_vtable_ptr (0),
            m_data_ptr (0),
            m_flags (0)
        {
        }
        
        void
        Clear()
        {
            m_isa = 0;
            m_superclass = 0;
            m_cache_ptr = 0;
            m_vtable_ptr = 0;
            m_data_ptr = 0;
            m_flags = 0;
        }

        bool Read(Process *process, lldb::addr_t addr)
        {
            size_t ptr_size = process->GetAddressByteSize();
            
            size_t objc_class_size = ptr_size   // uintptr_t isa;
            + ptr_size   // Class superclass;
            + ptr_size   // void *cache;
            + ptr_size   // IMP *vtable;
            + ptr_size;  // uintptr_t data_NEVER_USE;
            
            DataBufferHeap objc_class_buf (objc_class_size, '\0');
            Error error;
            
            process->ReadMemory(addr, objc_class_buf.GetBytes(), objc_class_size, error);
            if (error.Fail())
            {
                return false;
            }
            
            DataExtractor extractor(objc_class_buf.GetBytes(), objc_class_size, process->GetByteOrder(), process->GetAddressByteSize());
            
            lldb::offset_t cursor = 0;
            
            m_isa           = extractor.GetAddress_unchecked(&cursor);   // uintptr_t isa;
            m_superclass    = extractor.GetAddress_unchecked(&cursor);   // Class superclass;
            m_cache_ptr     = extractor.GetAddress_unchecked(&cursor);   // void *cache;
            m_vtable_ptr    = extractor.GetAddress_unchecked(&cursor);   // IMP *vtable;
            lldb::addr_t data_NEVER_USE = extractor.GetAddress_unchecked(&cursor);   // uintptr_t data_NEVER_USE;
            
            m_flags         = (uint8_t)(data_NEVER_USE & (lldb::addr_t)3);
            m_data_ptr      = data_NEVER_USE & ~(lldb::addr_t)3;
            
            return true;
        }
    };
    
    struct class_ro_t {
        uint32_t                        m_flags;
        uint32_t                        m_instanceStart;
        uint32_t                        m_instanceSize;
        uint32_t                        m_reserved;
        
        lldb::addr_t                    m_ivarLayout_ptr;
        lldb::addr_t                    m_name_ptr;
        lldb::addr_t                    m_baseMethods_ptr;
        lldb::addr_t                    m_baseProtocols_ptr;
        lldb::addr_t                    m_ivars_ptr;
        
        lldb::addr_t                    m_weakIvarLayout_ptr;
        lldb::addr_t                    m_baseProperties_ptr;
        
        std::string                     m_name;
        
        bool Read(Process *process, lldb::addr_t addr)
        {
            size_t ptr_size = process->GetAddressByteSize();
            
            size_t size = sizeof(uint32_t)             // uint32_t flags;
            + sizeof(uint32_t)                         // uint32_t instanceStart;
            + sizeof(uint32_t)                         // uint32_t instanceSize;
            + (ptr_size == 8 ? sizeof(uint32_t) : 0)   // uint32_t reserved; // __LP64__ only
            + ptr_size                                 // const uint8_t *ivarLayout;
            + ptr_size                                 // const char *name;
            + ptr_size                                 // const method_list_t *baseMethods;
            + ptr_size                                 // const protocol_list_t *baseProtocols;
            + ptr_size                                 // const ivar_list_t *ivars;
            + ptr_size                                 // const uint8_t *weakIvarLayout;
            + ptr_size;                                // const property_list_t *baseProperties;
            
            DataBufferHeap buffer (size, '\0');
            Error error;
            
            process->ReadMemory(addr, buffer.GetBytes(), size, error);
            if (error.Fail())
            {
                return false;
            }
            
            DataExtractor extractor(buffer.GetBytes(), size, process->GetByteOrder(), process->GetAddressByteSize());
            
            lldb::offset_t cursor = 0;
            
            m_flags             = extractor.GetU32_unchecked(&cursor);
            m_instanceStart     = extractor.GetU32_unchecked(&cursor);
            m_instanceSize      = extractor.GetU32_unchecked(&cursor);
            if (ptr_size == 8)
                m_reserved      = extractor.GetU32_unchecked(&cursor);
            else
                m_reserved      = 0;
            m_ivarLayout_ptr     = extractor.GetAddress_unchecked(&cursor);
            m_name_ptr           = extractor.GetAddress_unchecked(&cursor);
            m_baseMethods_ptr    = extractor.GetAddress_unchecked(&cursor);
            m_baseProtocols_ptr  = extractor.GetAddress_unchecked(&cursor);
            m_ivars_ptr          = extractor.GetAddress_unchecked(&cursor);
            m_weakIvarLayout_ptr = extractor.GetAddress_unchecked(&cursor);
            m_baseProperties_ptr = extractor.GetAddress_unchecked(&cursor);
            
            DataBufferHeap name_buf(1024, '\0');
            
            process->ReadCStringFromMemory(m_name_ptr, (char*)name_buf.GetBytes(), name_buf.GetByteSize(), error);
            
            if (error.Fail())
            {
                return false;
            }
            
            m_name.assign((char*)name_buf.GetBytes());
                
            return true;
        }
    };
    
    struct class_rw_t {
        uint32_t                        m_flags;
        uint32_t                        m_version;
        
        lldb::addr_t                    m_ro_ptr;
        union {
            lldb::addr_t                m_method_list_ptr;
            lldb::addr_t                m_method_lists_ptr;
        };
        lldb::addr_t                    m_properties_ptr;
        lldb::addr_t                    m_protocols_ptr;
        
        ObjCLanguageRuntime::ObjCISA    m_firstSubclass;
        ObjCLanguageRuntime::ObjCISA    m_nextSiblingClass;
        
        bool Read(Process *process, lldb::addr_t addr)
        {
            size_t ptr_size = process->GetAddressByteSize();
            
            size_t size = sizeof(uint32_t)  // uint32_t flags;
            + sizeof(uint32_t)  // uint32_t version;
            + ptr_size          // const class_ro_t *ro;
            + ptr_size          // union { method_list_t **method_lists; method_list_t *method_list; };
            + ptr_size          // struct chained_property_list *properties;
            + ptr_size          // const protocol_list_t **protocols;
            + ptr_size          // Class firstSubclass;
            + ptr_size;         // Class nextSiblingClass;
            
            DataBufferHeap buffer (size, '\0');
            Error error;
            
            process->ReadMemory(addr, buffer.GetBytes(), size, error);
            if (error.Fail())
            {
                return false;
            }
            
            DataExtractor extractor(buffer.GetBytes(), size, process->GetByteOrder(), process->GetAddressByteSize());
            
            lldb::offset_t cursor = 0;
            
            m_flags             = extractor.GetU32_unchecked(&cursor);
            m_version           = extractor.GetU32_unchecked(&cursor);
            m_ro_ptr            = extractor.GetAddress_unchecked(&cursor);
            m_method_list_ptr   = extractor.GetAddress_unchecked(&cursor);
            m_properties_ptr    = extractor.GetAddress_unchecked(&cursor);
            m_firstSubclass     = extractor.GetAddress_unchecked(&cursor);
            m_nextSiblingClass  = extractor.GetAddress_unchecked(&cursor);
            
            return true;
        }
    };
    
    struct method_list_t
    {
        uint32_t        m_entsize;
        uint32_t        m_count;
        lldb::addr_t    m_first_ptr;
        
        bool Read(Process *process, lldb::addr_t addr)
        {
            size_t size = sizeof(uint32_t)  // uint32_t entsize_NEVER_USE;
            + sizeof(uint32_t); // uint32_t count;
            
            DataBufferHeap buffer (size, '\0');
            Error error;
            
            process->ReadMemory(addr, buffer.GetBytes(), size, error);
            if (error.Fail())
            {
                return false;
            }
            
            DataExtractor extractor(buffer.GetBytes(), size, process->GetByteOrder(), process->GetAddressByteSize());
            
            lldb::offset_t cursor = 0;
            
            m_entsize   = extractor.GetU32_unchecked(&cursor) & ~(uint32_t)3;
            m_count     = extractor.GetU32_unchecked(&cursor);
            m_first_ptr  = addr + cursor;
            
            return true;
        }
    };
    
    struct method_t
    {
        lldb::addr_t    m_name_ptr;
        lldb::addr_t    m_types_ptr;
        lldb::addr_t    m_imp_ptr;
        
        std::string     m_name;
        std::string     m_types;
        
        static size_t GetSize(Process *process)
        {
            size_t ptr_size = process->GetAddressByteSize();
            
            return ptr_size     // SEL name;
            + ptr_size   // const char *types;
            + ptr_size;  // IMP imp;
        }
        
        bool Read(Process *process, lldb::addr_t addr)
        {
            size_t size = GetSize(process);
            
            DataBufferHeap buffer (size, '\0');
            Error error;
            
            process->ReadMemory(addr, buffer.GetBytes(), size, error);
            if (error.Fail())
            {
                return false;
            }
            
            DataExtractor extractor(buffer.GetBytes(), size, process->GetByteOrder(), process->GetAddressByteSize());
            
            lldb::offset_t cursor = 0;
            
            m_name_ptr   = extractor.GetAddress_unchecked(&cursor);
            m_types_ptr  = extractor.GetAddress_unchecked(&cursor);
            m_imp_ptr    = extractor.GetAddress_unchecked(&cursor);
            
            const size_t buffer_size = 1024;
            size_t count;
            
            DataBufferHeap string_buf(buffer_size, 0);
            
            count = process->ReadCStringFromMemory(m_name_ptr, (char*)string_buf.GetBytes(), buffer_size, error);
            m_name.assign((char*)string_buf.GetBytes(), count);
            
            count = process->ReadCStringFromMemory(m_types_ptr, (char*)string_buf.GetBytes(), buffer_size, error);
            m_types.assign((char*)string_buf.GetBytes(), count);
            
            return true;
        }
    };
    
    struct ivar_list_t
    {
        uint32_t        m_entsize;
        uint32_t        m_count;
        lldb::addr_t    m_first_ptr;
        
        bool Read(Process *process, lldb::addr_t addr)
        {
            size_t size = sizeof(uint32_t)  // uint32_t entsize;
                        + sizeof(uint32_t); // uint32_t count;
            
            DataBufferHeap buffer (size, '\0');
            Error error;
            
            process->ReadMemory(addr, buffer.GetBytes(), size, error);
            if (error.Fail())
            {
                return false;
            }
            
            DataExtractor extractor(buffer.GetBytes(), size, process->GetByteOrder(), process->GetAddressByteSize());
            
            lldb::offset_t cursor = 0;
            
            m_entsize   = extractor.GetU32_unchecked(&cursor);
            m_count     = extractor.GetU32_unchecked(&cursor);
            m_first_ptr = addr + cursor;
            
            return true;
        }
    };
    
    struct ivar_t
    {
        lldb::addr_t    m_offset_ptr;
        lldb::addr_t    m_name_ptr;
        lldb::addr_t    m_type_ptr;
        uint32_t        m_alignment;
        uint32_t        m_size;
        
        std::string     m_name;
        std::string     m_type;
        
        static size_t GetSize(Process *process)
        {
            size_t ptr_size = process->GetAddressByteSize();
            
            return ptr_size             // uintptr_t *offset;
                 + ptr_size             // const char *name;
                 + ptr_size             // const char *type;
                 + sizeof(uint32_t)     // uint32_t alignment;
                 + sizeof(uint32_t);    // uint32_t size;
        }
        
        bool Read(Process *process, lldb::addr_t addr)
        {
            size_t size = GetSize(process);
            
            DataBufferHeap buffer (size, '\0');
            Error error;
            
            process->ReadMemory(addr, buffer.GetBytes(), size, error);
            if (error.Fail())
            {
                return false;
            }
            
            DataExtractor extractor(buffer.GetBytes(), size, process->GetByteOrder(), process->GetAddressByteSize());
            
            lldb::offset_t cursor = 0;
            
            m_offset_ptr = extractor.GetAddress_unchecked(&cursor);
            m_name_ptr   = extractor.GetAddress_unchecked(&cursor);
            m_type_ptr   = extractor.GetAddress_unchecked(&cursor);
            m_alignment  = extractor.GetU32_unchecked(&cursor);
            m_size       = extractor.GetU32_unchecked(&cursor);
            
            const size_t buffer_size = 1024;
            size_t count;
            
            DataBufferHeap string_buf(buffer_size, 0);
            
            count = process->ReadCStringFromMemory(m_name_ptr, (char*)string_buf.GetBytes(), buffer_size, error);
            m_name.assign((char*)string_buf.GetBytes(), count);
            
            count = process->ReadCStringFromMemory(m_type_ptr, (char*)string_buf.GetBytes(), buffer_size, error);
            m_type.assign((char*)string_buf.GetBytes(), count);
            
            return true;
        }
    };
    
    bool Read_objc_class (Process* process, std::unique_ptr<objc_class_t> &objc_class)
    {
        objc_class.reset(new objc_class_t);
        
        bool ret = objc_class->Read (process, m_objc_class_ptr);
        
        if (!ret)
            objc_class.reset();
        
        return ret;
    }
    
    bool Read_class_row (Process* process, const objc_class_t &objc_class, std::unique_ptr<class_ro_t> &class_ro, std::unique_ptr<class_rw_t> &class_rw)
    {
        class_ro.reset();
        class_rw.reset();
        
        Error error;
        uint32_t class_row_t_flags = process->ReadUnsignedIntegerFromMemory(objc_class.m_data_ptr, sizeof(uint32_t), 0, error);
        if (!error.Success())
            return false;

        if (class_row_t_flags & RW_REALIZED)
        {
            class_rw.reset(new class_rw_t);
            
            if (!class_rw->Read(process, objc_class.m_data_ptr))
            {
                class_rw.reset();
                return false;
            }
            
            class_ro.reset(new class_ro_t);
            
            if (!class_ro->Read(process, class_rw->m_ro_ptr))
            {
                class_rw.reset();
                class_ro.reset();
                return false;
            }
        }
        else
        {
            class_ro.reset(new class_ro_t);
            
            if (!class_ro->Read(process, objc_class.m_data_ptr))
            {
                class_ro.reset();
                return false;
            }
        }
        
        return true;
    }

    AppleObjCRuntimeV2 &m_runtime;          // The runtime, so we can read information lazily.
    lldb::addr_t        m_objc_class_ptr;   // The address of the objc_class_t.  (I.e., objects of this class type have this as their ISA)
    ConstString         m_name;             // May be NULL
};

// tagged pointer descriptor
class ClassDescriptorV2Tagged : public ObjCLanguageRuntime::ClassDescriptor
{
public:
    ClassDescriptorV2Tagged (ConstString class_name,
                             uint64_t payload)
    {
        m_name = class_name;
        if (!m_name)
        {
            m_valid = false;
            return;
        }
        m_valid = true;
        m_payload = payload;
        m_info_bits = (m_payload & 0xF0ULL) >> 4;
        m_value_bits = (m_payload & ~0x0000000000000000FFULL) >> 8;
    }
    
    ClassDescriptorV2Tagged (ObjCLanguageRuntime::ClassDescriptorSP actual_class_sp,
                             uint64_t payload)
    {
        if (!actual_class_sp)
        {
            m_valid = false;
            return;
        }
        m_name = actual_class_sp->GetClassName();
        if (!m_name)
        {
            m_valid = false;
            return;
        }
        m_valid = true;
        m_payload = payload;
        m_info_bits = (m_payload & 0x0FULL);
        m_value_bits = (m_payload & ~0x0FULL) >> 4;
    }
    
    virtual ConstString
    GetClassName ()
    {
        return m_name;
    }
    
    virtual ObjCLanguageRuntime::ClassDescriptorSP
    GetSuperclass ()
    {
        // tagged pointers can represent a class that has a superclass, but since that information is not
        // stored in the object itself, we would have to query the runtime to discover the hierarchy
        // for the time being, we skip this step in the interest of static discovery
        return ObjCLanguageRuntime::ClassDescriptorSP();
    }
    
    virtual bool
    IsValid ()
    {
        return m_valid;
    }
    
    virtual bool
    IsKVO ()
    {
        return false; // tagged pointers are not KVO'ed
    }
    
    virtual bool
    IsCFType ()
    {
        return false; // tagged pointers are not CF objects
    }
    
    virtual bool
    GetTaggedPointerInfo (uint64_t* info_bits = NULL,
                          uint64_t* value_bits = NULL,
                          uint64_t* payload = NULL)
    {
        if (info_bits)
            *info_bits = GetInfoBits();
        if (value_bits)
            *value_bits = GetValueBits();
        if (payload)
            *payload = GetPayload();
        return true;
    }
    
    virtual uint64_t
    GetInstanceSize ()
    {
        return (IsValid() ? m_pointer_size : 0);
    }
    
    virtual ObjCLanguageRuntime::ObjCISA
    GetISA ()
    {
        return 0; // tagged pointers have no ISA
    }
    
    // these calls are not part of any formal tagged pointers specification
    virtual uint64_t
    GetValueBits ()
    {
        return (IsValid() ? m_value_bits : 0);
    }
    
    virtual uint64_t
    GetInfoBits ()
    {
        return (IsValid() ? m_info_bits : 0);
    }
    
    virtual uint64_t
    GetPayload ()
    {
        return (IsValid() ? m_payload : 0);
    }
    
    virtual
    ~ClassDescriptorV2Tagged ()
    {}

private:
    ConstString m_name;
    uint8_t m_pointer_size;
    bool m_valid;
    uint64_t m_info_bits;
    uint64_t m_value_bits;
    uint64_t m_payload;

};

ObjCLanguageRuntime::ClassDescriptorSP
AppleObjCRuntimeV2::GetClassDescriptor (ObjCISA isa)
{
    ObjCLanguageRuntime::ClassDescriptorSP class_descriptor_sp;
    if (m_non_pointer_isa_cache_ap.get())
        class_descriptor_sp = m_non_pointer_isa_cache_ap->GetClassDescriptor(isa);
    if (!class_descriptor_sp)
        class_descriptor_sp = ObjCLanguageRuntime::GetClassDescriptorFromISA(isa);
    return class_descriptor_sp;
}

ObjCLanguageRuntime::ClassDescriptorSP
AppleObjCRuntimeV2::GetClassDescriptor (ValueObject& valobj)
{
    ClassDescriptorSP objc_class_sp;
    // if we get an invalid VO (which might still happen when playing around
    // with pointers returned by the expression parser, don't consider this
    // a valid ObjC object)
    if (valobj.GetValue().GetContextType() != Value::eContextTypeInvalid)
    {
        addr_t isa_pointer = valobj.GetPointerValue();
        
        // tagged pointer
        if (IsTaggedPointer(isa_pointer))
        {
            return m_tagged_pointer_vendor_ap->GetClassDescriptor(isa_pointer);
        }
        else
        {
            ExecutionContext exe_ctx (valobj.GetExecutionContextRef());
            
            Process *process = exe_ctx.GetProcessPtr();
            if (process)
            {
                Error error;
                ObjCISA isa = process->ReadPointerFromMemory(isa_pointer, error);
                if (isa != LLDB_INVALID_ADDRESS)
                {
                    objc_class_sp = ObjCLanguageRuntime::GetClassDescriptorFromISA (isa);
                    if (isa && !objc_class_sp)
                    {
                        Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
                        if (log)
                            log->Printf("0x%" PRIx64 ": AppleObjCRuntimeV2::GetClassDescriptor() ISA was not in class descriptor cache 0x%" PRIx64,
                                        isa_pointer,
                                        isa);
                    }
                }
            }
        }
    }
    return objc_class_sp;
}

lldb::addr_t
AppleObjCRuntimeV2::GetISAHashTablePointer ()
{
    if (m_isa_hash_table_ptr == LLDB_INVALID_ADDRESS)
    {
        Process *process = GetProcess();

        ModuleSP objc_module_sp(GetObjCModule());
        
        if (!objc_module_sp)
            return LLDB_INVALID_ADDRESS;

        static ConstString g_gdb_objc_realized_classes("gdb_objc_realized_classes");
        
        const Symbol *symbol = objc_module_sp->FindFirstSymbolWithNameAndType(g_gdb_objc_realized_classes, lldb::eSymbolTypeData);
        if (symbol)
        {
            lldb::addr_t gdb_objc_realized_classes_ptr = symbol->GetAddress().GetLoadAddress(&process->GetTarget());
            
            if (gdb_objc_realized_classes_ptr != LLDB_INVALID_ADDRESS)
            {
                Error error;
                m_isa_hash_table_ptr = process->ReadPointerFromMemory(gdb_objc_realized_classes_ptr, error);
            }
        }
    }
    return m_isa_hash_table_ptr;
}

bool
AppleObjCRuntimeV2::UpdateISAToDescriptorMapDynamic(RemoteNXMapTable &hash_table)
{
    Process *process = GetProcess();
    
    if (process == NULL)
        return false;
    
    Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
    
    ExecutionContext exe_ctx;
    
    ThreadSP thread_sp = process->GetThreadList().GetSelectedThread();
    
    if (!thread_sp)
        return false;
    
    thread_sp->CalculateExecutionContext(exe_ctx);
    ClangASTContext *ast = process->GetTarget().GetScratchClangASTContext();
    
    if (!ast)
        return false;
    
    Address function_address;
    
    StreamString errors;
    
    const uint32_t addr_size = process->GetAddressByteSize();
    
    Error err;
    
    // Read the total number of classes from the hash table
    const uint32_t num_classes = hash_table.GetCount();
    if (num_classes == 0)
    {
        if (log)
            log->Printf ("No dynamic classes found in gdb_objc_realized_classes.");
        return false;
    }
    
    // Make some types for our arguments
    clang_type_t clang_uint32_t_type = ast->GetBuiltinTypeForEncodingAndBitSize(eEncodingUint, 32);
    clang_type_t clang_void_pointer_type = ast->CreatePointerType(ast->GetBuiltInType_void());
    
    if (!m_get_class_info_code.get())
    {
        m_get_class_info_code.reset (new ClangUtilityFunction (g_get_dynamic_class_info_body,
                                                               g_get_dynamic_class_info_name));
        
        errors.Clear();
        
        if (!m_get_class_info_code->Install(errors, exe_ctx))
        {
            if (log)
                log->Printf ("Failed to install implementation lookup: %s.", errors.GetData());
            m_get_class_info_code.reset();
        }
    }
    
    if (m_get_class_info_code.get())
        function_address.SetOffset(m_get_class_info_code->StartAddress());
    else
        return false;
    
    ValueList arguments;
    
    // Next make the runner function for our implementation utility function.
    if (!m_get_class_info_function.get())
    {
        Value value;
        value.SetValueType (Value::eValueTypeScalar);
        value.SetContext (Value::eContextTypeClangType, clang_void_pointer_type);
        arguments.PushValue (value);
        
        value.SetValueType (Value::eValueTypeScalar);
        value.SetContext (Value::eContextTypeClangType, clang_void_pointer_type);
        arguments.PushValue (value);
        
        value.SetValueType (Value::eValueTypeScalar);
        value.SetContext (Value::eContextTypeClangType, clang_uint32_t_type);
        arguments.PushValue (value);
        
        m_get_class_info_function.reset(new ClangFunction (*m_process,
                                                           ast,
                                                           clang_uint32_t_type,
                                                           function_address,
                                                           arguments));
        
        if (m_get_class_info_function.get() == NULL)
            return false;
        
        errors.Clear();
        
        unsigned num_errors = m_get_class_info_function->CompileFunction(errors);
        if (num_errors)
        {
            if (log)
                log->Printf ("Error compiling function: \"%s\".", errors.GetData());
            return false;
        }
        
        errors.Clear();
        
        if (!m_get_class_info_function->WriteFunctionWrapper(exe_ctx, errors))
        {
            if (log)
                log->Printf ("Error Inserting function: \"%s\".", errors.GetData());
            return false;
        }
    }
    else
    {
        arguments = m_get_class_info_function->GetArgumentValues ();
    }
    
    const uint32_t class_info_byte_size = addr_size + 4;
    const uint32_t class_infos_byte_size = num_classes * class_info_byte_size;
    lldb::addr_t class_infos_addr = process->AllocateMemory(class_infos_byte_size,
                                                            ePermissionsReadable | ePermissionsWritable,
                                                            err);
    
    if (class_infos_addr == LLDB_INVALID_ADDRESS)
        return false;
    
    Mutex::Locker locker(m_get_class_info_args_mutex);
    
    // Fill in our function argument values
    arguments.GetValueAtIndex(0)->GetScalar() = hash_table.GetTableLoadAddress();
    arguments.GetValueAtIndex(1)->GetScalar() = class_infos_addr;
    arguments.GetValueAtIndex(2)->GetScalar() = class_infos_byte_size;
    
    bool success = false;
    
    errors.Clear();
    
    // Write our function arguments into the process so we can run our function
    if (m_get_class_info_function->WriteFunctionArguments (exe_ctx,
                                                           m_get_class_info_args,
                                                           function_address,
                                                           arguments,
                                                           errors))
    {
        bool stop_others = true;
        bool try_all_threads = false;
        bool unwind_on_error = true;
        bool ignore_breakpoints = true;
        
        Value return_value;
        return_value.SetValueType (Value::eValueTypeScalar);
        return_value.SetContext (Value::eContextTypeClangType, clang_uint32_t_type);
        return_value.GetScalar() = 0;
        
        errors.Clear();
        
        // Run the function
        ExecutionResults results = m_get_class_info_function->ExecuteFunction (exe_ctx,
                                                                               &m_get_class_info_args,
                                                                               errors,
                                                                               stop_others,
                                                                               UTILITY_FUNCTION_TIMEOUT_USEC,
                                                                               try_all_threads,
                                                                               unwind_on_error,
                                                                               ignore_breakpoints,
                                                                               return_value);
        
        if (results == eExecutionCompleted)
        {
            // The result is the number of ClassInfo structures that were filled in
            uint32_t num_class_infos = return_value.GetScalar().ULong();
            if (log)
                log->Printf("Discovered %u ObjC classes\n",num_class_infos);
            if (num_class_infos > 0)
            {
                // Read the ClassInfo structures
                DataBufferHeap buffer (num_class_infos * class_info_byte_size, 0);
                if (process->ReadMemory(class_infos_addr, buffer.GetBytes(), buffer.GetByteSize(), err) == buffer.GetByteSize())
                {
                    DataExtractor class_infos_data (buffer.GetBytes(),
                                                    buffer.GetByteSize(),
                                                    process->GetByteOrder(),
                                                    addr_size);
                    ParseClassInfoArray (class_infos_data, num_class_infos);
                }
            }
            success = true;
        }
        else
        {
            if (log)
                log->Printf("Error evaluating our find class name function: %s.\n", errors.GetData());
        }
    }
    else
    {
        if (log)
            log->Printf ("Error writing function arguments: \"%s\".", errors.GetData());
    }
    
    // Deallocate the memory we allocated for the ClassInfo array
    process->DeallocateMemory(class_infos_addr);
    
    return success;
}

void
AppleObjCRuntimeV2::ParseClassInfoArray (const DataExtractor &data, uint32_t num_class_infos)
{
    // Parses an array of "num_class_infos" packed ClassInfo structures:
    //
    //    struct ClassInfo
    //    {
    //        Class isa;
    //        uint32_t hash;
    //    } __attribute__((__packed__));

    Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));

    // Iterate through all ClassInfo structures
    lldb::offset_t offset = 0;
    for (uint32_t i=0; i<num_class_infos; ++i)
    {
        ObjCISA isa = data.GetPointer(&offset);
        
        if (isa == 0)
        {
            if (log)
                log->Printf("AppleObjCRuntimeV2 found NULL isa, ignoring this class info");
            continue;
        }
        // Check if we already know about this ISA, if we do, the info will
        // never change, so we can just skip it.
        if (ISAIsCached(isa))
        {
            offset += 4;
        }
        else
        {
            // Read the 32 bit hash for the class name
            const uint32_t name_hash = data.GetU32(&offset);
            ClassDescriptorSP descriptor_sp (new ClassDescriptorV2(*this, isa, NULL));
            AddClass (isa, descriptor_sp, name_hash);
            if (log && log->GetVerbose())
                log->Printf("AppleObjCRuntimeV2 added isa=0x%" PRIx64 ", hash=0x%8.8x", isa, name_hash);
        }
    }
}

bool
AppleObjCRuntimeV2::UpdateISAToDescriptorMapSharedCache()
{
    Process *process = GetProcess();
    
    if (process == NULL)
        return false;
    
    Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
    
    ExecutionContext exe_ctx;
    
    ThreadSP thread_sp = process->GetThreadList().GetSelectedThread();
    
    if (!thread_sp)
        return false;
    
    thread_sp->CalculateExecutionContext(exe_ctx);
    ClangASTContext *ast = process->GetTarget().GetScratchClangASTContext();
    
    if (!ast)
        return false;
    
    Address function_address;
    
    StreamString errors;
    
    const uint32_t addr_size = process->GetAddressByteSize();
    
    Error err;
    
    const lldb::addr_t objc_opt_ptr = GetSharedCacheReadOnlyAddress();
    
    if (objc_opt_ptr == LLDB_INVALID_ADDRESS)
        return false;
    
    // Read the total number of classes from the hash table
    const uint32_t num_classes = 16*1024;
    if (num_classes == 0)
    {
        if (log)
            log->Printf ("No dynamic classes found in gdb_objc_realized_classes_addr.");
        return false;
    }
    
    // Make some types for our arguments
    clang_type_t clang_uint32_t_type = ast->GetBuiltinTypeForEncodingAndBitSize(eEncodingUint, 32);
    clang_type_t clang_void_pointer_type = ast->CreatePointerType(ast->GetBuiltInType_void());
    
    if (!m_get_shared_cache_class_info_code.get())
    {
        m_get_shared_cache_class_info_code.reset (new ClangUtilityFunction (g_get_shared_cache_class_info_body,
                                                                            g_get_shared_cache_class_info_name));
        
        errors.Clear();
        
        if (!m_get_shared_cache_class_info_code->Install(errors, exe_ctx))
        {
            if (log)
                log->Printf ("Failed to install implementation lookup: %s.", errors.GetData());
            m_get_shared_cache_class_info_code.reset();
        }
    }
    
    if (m_get_shared_cache_class_info_code.get())
        function_address.SetOffset(m_get_shared_cache_class_info_code->StartAddress());
    else
        return false;
    
    ValueList arguments;
    
    // Next make the runner function for our implementation utility function.
    if (!m_get_shared_cache_class_info_function.get())
    {
        Value value;
        value.SetValueType (Value::eValueTypeScalar);
        value.SetContext (Value::eContextTypeClangType, clang_void_pointer_type);
        arguments.PushValue (value);
        
        value.SetValueType (Value::eValueTypeScalar);
        value.SetContext (Value::eContextTypeClangType, clang_void_pointer_type);
        arguments.PushValue (value);
        
        value.SetValueType (Value::eValueTypeScalar);
        value.SetContext (Value::eContextTypeClangType, clang_uint32_t_type);
        arguments.PushValue (value);
        
        m_get_shared_cache_class_info_function.reset(new ClangFunction (*m_process,
                                                                        ast,
                                                                        clang_uint32_t_type,
                                                                        function_address,
                                                                        arguments));
        
        if (m_get_shared_cache_class_info_function.get() == NULL)
            return false;
        
        errors.Clear();
        
        unsigned num_errors = m_get_shared_cache_class_info_function->CompileFunction(errors);
        if (num_errors)
        {
            if (log)
                log->Printf ("Error compiling function: \"%s\".", errors.GetData());
            return false;
        }
        
        errors.Clear();
        
        if (!m_get_shared_cache_class_info_function->WriteFunctionWrapper(exe_ctx, errors))
        {
            if (log)
                log->Printf ("Error Inserting function: \"%s\".", errors.GetData());
            return false;
        }
    }
    else
    {
        arguments = m_get_shared_cache_class_info_function->GetArgumentValues ();
    }
    
    const uint32_t class_info_byte_size = addr_size + 4;
    const uint32_t class_infos_byte_size = num_classes * class_info_byte_size;
    lldb::addr_t class_infos_addr = process->AllocateMemory (class_infos_byte_size,
                                                             ePermissionsReadable | ePermissionsWritable,
                                                             err);
    
    if (class_infos_addr == LLDB_INVALID_ADDRESS)
        return false;
    
    Mutex::Locker locker(m_get_shared_cache_class_info_args_mutex);
    
    // Fill in our function argument values
    arguments.GetValueAtIndex(0)->GetScalar() = objc_opt_ptr;
    arguments.GetValueAtIndex(1)->GetScalar() = class_infos_addr;
    arguments.GetValueAtIndex(2)->GetScalar() = class_infos_byte_size;
    
    bool success = false;
    
    errors.Clear();
    
    // Write our function arguments into the process so we can run our function
    if (m_get_shared_cache_class_info_function->WriteFunctionArguments (exe_ctx,
                                                                        m_get_shared_cache_class_info_args,
                                                                        function_address,
                                                                        arguments,
                                                                        errors))
    {
        bool stop_others = true;
        bool try_all_threads = false;
        bool unwind_on_error = true;
        bool ignore_breakpoints = true;
        
        Value return_value;
        return_value.SetValueType (Value::eValueTypeScalar);
        return_value.SetContext (Value::eContextTypeClangType, clang_uint32_t_type);
        return_value.GetScalar() = 0;
        
        errors.Clear();
        
        // Run the function
        ExecutionResults results = m_get_shared_cache_class_info_function->ExecuteFunction (exe_ctx,
                                                                                            &m_get_shared_cache_class_info_args,
                                                                                            errors,
                                                                                            stop_others,
                                                                                            UTILITY_FUNCTION_TIMEOUT_USEC,
                                                                                            try_all_threads,
                                                                                            unwind_on_error,
                                                                                            ignore_breakpoints,
                                                                                            return_value);
        
        if (results == eExecutionCompleted)
        {
            // The result is the number of ClassInfo structures that were filled in
            uint32_t num_class_infos = return_value.GetScalar().ULong();
            if (log)
                log->Printf("Discovered %u ObjC classes in shared cache\n",num_class_infos);
            if (num_class_infos > 0)
            {
                // Read the ClassInfo structures
                DataBufferHeap buffer (num_class_infos * class_info_byte_size, 0);
                if (process->ReadMemory(class_infos_addr,
                                        buffer.GetBytes(),
                                        buffer.GetByteSize(),
                                        err) == buffer.GetByteSize())
                {
                    DataExtractor class_infos_data (buffer.GetBytes(),
                                                    buffer.GetByteSize(),
                                                    process->GetByteOrder(),
                                                    addr_size);
                    
                    ParseClassInfoArray (class_infos_data, num_class_infos);
                }
            }
            success = true;
        }
        else
        {
            if (log)
                log->Printf("Error evaluating our find class name function: %s.\n", errors.GetData());
        }
    }
    else
    {
        if (log)
            log->Printf ("Error writing function arguments: \"%s\".", errors.GetData());
    }
    
    // Deallocate the memory we allocated for the ClassInfo array
    process->DeallocateMemory(class_infos_addr);
    
    return success;
}


bool
AppleObjCRuntimeV2::UpdateISAToDescriptorMapFromMemory (RemoteNXMapTable &hash_table)
{
    Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
    
    Process *process = GetProcess();

    if (process == NULL)
        return false;
    
    uint32_t num_map_table_isas = 0;
    
    ModuleSP objc_module_sp(GetObjCModule());
    
    if (objc_module_sp)
    {
        for (RemoteNXMapTable::element elt : hash_table)
        {
            ++num_map_table_isas;
            
            if (ISAIsCached(elt.second))
                continue;
            
            ClassDescriptorSP descriptor_sp = ClassDescriptorSP(new ClassDescriptorV2(*this, elt.second, elt.first.AsCString()));
            
            if (log && log->GetVerbose())
                log->Printf("AppleObjCRuntimeV2 added (ObjCISA)0x%" PRIx64 " (%s) from dynamic table to isa->descriptor cache", elt.second, elt.first.AsCString());
            
            AddClass (elt.second, descriptor_sp, elt.first.AsCString());
        }
    }
    
    return num_map_table_isas > 0;
}

lldb::addr_t
AppleObjCRuntimeV2::GetSharedCacheReadOnlyAddress()
{
    Process *process = GetProcess();
    
    if (process)
    {
        ModuleSP objc_module_sp(GetObjCModule());
        
        if (objc_module_sp)
        {
            ObjectFile *objc_object = objc_module_sp->GetObjectFile();
            
            if (objc_object)
            {
                SectionList *section_list = objc_object->GetSectionList();
                
                if (section_list)
                {
                    SectionSP text_segment_sp (section_list->FindSectionByName(ConstString("__TEXT")));
                    
                    if (text_segment_sp)
                    {
                        SectionSP objc_opt_section_sp (text_segment_sp->GetChildren().FindSectionByName(ConstString("__objc_opt_ro")));
                        
                        if (objc_opt_section_sp)
                        {
                            return objc_opt_section_sp->GetLoadBaseAddress(&process->GetTarget());
                        }
                    }
                }
            }
        }
    }
    return LLDB_INVALID_ADDRESS;
}

void
AppleObjCRuntimeV2::UpdateISAToDescriptorMapIfNeeded()
{
    Timer scoped_timer (__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);
    
    // Else we need to check with our process to see when the map was updated.
    Process *process = GetProcess();

    if (process)
    {
        RemoteNXMapTable hash_table;
        
        // Update the process stop ID that indicates the last time we updated the
        // map, wether it was successful or not.
        m_isa_to_descriptor_stop_id = process->GetStopID();
        
        if (!m_hash_signature.NeedsUpdate(process, this, hash_table))
            return;
        
        m_hash_signature.UpdateSignature (hash_table);

        // Grab the dynamicly loaded objc classes from the hash table in memory
        UpdateISAToDescriptorMapDynamic(hash_table);

        // Now get the objc classes that are baked into the Objective C runtime
        // in the shared cache, but only once per process as this data never
        // changes
        if (!m_loaded_objc_opt)
            UpdateISAToDescriptorMapSharedCache();
    }
    else
    {
        m_isa_to_descriptor_stop_id = UINT32_MAX;
    }
}


// TODO: should we have a transparent_kvo parameter here to say if we
// want to replace the KVO swizzled class with the actual user-level type?
ConstString
AppleObjCRuntimeV2::GetActualTypeName(ObjCLanguageRuntime::ObjCISA isa)
{
    if (isa == g_objc_Tagged_ISA)
    {
        static const ConstString g_objc_tagged_isa_name ("_lldb_Tagged_ObjC_ISA");
        return g_objc_tagged_isa_name;
    }
    if (isa == g_objc_Tagged_ISA_NSAtom)
    {
        static const ConstString g_objc_tagged_isa_nsatom_name ("NSAtom");
        return g_objc_tagged_isa_nsatom_name;
    }
    if (isa == g_objc_Tagged_ISA_NSNumber)
    {
        static const ConstString g_objc_tagged_isa_nsnumber_name ("NSNumber");
        return g_objc_tagged_isa_nsnumber_name;
    }
    if (isa == g_objc_Tagged_ISA_NSDateTS)
    {
        static const ConstString g_objc_tagged_isa_nsdatets_name ("NSDateTS");
        return g_objc_tagged_isa_nsdatets_name;
    }
    if (isa == g_objc_Tagged_ISA_NSManagedObject)
    {
        static const ConstString g_objc_tagged_isa_nsmanagedobject_name ("NSManagedObject");
        return g_objc_tagged_isa_nsmanagedobject_name;
    }
    if (isa == g_objc_Tagged_ISA_NSDate)
    {
        static const ConstString g_objc_tagged_isa_nsdate_name ("NSDate");
        return g_objc_tagged_isa_nsdate_name;
    }
    return ObjCLanguageRuntime::GetActualTypeName(isa);
}

TypeVendor *
AppleObjCRuntimeV2::GetTypeVendor()
{
    if (!m_type_vendor_ap.get())
        m_type_vendor_ap.reset(new AppleObjCTypeVendor(*this));
    
    return m_type_vendor_ap.get();
}

lldb::addr_t
AppleObjCRuntimeV2::LookupRuntimeSymbol (const ConstString &name)
{
    lldb::addr_t ret = LLDB_INVALID_ADDRESS;

    const char *name_cstr = name.AsCString();    
    
    if (name_cstr)
    {
        llvm::StringRef name_strref(name_cstr);
        
        static const llvm::StringRef ivar_prefix("OBJC_IVAR_$_");
        static const llvm::StringRef class_prefix("OBJC_CLASS_$_");
        
        if (name_strref.startswith(ivar_prefix))
        {
            llvm::StringRef ivar_skipped_prefix = name_strref.substr(ivar_prefix.size());
            std::pair<llvm::StringRef, llvm::StringRef> class_and_ivar = ivar_skipped_prefix.split('.');
            
            if (class_and_ivar.first.size() && class_and_ivar.second.size())
            {
                const ConstString class_name_cs(class_and_ivar.first);
                ClassDescriptorSP descriptor = ObjCLanguageRuntime::GetClassDescriptorFromClassName(class_name_cs);
                                
                if (descriptor)
                {
                    const ConstString ivar_name_cs(class_and_ivar.second);
                    const char *ivar_name_cstr = ivar_name_cs.AsCString();
                    
                    auto ivar_func = [&ret, ivar_name_cstr](const char *name, const char *type, lldb::addr_t offset_addr, uint64_t size) -> lldb::addr_t
                    {
                        if (!strcmp(name, ivar_name_cstr))
                        {
                            ret = offset_addr;
                            return true;
                        }
                        return false;
                    };

                    descriptor->Describe(std::function<void (ObjCISA)>(nullptr),
                                         std::function<bool (const char *, const char *)>(nullptr),
                                         std::function<bool (const char *, const char *)>(nullptr),
                                         ivar_func);
                }
            }
        }
        else if (name_strref.startswith(class_prefix))
        {
            llvm::StringRef class_skipped_prefix = name_strref.substr(class_prefix.size());
            const ConstString class_name_cs(class_skipped_prefix);
            ClassDescriptorSP descriptor = GetClassDescriptorFromClassName(class_name_cs);
            
            if (descriptor)
                ret = descriptor->GetISA();
        }
    }
    
    return ret;
}

AppleObjCRuntimeV2::NonPointerISACache*
AppleObjCRuntimeV2::NonPointerISACache::CreateInstance (AppleObjCRuntimeV2& runtime, const lldb::ModuleSP& objc_module_sp)
{
    Process* process(runtime.GetProcess());
    
    Error error;
    
    auto objc_debug_isa_magic_mask = ExtractRuntimeGlobalSymbol(process,
                                                                ConstString("objc_debug_isa_magic_mask"),
                                                                objc_module_sp,
                                                                error);
    if (error.Fail())
        return NULL;

    auto objc_debug_isa_magic_value = ExtractRuntimeGlobalSymbol(process,
                                                                 ConstString("objc_debug_isa_magic_value"),
                                                                 objc_module_sp,
                                                                 error);
    if (error.Fail())
        return NULL;

    auto objc_debug_isa_class_mask = ExtractRuntimeGlobalSymbol(process,
                                                                ConstString("objc_debug_isa_class_mask"),
                                                                objc_module_sp,
                                                                error);
    if (error.Fail())
        return NULL;

    // we might want to have some rules to outlaw these other values (e.g if the mask is zero but the value is non-zero, ...)
    
    return new NonPointerISACache(runtime,
                                  objc_debug_isa_class_mask,
                                  objc_debug_isa_magic_mask,
                                  objc_debug_isa_magic_value);
}

AppleObjCRuntimeV2::TaggedPointerVendor*
AppleObjCRuntimeV2::TaggedPointerVendor::CreateInstance (AppleObjCRuntimeV2& runtime, const lldb::ModuleSP& objc_module_sp)
{
    Process* process(runtime.GetProcess());
    
    Error error;
    
    auto objc_debug_taggedpointer_mask = ExtractRuntimeGlobalSymbol(process,
                                                                    ConstString("objc_debug_taggedpointer_mask"),
                                                                    objc_module_sp,
                                                                    error);
    if (error.Fail())
        return new TaggedPointerVendorLegacy(runtime);
    
    auto objc_debug_taggedpointer_slot_shift = ExtractRuntimeGlobalSymbol(process,
                                                                          ConstString("objc_debug_taggedpointer_slot_shift"),
                                                                          objc_module_sp,
                                                                          error,
                                                                          true,
                                                                          4);
    if (error.Fail())
        return new TaggedPointerVendorLegacy(runtime);
    
    auto objc_debug_taggedpointer_slot_mask = ExtractRuntimeGlobalSymbol(process,
                                                                          ConstString("objc_debug_taggedpointer_slot_mask"),
                                                                          objc_module_sp,
                                                                          error,
                                                                          true,
                                                                          4);
    if (error.Fail())
        return new TaggedPointerVendorLegacy(runtime);

    auto objc_debug_taggedpointer_payload_lshift = ExtractRuntimeGlobalSymbol(process,
                                                                              ConstString("objc_debug_taggedpointer_payload_lshift"),
                                                                              objc_module_sp,
                                                                              error,
                                                                              true,
                                                                              4);
    if (error.Fail())
        return new TaggedPointerVendorLegacy(runtime);
    
    auto objc_debug_taggedpointer_payload_rshift = ExtractRuntimeGlobalSymbol(process,
                                                                              ConstString("objc_debug_taggedpointer_payload_rshift"),
                                                                              objc_module_sp,
                                                                              error,
                                                                              true,
                                                                              4);
    if (error.Fail())
        return new TaggedPointerVendorLegacy(runtime);
    
    auto objc_debug_taggedpointer_classes = ExtractRuntimeGlobalSymbol(process,
                                                                       ConstString("objc_debug_taggedpointer_classes"),
                                                                       objc_module_sp,
                                                                       error,
                                                                       false);
    if (error.Fail())
        return new TaggedPointerVendorLegacy(runtime);

    
    // we might want to have some rules to outlaw these values (e.g if the table's address is zero)
    
    return new TaggedPointerVendorRuntimeAssisted(runtime,
                                                  objc_debug_taggedpointer_mask,
                                                  objc_debug_taggedpointer_slot_shift,
                                                  objc_debug_taggedpointer_slot_mask,
                                                  objc_debug_taggedpointer_payload_lshift,
                                                  objc_debug_taggedpointer_payload_rshift,
                                                  objc_debug_taggedpointer_classes);
}

bool
AppleObjCRuntimeV2::TaggedPointerVendorLegacy::IsPossibleTaggedPointer (lldb::addr_t ptr)
{
    return (ptr & 1);
}

// we use the version of Foundation to make assumptions about the ObjC runtime on a target
uint32_t
AppleObjCRuntimeV2::TaggedPointerVendorLegacy::GetFoundationVersion (Target &target)
{
    const ModuleList& modules = target.GetImages();
    uint32_t major = UINT32_MAX;
    for (uint32_t idx = 0; idx < modules.GetSize(); idx++)
    {
        lldb::ModuleSP module_sp = modules.GetModuleAtIndex(idx);
        if (!module_sp)
            continue;
        if (strcmp(module_sp->GetFileSpec().GetFilename().AsCString(""),"Foundation") == 0)
        {
            module_sp->GetVersion(&major,1);
            break;
        }
    }
    return major;
}

ObjCLanguageRuntime::ClassDescriptorSP
AppleObjCRuntimeV2::TaggedPointerVendorLegacy::GetClassDescriptor (lldb::addr_t ptr)
{
    if (!IsPossibleTaggedPointer(ptr))
        return ObjCLanguageRuntime::ClassDescriptorSP();

    Process* process(m_runtime.GetProcess());
    
    if (m_Foundation_version == 0)
        m_Foundation_version = GetFoundationVersion(process->GetTarget());
    
    if (m_Foundation_version == UINT32_MAX)
        return ObjCLanguageRuntime::ClassDescriptorSP();
    
    uint64_t class_bits = (ptr & 0xE) >> 1;
    ConstString name;
    
    // TODO: make a table
    if (m_Foundation_version >= 900)
    {
        switch (class_bits)
        {
            case 0:
                name = ConstString("NSAtom");
                break;
            case 3:
                name = ConstString("NSNumber");
                break;
            case 4:
                name = ConstString("NSDateTS");
                break;
            case 5:
                name = ConstString("NSManagedObject");
                break;
            case 6:
                name = ConstString("NSDate");
                break;
            default:
                return ObjCLanguageRuntime::ClassDescriptorSP();
        }
    }
    else
    {
        switch (class_bits)
        {
            case 1:
                name = ConstString("NSNumber");
                break;
            case 5:
                name = ConstString("NSManagedObject");
                break;
            case 6:
                name = ConstString("NSDate");
                break;
            case 7:
                name = ConstString("NSDateTS");
                break;
            default:
                return ObjCLanguageRuntime::ClassDescriptorSP();
        }
    }
    return ClassDescriptorSP(new ClassDescriptorV2Tagged(name,ptr));
}

AppleObjCRuntimeV2::TaggedPointerVendorRuntimeAssisted::TaggedPointerVendorRuntimeAssisted (AppleObjCRuntimeV2& runtime,
                                                                                            uint64_t objc_debug_taggedpointer_mask,
                                                                                            uint32_t objc_debug_taggedpointer_slot_shift,
                                                                                            uint32_t objc_debug_taggedpointer_slot_mask,
                                                                                            uint32_t objc_debug_taggedpointer_payload_lshift,
                                                                                            uint32_t objc_debug_taggedpointer_payload_rshift,
                                                                                            lldb::addr_t objc_debug_taggedpointer_classes) :
TaggedPointerVendor(runtime),
m_cache(),
m_objc_debug_taggedpointer_mask(objc_debug_taggedpointer_mask),
m_objc_debug_taggedpointer_slot_shift(objc_debug_taggedpointer_slot_shift),
m_objc_debug_taggedpointer_slot_mask(objc_debug_taggedpointer_slot_mask),
m_objc_debug_taggedpointer_payload_lshift(objc_debug_taggedpointer_payload_lshift),
m_objc_debug_taggedpointer_payload_rshift(objc_debug_taggedpointer_payload_rshift),
m_objc_debug_taggedpointer_classes(objc_debug_taggedpointer_classes)
{
}

bool
AppleObjCRuntimeV2::TaggedPointerVendorRuntimeAssisted::IsPossibleTaggedPointer (lldb::addr_t ptr)
{
    return (ptr & m_objc_debug_taggedpointer_mask) != 0;
}

ObjCLanguageRuntime::ClassDescriptorSP
AppleObjCRuntimeV2::TaggedPointerVendorRuntimeAssisted::GetClassDescriptor (lldb::addr_t ptr)
{
    ClassDescriptorSP actual_class_descriptor_sp;
    uint64_t data_payload;

    if (!IsPossibleTaggedPointer(ptr))
        return ObjCLanguageRuntime::ClassDescriptorSP();
    
    uintptr_t slot = (ptr >> m_objc_debug_taggedpointer_slot_shift) & m_objc_debug_taggedpointer_slot_mask;
    
    CacheIterator iterator = m_cache.find(slot),
    end = m_cache.end();
    if (iterator != end)
    {
        actual_class_descriptor_sp = iterator->second;
    }
    else
    {
        Process* process(m_runtime.GetProcess());
        uintptr_t slot_ptr = slot*process->GetAddressByteSize()+m_objc_debug_taggedpointer_classes;
        Error error;
        uintptr_t slot_data = process->ReadPointerFromMemory(slot_ptr, error);
        if (error.Fail() || slot_data == 0 || slot_data == LLDB_INVALID_ADDRESS)
            return nullptr;
        actual_class_descriptor_sp = m_runtime.GetClassDescriptor(slot_data);
        if (!actual_class_descriptor_sp)
            return ObjCLanguageRuntime::ClassDescriptorSP();
        m_cache[slot] = actual_class_descriptor_sp;
    }
    
    data_payload = (((uint64_t)ptr << m_objc_debug_taggedpointer_payload_lshift) >> m_objc_debug_taggedpointer_payload_rshift);
    
    return ClassDescriptorSP(new ClassDescriptorV2Tagged(actual_class_descriptor_sp,data_payload));
}

AppleObjCRuntimeV2::NonPointerISACache::NonPointerISACache (AppleObjCRuntimeV2& runtime,
                                                            uint64_t objc_debug_isa_class_mask,
                                                            uint64_t objc_debug_isa_magic_mask,
                                                            uint64_t objc_debug_isa_magic_value) :
m_runtime(runtime),
m_cache(),
m_objc_debug_isa_class_mask(objc_debug_isa_class_mask),
m_objc_debug_isa_magic_mask(objc_debug_isa_magic_mask),
m_objc_debug_isa_magic_value(objc_debug_isa_magic_value)
{
}

ObjCLanguageRuntime::ClassDescriptorSP
AppleObjCRuntimeV2::NonPointerISACache::GetClassDescriptor (ObjCISA isa)
{
    ObjCISA real_isa = 0;
    if (EvaluateNonPointerISA(isa, real_isa) == false)
        return ObjCLanguageRuntime::ClassDescriptorSP();
    auto cache_iter = m_cache.find(real_isa);
    if (cache_iter != m_cache.end())
        return cache_iter->second;
    auto descriptor_sp = m_runtime.ObjCLanguageRuntime::GetClassDescriptorFromISA(real_isa);
    if (descriptor_sp) // cache only positive matches since the table might grow
        m_cache[real_isa] = descriptor_sp;
    return descriptor_sp;
}

bool
AppleObjCRuntimeV2::NonPointerISACache::EvaluateNonPointerISA (ObjCISA isa, ObjCISA& ret_isa)
{
    if ( (isa & ~m_objc_debug_isa_class_mask) == 0)
        return false;
    if ( (isa & m_objc_debug_isa_magic_mask) == m_objc_debug_isa_magic_value)
    {
        ret_isa = isa & m_objc_debug_isa_class_mask;
        return (ret_isa != 0); // this is a pointer so 0 is not a valid value
    }
    return false;
}
