//===-- RenderScriptRuntime.cpp ---------------------------------*- C++ -*-===//
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
#include "RenderScriptRuntime.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/DataFormatters/DumpValueObjectOptions.h"
#include "lldb/Host/StringConvert.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Expression/UserExpression.h"
#include "lldb/Symbol/VariableList.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_renderscript;

namespace {

// The empirical_type adds a basic level of validation to arbitrary data
// allowing us to track if data has been discovered and stored or not.
// An empirical_type will be marked as valid only if it has been explicitly assigned to.
template <typename type_t>
class empirical_type
{
public:
    // Ctor. Contents is invalid when constructed.
    empirical_type()
        : valid(false)
    {}

    // Return true and copy contents to out if valid, else return false.
    bool get(type_t& out) const
    {
        if (valid)
            out = data;
        return valid;
    }

    // Return a pointer to the contents or nullptr if it was not valid.
    const type_t* get() const
    {
        return valid ? &data : nullptr;
    }

    // Assign data explicitly.
    void set(const type_t in)
    {
        data = in;
        valid = true;
    }

    // Mark contents as invalid.
    void invalidate()
    {
        valid = false;
    }

    // Returns true if this type contains valid data.
    bool isValid() const
    {
        return valid;
    }

    // Assignment operator.
    empirical_type<type_t>& operator = (const type_t in)
    {
        set(in);
        return *this;
    }

    // Dereference operator returns contents.
    // Warning: Will assert if not valid so use only when you know data is valid.
    const type_t& operator * () const
    {
        assert(valid);
        return data;
    }

protected:
    bool valid;
    type_t data;
};

} // anonymous namespace

// The ScriptDetails class collects data associated with a single script instance.
struct RenderScriptRuntime::ScriptDetails
{
    ~ScriptDetails() = default;

    enum ScriptType
    {
        eScript,
        eScriptC
    };

    // The derived type of the script.
    empirical_type<ScriptType> type;
    // The name of the original source file.
    empirical_type<std::string> resName;
    // Path to script .so file on the device.
    empirical_type<std::string> scriptDyLib;
    // Directory where kernel objects are cached on device.
    empirical_type<std::string> cacheDir;
    // Pointer to the context which owns this script.
    empirical_type<lldb::addr_t> context;
    // Pointer to the script object itself.
    empirical_type<lldb::addr_t> script;
};

// This Element class represents the Element object in RS,
// defining the type associated with an Allocation.
struct RenderScriptRuntime::Element
{
    // Taken from rsDefines.h
    enum DataKind
    {
        RS_KIND_USER,
        RS_KIND_PIXEL_L = 7,
        RS_KIND_PIXEL_A,
        RS_KIND_PIXEL_LA,
        RS_KIND_PIXEL_RGB,
        RS_KIND_PIXEL_RGBA,
        RS_KIND_PIXEL_DEPTH,
        RS_KIND_PIXEL_YUV,
        RS_KIND_INVALID = 100
    };

    // Taken from rsDefines.h
    enum DataType
    {
        RS_TYPE_NONE = 0,
        RS_TYPE_FLOAT_16,
        RS_TYPE_FLOAT_32,
        RS_TYPE_FLOAT_64,
        RS_TYPE_SIGNED_8,
        RS_TYPE_SIGNED_16,
        RS_TYPE_SIGNED_32,
        RS_TYPE_SIGNED_64,
        RS_TYPE_UNSIGNED_8,
        RS_TYPE_UNSIGNED_16,
        RS_TYPE_UNSIGNED_32,
        RS_TYPE_UNSIGNED_64,
        RS_TYPE_BOOLEAN,

        RS_TYPE_UNSIGNED_5_6_5,
        RS_TYPE_UNSIGNED_5_5_5_1,
        RS_TYPE_UNSIGNED_4_4_4_4,

        RS_TYPE_MATRIX_4X4,
        RS_TYPE_MATRIX_3X3,
        RS_TYPE_MATRIX_2X2,

        RS_TYPE_ELEMENT = 1000,
        RS_TYPE_TYPE,
        RS_TYPE_ALLOCATION,
        RS_TYPE_SAMPLER,
        RS_TYPE_SCRIPT,
        RS_TYPE_MESH,
        RS_TYPE_PROGRAM_FRAGMENT,
        RS_TYPE_PROGRAM_VERTEX,
        RS_TYPE_PROGRAM_RASTER,
        RS_TYPE_PROGRAM_STORE,
        RS_TYPE_FONT,

        RS_TYPE_INVALID = 10000
    };

    std::vector<Element> children;                       // Child Element fields for structs
    empirical_type<lldb::addr_t> element_ptr;            // Pointer to the RS Element of the Type
    empirical_type<DataType> type;                       // Type of each data pointer stored by the allocation
    empirical_type<DataKind> type_kind;                  // Defines pixel type if Allocation is created from an image
    empirical_type<uint32_t> type_vec_size;              // Vector size of each data point, e.g '4' for uchar4
    empirical_type<uint32_t> field_count;                // Number of Subelements
    empirical_type<uint32_t> datum_size;                 // Size of a single Element with padding
    empirical_type<uint32_t> padding;                    // Number of padding bytes
    empirical_type<uint32_t> array_size;                 // Number of items in array, only needed for strucrs
    ConstString type_name;                               // Name of type, only needed for structs

    static const ConstString &GetFallbackStructName();   // Print this as the type name of a struct Element
                                                         // If we can't resolve the actual struct name

    bool shouldRefresh() const
    {
        const bool valid_ptr = element_ptr.isValid() && *element_ptr.get() != 0x0;
        const bool valid_type = type.isValid() && type_vec_size.isValid() && type_kind.isValid();
        return !valid_ptr || !valid_type || !datum_size.isValid();
    }
};

// This AllocationDetails class collects data associated with a single
// allocation instance.
struct RenderScriptRuntime::AllocationDetails
{
    struct Dimension
    {
        uint32_t dim_1;
        uint32_t dim_2;
        uint32_t dim_3;
        uint32_t cubeMap;

        Dimension()
        {
             dim_1 = 0;
             dim_2 = 0;
             dim_3 = 0;
             cubeMap = 0;
        }
    };

    // The FileHeader struct specifies the header we use for writing allocations to a binary file.
    // Our format begins with the ASCII characters "RSAD", identifying the file as an allocation dump.
    // Member variables dims and hdr_size are then written consecutively, immediately followed by an instance of
    // the ElementHeader struct. Because Elements can contain subelements, there may be more than one instance
    // of the ElementHeader struct. With this first instance being the root element, and the other instances being
    // the root's descendants. To identify which instances are an ElementHeader's children, each struct
    // is immediately followed by a sequence of consecutive offsets to the start of its child structs.
    // These offsets are 4 bytes in size, and the 0 offset signifies no more children.
    struct FileHeader
    {
        uint8_t ident[4];      // ASCII 'RSAD' identifying the file
        uint32_t dims[3];      // Dimensions
        uint16_t hdr_size;     // Header size in bytes, including all element headers
    };

    struct ElementHeader
    {
        uint16_t type;          // DataType enum
        uint32_t kind;          // DataKind enum
        uint32_t element_size;  // Size of a single element, including padding
        uint16_t vector_size;   // Vector width
        uint32_t array_size;    // Number of elements in array
    };

    // Monotonically increasing from 1
    static unsigned int ID;

    // Maps Allocation DataType enum and vector size to printable strings
    // using mapping from RenderScript numerical types summary documentation
    static const char* RsDataTypeToString[][4];

    // Maps Allocation DataKind enum to printable strings
    static const char* RsDataKindToString[];

    // Maps allocation types to format sizes for printing.
    static const unsigned int RSTypeToFormat[][3];

    // Give each allocation an ID as a way
    // for commands to reference it.
    const unsigned int id;

    RenderScriptRuntime::Element element;     // Allocation Element type
    empirical_type<Dimension> dimension;      // Dimensions of the Allocation
    empirical_type<lldb::addr_t> address;     // Pointer to address of the RS Allocation
    empirical_type<lldb::addr_t> data_ptr;    // Pointer to the data held by the Allocation
    empirical_type<lldb::addr_t> type_ptr;    // Pointer to the RS Type of the Allocation
    empirical_type<lldb::addr_t> context;     // Pointer to the RS Context of the Allocation
    empirical_type<uint32_t> size;            // Size of the allocation
    empirical_type<uint32_t> stride;          // Stride between rows of the allocation

    // Give each allocation an id, so we can reference it in user commands.
    AllocationDetails(): id(ID++)
    {
    }

    bool shouldRefresh() const
    {
        bool valid_ptrs = data_ptr.isValid() && *data_ptr.get() != 0x0;
        valid_ptrs = valid_ptrs && type_ptr.isValid() && *type_ptr.get() != 0x0;
        return !valid_ptrs || !dimension.isValid() || !size.isValid() || element.shouldRefresh();
    }
};

const ConstString &
RenderScriptRuntime::Element::GetFallbackStructName()
{
    static const ConstString FallbackStructName("struct");
    return FallbackStructName;
}

unsigned int RenderScriptRuntime::AllocationDetails::ID = 1;

const char* RenderScriptRuntime::AllocationDetails::RsDataKindToString[] =
{
   "User",
   "Undefined", "Undefined", "Undefined", // Enum jumps from 0 to 7
   "Undefined", "Undefined", "Undefined",
   "L Pixel",
   "A Pixel",
   "LA Pixel",
   "RGB Pixel",
   "RGBA Pixel",
   "Pixel Depth",
   "YUV Pixel"
};

const char* RenderScriptRuntime::AllocationDetails::RsDataTypeToString[][4] =
{
    {"None", "None", "None", "None"},
    {"half", "half2", "half3", "half4"},
    {"float", "float2", "float3", "float4"},
    {"double", "double2", "double3", "double4"},
    {"char", "char2", "char3", "char4"},
    {"short", "short2", "short3", "short4"},
    {"int", "int2", "int3", "int4"},
    {"long", "long2", "long3", "long4"},
    {"uchar", "uchar2", "uchar3", "uchar4"},
    {"ushort", "ushort2", "ushort3", "ushort4"},
    {"uint", "uint2", "uint3", "uint4"},
    {"ulong", "ulong2", "ulong3", "ulong4"},
    {"bool", "bool2", "bool3", "bool4"},
    {"packed_565", "packed_565", "packed_565", "packed_565"},
    {"packed_5551", "packed_5551", "packed_5551", "packed_5551"},
    {"packed_4444", "packed_4444", "packed_4444", "packed_4444"},
    {"rs_matrix4x4", "rs_matrix4x4", "rs_matrix4x4", "rs_matrix4x4"},
    {"rs_matrix3x3", "rs_matrix3x3", "rs_matrix3x3", "rs_matrix3x3"},
    {"rs_matrix2x2", "rs_matrix2x2", "rs_matrix2x2", "rs_matrix2x2"},

    // Handlers
    {"RS Element", "RS Element", "RS Element", "RS Element"},
    {"RS Type", "RS Type", "RS Type", "RS Type"},
    {"RS Allocation", "RS Allocation", "RS Allocation", "RS Allocation"},
    {"RS Sampler", "RS Sampler", "RS Sampler", "RS Sampler"},
    {"RS Script", "RS Script", "RS Script", "RS Script"},

    // Deprecated
    {"RS Mesh", "RS Mesh", "RS Mesh", "RS Mesh"},
    {"RS Program Fragment", "RS Program Fragment", "RS Program Fragment", "RS Program Fragment"},
    {"RS Program Vertex", "RS Program Vertex", "RS Program Vertex", "RS Program Vertex"},
    {"RS Program Raster", "RS Program Raster", "RS Program Raster", "RS Program Raster"},
    {"RS Program Store", "RS Program Store", "RS Program Store", "RS Program Store"},
    {"RS Font", "RS Font", "RS Font", "RS Font"}
};

// Used as an index into the RSTypeToFormat array elements
enum TypeToFormatIndex {
   eFormatSingle = 0,
   eFormatVector,
   eElementSize
};

// { format enum of single element, format enum of element vector, size of element}
const unsigned int RenderScriptRuntime::AllocationDetails::RSTypeToFormat[][3] =
{
    {eFormatHex, eFormatHex, 1}, // RS_TYPE_NONE
    {eFormatFloat, eFormatVectorOfFloat16, 2}, // RS_TYPE_FLOAT_16
    {eFormatFloat, eFormatVectorOfFloat32, sizeof(float)}, // RS_TYPE_FLOAT_32
    {eFormatFloat, eFormatVectorOfFloat64, sizeof(double)}, // RS_TYPE_FLOAT_64
    {eFormatDecimal, eFormatVectorOfSInt8, sizeof(int8_t)}, // RS_TYPE_SIGNED_8
    {eFormatDecimal, eFormatVectorOfSInt16, sizeof(int16_t)}, // RS_TYPE_SIGNED_16
    {eFormatDecimal, eFormatVectorOfSInt32, sizeof(int32_t)}, // RS_TYPE_SIGNED_32
    {eFormatDecimal, eFormatVectorOfSInt64, sizeof(int64_t)}, // RS_TYPE_SIGNED_64
    {eFormatDecimal, eFormatVectorOfUInt8, sizeof(uint8_t)}, // RS_TYPE_UNSIGNED_8
    {eFormatDecimal, eFormatVectorOfUInt16, sizeof(uint16_t)}, // RS_TYPE_UNSIGNED_16
    {eFormatDecimal, eFormatVectorOfUInt32, sizeof(uint32_t)}, // RS_TYPE_UNSIGNED_32
    {eFormatDecimal, eFormatVectorOfUInt64, sizeof(uint64_t)}, // RS_TYPE_UNSIGNED_64
    {eFormatBoolean, eFormatBoolean, 1}, // RS_TYPE_BOOL
    {eFormatHex, eFormatHex, sizeof(uint16_t)}, // RS_TYPE_UNSIGNED_5_6_5
    {eFormatHex, eFormatHex, sizeof(uint16_t)}, // RS_TYPE_UNSIGNED_5_5_5_1
    {eFormatHex, eFormatHex, sizeof(uint16_t)}, // RS_TYPE_UNSIGNED_4_4_4_4
    {eFormatVectorOfFloat32, eFormatVectorOfFloat32, sizeof(float) * 16}, // RS_TYPE_MATRIX_4X4
    {eFormatVectorOfFloat32, eFormatVectorOfFloat32, sizeof(float) * 9}, // RS_TYPE_MATRIX_3X3
    {eFormatVectorOfFloat32, eFormatVectorOfFloat32, sizeof(float) * 4} // RS_TYPE_MATRIX_2X2
};

const std::string RenderScriptRuntime::s_runtimeExpandSuffix(".expand");
const std::array<const char *, 3> RenderScriptRuntime::s_runtimeCoordVars{{"rsIndex", "p->current.y", "p->current.z"}};
//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
LanguageRuntime *
RenderScriptRuntime::CreateInstance(Process *process, lldb::LanguageType language)
{

    if (language == eLanguageTypeExtRenderScript)
        return new RenderScriptRuntime(process);
    else
        return NULL;
}

// Callback with a module to search for matching symbols.
// We first check that the module contains RS kernels.
// Then look for a symbol which matches our kernel name.
// The breakpoint address is finally set using the address of this symbol.
Searcher::CallbackReturn
RSBreakpointResolver::SearchCallback(SearchFilter &filter,
                                     SymbolContext &context,
                                     Address*,
                                     bool)
{
    ModuleSP module = context.module_sp;

    if (!module)
        return Searcher::eCallbackReturnContinue;

    // Is this a module containing renderscript kernels?
    if (nullptr == module->FindFirstSymbolWithNameAndType(ConstString(".rs.info"), eSymbolTypeData))
        return Searcher::eCallbackReturnContinue;

    // Attempt to set a breakpoint on the kernel name symbol within the module library.
    // If it's not found, it's likely debug info is unavailable - try to set a
    // breakpoint on <name>.expand.

    const Symbol* kernel_sym = module->FindFirstSymbolWithNameAndType(m_kernel_name, eSymbolTypeCode);
    if (!kernel_sym)
    {
        std::string kernel_name_expanded(m_kernel_name.AsCString());
        kernel_name_expanded.append(".expand");
        kernel_sym = module->FindFirstSymbolWithNameAndType(ConstString(kernel_name_expanded.c_str()), eSymbolTypeCode);
    }

    if (kernel_sym)
    {
        Address bp_addr = kernel_sym->GetAddress();
        if (filter.AddressPasses(bp_addr))
            m_breakpoint->AddLocation(bp_addr);
    }

    return Searcher::eCallbackReturnContinue;
}

void
RenderScriptRuntime::Initialize()
{
    PluginManager::RegisterPlugin(GetPluginNameStatic(), "RenderScript language support", CreateInstance, GetCommandObject);
}

void
RenderScriptRuntime::Terminate()
{
    PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::ConstString
RenderScriptRuntime::GetPluginNameStatic()
{
    static ConstString g_name("renderscript");
    return g_name;
}

RenderScriptRuntime::ModuleKind
RenderScriptRuntime::GetModuleKind(const lldb::ModuleSP &module_sp)
{
    if (module_sp)
    {
        // Is this a module containing renderscript kernels?
        const Symbol *info_sym = module_sp->FindFirstSymbolWithNameAndType(ConstString(".rs.info"), eSymbolTypeData);
        if (info_sym)
        {
            return eModuleKindKernelObj;
        }

        // Is this the main RS runtime library
        const ConstString rs_lib("libRS.so");
        if (module_sp->GetFileSpec().GetFilename() == rs_lib)
        {
            return eModuleKindLibRS;
        }

        const ConstString rs_driverlib("libRSDriver.so");
        if (module_sp->GetFileSpec().GetFilename() == rs_driverlib)
        {
            return eModuleKindDriver;
        }

        const ConstString rs_cpureflib("libRSCpuRef.so");
        if (module_sp->GetFileSpec().GetFilename() == rs_cpureflib)
        {
            return eModuleKindImpl;
        }

    }
    return eModuleKindIgnored;
}

bool
RenderScriptRuntime::IsRenderScriptModule(const lldb::ModuleSP &module_sp)
{
    return GetModuleKind(module_sp) != eModuleKindIgnored;
}

void
RenderScriptRuntime::ModulesDidLoad(const ModuleList &module_list )
{
    Mutex::Locker locker (module_list.GetMutex ());

    size_t num_modules = module_list.GetSize();
    for (size_t i = 0; i < num_modules; i++)
    {
        auto mod = module_list.GetModuleAtIndex (i);
        if (IsRenderScriptModule (mod))
        {
            LoadModule(mod);
        }
    }
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
lldb_private::ConstString
RenderScriptRuntime::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
RenderScriptRuntime::GetPluginVersion()
{
    return 1;
}

bool
RenderScriptRuntime::IsVTableName(const char *name)
{
    return false;
}

bool
RenderScriptRuntime::GetDynamicTypeAndAddress(ValueObject &in_value, lldb::DynamicValueType use_dynamic,
                                              TypeAndOrName &class_type_or_name, Address &address,
                                              Value::ValueType &value_type)
{
    return false;
}

TypeAndOrName
RenderScriptRuntime::FixUpDynamicType (const TypeAndOrName& type_and_or_name,
                                       ValueObject& static_value)
{
    return type_and_or_name;
}

bool
RenderScriptRuntime::CouldHaveDynamicValue(ValueObject &in_value)
{
    return false;
}

lldb::BreakpointResolverSP
RenderScriptRuntime::CreateExceptionResolver(Breakpoint *bkpt, bool catch_bp, bool throw_bp)
{
    BreakpointResolverSP resolver_sp;
    return resolver_sp;
}

const RenderScriptRuntime::HookDefn RenderScriptRuntime::s_runtimeHookDefns[] =
{
    //rsdScript
    {
        "rsdScriptInit", //name
        "_Z13rsdScriptInitPKN7android12renderscript7ContextEPNS0_7ScriptCEPKcS7_PKhjj", // symbol name 32 bit
        "_Z13rsdScriptInitPKN7android12renderscript7ContextEPNS0_7ScriptCEPKcS7_PKhmj", // symbol name 64 bit
        0, // version
        RenderScriptRuntime::eModuleKindDriver, // type
        &lldb_private::RenderScriptRuntime::CaptureScriptInit1 // handler
    },
    {
        "rsdScriptInvokeForEachMulti", // name
        "_Z27rsdScriptInvokeForEachMultiPKN7android12renderscript7ContextEPNS0_6ScriptEjPPKNS0_10AllocationEjPS6_PKvjPK12RsScriptCall", // symbol name 32bit
        "_Z27rsdScriptInvokeForEachMultiPKN7android12renderscript7ContextEPNS0_6ScriptEjPPKNS0_10AllocationEmPS6_PKvmPK12RsScriptCall", // symbol name 64bit
        0, // version
        RenderScriptRuntime::eModuleKindDriver, // type
        &lldb_private::RenderScriptRuntime::CaptureScriptInvokeForEachMulti // handler
    },
    {
        "rsdScriptSetGlobalVar", // name
        "_Z21rsdScriptSetGlobalVarPKN7android12renderscript7ContextEPKNS0_6ScriptEjPvj", // symbol name 32bit
        "_Z21rsdScriptSetGlobalVarPKN7android12renderscript7ContextEPKNS0_6ScriptEjPvm", // symbol name 64bit
        0, // version
        RenderScriptRuntime::eModuleKindDriver, // type
        &lldb_private::RenderScriptRuntime::CaptureSetGlobalVar1 // handler
    },

    //rsdAllocation
    {
        "rsdAllocationInit", // name
        "_Z17rsdAllocationInitPKN7android12renderscript7ContextEPNS0_10AllocationEb", // symbol name 32bit
        "_Z17rsdAllocationInitPKN7android12renderscript7ContextEPNS0_10AllocationEb", // symbol name 64bit
        0, // version
        RenderScriptRuntime::eModuleKindDriver, // type
        &lldb_private::RenderScriptRuntime::CaptureAllocationInit1 // handler
    },
    {
        "rsdAllocationRead2D", //name
        "_Z19rsdAllocationRead2DPKN7android12renderscript7ContextEPKNS0_10AllocationEjjj23RsAllocationCubemapFacejjPvjj", // symbol name 32bit
        "_Z19rsdAllocationRead2DPKN7android12renderscript7ContextEPKNS0_10AllocationEjjj23RsAllocationCubemapFacejjPvmm", // symbol name 64bit
        0, // version
        RenderScriptRuntime::eModuleKindDriver, // type
        nullptr // handler
    },
    {
        "rsdAllocationDestroy", // name
        "_Z20rsdAllocationDestroyPKN7android12renderscript7ContextEPNS0_10AllocationE", // symbol name 32bit
        "_Z20rsdAllocationDestroyPKN7android12renderscript7ContextEPNS0_10AllocationE", // symbol name 64bit
        0, // version
        RenderScriptRuntime::eModuleKindDriver, // type
        &lldb_private::RenderScriptRuntime::CaptureAllocationDestroy // handler
    },
};

const size_t RenderScriptRuntime::s_runtimeHookCount = sizeof(s_runtimeHookDefns)/sizeof(s_runtimeHookDefns[0]);

bool
RenderScriptRuntime::HookCallback(void *baton, StoppointCallbackContext *ctx, lldb::user_id_t break_id, lldb::user_id_t break_loc_id)
{
    RuntimeHook* hook_info = (RuntimeHook*)baton;
    ExecutionContext context(ctx->exe_ctx_ref);

    RenderScriptRuntime *lang_rt = (RenderScriptRuntime *)context.GetProcessPtr()->GetLanguageRuntime(eLanguageTypeExtRenderScript);

    lang_rt->HookCallback(hook_info, context);

    return false;
}

void
RenderScriptRuntime::HookCallback(RuntimeHook* hook_info, ExecutionContext& context)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    if (log)
        log->Printf ("RenderScriptRuntime::HookCallback - '%s' .", hook_info->defn->name);

    if (hook_info->defn->grabber)
    {
        (this->*(hook_info->defn->grabber))(hook_info, context);
    }
}

bool
RenderScriptRuntime::GetArgSimple(ExecutionContext &context, uint32_t arg, uint64_t *data)
{
    // Get a positional integer argument.
    // Given an ExecutionContext, ``context`` which should be a RenderScript
    // frame, get the value of the positional argument ``arg`` and save its value
    // to the address pointed to by ``data``.
    // returns true on success, false otherwise.
    // If unsuccessful, the value pointed to by ``data`` is undefined. Otherwise,
    // ``data`` will be set to the value of the the given ``arg``.
    // NOTE: only natural width integer arguments for the machine are supported.
    // Behaviour with non primitive arguments is undefined.

    if (!data)
        return false;

    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));
    Error error;
    RegisterContext* reg_ctx = context.GetRegisterContext();
    Process* process = context.GetProcessPtr();
    bool success = false; // return value

    if (!context.GetTargetPtr())
    {
        if (log)
            log->Printf("RenderScriptRuntime::GetArgSimple - Invalid target");

        return false;
    }

    switch (context.GetTargetPtr()->GetArchitecture().GetMachine())
    {
        case llvm::Triple::ArchType::x86:
        {
            uint64_t sp = reg_ctx->GetSP();
            uint32_t offset = (1 + arg) * sizeof(uint32_t);
            uint32_t result = 0;
            process->ReadMemory(sp + offset, &result, sizeof(uint32_t), error);
            if (error.Fail())
            {
                if (log)
                    log->Printf("RenderScriptRuntime::GetArgSimple - error reading X86 stack: %s.", error.AsCString());
            }
            else
            {
                *data = result;
                success = true;
            }
            break;
        }
        case llvm::Triple::ArchType::x86_64:
        {
            // amd64 has 6 integer registers, and 8 XMM registers for parameter passing.
            // Surplus args are spilled onto the stack.
            // rdi, rsi, rdx, rcx, r8, r9, (zmm0 - 7 for vectors)
            // ref: AMD64 ABI Draft 0.99.6 – October 7, 2013 – 10:35; Figure 3.4. Retrieved from
            // http://www.x86-64.org/documentation/abi.pdf
            if (arg > 5)
            {
                if (log)
                    log->Warning("X86_64 register spill is not supported.");
                break;
            }
            const char * regnames[] = {"rdi", "rsi", "rdx", "rcx", "r8", "r9"};
            assert((sizeof(regnames) / sizeof(const char *)) > arg);
            const RegisterInfo *rArg = reg_ctx->GetRegisterInfoByName(regnames[arg]);
            RegisterValue rVal;
            success = reg_ctx->ReadRegister(rArg, rVal);
            if (success)
            {
                *data = rVal.GetAsUInt64(0u, &success);
            }
            else
            {
                if (log)
                    log->Printf("RenderScriptRuntime::GetArgSimple - error reading x86_64 register: %d.", arg);
            }
            break;
        }
        case llvm::Triple::ArchType::arm:
        {
            // arm 32 bit
            // first 4 arguments are passed via registers
            if (arg < 4)
            {
                const RegisterInfo* rArg = reg_ctx->GetRegisterInfoAtIndex(arg);
                RegisterValue rVal;
                success = reg_ctx->ReadRegister(rArg, rVal);
                if (success)
                {
                    (*data) = rVal.GetAsUInt32(0u, &success);
                }
                else
                {
                    if (log)
                        log->Printf("RenderScriptRuntime::GetArgSimple - error reading ARM register: %d.", arg);
                }
            }
            else
            {
                uint64_t sp = reg_ctx->GetSP();
                uint32_t offset = (arg-4) * sizeof(uint32_t);
                uint32_t value = 0;
                size_t bytes_read = process->ReadMemory(sp + offset, &value, sizeof(value), error);
                if (error.Fail() || bytes_read != sizeof(value))
                {
                    if (log)
                        log->Printf("RenderScriptRuntime::GetArgSimple - error reading ARM stack: %s.", error.AsCString());
                }
                else
                {
                    *data = value;
                    success = true;
                }
            }
            break;
        }
        case llvm::Triple::ArchType::aarch64:
        {
            // arm 64 bit
            // first 8 arguments are in the registers
            if (arg < 8)
            {
                const RegisterInfo* rArg = reg_ctx->GetRegisterInfoAtIndex(arg);
                RegisterValue rVal;
                success = reg_ctx->ReadRegister(rArg, rVal);
                if (success)
                {
                    *data = rVal.GetAsUInt64(0u, &success);
                }
                else
                {
                    if (log)
                        log->Printf("RenderScriptRuntime::GetArgSimple() - AARCH64 - Error while reading the argument #%d", arg);
                }
            }
            else
            {
                // @TODO: need to find the argument in the stack
                if (log)
                    log->Printf("RenderScriptRuntime::GetArgSimple - AARCH64 - FOR #ARG >= 8 NOT IMPLEMENTED YET. Argument number: %d", arg);
            }
            break;
        }
        case llvm::Triple::ArchType::mipsel:
        {
            // read from the registers
            // first 4 arguments are passed in registers
            if (arg < 4){
                const RegisterInfo* rArg = reg_ctx->GetRegisterInfoAtIndex(arg + 4);
                RegisterValue rVal;
                success = reg_ctx->ReadRegister(rArg, rVal);
                if (success)
                {
                    *data = rVal.GetAsUInt64(0u, &success);
                }
                else
                {
                    if (log)
                        log->Printf("RenderScriptRuntime::GetArgSimple() - Mips - Error while reading the argument #%d", arg);
                }
            }
            // arguments > 4 are read from the stack
            else
            {
                uint64_t sp = reg_ctx->GetSP();
                uint32_t offset = arg * sizeof(uint32_t);
                uint32_t value = 0;
                size_t bytes_read = process->ReadMemory(sp + offset, &value, sizeof(value), error);
                if (error.Fail() || bytes_read != sizeof(value))
                {
                    if (log)
                        log->Printf("RenderScriptRuntime::GetArgSimple - error reading Mips stack: %s.", error.AsCString());
                }
                else
                {
                    *data = value;
                    success = true;
                }
            }
            break;
        }
        case llvm::Triple::ArchType::mips64el:
        {
            // read from the registers
            if (arg < 8)
            {
                const RegisterInfo* rArg = reg_ctx->GetRegisterInfoAtIndex(arg + 4);
                RegisterValue rVal;
                success = reg_ctx->ReadRegister(rArg, rVal);
                if (success)
                {
                    (*data) = rVal.GetAsUInt64(0u, &success);
                }
                else
                {
                    if (log)
                        log->Printf("RenderScriptRuntime::GetArgSimple - Mips64 - Error reading the argument #%d", arg);
                }
            }
            // arguments > 8 are read from the stack
            else
            {
                uint64_t sp = reg_ctx->GetSP();
                uint32_t offset = (arg - 8) * sizeof(uint64_t);
                uint64_t value = 0;
                size_t bytes_read = process->ReadMemory(sp + offset, &value, sizeof(value), error);
                if (error.Fail() || bytes_read != sizeof(value))
                {
                    if (log)
                        log->Printf("RenderScriptRuntime::GetArgSimple - Mips64 - Error reading Mips64 stack: %s.", error.AsCString());
                }
                else
                {
                    *data = value;
                    success = true;
                }
            }
            break;
        }
        default:
        {
            // invalid architecture
            if (log)
                log->Printf("RenderScriptRuntime::GetArgSimple - Architecture not supported");
        }
    }

    if (!success)
    {
        if (log)
            log->Printf("RenderScriptRuntime::GetArgSimple - failed to get argument at index %" PRIu32, arg);
    }
    return success;
}

void
RenderScriptRuntime::CaptureScriptInvokeForEachMulti(RuntimeHook* hook_info,
                                                     ExecutionContext& context)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    struct args_t
    {
        uint64_t context;   // const Context       *rsc
        uint64_t script;    // Script              *s
        uint64_t slot;      // uint32_t             slot
        uint64_t aIns;      // const Allocation   **aIns
        uint64_t inLen;     // size_t               inLen
        uint64_t aOut;      // Allocation          *aout
        uint64_t usr;       // const void          *usr
        uint64_t usrLen;    // size_t               usrLen
        uint64_t sc;        // const RsScriptCall  *sc
    }
    args;

    bool success =
        GetArgSimple(context, 0, &args.context) &&
        GetArgSimple(context, 1, &args.script) &&
        GetArgSimple(context, 2, &args.slot) &&
        GetArgSimple(context, 3, &args.aIns) &&
        GetArgSimple(context, 4, &args.inLen) &&
        GetArgSimple(context, 5, &args.aOut) &&
        GetArgSimple(context, 6, &args.usr) &&
        GetArgSimple(context, 7, &args.usrLen) &&
        GetArgSimple(context, 8, &args.sc);

    if (!success)
    {
        if (log)
            log->Printf("RenderScriptRuntime::CaptureScriptInvokeForEachMulti()"
                        " - Error while reading the function parameters");
        return;
    }

    const uint32_t target_ptr_size = m_process->GetAddressByteSize();
    Error error;
    std::vector<uint64_t> allocs;

    // traverse allocation list
    for (uint64_t i = 0; i < args.inLen; ++i)
    {
        // calculate offest to allocation pointer
        const lldb::addr_t addr = args.aIns + i * target_ptr_size;

        // Note: due to little endian layout, reading 32bits or 64bits into res64 will
        //       give the correct results.

        uint64_t res64 = 0;
        size_t read = m_process->ReadMemory(addr, &res64, target_ptr_size, error);
        if (read != target_ptr_size || !error.Success())
        {
            if (log)
                log->Printf("RenderScriptRuntime::CaptureScriptInvokeForEachMulti()"
                            " - Error while reading allocation list argument %" PRId64, i);
        }
        else
        {
            allocs.push_back(res64);
        }
    }

    // if there is an output allocation track it
    if (args.aOut)
    {
        allocs.push_back(args.aOut);
    }

    // for all allocations we have found
    for (const uint64_t alloc_addr : allocs)
    {
        AllocationDetails* alloc = LookUpAllocation(alloc_addr, true);
        if (alloc)
        {
            // save the allocation address
            if (alloc->address.isValid())
            {
                // check the allocation address we already have matches
                assert(*alloc->address.get() == alloc_addr);
            }
            else
            {
                alloc->address = alloc_addr;
            }

            // save the context
            if (log)
            {
                if (alloc->context.isValid() && *alloc->context.get() != args.context)
                    log->Printf("RenderScriptRuntime::CaptureScriptInvokeForEachMulti"
                                " - Allocation used by multiple contexts");
            }
            alloc->context = args.context;
        }
    }

    // make sure we track this script object
    if (lldb_private::RenderScriptRuntime::ScriptDetails * script = LookUpScript(args.script, true))
    {
        if (log)
        {
            if (script->context.isValid() && *script->context.get() != args.context)
                log->Printf("RenderScriptRuntime::CaptureScriptInvokeForEachMulti"
                            " - Script used by multiple contexts");
        }
        script->context = args.context;
    }
}

void
RenderScriptRuntime::CaptureSetGlobalVar1(RuntimeHook* hook_info, ExecutionContext& context)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    //Context, Script, int, data, length

    uint64_t rs_context_u64 = 0U;
    uint64_t rs_script_u64 = 0U;
    uint64_t rs_id_u64 = 0U;
    uint64_t rs_data_u64 = 0U;
    uint64_t rs_length_u64 = 0U;

    bool success =
        GetArgSimple(context, 0, &rs_context_u64) &&
        GetArgSimple(context, 1, &rs_script_u64) &&
        GetArgSimple(context, 2, &rs_id_u64) &&
        GetArgSimple(context, 3, &rs_data_u64) &&
        GetArgSimple(context, 4, &rs_length_u64);

    if (!success)
    {
        if (log)
            log->Printf("RenderScriptRuntime::CaptureSetGlobalVar1 - Error while reading the function parameters");
        return;
    }

    if (log)
    {
        log->Printf ("RenderScriptRuntime::CaptureSetGlobalVar1 - 0x%" PRIx64 ",0x%" PRIx64 " slot %" PRIu64 " = 0x%" PRIx64 ":%" PRIu64 "bytes.",
                        rs_context_u64, rs_script_u64, rs_id_u64, rs_data_u64, rs_length_u64);

        addr_t script_addr =  (addr_t)rs_script_u64;
        if (m_scriptMappings.find( script_addr ) != m_scriptMappings.end())
        {
            auto rsm = m_scriptMappings[script_addr];
            if (rs_id_u64 < rsm->m_globals.size())
            {
                auto rsg = rsm->m_globals[rs_id_u64];
                log->Printf ("RenderScriptRuntime::CaptureSetGlobalVar1 - Setting of '%s' within '%s' inferred", rsg.m_name.AsCString(),
                                rsm->m_module->GetFileSpec().GetFilename().AsCString());
            }
        }
    }
}

void
RenderScriptRuntime::CaptureAllocationInit1(RuntimeHook* hook_info, ExecutionContext& context)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    //Context, Alloc, bool

    uint64_t rs_context_u64 = 0U;
    uint64_t rs_alloc_u64 = 0U;
    uint64_t rs_forceZero_u64 = 0U;

    bool success =
        GetArgSimple(context, 0, &rs_context_u64) &&
        GetArgSimple(context, 1, &rs_alloc_u64) &&
        GetArgSimple(context, 2, &rs_forceZero_u64);
    if (!success) // error case
    {
        if (log)
            log->Printf("RenderScriptRuntime::CaptureAllocationInit1 - Error while reading the function parameters");
        return; // abort
    }

    if (log)
        log->Printf ("RenderScriptRuntime::CaptureAllocationInit1 - 0x%" PRIx64 ",0x%" PRIx64 ",0x%" PRIx64 " .",
                        rs_context_u64, rs_alloc_u64, rs_forceZero_u64);

    AllocationDetails* alloc = LookUpAllocation(rs_alloc_u64, true);
    if (alloc)
        alloc->context = rs_context_u64;
}

void
RenderScriptRuntime::CaptureAllocationDestroy(RuntimeHook* hook_info, ExecutionContext& context)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    // Context, Alloc
    uint64_t rs_context_u64 = 0U;
    uint64_t rs_alloc_u64 = 0U;

    bool success = GetArgSimple(context, 0, &rs_context_u64) && GetArgSimple(context, 1, &rs_alloc_u64);
    if (!success) // error case
    {
        if (log)
            log->Printf("RenderScriptRuntime::CaptureAllocationDestroy - Error while reading the function parameters");
        return; // abort
    }

    if (log)
        log->Printf("RenderScriptRuntime::CaptureAllocationDestroy - 0x%" PRIx64 ", 0x%" PRIx64 ".",
                    rs_context_u64, rs_alloc_u64);

    for (auto iter = m_allocations.begin(); iter != m_allocations.end(); ++iter)
    {
        auto& allocation_ap = *iter; // get the unique pointer
        if (allocation_ap->address.isValid() && *allocation_ap->address.get() == rs_alloc_u64)
        {
            m_allocations.erase(iter);
            if (log)
                log->Printf("RenderScriptRuntime::CaptureAllocationDestroy - Deleted allocation entry");
            return;
        }
    }

    if (log)
        log->Printf("RenderScriptRuntime::CaptureAllocationDestroy - Couldn't find destroyed allocation");
}

void
RenderScriptRuntime::CaptureScriptInit1(RuntimeHook* hook_info, ExecutionContext& context)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    //Context, Script, resname Str, cachedir Str
    Error error;
    Process* process = context.GetProcessPtr();

    uint64_t rs_context_u64 = 0U;
    uint64_t rs_script_u64 = 0U;
    uint64_t rs_resnameptr_u64 = 0U;
    uint64_t rs_cachedirptr_u64 = 0U;

    std::string resname;
    std::string cachedir;

    // read the function parameters
    bool success =
        GetArgSimple(context, 0, &rs_context_u64) &&
        GetArgSimple(context, 1, &rs_script_u64) &&
        GetArgSimple(context, 2, &rs_resnameptr_u64) &&
        GetArgSimple(context, 3, &rs_cachedirptr_u64);

    if (!success)
    {
        if (log)
            log->Printf("RenderScriptRuntime::CaptureScriptInit1 - Error while reading the function parameters");
        return;
    }

    process->ReadCStringFromMemory((lldb::addr_t)rs_resnameptr_u64, resname, error);
    if (error.Fail())
    {
        if (log)
            log->Printf ("RenderScriptRuntime::CaptureScriptInit1 - error reading resname: %s.", error.AsCString());

    }

    process->ReadCStringFromMemory((lldb::addr_t)rs_cachedirptr_u64, cachedir, error);
    if (error.Fail())
    {
        if (log)
            log->Printf ("RenderScriptRuntime::CaptureScriptInit1 - error reading cachedir: %s.", error.AsCString());
    }

    if (log)
        log->Printf ("RenderScriptRuntime::CaptureScriptInit1 - 0x%" PRIx64 ",0x%" PRIx64 " => '%s' at '%s' .",
                     rs_context_u64, rs_script_u64, resname.c_str(), cachedir.c_str());

    if (resname.size() > 0)
    {
        StreamString strm;
        strm.Printf("librs.%s.so", resname.c_str());

        ScriptDetails* script = LookUpScript(rs_script_u64, true);
        if (script)
        {
            script->type = ScriptDetails::eScriptC;
            script->cacheDir = cachedir;
            script->resName = resname;
            script->scriptDyLib = strm.GetData();
            script->context = addr_t(rs_context_u64);
        }

        if (log)
            log->Printf ("RenderScriptRuntime::CaptureScriptInit1 - '%s' tagged with context 0x%" PRIx64 " and script 0x%" PRIx64 ".",
                         strm.GetData(), rs_context_u64, rs_script_u64);
    }
    else if (log)
    {
        log->Printf ("RenderScriptRuntime::CaptureScriptInit1 - resource name invalid, Script not tagged");
    }
}

void
RenderScriptRuntime::LoadRuntimeHooks(lldb::ModuleSP module, ModuleKind kind)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    if (!module)
    {
        return;
    }

    Target &target = GetProcess()->GetTarget();
    llvm::Triple::ArchType targetArchType = target.GetArchitecture().GetMachine();

    if (targetArchType != llvm::Triple::ArchType::x86
        && targetArchType != llvm::Triple::ArchType::arm
        && targetArchType != llvm::Triple::ArchType::aarch64
        && targetArchType != llvm::Triple::ArchType::mipsel
        && targetArchType != llvm::Triple::ArchType::mips64el
        && targetArchType != llvm::Triple::ArchType::x86_64
    )
    {
        if (log)
            log->Printf ("RenderScriptRuntime::LoadRuntimeHooks - Unable to hook runtime. Only X86, ARM, Mips supported currently.");

        return;
    }

    uint32_t archByteSize = target.GetArchitecture().GetAddressByteSize();

    for (size_t idx = 0; idx < s_runtimeHookCount; idx++)
    {
        const HookDefn* hook_defn = &s_runtimeHookDefns[idx];
        if (hook_defn->kind != kind) {
            continue;
        }

        const char* symbol_name = (archByteSize == 4) ? hook_defn->symbol_name_m32 : hook_defn->symbol_name_m64;

        const Symbol *sym = module->FindFirstSymbolWithNameAndType(ConstString(symbol_name), eSymbolTypeCode);
        if (!sym){
            if (log){
                log->Printf("RenderScriptRuntime::LoadRuntimeHooks - ERROR: Symbol '%s' related to the function %s not found", symbol_name, hook_defn->name);
            }
            continue;
        }

        addr_t addr = sym->GetLoadAddress(&target);
        if (addr == LLDB_INVALID_ADDRESS)
        {
            if (log)
                log->Printf ("RenderScriptRuntime::LoadRuntimeHooks - Unable to resolve the address of hook function '%s' with symbol '%s'.",
                             hook_defn->name, symbol_name);
            continue;
        }
        else
        {
            if (log)
                log->Printf("RenderScriptRuntime::LoadRuntimeHooks - Function %s, address resolved at 0x%" PRIx64, hook_defn->name, addr);
        }

        RuntimeHookSP hook(new RuntimeHook());
        hook->address = addr;
        hook->defn = hook_defn;
        hook->bp_sp = target.CreateBreakpoint(addr, true, false);
        hook->bp_sp->SetCallback(HookCallback, hook.get(), true);
        m_runtimeHooks[addr] = hook;
        if (log)
        {
            log->Printf ("RenderScriptRuntime::LoadRuntimeHooks - Successfully hooked '%s' in '%s' version %" PRIu64 " at 0x%" PRIx64 ".",
                hook_defn->name, module->GetFileSpec().GetFilename().AsCString(), (uint64_t)hook_defn->version, (uint64_t)addr);
        }
    }
}

void
RenderScriptRuntime::FixupScriptDetails(RSModuleDescriptorSP rsmodule_sp)
{
    if (!rsmodule_sp)
        return;

    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    const ModuleSP module = rsmodule_sp->m_module;
    const FileSpec& file = module->GetPlatformFileSpec();

    // Iterate over all of the scripts that we currently know of.
    // Note: We cant push or pop to m_scripts here or it may invalidate rs_script.
    for (const auto & rs_script : m_scripts)
    {
        // Extract the expected .so file path for this script.
        std::string dylib;
        if (!rs_script->scriptDyLib.get(dylib))
            continue;

        // Only proceed if the module that has loaded corresponds to this script.
        if (file.GetFilename() != ConstString(dylib.c_str()))
            continue;

        // Obtain the script address which we use as a key.
        lldb::addr_t script;
        if (!rs_script->script.get(script))
            continue;

        // If we have a script mapping for the current script.
        if (m_scriptMappings.find(script) != m_scriptMappings.end())
        {
            // if the module we have stored is different to the one we just received.
            if (m_scriptMappings[script] != rsmodule_sp)
            {
                if (log)
                    log->Printf ("RenderScriptRuntime::FixupScriptDetails - Error: script %" PRIx64 " wants reassigned to new rsmodule '%s'.",
                                    (uint64_t)script, rsmodule_sp->m_module->GetFileSpec().GetFilename().AsCString());
            }
        }
        // We don't have a script mapping for the current script.
        else
        {
            // Obtain the script resource name.
            std::string resName;
            if (rs_script->resName.get(resName))
                // Set the modules resource name.
                rsmodule_sp->m_resname = resName;
            // Add Script/Module pair to map.
            m_scriptMappings[script] = rsmodule_sp;
            if (log)
                log->Printf ("RenderScriptRuntime::FixupScriptDetails - script %" PRIx64 " associated with rsmodule '%s'.",
                                (uint64_t)script, rsmodule_sp->m_module->GetFileSpec().GetFilename().AsCString());
        }
    }
}

// Uses the Target API to evaluate the expression passed as a parameter to the function
// The result of that expression is returned an unsigned 64 bit int, via the result* paramter.
// Function returns true on success, and false on failure
bool
RenderScriptRuntime::EvalRSExpression(const char* expression, StackFrame* frame_ptr, uint64_t* result)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));
    if (log)
        log->Printf("RenderScriptRuntime::EvalRSExpression(%s)", expression);

    ValueObjectSP expr_result;
    // Perform the actual expression evaluation
    GetProcess()->GetTarget().EvaluateExpression(expression, frame_ptr, expr_result);

    if (!expr_result)
    {
       if (log)
           log->Printf("RenderScriptRuntime::EvalRSExpression -  Error: Couldn't evaluate expression");
       return false;
    }

    // The result of the expression is invalid
    if (!expr_result->GetError().Success())
    {
        Error err = expr_result->GetError();
        if (err.GetError() == UserExpression::kNoResult) // Expression returned void, so this is actually a success
        {
            if (log)
                log->Printf("RenderScriptRuntime::EvalRSExpression - Expression returned void");

            result = nullptr;
            return true;
        }

        if (log)
            log->Printf("RenderScriptRuntime::EvalRSExpression - Error evaluating expression result: %s", err.AsCString());
        return false;
    }

    bool success = false;
    *result = expr_result->GetValueAsUnsigned(0, &success); // We only read the result as an unsigned int.

    if (!success)
    {
       if (log)
           log->Printf("RenderScriptRuntime::EvalRSExpression -  Error: Couldn't convert expression result to unsigned int");
       return false;
    }

    return true;
}

// Used to index expression format strings
enum ExpressionStrings
{
   eExprGetOffsetPtr = 0,
   eExprAllocGetType,
   eExprTypeDimX,
   eExprTypeDimY,
   eExprTypeDimZ,
   eExprTypeElemPtr,
   eExprElementType,
   eExprElementKind,
   eExprElementVec,
   eExprElementFieldCount,
   eExprSubelementsId,
   eExprSubelementsName,
   eExprSubelementsArrSize
};

// Format strings containing the expressions we may need to evaluate.
const char runtimeExpressions[][256] =
{
 // Mangled GetOffsetPointer(Allocation*, xoff, yoff, zoff, lod, cubemap)
 "(int*)_Z12GetOffsetPtrPKN7android12renderscript10AllocationEjjjj23RsAllocationCubemapFace(0x%lx, %u, %u, %u, 0, 0)",

 // Type* rsaAllocationGetType(Context*, Allocation*)
 "(void*)rsaAllocationGetType(0x%lx, 0x%lx)",

 // rsaTypeGetNativeData(Context*, Type*, void* typeData, size)
 // Pack the data in the following way mHal.state.dimX; mHal.state.dimY; mHal.state.dimZ;
 // mHal.state.lodCount; mHal.state.faces; mElement; into typeData
 // Need to specify 32 or 64 bit for uint_t since this differs between devices
 "uint%u_t data[6]; (void*)rsaTypeGetNativeData(0x%lx, 0x%lx, data, 6); data[0]", // X dim
 "uint%u_t data[6]; (void*)rsaTypeGetNativeData(0x%lx, 0x%lx, data, 6); data[1]", // Y dim
 "uint%u_t data[6]; (void*)rsaTypeGetNativeData(0x%lx, 0x%lx, data, 6); data[2]", // Z dim
 "uint%u_t data[6]; (void*)rsaTypeGetNativeData(0x%lx, 0x%lx, data, 6); data[5]", // Element ptr

 // rsaElementGetNativeData(Context*, Element*, uint32_t* elemData,size)
 // Pack mType; mKind; mNormalized; mVectorSize; NumSubElements into elemData
 "uint32_t data[5]; (void*)rsaElementGetNativeData(0x%lx, 0x%lx, data, 5); data[0]", // Type
 "uint32_t data[5]; (void*)rsaElementGetNativeData(0x%lx, 0x%lx, data, 5); data[1]", // Kind
 "uint32_t data[5]; (void*)rsaElementGetNativeData(0x%lx, 0x%lx, data, 5); data[3]", // Vector Size
 "uint32_t data[5]; (void*)rsaElementGetNativeData(0x%lx, 0x%lx, data, 5); data[4]", // Field Count

  // rsaElementGetSubElements(RsContext con, RsElement elem, uintptr_t *ids, const char **names,
  // size_t *arraySizes, uint32_t dataSize)
  // Needed for Allocations of structs to gather details about fields/Subelements
 "void* ids[%u]; const char* names[%u]; size_t arr_size[%u];"
 "(void*)rsaElementGetSubElements(0x%lx, 0x%lx, ids, names, arr_size, %u); ids[%u]",     // Element* of field

 "void* ids[%u]; const char* names[%u]; size_t arr_size[%u];"
 "(void*)rsaElementGetSubElements(0x%lx, 0x%lx, ids, names, arr_size, %u); names[%u]",   // Name of field

 "void* ids[%u]; const char* names[%u]; size_t arr_size[%u];"
 "(void*)rsaElementGetSubElements(0x%lx, 0x%lx, ids, names, arr_size, %u); arr_size[%u]" // Array size of field
};

// JITs the RS runtime for the internal data pointer of an allocation.
// Is passed x,y,z coordinates for the pointer to a specific element.
// Then sets the data_ptr member in Allocation with the result.
// Returns true on success, false otherwise
bool
RenderScriptRuntime::JITDataPointer(AllocationDetails* allocation, StackFrame* frame_ptr,
                                    unsigned int x, unsigned int y, unsigned int z)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    if (!allocation->address.isValid())
    {
        if (log)
            log->Printf("RenderScriptRuntime::JITDataPointer - Failed to find allocation details");
        return false;
    }

    const char* expr_cstr = runtimeExpressions[eExprGetOffsetPtr];
    const int max_expr_size = 512; // Max expression size
    char buffer[max_expr_size];

    int chars_written = snprintf(buffer, max_expr_size, expr_cstr, *allocation->address.get(), x, y, z);
    if (chars_written < 0)
    {
        if (log)
            log->Printf("RenderScriptRuntime::JITDataPointer - Encoding error in snprintf()");
        return false;
    }
    else if (chars_written >= max_expr_size)
    {
        if (log)
            log->Printf("RenderScriptRuntime::JITDataPointer - Expression too long");
        return false;
    }

    uint64_t result = 0;
    if (!EvalRSExpression(buffer, frame_ptr, &result))
        return false;

    addr_t mem_ptr = static_cast<lldb::addr_t>(result);
    allocation->data_ptr = mem_ptr;

    return true;
}

// JITs the RS runtime for the internal pointer to the RS Type of an allocation
// Then sets the type_ptr member in Allocation with the result.
// Returns true on success, false otherwise
bool
RenderScriptRuntime::JITTypePointer(AllocationDetails* allocation, StackFrame* frame_ptr)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    if (!allocation->address.isValid() || !allocation->context.isValid())
    {
        if (log)
            log->Printf("RenderScriptRuntime::JITTypePointer - Failed to find allocation details");
        return false;
    }

    const char* expr_cstr = runtimeExpressions[eExprAllocGetType];
    const int max_expr_size = 512; // Max expression size
    char buffer[max_expr_size];

    int chars_written = snprintf(buffer, max_expr_size, expr_cstr, *allocation->context.get(), *allocation->address.get());
    if (chars_written < 0)
    {
        if (log)
            log->Printf("RenderScriptRuntime::JITDataPointer - Encoding error in snprintf()");
        return false;
    }
    else if (chars_written >= max_expr_size)
    {
        if (log)
            log->Printf("RenderScriptRuntime::JITTypePointer - Expression too long");
        return false;
    }

    uint64_t result = 0;
    if (!EvalRSExpression(buffer, frame_ptr, &result))
        return false;

    addr_t type_ptr = static_cast<lldb::addr_t>(result);
    allocation->type_ptr = type_ptr;

    return true;
}

// JITs the RS runtime for information about the dimensions and type of an allocation
// Then sets dimension and element_ptr members in Allocation with the result.
// Returns true on success, false otherwise
bool
RenderScriptRuntime::JITTypePacked(AllocationDetails* allocation, StackFrame* frame_ptr)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    if (!allocation->type_ptr.isValid() || !allocation->context.isValid())
    {
        if (log)
            log->Printf("RenderScriptRuntime::JITTypePacked - Failed to find allocation details");
        return false;
    }

    // Expression is different depending on if device is 32 or 64 bit
    uint32_t archByteSize = GetProcess()->GetTarget().GetArchitecture().GetAddressByteSize();
    const unsigned int bits = archByteSize == 4 ? 32 : 64;

    // We want 4 elements from packed data
    const unsigned int num_exprs = 4;
    assert(num_exprs == (eExprTypeElemPtr - eExprTypeDimX + 1) && "Invalid number of expressions");

    const int max_expr_size = 512; // Max expression size
    char buffer[num_exprs][max_expr_size];
    uint64_t results[num_exprs];

    for (unsigned int i = 0; i < num_exprs; ++i)
    {
        int chars_written = snprintf(buffer[i], max_expr_size, runtimeExpressions[eExprTypeDimX + i], bits,
                                     *allocation->context.get(), *allocation->type_ptr.get());
        if (chars_written < 0)
        {
            if (log)
                log->Printf("RenderScriptRuntime::JITDataPointer - Encoding error in snprintf()");
            return false;
        }
        else if (chars_written >= max_expr_size)
        {
            if (log)
                log->Printf("RenderScriptRuntime::JITTypePacked - Expression too long");
            return false;
        }

        // Perform expression evaluation
        if (!EvalRSExpression(buffer[i], frame_ptr, &results[i]))
            return false;
    }

    // Assign results to allocation members
    AllocationDetails::Dimension dims;
    dims.dim_1 = static_cast<uint32_t>(results[0]);
    dims.dim_2 = static_cast<uint32_t>(results[1]);
    dims.dim_3 = static_cast<uint32_t>(results[2]);
    allocation->dimension = dims;

    addr_t elem_ptr = static_cast<lldb::addr_t>(results[3]);
    allocation->element.element_ptr = elem_ptr;

    if (log)
        log->Printf("RenderScriptRuntime::JITTypePacked - dims (%u, %u, %u) Element*: 0x%" PRIx64,
                    dims.dim_1, dims.dim_2, dims.dim_3, elem_ptr);

    return true;
}

// JITs the RS runtime for information about the Element of an allocation
// Then sets type, type_vec_size, field_count and type_kind members in Element with the result.
// Returns true on success, false otherwise
bool
RenderScriptRuntime::JITElementPacked(Element& elem, const lldb::addr_t context, StackFrame* frame_ptr)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    if (!elem.element_ptr.isValid())
    {
        if (log)
            log->Printf("RenderScriptRuntime::JITElementPacked - Failed to find allocation details");
        return false;
    }

    // We want 4 elements from packed data
    const unsigned int num_exprs = 4;
    assert(num_exprs == (eExprElementFieldCount - eExprElementType + 1) && "Invalid number of expressions");

    const int max_expr_size = 512; // Max expression size
    char buffer[num_exprs][max_expr_size];
    uint64_t results[num_exprs];

    for (unsigned int i = 0; i < num_exprs; i++)
    {
        int chars_written = snprintf(buffer[i], max_expr_size, runtimeExpressions[eExprElementType + i], context, *elem.element_ptr.get());
        if (chars_written < 0)
        {
            if (log)
                log->Printf("RenderScriptRuntime::JITElementPacked - Encoding error in snprintf()");
            return false;
        }
        else if (chars_written >= max_expr_size)
        {
            if (log)
                log->Printf("RenderScriptRuntime::JITElementPacked - Expression too long");
            return false;
        }

        // Perform expression evaluation
        if (!EvalRSExpression(buffer[i], frame_ptr, &results[i]))
            return false;
    }

    // Assign results to allocation members
    elem.type = static_cast<RenderScriptRuntime::Element::DataType>(results[0]);
    elem.type_kind = static_cast<RenderScriptRuntime::Element::DataKind>(results[1]);
    elem.type_vec_size = static_cast<uint32_t>(results[2]);
    elem.field_count = static_cast<uint32_t>(results[3]);

    if (log)
        log->Printf("RenderScriptRuntime::JITElementPacked - data type %u, pixel type %u, vector size %u, field count %u",
                    *elem.type.get(), *elem.type_kind.get(), *elem.type_vec_size.get(), *elem.field_count.get());

    // If this Element has subelements then JIT rsaElementGetSubElements() for details about its fields
    if (*elem.field_count.get() > 0 && !JITSubelements(elem, context, frame_ptr))
        return false;

    return true;
}

// JITs the RS runtime for information about the subelements/fields of a struct allocation
// This is necessary for infering the struct type so we can pretty print the allocation's contents.
// Returns true on success, false otherwise
bool
RenderScriptRuntime::JITSubelements(Element& elem, const lldb::addr_t context, StackFrame* frame_ptr)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    if (!elem.element_ptr.isValid() || !elem.field_count.isValid())
    {
        if (log)
            log->Printf("RenderScriptRuntime::JITSubelements - Failed to find allocation details");
        return false;
    }

    const short num_exprs = 3;
    assert(num_exprs == (eExprSubelementsArrSize - eExprSubelementsId + 1) && "Invalid number of expressions");

    const int max_expr_size = 512; // Max expression size
    char expr_buffer[max_expr_size];
    uint64_t results;

    // Iterate over struct fields.
    const uint32_t field_count = *elem.field_count.get();
    for (unsigned int field_index = 0; field_index < field_count; ++field_index)
    {
        Element child;
        for (unsigned int expr_index = 0; expr_index < num_exprs; ++expr_index)
        {
            int chars_written = snprintf(expr_buffer, max_expr_size, runtimeExpressions[eExprSubelementsId + expr_index],
                                         field_count, field_count, field_count,
                                         context, *elem.element_ptr.get(), field_count, field_index);
            if (chars_written < 0)
            {
                if (log)
                    log->Printf("RenderScriptRuntime::JITSubelements - Encoding error in snprintf()");
                return false;
            }
            else if (chars_written >= max_expr_size)
            {
                if (log)
                    log->Printf("RenderScriptRuntime::JITSubelements - Expression too long");
                return false;
            }

            // Perform expression evaluation
            if (!EvalRSExpression(expr_buffer, frame_ptr, &results))
                return false;

            if (log)
                log->Printf("RenderScriptRuntime::JITSubelements - Expr result 0x%" PRIx64, results);

            switch(expr_index)
            {
                case 0: // Element* of child
                    child.element_ptr = static_cast<addr_t>(results);
                    break;
                case 1: // Name of child
                {
                    lldb::addr_t address = static_cast<addr_t>(results);
                    Error err;
                    std::string name;
                    GetProcess()->ReadCStringFromMemory(address, name, err);
                    if (!err.Fail())
                        child.type_name = ConstString(name);
                    else
                    {
                        if (log)
                            log->Printf("RenderScriptRuntime::JITSubelements - Warning: Couldn't read field name");
                    }
                    break;
                }
                case 2: // Array size of child
                    child.array_size = static_cast<uint32_t>(results);
                    break;
            }
        }

        // We need to recursively JIT each Element field of the struct since
        // structs can be nested inside structs.
        if (!JITElementPacked(child, context, frame_ptr))
            return false;
        elem.children.push_back(child);
    }

    // Try to infer the name of the struct type so we can pretty print the allocation contents.
    FindStructTypeName(elem, frame_ptr);

    return true;
}

// JITs the RS runtime for the address of the last element in the allocation.
// The `elem_size` paramter represents the size of a single element, including padding.
// Which is needed as an offset from the last element pointer.
// Using this offset minus the starting address we can calculate the size of the allocation.
// Returns true on success, false otherwise
bool
RenderScriptRuntime::JITAllocationSize(AllocationDetails* allocation, StackFrame* frame_ptr)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    if (!allocation->address.isValid() || !allocation->dimension.isValid()
        || !allocation->data_ptr.isValid() || !allocation->element.datum_size.isValid())
    {
        if (log)
            log->Printf("RenderScriptRuntime::JITAllocationSize - Failed to find allocation details");
        return false;
    }

    // Find dimensions
    unsigned int dim_x = allocation->dimension.get()->dim_1;
    unsigned int dim_y = allocation->dimension.get()->dim_2;
    unsigned int dim_z = allocation->dimension.get()->dim_3;

    // Our plan of jitting the last element address doesn't seem to work for struct Allocations
    // Instead try to infer the size ourselves without any inter element padding.
    if (allocation->element.children.size() > 0)
    {
        if (dim_x == 0) dim_x = 1;
        if (dim_y == 0) dim_y = 1;
        if (dim_z == 0) dim_z = 1;

        allocation->size = dim_x * dim_y * dim_z * *allocation->element.datum_size.get();

        if (log)
            log->Printf("RenderScriptRuntime::JITAllocationSize - Infered size of struct allocation %u", *allocation->size.get());

        return true;
    }

    const char* expr_cstr = runtimeExpressions[eExprGetOffsetPtr];
    const int max_expr_size = 512;
    char buffer[max_expr_size];

    // Calculate last element
    dim_x = dim_x == 0 ? 0 : dim_x - 1;
    dim_y = dim_y == 0 ? 0 : dim_y - 1;
    dim_z = dim_z == 0 ? 0 : dim_z - 1;

    int chars_written = snprintf(buffer, max_expr_size, expr_cstr, *allocation->address.get(),
                                 dim_x, dim_y, dim_z);
    if (chars_written < 0)
    {
        if (log)
            log->Printf("RenderScriptRuntime::JITAllocationSize - Encoding error in snprintf()");
        return false;
    }
    else if (chars_written >= max_expr_size)
    {
        if (log)
            log->Printf("RenderScriptRuntime::JITAllocationSize - Expression too long");
        return false;
    }

    uint64_t result = 0;
    if (!EvalRSExpression(buffer, frame_ptr, &result))
        return false;

    addr_t mem_ptr = static_cast<lldb::addr_t>(result);
    // Find pointer to last element and add on size of an element
    allocation->size = static_cast<uint32_t>(mem_ptr - *allocation->data_ptr.get()) + *allocation->element.datum_size.get();

    return true;
}

// JITs the RS runtime for information about the stride between rows in the allocation.
// This is done to detect padding, since allocated memory is 16-byte aligned.
// Returns true on success, false otherwise
bool
RenderScriptRuntime::JITAllocationStride(AllocationDetails* allocation, StackFrame* frame_ptr)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    if (!allocation->address.isValid() || !allocation->data_ptr.isValid())
    {
        if (log)
            log->Printf("RenderScriptRuntime::JITAllocationStride - Failed to find allocation details");
        return false;
    }

    const char* expr_cstr = runtimeExpressions[eExprGetOffsetPtr];
    const int max_expr_size = 512; // Max expression size
    char buffer[max_expr_size];

    int chars_written = snprintf(buffer, max_expr_size, expr_cstr, *allocation->address.get(),
                                 0, 1, 0);
    if (chars_written < 0)
    {
        if (log)
            log->Printf("RenderScriptRuntime::JITAllocationStride - Encoding error in snprintf()");
        return false;
    }
    else if (chars_written >= max_expr_size)
    {
        if (log)
            log->Printf("RenderScriptRuntime::JITAllocationStride - Expression too long");
        return false;
    }

    uint64_t result = 0;
    if (!EvalRSExpression(buffer, frame_ptr, &result))
        return false;

    addr_t mem_ptr = static_cast<lldb::addr_t>(result);
    allocation->stride = static_cast<uint32_t>(mem_ptr - *allocation->data_ptr.get());

    return true;
}

// JIT all the current runtime info regarding an allocation
bool
RenderScriptRuntime::RefreshAllocation(AllocationDetails* allocation, StackFrame* frame_ptr)
{
    // GetOffsetPointer()
    if (!JITDataPointer(allocation, frame_ptr))
        return false;

    // rsaAllocationGetType()
    if (!JITTypePointer(allocation, frame_ptr))
        return false;

    // rsaTypeGetNativeData()
    if (!JITTypePacked(allocation, frame_ptr))
        return false;

    // rsaElementGetNativeData()
    if (!JITElementPacked(allocation->element, *allocation->context.get(), frame_ptr))
        return false;

    // Sets the datum_size member in Element
    SetElementSize(allocation->element);

    // Use GetOffsetPointer() to infer size of the allocation
    if (!JITAllocationSize(allocation, frame_ptr))
        return false;

    return true;
}

// Function attempts to set the type_name member of the paramaterised Element object.
// This string should be the name of the struct type the Element represents.
// We need this string for pretty printing the Element to users.
void
RenderScriptRuntime::FindStructTypeName(Element& elem, StackFrame* frame_ptr)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    if (!elem.type_name.IsEmpty()) // Name already set
        return;
    else
        elem.type_name = Element::GetFallbackStructName(); // Default type name if we don't succeed

    // Find all the global variables from the script rs modules
    VariableList variable_list;
    for (auto module_sp : m_rsmodules)
        module_sp->m_module->FindGlobalVariables(RegularExpression("."), true, UINT32_MAX, variable_list);

    // Iterate over all the global variables looking for one with a matching type to the Element.
    // We make the assumption a match exists since there needs to be a global variable to reflect the
    // struct type back into java host code.
    for (uint32_t var_index = 0; var_index < variable_list.GetSize(); ++var_index)
    {
        const VariableSP var_sp(variable_list.GetVariableAtIndex(var_index));
        if (!var_sp)
           continue;

        ValueObjectSP valobj_sp = ValueObjectVariable::Create(frame_ptr, var_sp);
        if (!valobj_sp)
            continue;

        // Find the number of variable fields.
        // If it has no fields, or more fields than our Element, then it can't be the struct we're looking for.
        // Don't check for equality since RS can add extra struct members for padding.
        size_t num_children = valobj_sp->GetNumChildren();
        if (num_children > elem.children.size() || num_children == 0)
            continue;

        // Iterate over children looking for members with matching field names.
        // If all the field names match, this is likely the struct we want.
        //
        //   TODO: This could be made more robust by also checking children data sizes, or array size
        bool found = true;
        for (size_t child_index = 0; child_index < num_children; ++child_index)
        {
            ValueObjectSP child = valobj_sp->GetChildAtIndex(child_index, true);
            if (!child || (child->GetName() != elem.children[child_index].type_name))
            {
                found = false;
                break;
            }
        }

        // RS can add extra struct members for padding in the format '#rs_padding_[0-9]+'
        if (found && num_children < elem.children.size())
        {
            const unsigned int size_diff = elem.children.size() - num_children;
            if (log)
                log->Printf("RenderScriptRuntime::FindStructTypeName - %u padding struct entries", size_diff);

            for (unsigned int padding_index = 0; padding_index < size_diff; ++padding_index)
            {
                const ConstString& name = elem.children[num_children + padding_index].type_name;
                if (strcmp(name.AsCString(), "#rs_padding") < 0)
                    found = false;
            }
        }

        // We've found a global var with matching type
        if (found)
        {
            // Dereference since our Element type isn't a pointer.
            if (valobj_sp->IsPointerType())
            {
                Error err;
                ValueObjectSP deref_valobj = valobj_sp->Dereference(err);
                if (!err.Fail())
                    valobj_sp = deref_valobj;
            }

            // Save name of variable in Element.
            elem.type_name = valobj_sp->GetTypeName();
            if (log)
                log->Printf("RenderScriptRuntime::FindStructTypeName - Element name set to %s", elem.type_name.AsCString());

            return;
        }
    }
}

// Function sets the datum_size member of Element. Representing the size of a single instance including padding.
// Assumes the relevant allocation information has already been jitted.
void
RenderScriptRuntime::SetElementSize(Element& elem)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));
    const Element::DataType type = *elem.type.get();
    assert(type >= Element::RS_TYPE_NONE && type <= Element::RS_TYPE_FONT
                                                   && "Invalid allocation type");

    const unsigned int vec_size = *elem.type_vec_size.get();
    unsigned int data_size = 0;
    unsigned int padding = 0;

    // Element is of a struct type, calculate size recursively.
    if ((type == Element::RS_TYPE_NONE) && (elem.children.size() > 0))
    {
        for (Element& child : elem.children)
        {
            SetElementSize(child);
            const unsigned int array_size = child.array_size.isValid() ? *child.array_size.get() : 1;
            data_size += *child.datum_size.get() * array_size;
        }
    }
    else if (type == Element::RS_TYPE_UNSIGNED_5_6_5 || type == Element::RS_TYPE_UNSIGNED_5_5_5_1 ||
             type == Element::RS_TYPE_UNSIGNED_4_4_4_4) // These have been packed already
    {
        data_size = AllocationDetails::RSTypeToFormat[type][eElementSize];
    }
    else if (type < Element::RS_TYPE_ELEMENT)
    {
        data_size = vec_size * AllocationDetails::RSTypeToFormat[type][eElementSize];
        if (vec_size == 3)
            padding = AllocationDetails::RSTypeToFormat[type][eElementSize];
    }
    else
        data_size = GetProcess()->GetTarget().GetArchitecture().GetAddressByteSize();

    elem.padding = padding;
    elem.datum_size = data_size + padding;
    if (log)
        log->Printf("RenderScriptRuntime::SetElementSize - element size set to %u", data_size + padding);
}

// Given an allocation, this function copies the allocation contents from device into a buffer on the heap.
// Returning a shared pointer to the buffer containing the data.
std::shared_ptr<uint8_t>
RenderScriptRuntime::GetAllocationData(AllocationDetails* allocation, StackFrame* frame_ptr)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    // JIT all the allocation details
    if (allocation->shouldRefresh())
    {
        if (log)
            log->Printf("RenderScriptRuntime::GetAllocationData - Allocation details not calculated yet, jitting info");

        if (!RefreshAllocation(allocation, frame_ptr))
        {
            if (log)
                log->Printf("RenderScriptRuntime::GetAllocationData - Couldn't JIT allocation details");
            return nullptr;
        }
    }

    assert(allocation->data_ptr.isValid() && allocation->element.type.isValid() && allocation->element.type_vec_size.isValid()
           && allocation->size.isValid() && "Allocation information not available");

    // Allocate a buffer to copy data into
    const unsigned int size = *allocation->size.get();
    std::shared_ptr<uint8_t> buffer(new uint8_t[size]);
    if (!buffer)
    {
        if (log)
            log->Printf("RenderScriptRuntime::GetAllocationData - Couldn't allocate a %u byte buffer", size);
        return nullptr;
    }

    // Read the inferior memory
    Error error;
    lldb::addr_t data_ptr = *allocation->data_ptr.get();
    GetProcess()->ReadMemory(data_ptr, buffer.get(), size, error);
    if (error.Fail())
    {
        if (log)
            log->Printf("RenderScriptRuntime::GetAllocationData - '%s' Couldn't read %u bytes of allocation data from 0x%" PRIx64,
                        error.AsCString(), size, data_ptr);
        return nullptr;
    }

    return buffer;
}

// Function copies data from a binary file into an allocation.
// There is a header at the start of the file, FileHeader, before the data content itself.
// Information from this header is used to display warnings to the user about incompatabilities
bool
RenderScriptRuntime::LoadAllocation(Stream &strm, const uint32_t alloc_id, const char* filename, StackFrame* frame_ptr)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    // Find allocation with the given id
    AllocationDetails* alloc = FindAllocByID(strm, alloc_id);
    if (!alloc)
        return false;

    if (log)
        log->Printf("RenderScriptRuntime::LoadAllocation - Found allocation 0x%" PRIx64, *alloc->address.get());

    // JIT all the allocation details
    if (alloc->shouldRefresh())
    {
        if (log)
            log->Printf("RenderScriptRuntime::LoadAllocation - Allocation details not calculated yet, jitting info");

        if (!RefreshAllocation(alloc, frame_ptr))
        {
            if (log)
                log->Printf("RenderScriptRuntime::LoadAllocation - Couldn't JIT allocation details");
            return false;
        }
    }

    assert(alloc->data_ptr.isValid() && alloc->element.type.isValid() && alloc->element.type_vec_size.isValid()
           && alloc->size.isValid() && alloc->element.datum_size.isValid() && "Allocation information not available");

    // Check we can read from file
    FileSpec file(filename, true);
    if (!file.Exists())
    {
        strm.Printf("Error: File %s does not exist", filename);
        strm.EOL();
        return false;
    }

    if (!file.Readable())
    {
        strm.Printf("Error: File %s does not have readable permissions", filename);
        strm.EOL();
        return false;
    }

    // Read file into data buffer
    DataBufferSP data_sp(file.ReadFileContents());

    // Cast start of buffer to FileHeader and use pointer to read metadata
    void* file_buffer = data_sp->GetBytes();
    if (file_buffer == NULL || data_sp->GetByteSize() <
        (sizeof(AllocationDetails::FileHeader) + sizeof(AllocationDetails::ElementHeader)))
    {
        strm.Printf("Error: File %s does not contain enough data for header", filename);
        strm.EOL();
        return false;
    }
    const AllocationDetails::FileHeader* file_header = static_cast<AllocationDetails::FileHeader*>(file_buffer);

    // Check file starts with ascii characters "RSAD"
    if (file_header->ident[0] != 'R' || file_header->ident[1] != 'S' || file_header->ident[2] != 'A'
        || file_header->ident[3] != 'D')
    {
        strm.Printf("Error: File doesn't contain identifier for an RS allocation dump. Are you sure this is the correct file?");
        strm.EOL();
        return false;
    }

    // Look at the type of the root element in the header
    AllocationDetails::ElementHeader root_element_header;
    memcpy(&root_element_header, static_cast<uint8_t*>(file_buffer) + sizeof(AllocationDetails::FileHeader),
           sizeof(AllocationDetails::ElementHeader));

    if (log)
        log->Printf("RenderScriptRuntime::LoadAllocation - header type %u, element size %u",
                    root_element_header.type, root_element_header.element_size);

    // Check if the target allocation and file both have the same number of bytes for an Element
    if (*alloc->element.datum_size.get() != root_element_header.element_size)
    {
        strm.Printf("Warning: Mismatched Element sizes - file %u bytes, allocation %u bytes",
                    root_element_header.element_size, *alloc->element.datum_size.get());
        strm.EOL();
    }

    // Check if the target allocation and file both have the same type
    const unsigned int alloc_type = static_cast<unsigned int>(*alloc->element.type.get());
    const unsigned int file_type = root_element_header.type;

    if (file_type > Element::RS_TYPE_FONT)
    {
        strm.Printf("Warning: File has unknown allocation type");
        strm.EOL();
    }
    else if (alloc_type != file_type)
    {
        // Enum value isn't monotonous, so doesn't always index RsDataTypeToString array
        unsigned int printable_target_type_index = alloc_type;
        unsigned int printable_head_type_index = file_type;
        if (alloc_type >= Element::RS_TYPE_ELEMENT && alloc_type <= Element::RS_TYPE_FONT)
            printable_target_type_index = static_cast<Element::DataType>(
                                         (alloc_type - Element::RS_TYPE_ELEMENT) + Element::RS_TYPE_MATRIX_2X2 + 1);

        if (file_type >= Element::RS_TYPE_ELEMENT && file_type <= Element::RS_TYPE_FONT)
            printable_head_type_index = static_cast<Element::DataType>(
                                        (file_type - Element::RS_TYPE_ELEMENT) + Element::RS_TYPE_MATRIX_2X2 + 1);

        const char* file_type_cstr = AllocationDetails::RsDataTypeToString[printable_head_type_index][0];
        const char* target_type_cstr = AllocationDetails::RsDataTypeToString[printable_target_type_index][0];

        strm.Printf("Warning: Mismatched Types - file '%s' type, allocation '%s' type",
                    file_type_cstr, target_type_cstr);
        strm.EOL();
    }

    // Advance buffer past header
    file_buffer = static_cast<uint8_t*>(file_buffer) + file_header->hdr_size;

    // Calculate size of allocation data in file
    size_t length = data_sp->GetByteSize() - file_header->hdr_size;

    // Check if the target allocation and file both have the same total data size.
    const unsigned int alloc_size = *alloc->size.get();
    if (alloc_size != length)
    {
        strm.Printf("Warning: Mismatched allocation sizes - file 0x%" PRIx64 " bytes, allocation 0x%x bytes",
                    (uint64_t) length, alloc_size);
        strm.EOL();
        length = alloc_size < length ? alloc_size : length; // Set length to copy to minimum
    }

    // Copy file data from our buffer into the target allocation.
    lldb::addr_t alloc_data = *alloc->data_ptr.get();
    Error error;
    size_t bytes_written = GetProcess()->WriteMemory(alloc_data, file_buffer, length, error);
    if (!error.Success() || bytes_written != length)
    {
        strm.Printf("Error: Couldn't write data to allocation %s", error.AsCString());
        strm.EOL();
        return false;
    }

    strm.Printf("Contents of file '%s' read into allocation %u", filename, alloc->id);
    strm.EOL();

    return true;
}

// Function takes as parameters a byte buffer, which will eventually be written to file as the element header,
// an offset into that buffer, and an Element that will be saved into the buffer at the parametrised offset.
// Return value is the new offset after writing the element into the buffer.
// Elements are saved to the file as the ElementHeader struct followed by offsets to the structs of all the element's children.
size_t
RenderScriptRuntime::PopulateElementHeaders(const std::shared_ptr<uint8_t> header_buffer, size_t offset, const Element& elem)
{
    // File struct for an element header with all the relevant details copied from elem.
    // We assume members are valid already.
    AllocationDetails::ElementHeader elem_header;
    elem_header.type = *elem.type.get();
    elem_header.kind = *elem.type_kind.get();
    elem_header.element_size = *elem.datum_size.get();
    elem_header.vector_size = *elem.type_vec_size.get();
    elem_header.array_size = elem.array_size.isValid() ? *elem.array_size.get() : 0;
    const size_t elem_header_size = sizeof(AllocationDetails::ElementHeader);

    // Copy struct into buffer and advance offset
    // We assume that header_buffer has been checked for NULL before this method is called
    memcpy(header_buffer.get() + offset, &elem_header, elem_header_size);
    offset += elem_header_size;

    // Starting offset of child ElementHeader struct
    size_t child_offset = offset + ((elem.children.size() + 1) * sizeof(uint32_t));
    for (const RenderScriptRuntime::Element& child : elem.children)
    {
        // Recursively populate the buffer with the element header structs of children.
        // Then save the offsets where they were set after the parent element header.
        memcpy(header_buffer.get() + offset, &child_offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);

        child_offset = PopulateElementHeaders(header_buffer, child_offset, child);
    }

    // Zero indicates no more children
    memset(header_buffer.get() + offset, 0, sizeof(uint32_t));

    return child_offset;
}

// Given an Element object this function returns the total size needed in the file header to store the element's details.
// Taking into account the size of the element header struct, plus the offsets to all the element's children.
// Function is recursive so that the size of all ancestors is taken into account.
size_t
RenderScriptRuntime::CalculateElementHeaderSize(const Element& elem)
{
    size_t size = (elem.children.size() + 1) * sizeof(uint32_t); // Offsets to children plus zero terminator
    size += sizeof(AllocationDetails::ElementHeader); // Size of header struct with type details

    // Calculate recursively for all descendants
    for (const Element& child : elem.children)
        size += CalculateElementHeaderSize(child);

    return size;
}

// Function copies allocation contents into a binary file.
// This file can then be loaded later into a different allocation.
// There is a header, FileHeader, before the allocation data containing meta-data.
bool
RenderScriptRuntime::SaveAllocation(Stream &strm, const uint32_t alloc_id, const char* filename, StackFrame* frame_ptr)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    // Find allocation with the given id
    AllocationDetails* alloc = FindAllocByID(strm, alloc_id);
    if (!alloc)
        return false;

    if (log)
        log->Printf("RenderScriptRuntime::SaveAllocation - Found allocation 0x%" PRIx64, *alloc->address.get());

     // JIT all the allocation details
    if (alloc->shouldRefresh())
    {
        if (log)
            log->Printf("RenderScriptRuntime::SaveAllocation - Allocation details not calculated yet, jitting info");

        if (!RefreshAllocation(alloc, frame_ptr))
        {
            if (log)
                log->Printf("RenderScriptRuntime::SaveAllocation - Couldn't JIT allocation details");
            return false;
        }
    }

    assert(alloc->data_ptr.isValid() && alloc->element.type.isValid() && alloc->element.type_vec_size.isValid() && alloc->element.datum_size.get()
           && alloc->element.type_kind.isValid() && alloc->dimension.isValid() && "Allocation information not available");

    // Check we can create writable file
    FileSpec file_spec(filename, true);
    File file(file_spec, File::eOpenOptionWrite | File::eOpenOptionCanCreate | File::eOpenOptionTruncate);
    if (!file)
    {
        strm.Printf("Error: Failed to open '%s' for writing", filename);
        strm.EOL();
        return false;
    }

    // Read allocation into buffer of heap memory
    const std::shared_ptr<uint8_t> buffer = GetAllocationData(alloc, frame_ptr);
    if (!buffer)
    {
        strm.Printf("Error: Couldn't read allocation data into buffer");
        strm.EOL();
        return false;
    }

    // Create the file header
    AllocationDetails::FileHeader head;
    head.ident[0] = 'R'; head.ident[1] = 'S'; head.ident[2] = 'A'; head.ident[3] = 'D';
    head.dims[0] = static_cast<uint32_t>(alloc->dimension.get()->dim_1);
    head.dims[1] = static_cast<uint32_t>(alloc->dimension.get()->dim_2);
    head.dims[2] = static_cast<uint32_t>(alloc->dimension.get()->dim_3);

    const size_t element_header_size = CalculateElementHeaderSize(alloc->element);
    assert((sizeof(AllocationDetails::FileHeader) + element_header_size) < UINT16_MAX && "Element header too large");
    head.hdr_size = static_cast<uint16_t>(sizeof(AllocationDetails::FileHeader) + element_header_size);

    // Write the file header
    size_t num_bytes = sizeof(AllocationDetails::FileHeader);
    if (log)
        log->Printf("RenderScriptRuntime::SaveAllocation - Writing File Header, 0x%zX bytes", num_bytes);

    Error err = file.Write(&head, num_bytes);
    if (!err.Success())
    {
        strm.Printf("Error: '%s' when writing to file '%s'", err.AsCString(), filename);
        strm.EOL();
        return false;
    }

    // Create the headers describing the element type of the allocation.
    std::shared_ptr<uint8_t> element_header_buffer(new uint8_t[element_header_size]);
    if (element_header_buffer == nullptr)
    {
        strm.Printf("Internal Error: Couldn't allocate %zu bytes on the heap", element_header_size);
        strm.EOL();
        return false;
    }

    PopulateElementHeaders(element_header_buffer, 0, alloc->element);

    // Write headers for allocation element type to file
    num_bytes = element_header_size;
    if (log)
        log->Printf("RenderScriptRuntime::SaveAllocation - Writing Element Headers, 0x%zX bytes", num_bytes);

    err = file.Write(element_header_buffer.get(), num_bytes);
    if (!err.Success())
    {
        strm.Printf("Error: '%s' when writing to file '%s'", err.AsCString(), filename);
        strm.EOL();
        return false;
    }

    // Write allocation data to file
    num_bytes = static_cast<size_t>(*alloc->size.get());
    if (log)
        log->Printf("RenderScriptRuntime::SaveAllocation - Writing 0x%zX bytes", num_bytes);

    err = file.Write(buffer.get(), num_bytes);
    if (!err.Success())
    {
        strm.Printf("Error: '%s' when writing to file '%s'", err.AsCString(), filename);
        strm.EOL();
        return false;
    }

    strm.Printf("Allocation written to file '%s'", filename);
    strm.EOL();
    return true;
}

bool
RenderScriptRuntime::LoadModule(const lldb::ModuleSP &module_sp)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    if (module_sp)
    {
        for (const auto &rs_module : m_rsmodules)
        {
            if (rs_module->m_module == module_sp)
            {
                // Check if the user has enabled automatically breaking on
                // all RS kernels.
                if (m_breakAllKernels)
                    BreakOnModuleKernels(rs_module);

                return false;
            }
        }
        bool module_loaded = false;
        switch (GetModuleKind(module_sp))
        {
            case eModuleKindKernelObj:
            {
                RSModuleDescriptorSP module_desc;
                module_desc.reset(new RSModuleDescriptor(module_sp));
                if (module_desc->ParseRSInfo())
                {
                    m_rsmodules.push_back(module_desc);
                    module_loaded = true;
                }
                if (module_loaded)
                {
                    FixupScriptDetails(module_desc);
                }
                break;
            }
            case eModuleKindDriver:
            {
                if (!m_libRSDriver)
                {
                    m_libRSDriver = module_sp;
                    LoadRuntimeHooks(m_libRSDriver, RenderScriptRuntime::eModuleKindDriver);
                }
                break;
            }
            case eModuleKindImpl:
            {
                m_libRSCpuRef = module_sp;
                break;
            }
            case eModuleKindLibRS:
            {
                if (!m_libRS)
                {
                    m_libRS = module_sp;
                    static ConstString gDbgPresentStr("gDebuggerPresent");
                    const Symbol* debug_present = m_libRS->FindFirstSymbolWithNameAndType(gDbgPresentStr, eSymbolTypeData);
                    if (debug_present)
                    {
                        Error error;
                        uint32_t flag = 0x00000001U;
                        Target &target = GetProcess()->GetTarget();
                        addr_t addr = debug_present->GetLoadAddress(&target);
                        GetProcess()->WriteMemory(addr, &flag, sizeof(flag), error);
                        if(error.Success())
                        {
                            if (log)
                                log->Printf ("RenderScriptRuntime::LoadModule - Debugger present flag set on debugee");

                            m_debuggerPresentFlagged = true;
                        }
                        else if (log)
                        {
                            log->Printf ("RenderScriptRuntime::LoadModule - Error writing debugger present flags '%s' ", error.AsCString());
                        }
                    }
                    else if (log)
                    {
                        log->Printf ("RenderScriptRuntime::LoadModule - Error writing debugger present flags - symbol not found");
                    }
                }
                break;
            }
            default:
                break;
        }
        if (module_loaded)
            Update();
        return module_loaded;
    }
    return false;
}

void
RenderScriptRuntime::Update()
{
    if (m_rsmodules.size() > 0)
    {
        if (!m_initiated)
        {
            Initiate();
        }
    }
}

// The maximum line length of an .rs.info packet
#define MAXLINE 500

// The .rs.info symbol in renderscript modules contains a string which needs to be parsed.
// The string is basic and is parsed on a line by line basis.
bool
RSModuleDescriptor::ParseRSInfo()
{
    const Symbol *info_sym = m_module->FindFirstSymbolWithNameAndType(ConstString(".rs.info"), eSymbolTypeData);
    if (info_sym)
    {
        const addr_t addr = info_sym->GetAddressRef().GetFileAddress();
        const addr_t size = info_sym->GetByteSize();
        const FileSpec fs = m_module->GetFileSpec();

        DataBufferSP buffer = fs.ReadFileContents(addr, size);

        if (!buffer)
            return false;

        std::string info((const char *)buffer->GetBytes());

        std::vector<std::string> info_lines;
        size_t lpos = info.find('\n');
        while (lpos != std::string::npos)
        {
            info_lines.push_back(info.substr(0, lpos));
            info = info.substr(lpos + 1);
            lpos = info.find('\n');
        }
        size_t offset = 0;
        while (offset < info_lines.size())
        {
            std::string line = info_lines[offset];
            // Parse directives
            uint32_t numDefns = 0;
            if (sscanf(line.c_str(), "exportVarCount: %u", &numDefns) == 1)
            {
                while (numDefns--)
                    m_globals.push_back(RSGlobalDescriptor(this, info_lines[++offset].c_str()));
            }
            else if (sscanf(line.c_str(), "exportFuncCount: %u", &numDefns) == 1)
            {
            }
            else if (sscanf(line.c_str(), "exportForEachCount: %u", &numDefns) == 1)
            {
                char name[MAXLINE];
                while (numDefns--)
                {
                    uint32_t slot = 0;
                    name[0] = '\0';
                    if (sscanf(info_lines[++offset].c_str(), "%u - %s", &slot, &name[0]) == 2)
                    {
                        m_kernels.push_back(RSKernelDescriptor(this, name, slot));
                    }
                }
            }
            else if (sscanf(line.c_str(), "pragmaCount: %u", &numDefns) == 1)
            {
                char name[MAXLINE];
                char value[MAXLINE];
                while (numDefns--)
                {
                    name[0] = '\0';
                    value[0] = '\0';
                    if (sscanf(info_lines[++offset].c_str(), "%s - %s", &name[0], &value[0]) != 0
                        && (name[0] != '\0'))
                    {
                        m_pragmas[std::string(name)] = value;
                    }
                }
            }
            else if (sscanf(line.c_str(), "objectSlotCount: %u", &numDefns) == 1)
            {
            }

            offset++;
        }
        return m_kernels.size() > 0;
    }
    return false;
}

bool
RenderScriptRuntime::ProbeModules(const ModuleList module_list)
{
    bool rs_found = false;
    size_t num_modules = module_list.GetSize();
    for (size_t i = 0; i < num_modules; i++)
    {
        auto module = module_list.GetModuleAtIndex(i);
        rs_found |= LoadModule(module);
    }
    return rs_found;
}

void
RenderScriptRuntime::Status(Stream &strm) const
{
    if (m_libRS)
    {
        strm.Printf("Runtime Library discovered.");
        strm.EOL();
    }
    if (m_libRSDriver)
    {
        strm.Printf("Runtime Driver discovered.");
        strm.EOL();
    }
    if (m_libRSCpuRef)
    {
        strm.Printf("CPU Reference Implementation discovered.");
        strm.EOL();
    }

    if (m_runtimeHooks.size())
    {
        strm.Printf("Runtime functions hooked:");
        strm.EOL();
        for (auto b : m_runtimeHooks)
        {
            strm.Indent(b.second->defn->name);
            strm.EOL();
        }
    }
    else
    {
        strm.Printf("Runtime is not hooked.");
        strm.EOL();
    }
}

void
RenderScriptRuntime::DumpContexts(Stream &strm) const
{
    strm.Printf("Inferred RenderScript Contexts:");
    strm.EOL();
    strm.IndentMore();

    std::map<addr_t, uint64_t> contextReferences;

    // Iterate over all of the currently discovered scripts.
    // Note: We cant push or pop from m_scripts inside this loop or it may invalidate script.
    for (const auto & script : m_scripts)
    {
        if (!script->context.isValid())
            continue;
        lldb::addr_t context = *script->context;

        if (contextReferences.find(context) != contextReferences.end())
        {
            contextReferences[context]++;
        }
        else
        {
            contextReferences[context] = 1;
        }
    }

    for (const auto& cRef : contextReferences)
    {
        strm.Printf("Context 0x%" PRIx64 ": %" PRIu64 " script instances", cRef.first, cRef.second);
        strm.EOL();
    }
    strm.IndentLess();
}

void
RenderScriptRuntime::DumpKernels(Stream &strm) const
{
    strm.Printf("RenderScript Kernels:");
    strm.EOL();
    strm.IndentMore();
    for (const auto &module : m_rsmodules)
    {
        strm.Printf("Resource '%s':",module->m_resname.c_str());
        strm.EOL();
        for (const auto &kernel : module->m_kernels)
        {
            strm.Indent(kernel.m_name.AsCString());
            strm.EOL();
        }
    }
    strm.IndentLess();
}

RenderScriptRuntime::AllocationDetails*
RenderScriptRuntime::FindAllocByID(Stream &strm, const uint32_t alloc_id)
{
    AllocationDetails* alloc = nullptr;

    // See if we can find allocation using id as an index;
    if (alloc_id <= m_allocations.size() && alloc_id != 0
        && m_allocations[alloc_id-1]->id == alloc_id)
    {
        alloc = m_allocations[alloc_id-1].get();
        return alloc;
    }

    // Fallback to searching
    for (const auto & a : m_allocations)
    {
       if (a->id == alloc_id)
       {
           alloc = a.get();
           break;
       }
    }

    if (alloc == nullptr)
    {
        strm.Printf("Error: Couldn't find allocation with id matching %u", alloc_id);
        strm.EOL();
    }

    return alloc;
}

// Prints the contents of an allocation to the output stream, which may be a file
bool
RenderScriptRuntime::DumpAllocation(Stream &strm, StackFrame* frame_ptr, const uint32_t id)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    // Check we can find the desired allocation
    AllocationDetails* alloc = FindAllocByID(strm, id);
    if (!alloc)
        return false; // FindAllocByID() will print error message for us here

    if (log)
        log->Printf("RenderScriptRuntime::DumpAllocation - Found allocation 0x%" PRIx64, *alloc->address.get());

    // Check we have information about the allocation, if not calculate it
    if (alloc->shouldRefresh())
    {
        if (log)
            log->Printf("RenderScriptRuntime::DumpAllocation - Allocation details not calculated yet, jitting info");

        // JIT all the allocation information
        if (!RefreshAllocation(alloc, frame_ptr))
        {
            strm.Printf("Error: Couldn't JIT allocation details");
            strm.EOL();
            return false;
        }
    }

    // Establish format and size of each data element
    const unsigned int vec_size = *alloc->element.type_vec_size.get();
    const Element::DataType type = *alloc->element.type.get();

    assert(type >= Element::RS_TYPE_NONE && type <= Element::RS_TYPE_FONT
                                                   && "Invalid allocation type");

    lldb::Format format;
    if (type >= Element::RS_TYPE_ELEMENT)
        format = eFormatHex;
    else
        format = vec_size == 1 ? static_cast<lldb::Format>(AllocationDetails::RSTypeToFormat[type][eFormatSingle])
                               : static_cast<lldb::Format>(AllocationDetails::RSTypeToFormat[type][eFormatVector]);

    const unsigned int data_size = *alloc->element.datum_size.get();

    if (log)
        log->Printf("RenderScriptRuntime::DumpAllocation - Element size %u bytes, including padding", data_size);

    // Allocate a buffer to copy data into
    std::shared_ptr<uint8_t> buffer = GetAllocationData(alloc, frame_ptr);
    if (!buffer)
    {
        strm.Printf("Error: Couldn't read allocation data");
        strm.EOL();
        return false;
    }

    // Calculate stride between rows as there may be padding at end of rows since
    // allocated memory is 16-byte aligned
    if (!alloc->stride.isValid())
    {
        if (alloc->dimension.get()->dim_2 == 0) // We only have one dimension
            alloc->stride = 0;
        else if (!JITAllocationStride(alloc, frame_ptr))
        {
            strm.Printf("Error: Couldn't calculate allocation row stride");
            strm.EOL();
            return false;
        }
    }
    const unsigned int stride = *alloc->stride.get();
    const unsigned int size = *alloc->size.get(); // Size of whole allocation
    const unsigned int padding = alloc->element.padding.isValid() ? *alloc->element.padding.get() : 0;
    if (log)
        log->Printf("RenderScriptRuntime::DumpAllocation - stride %u bytes, size %u bytes, padding %u", stride, size, padding);

    // Find dimensions used to index loops, so need to be non-zero
    unsigned int dim_x = alloc->dimension.get()->dim_1;
    dim_x = dim_x == 0 ? 1 : dim_x;

    unsigned int dim_y = alloc->dimension.get()->dim_2;
    dim_y = dim_y == 0 ? 1 : dim_y;

    unsigned int dim_z = alloc->dimension.get()->dim_3;
    dim_z = dim_z == 0 ? 1 : dim_z;

    // Use data extractor to format output
    const uint32_t archByteSize = GetProcess()->GetTarget().GetArchitecture().GetAddressByteSize();
    DataExtractor alloc_data(buffer.get(), size, GetProcess()->GetByteOrder(), archByteSize);

    unsigned int offset = 0;   // Offset in buffer to next element to be printed
    unsigned int prev_row = 0; // Offset to the start of the previous row

    // Iterate over allocation dimensions, printing results to user
    strm.Printf("Data (X, Y, Z):");
    for (unsigned int z = 0; z < dim_z; ++z)
    {
        for (unsigned int y = 0; y < dim_y; ++y)
        {
            // Use stride to index start of next row.
            if (!(y==0 && z==0))
                offset = prev_row + stride;
            prev_row = offset;

            // Print each element in the row individually
            for (unsigned int x = 0; x < dim_x; ++x)
            {
                strm.Printf("\n(%u, %u, %u) = ", x, y, z);
                if ((type == Element::RS_TYPE_NONE) && (alloc->element.children.size() > 0) &&
                    (alloc->element.type_name != Element::GetFallbackStructName()))
                {
                    // Here we are dumping an Element of struct type.
                    // This is done using expression evaluation with the name of the struct type and pointer to element.

                    // Don't print the name of the resulting expression, since this will be '$[0-9]+'
                    DumpValueObjectOptions expr_options;
                    expr_options.SetHideName(true);

                    // Setup expression as derefrencing a pointer cast to element address.
                    const int max_expr_size = 512;
                    char expr_char_buffer[max_expr_size];
                    int chars_written = snprintf(expr_char_buffer, max_expr_size, "*(%s*) 0x%" PRIx64,
                                        alloc->element.type_name.AsCString(), *alloc->data_ptr.get() + offset);

                    if (chars_written < 0 || chars_written >= max_expr_size)
                    {
                        if (log)
                            log->Printf("RenderScriptRuntime::DumpAllocation- Error in snprintf()");
                        continue;
                    }

                    // Evaluate expression
                    ValueObjectSP expr_result;
                    GetProcess()->GetTarget().EvaluateExpression(expr_char_buffer, frame_ptr, expr_result);

                    // Print the results to our stream.
                    expr_result->Dump(strm, expr_options);
                }
                else
                {
                    alloc_data.Dump(&strm, offset, format, data_size - padding, 1, 1, LLDB_INVALID_ADDRESS, 0, 0);
                }
                offset += data_size;
            }
        }
    }
    strm.EOL();

    return true;
}

// Prints infomation regarding all the currently loaded allocations.
// These details are gathered by jitting the runtime, which has as latency.
void
RenderScriptRuntime::ListAllocations(Stream &strm, StackFrame* frame_ptr, bool recompute)
{
    strm.Printf("RenderScript Allocations:");
    strm.EOL();
    strm.IndentMore();

    for (auto &alloc : m_allocations)
    {
        // JIT the allocation info if we haven't done it, or the user forces us to.
        bool do_refresh = alloc->shouldRefresh() || recompute;

        // JIT current allocation information
        if (do_refresh && !RefreshAllocation(alloc.get(), frame_ptr))
        {
            strm.Printf("Error: Couldn't evaluate details for allocation %u\n", alloc->id);
            continue;
        }

        strm.Printf("%u:\n",alloc->id);
        strm.IndentMore();

        strm.Indent("Context: ");
        if (!alloc->context.isValid())
            strm.Printf("unknown\n");
        else
            strm.Printf("0x%" PRIx64 "\n", *alloc->context.get());

        strm.Indent("Address: ");
        if (!alloc->address.isValid())
            strm.Printf("unknown\n");
        else
            strm.Printf("0x%" PRIx64 "\n", *alloc->address.get());

        strm.Indent("Data pointer: ");
        if (!alloc->data_ptr.isValid())
            strm.Printf("unknown\n");
        else
            strm.Printf("0x%" PRIx64 "\n", *alloc->data_ptr.get());

        strm.Indent("Dimensions: ");
        if (!alloc->dimension.isValid())
            strm.Printf("unknown\n");
        else
            strm.Printf("(%d, %d, %d)\n", alloc->dimension.get()->dim_1,
                                          alloc->dimension.get()->dim_2,
                                          alloc->dimension.get()->dim_3);

        strm.Indent("Data Type: ");
        if (!alloc->element.type.isValid() || !alloc->element.type_vec_size.isValid())
            strm.Printf("unknown\n");
        else
        {
            const int vector_size = *alloc->element.type_vec_size.get();
            Element::DataType type = *alloc->element.type.get();

            if (!alloc->element.type_name.IsEmpty())
                strm.Printf("%s\n", alloc->element.type_name.AsCString());
            else
            {
                // Enum value isn't monotonous, so doesn't always index RsDataTypeToString array
                if (type >= Element::RS_TYPE_ELEMENT && type <= Element::RS_TYPE_FONT)
                    type = static_cast<Element::DataType>((type - Element::RS_TYPE_ELEMENT) +  Element::RS_TYPE_MATRIX_2X2 + 1);

                if (type >= (sizeof(AllocationDetails::RsDataTypeToString) / sizeof(AllocationDetails::RsDataTypeToString[0]))
                    || vector_size > 4 || vector_size < 1)
                    strm.Printf("invalid type\n");
                else
                    strm.Printf("%s\n", AllocationDetails::RsDataTypeToString[static_cast<unsigned int>(type)][vector_size-1]);
            }
        }

        strm.Indent("Data Kind: ");
        if (!alloc->element.type_kind.isValid())
            strm.Printf("unknown\n");
        else
        {
            const Element::DataKind kind = *alloc->element.type_kind.get();
            if (kind < Element::RS_KIND_USER || kind > Element::RS_KIND_PIXEL_YUV)
                strm.Printf("invalid kind\n");
            else
                strm.Printf("%s\n", AllocationDetails::RsDataKindToString[static_cast<unsigned int>(kind)]);
        }

        strm.EOL();
        strm.IndentLess();
    }
    strm.IndentLess();
}

// Set breakpoints on every kernel found in RS module
void
RenderScriptRuntime::BreakOnModuleKernels(const RSModuleDescriptorSP rsmodule_sp)
{
    for (const auto &kernel : rsmodule_sp->m_kernels)
    {
        // Don't set breakpoint on 'root' kernel
        if (strcmp(kernel.m_name.AsCString(), "root") == 0)
            continue;

        CreateKernelBreakpoint(kernel.m_name);
    }
}

// Method is internally called by the 'kernel breakpoint all' command to
// enable or disable breaking on all kernels.
//
// When do_break is true we want to enable this functionality.
// When do_break is false we want to disable it.
void
RenderScriptRuntime::SetBreakAllKernels(bool do_break, TargetSP target)
{
    Log* log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_LANGUAGE | LIBLLDB_LOG_BREAKPOINTS));

    InitSearchFilter(target);

    // Set breakpoints on all the kernels
    if (do_break && !m_breakAllKernels)
    {
        m_breakAllKernels = true;

        for (const auto &module : m_rsmodules)
            BreakOnModuleKernels(module);

        if (log)
            log->Printf("RenderScriptRuntime::SetBreakAllKernels(True)"
                        "- breakpoints set on all currently loaded kernels");
    }
    else if (!do_break && m_breakAllKernels) // Breakpoints won't be set on any new kernels.
    {
        m_breakAllKernels = false;

        if (log)
            log->Printf("RenderScriptRuntime::SetBreakAllKernels(False) - breakpoints no longer automatically set");
    }
}

// Given the name of a kernel this function creates a breakpoint using our
// own breakpoint resolver, and returns the Breakpoint shared pointer.
BreakpointSP
RenderScriptRuntime::CreateKernelBreakpoint(const ConstString& name)
{
    Log* log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_LANGUAGE | LIBLLDB_LOG_BREAKPOINTS));

    if (!m_filtersp)
    {
        if (log)
            log->Printf("RenderScriptRuntime::CreateKernelBreakpoint - Error: No breakpoint search filter set");
        return nullptr;
    }

    BreakpointResolverSP resolver_sp(new RSBreakpointResolver(nullptr, name));
    BreakpointSP bp = GetProcess()->GetTarget().CreateBreakpoint(m_filtersp, resolver_sp, false, false, false);

    // Give RS breakpoints a specific name, so the user can manipulate them as a group.
    Error err;
    if (!bp->AddName("RenderScriptKernel", err) && log)
        log->Printf("RenderScriptRuntime::CreateKernelBreakpoint: Error setting break name, %s", err.AsCString());

    return bp;
}

// Given an expression for a variable this function tries to calculate the variable's value.
// If this is possible it returns true and sets the uint64_t parameter to the variables unsigned value.
// Otherwise function returns false.
bool
RenderScriptRuntime::GetFrameVarAsUnsigned(const StackFrameSP frame_sp, const char* var_name, uint64_t& val)
{
    Log* log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_LANGUAGE));
    Error error;
    VariableSP var_sp;

    // Find variable in stack frame
    ValueObjectSP value_sp(frame_sp->GetValueForVariableExpressionPath(var_name,
                                                                       eNoDynamicValues,
                                                                       StackFrame::eExpressionPathOptionCheckPtrVsMember |
                                                                       StackFrame::eExpressionPathOptionsAllowDirectIVarAccess,
                                                                       var_sp,
                                                                       error));
    if (!error.Success())
    {
        if (log)
            log->Printf("RenderScriptRuntime::GetFrameVarAsUnsigned - Error, couldn't find '%s' in frame", var_name);

        return false;
    }

    // Find the unsigned int value for the variable
    bool success = false;
    val = value_sp->GetValueAsUnsigned(0, &success);
    if (!success)
    {
        if (log)
            log->Printf("RenderScriptRuntime::GetFrameVarAsUnsigned - Error, couldn't parse '%s' as an unsigned int", var_name);

        return false;
    }

    return true;
}

// Function attempts to find the current coordinate of a kernel invocation by investigating the
// values of frame variables in the .expand function. These coordinates are returned via the coord
// array reference parameter. Returns true if the coordinates could be found, and false otherwise.
bool
RenderScriptRuntime::GetKernelCoordinate(RSCoordinate &coord, Thread *thread_ptr)
{
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    if (!thread_ptr)
    {
        if (log)
            log->Printf("%s - Error, No thread pointer", __FUNCTION__);

        return false;
    }

    // Walk the call stack looking for a function whose name has the suffix '.expand'
    // and contains the variables we're looking for.
    for (uint32_t i = 0; i < thread_ptr->GetStackFrameCount(); ++i)
    {
        if (!thread_ptr->SetSelectedFrameByIndex(i))
            continue;

        StackFrameSP frame_sp = thread_ptr->GetSelectedFrame();
        if (!frame_sp)
            continue;

        // Find the function name
        const SymbolContext sym_ctx = frame_sp->GetSymbolContext(false);
        const char *func_name_cstr = sym_ctx.GetFunctionName().AsCString();
        if (!func_name_cstr)
            continue;

        if (log)
            log->Printf("%s - Inspecting function '%s'", __FUNCTION__, func_name_cstr);

        // Check if function name has .expand suffix
        std::string func_name(func_name_cstr);
        const int length_difference = func_name.length() - RenderScriptRuntime::s_runtimeExpandSuffix.length();
        if (length_difference <= 0)
            continue;

        const int32_t has_expand_suffix = func_name.compare(length_difference,
                                                            RenderScriptRuntime::s_runtimeExpandSuffix.length(),
                                                            RenderScriptRuntime::s_runtimeExpandSuffix);

        if (has_expand_suffix != 0)
            continue;

        if (log)
            log->Printf("%s - Found .expand function '%s'", __FUNCTION__, func_name_cstr);

        // Get values for variables in .expand frame that tell us the current kernel invocation
        bool found_coord_variables = true;
        assert(RenderScriptRuntime::s_runtimeCoordVars.size() == coord.size());

        for (uint32_t i = 0; i < coord.size(); ++i)
        {
            uint64_t value = 0;
            if (!GetFrameVarAsUnsigned(frame_sp, RenderScriptRuntime::s_runtimeCoordVars[i], value))
            {
                found_coord_variables = false;
                break;
            }
            coord[i] = value;
        }

        if (found_coord_variables)
            return true;
    }
    return false;
}

// Callback when a kernel breakpoint hits and we're looking for a specific coordinate.
// Baton parameter contains a pointer to the target coordinate we want to break on.
// Function then checks the .expand frame for the current coordinate and breaks to user if it matches.
// Parameter 'break_id' is the id of the Breakpoint which made the callback.
// Parameter 'break_loc_id' is the id for the BreakpointLocation which was hit,
// a single logical breakpoint can have multiple addresses.
bool
RenderScriptRuntime::KernelBreakpointHit(void *baton, StoppointCallbackContext *ctx,
                                         user_id_t break_id, user_id_t break_loc_id)
{
    Log* log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_LANGUAGE | LIBLLDB_LOG_BREAKPOINTS));

    assert(baton && "Error: null baton in conditional kernel breakpoint callback");

    // Coordinate we want to stop on
    const uint32_t *target_coord = static_cast<const uint32_t *>(baton);

    if (log)
        log->Printf("%s - Break ID %" PRIu64 ", (%" PRIu32 ", %" PRIu32 ", %" PRIu32 ")", __FUNCTION__, break_id,
                    target_coord[0], target_coord[1], target_coord[2]);

    // Select current thread
    ExecutionContext context(ctx->exe_ctx_ref);
    Thread *thread_ptr = context.GetThreadPtr();
    assert(thread_ptr && "Null thread pointer");

    // Find current kernel invocation from .expand frame variables
    RSCoordinate current_coord{}; // Zero initialise array
    if (!GetKernelCoordinate(current_coord, thread_ptr))
    {
        if (log)
            log->Printf("%s - Error, couldn't select .expand stack frame", __FUNCTION__);

        return false;
    }

    if (log)
        log->Printf("%s - (%" PRIu32 ",%" PRIu32 ",%" PRIu32 ")", __FUNCTION__, current_coord[0], current_coord[1],
                    current_coord[2]);

    // Check if the current kernel invocation coordinate matches our target coordinate
    if (current_coord[0] == target_coord[0] && current_coord[1] == target_coord[1] &&
        current_coord[2] == target_coord[2])
    {
        if (log)
            log->Printf("%s, BREAKING (%" PRIu32 ",%" PRIu32 ",%" PRIu32 ")", __FUNCTION__, current_coord[0],
                        current_coord[1], current_coord[2]);

        BreakpointSP breakpoint_sp = context.GetTargetPtr()->GetBreakpointByID(break_id);
        assert(breakpoint_sp != nullptr && "Error: Couldn't find breakpoint matching break id for callback");
        breakpoint_sp->SetEnabled(false); // Optimise since conditional breakpoint should only be hit once.
        return true;
    }

    // No match on coordinate
    return false;
}

// Tries to set a breakpoint on the start of a kernel, resolved using the kernel name.
// Argument 'coords', represents a three dimensional coordinate which can be used to specify
// a single kernel instance to break on. If this is set then we add a callback to the breakpoint.
void
RenderScriptRuntime::PlaceBreakpointOnKernel(Stream &strm, const char* name, const std::array<int,3> coords,
                                             Error& error, TargetSP target)
{
    if (!name)
    {
        error.SetErrorString("invalid kernel name");
        return;
    }

    InitSearchFilter(target);

    ConstString kernel_name(name);
    BreakpointSP bp = CreateKernelBreakpoint(kernel_name);

    // We have a conditional breakpoint on a specific coordinate
    if (coords[0] != -1)
    {
        strm.Printf("Conditional kernel breakpoint on coordinate %d, %d, %d", coords[0], coords[1], coords[2]);
        strm.EOL();

        // Allocate memory for the baton, and copy over coordinate
        uint32_t *baton = new uint32_t[coords.size()];
        baton[0] = coords[0]; baton[1] = coords[1]; baton[2] = coords[2];

        // Create a callback that will be invoked everytime the breakpoint is hit.
        // The baton object passed to the handler is the target coordinate we want to break on.
        bp->SetCallback(KernelBreakpointHit, baton, true);

        // Store a shared pointer to the baton, so the memory will eventually be cleaned up after destruction
        m_conditional_breaks[bp->GetID()] = std::shared_ptr<uint32_t>(baton);
    }

    if (bp)
        bp->GetDescription(&strm, lldb::eDescriptionLevelInitial, false);
}

void
RenderScriptRuntime::DumpModules(Stream &strm) const
{
    strm.Printf("RenderScript Modules:");
    strm.EOL();
    strm.IndentMore();
    for (const auto &module : m_rsmodules)
    {
        module->Dump(strm);
    }
    strm.IndentLess();
}

RenderScriptRuntime::ScriptDetails*
RenderScriptRuntime::LookUpScript(addr_t address, bool create)
{
    for (const auto & s : m_scripts)
    {
        if (s->script.isValid())
            if (*s->script == address)
                return s.get();
    }
    if (create)
    {
        std::unique_ptr<ScriptDetails> s(new ScriptDetails);
        s->script = address;
        m_scripts.push_back(std::move(s));
        return m_scripts.back().get();
    }
    return nullptr;
}

RenderScriptRuntime::AllocationDetails*
RenderScriptRuntime::LookUpAllocation(addr_t address, bool create)
{
    for (const auto & a : m_allocations)
    {
        if (a->address.isValid())
            if (*a->address == address)
                return a.get();
    }
    if (create)
    {
        std::unique_ptr<AllocationDetails> a(new AllocationDetails);
        a->address = address;
        m_allocations.push_back(std::move(a));
        return m_allocations.back().get();
    }
    return nullptr;
}

void
RSModuleDescriptor::Dump(Stream &strm) const
{
    strm.Indent();
    m_module->GetFileSpec().Dump(&strm);
    if(m_module->GetNumCompileUnits())
    {
        strm.Indent("Debug info loaded.");
    }
    else
    {
        strm.Indent("Debug info does not exist.");
    }
    strm.EOL();
    strm.IndentMore();
    strm.Indent();
    strm.Printf("Globals: %" PRIu64, static_cast<uint64_t>(m_globals.size()));
    strm.EOL();
    strm.IndentMore();
    for (const auto &global : m_globals)
    {
        global.Dump(strm);
    }
    strm.IndentLess();
    strm.Indent();
    strm.Printf("Kernels: %" PRIu64, static_cast<uint64_t>(m_kernels.size()));
    strm.EOL();
    strm.IndentMore();
    for (const auto &kernel : m_kernels)
    {
        kernel.Dump(strm);
    }
    strm.Printf("Pragmas: %"  PRIu64 , static_cast<uint64_t>(m_pragmas.size()));
    strm.EOL();
    strm.IndentMore();
    for (const auto &key_val : m_pragmas)
    {
        strm.Printf("%s: %s", key_val.first.c_str(), key_val.second.c_str());
        strm.EOL();
    }
    strm.IndentLess(4);
}

void
RSGlobalDescriptor::Dump(Stream &strm) const
{
    strm.Indent(m_name.AsCString());
    VariableList var_list;
    m_module->m_module->FindGlobalVariables(m_name, nullptr, true, 1U, var_list);
    if (var_list.GetSize() == 1)
    {
        auto var = var_list.GetVariableAtIndex(0);
        auto type = var->GetType();
        if(type)
        {
            strm.Printf(" - ");
            type->DumpTypeName(&strm);
        }
        else
        {
            strm.Printf(" - Unknown Type");
        }
    }
    else
    {
        strm.Printf(" - variable identified, but not found in binary");
        const Symbol* s = m_module->m_module->FindFirstSymbolWithNameAndType(m_name, eSymbolTypeData);
        if (s)
        {
            strm.Printf(" (symbol exists) ");
        }
    }

    strm.EOL();
}

void
RSKernelDescriptor::Dump(Stream &strm) const
{
    strm.Indent(m_name.AsCString());
    strm.EOL();
}

class CommandObjectRenderScriptRuntimeModuleProbe : public CommandObjectParsed
{
public:
    CommandObjectRenderScriptRuntimeModuleProbe(CommandInterpreter &interpreter)
        : CommandObjectParsed(interpreter, "renderscript module probe",
                              "Initiates a Probe of all loaded modules for kernels and other renderscript objects.",
                              "renderscript module probe",
                              eCommandRequiresTarget | eCommandRequiresProcess | eCommandProcessMustBeLaunched)
    {
    }

    ~CommandObjectRenderScriptRuntimeModuleProbe() override = default;

    bool
    DoExecute(Args &command, CommandReturnObject &result) override
    {
        const size_t argc = command.GetArgumentCount();
        if (argc == 0)
        {
            Target *target = m_exe_ctx.GetTargetPtr();
            RenderScriptRuntime *runtime =
                (RenderScriptRuntime *)m_exe_ctx.GetProcessPtr()->GetLanguageRuntime(eLanguageTypeExtRenderScript);
            auto module_list = target->GetImages();
            bool new_rs_details = runtime->ProbeModules(module_list);
            if (new_rs_details)
            {
                result.AppendMessage("New renderscript modules added to runtime model.");
            }
            result.SetStatus(eReturnStatusSuccessFinishResult);
            return true;
        }

        result.AppendErrorWithFormat("'%s' takes no arguments", m_cmd_name.c_str());
        result.SetStatus(eReturnStatusFailed);
        return false;
    }
};

class CommandObjectRenderScriptRuntimeModuleDump : public CommandObjectParsed
{
public:
    CommandObjectRenderScriptRuntimeModuleDump(CommandInterpreter &interpreter)
        : CommandObjectParsed(interpreter, "renderscript module dump",
                              "Dumps renderscript specific information for all modules.", "renderscript module dump",
                              eCommandRequiresProcess | eCommandProcessMustBeLaunched)
    {
    }

    ~CommandObjectRenderScriptRuntimeModuleDump() override = default;

    bool
    DoExecute(Args &command, CommandReturnObject &result) override
    {
        RenderScriptRuntime *runtime =
            (RenderScriptRuntime *)m_exe_ctx.GetProcessPtr()->GetLanguageRuntime(eLanguageTypeExtRenderScript);
        runtime->DumpModules(result.GetOutputStream());
        result.SetStatus(eReturnStatusSuccessFinishResult);
        return true;
    }
};

class CommandObjectRenderScriptRuntimeModule : public CommandObjectMultiword
{
public:
    CommandObjectRenderScriptRuntimeModule(CommandInterpreter &interpreter)
        : CommandObjectMultiword(interpreter, "renderscript module", "Commands that deal with renderscript modules.",
                                 NULL)
    {
        LoadSubCommand("probe", CommandObjectSP(new CommandObjectRenderScriptRuntimeModuleProbe(interpreter)));
        LoadSubCommand("dump", CommandObjectSP(new CommandObjectRenderScriptRuntimeModuleDump(interpreter)));
    }

    ~CommandObjectRenderScriptRuntimeModule() override = default;
};

class CommandObjectRenderScriptRuntimeKernelList : public CommandObjectParsed
{
public:
    CommandObjectRenderScriptRuntimeKernelList(CommandInterpreter &interpreter)
        : CommandObjectParsed(interpreter, "renderscript kernel list",
                              "Lists renderscript kernel names and associated script resources.", "renderscript kernel list",
                              eCommandRequiresProcess | eCommandProcessMustBeLaunched)
    {
    }

    ~CommandObjectRenderScriptRuntimeKernelList() override = default;

    bool
    DoExecute(Args &command, CommandReturnObject &result) override
    {
        RenderScriptRuntime *runtime =
            (RenderScriptRuntime *)m_exe_ctx.GetProcessPtr()->GetLanguageRuntime(eLanguageTypeExtRenderScript);
        runtime->DumpKernels(result.GetOutputStream());
        result.SetStatus(eReturnStatusSuccessFinishResult);
        return true;
    }
};

class CommandObjectRenderScriptRuntimeKernelBreakpointSet : public CommandObjectParsed
{
public:
    CommandObjectRenderScriptRuntimeKernelBreakpointSet(CommandInterpreter &interpreter)
        : CommandObjectParsed(interpreter, "renderscript kernel breakpoint set",
                              "Sets a breakpoint on a renderscript kernel.", "renderscript kernel breakpoint set <kernel_name> [-c x,y,z]",
                              eCommandRequiresProcess | eCommandProcessMustBeLaunched | eCommandProcessMustBePaused), m_options(interpreter)
    {
    }

    ~CommandObjectRenderScriptRuntimeKernelBreakpointSet() override = default;

    Options*
    GetOptions() override
    {
        return &m_options;
    }

    class CommandOptions : public Options
    {
    public:
        CommandOptions(CommandInterpreter &interpreter) : Options(interpreter)
        {
        }

        ~CommandOptions() override = default;

        Error
        SetOptionValue(uint32_t option_idx, const char *option_arg) override
        {
            Error error;
            const int short_option = m_getopt_table[option_idx].val;

            switch (short_option)
            {
                case 'c':
                    if (!ParseCoordinate(option_arg))
                        error.SetErrorStringWithFormat("Couldn't parse coordinate '%s', should be in format 'x,y,z'.", option_arg);
                    break;
                default:
                    error.SetErrorStringWithFormat("unrecognized option '%c'", short_option);
                    break;
            }
            return error;
        }

        // -c takes an argument of the form 'num[,num][,num]'.
        // Where 'id_cstr' is this argument with the whitespace trimmed.
        // Missing coordinates are defaulted to zero.
        bool
        ParseCoordinate(const char* id_cstr)
        {
            RegularExpression regex;
            RegularExpression::Match regex_match(3);

            bool matched = false;
            if(regex.Compile("^([0-9]+),([0-9]+),([0-9]+)$") && regex.Execute(id_cstr, &regex_match))
                matched = true;
            else if(regex.Compile("^([0-9]+),([0-9]+)$") && regex.Execute(id_cstr, &regex_match))
                matched = true;
            else if(regex.Compile("^([0-9]+)$") && regex.Execute(id_cstr, &regex_match))
                matched = true;
            for(uint32_t i = 0; i < 3; i++)
            {
                std::string group;
                if(regex_match.GetMatchAtIndex(id_cstr, i + 1, group))
                    m_coord[i] = (uint32_t)strtoul(group.c_str(), NULL, 0);
                else
                    m_coord[i] = 0;
            }
            return matched;
        }

        void
        OptionParsingStarting() override
        {
            // -1 means the -c option hasn't been set
            m_coord[0] = -1;
            m_coord[1] = -1;
            m_coord[2] = -1;
        }

        const OptionDefinition*
        GetDefinitions() override
        {
            return g_option_table;
        }

        static OptionDefinition g_option_table[];
        std::array<int,3> m_coord;
    };

    bool
    DoExecute(Args &command, CommandReturnObject &result) override
    {
        const size_t argc = command.GetArgumentCount();
        if (argc < 1)
        {
            result.AppendErrorWithFormat("'%s' takes 1 argument of kernel name, and an optional coordinate.", m_cmd_name.c_str());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        RenderScriptRuntime *runtime =
                (RenderScriptRuntime *)m_exe_ctx.GetProcessPtr()->GetLanguageRuntime(eLanguageTypeExtRenderScript);

        Error error;
        runtime->PlaceBreakpointOnKernel(result.GetOutputStream(), command.GetArgumentAtIndex(0), m_options.m_coord,
                                         error, m_exe_ctx.GetTargetSP());

        if (error.Success())
        {
            result.AppendMessage("Breakpoint(s) created");
            result.SetStatus(eReturnStatusSuccessFinishResult);
            return true;
        }
        result.SetStatus(eReturnStatusFailed);
        result.AppendErrorWithFormat("Error: %s", error.AsCString());
        return false;
    }

private:
    CommandOptions m_options;
};

OptionDefinition
CommandObjectRenderScriptRuntimeKernelBreakpointSet::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_1, false, "coordinate", 'c', OptionParser::eRequiredArgument, NULL, NULL, 0, eArgTypeValue,
      "Set a breakpoint on a single invocation of the kernel with specified coordinate.\n"
      "Coordinate takes the form 'x[,y][,z] where x,y,z are positive integers representing kernel dimensions. "
      "Any unset dimensions will be defaulted to zero."},
    { 0, false, NULL, 0, 0, NULL, NULL, 0, eArgTypeNone, NULL }
};

class CommandObjectRenderScriptRuntimeKernelBreakpointAll : public CommandObjectParsed
{
public:
    CommandObjectRenderScriptRuntimeKernelBreakpointAll(CommandInterpreter &interpreter)
        : CommandObjectParsed(interpreter, "renderscript kernel breakpoint all",
                              "Automatically sets a breakpoint on all renderscript kernels that are or will be loaded.\n"
                              "Disabling option means breakpoints will no longer be set on any kernels loaded in the future, "
                              "but does not remove currently set breakpoints.",
                              "renderscript kernel breakpoint all <enable/disable>",
                              eCommandRequiresProcess | eCommandProcessMustBeLaunched | eCommandProcessMustBePaused)
    {
    }

    ~CommandObjectRenderScriptRuntimeKernelBreakpointAll() override = default;

    bool
    DoExecute(Args &command, CommandReturnObject &result) override
    {
        const size_t argc = command.GetArgumentCount();
        if (argc != 1)
        {
            result.AppendErrorWithFormat("'%s' takes 1 argument of 'enable' or 'disable'", m_cmd_name.c_str());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        RenderScriptRuntime *runtime =
          static_cast<RenderScriptRuntime *>(m_exe_ctx.GetProcessPtr()->GetLanguageRuntime(eLanguageTypeExtRenderScript));

        bool do_break = false;
        const char* argument = command.GetArgumentAtIndex(0);
        if (strcmp(argument, "enable") == 0)
        {
            do_break = true;
            result.AppendMessage("Breakpoints will be set on all kernels.");
        }
        else if (strcmp(argument, "disable") == 0)
        {
            do_break = false;
            result.AppendMessage("Breakpoints will not be set on any new kernels.");
        }
        else
        {
            result.AppendErrorWithFormat("Argument must be either 'enable' or 'disable'");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        runtime->SetBreakAllKernels(do_break, m_exe_ctx.GetTargetSP());

        result.SetStatus(eReturnStatusSuccessFinishResult);
        return true;
    }
};

class CommandObjectRenderScriptRuntimeKernelCoordinate : public CommandObjectParsed
{
public:
    CommandObjectRenderScriptRuntimeKernelCoordinate(CommandInterpreter &interpreter)
        : CommandObjectParsed(interpreter, "renderscript kernel coordinate",
                              "Shows the (x,y,z) coordinate of the current kernel invocation.",
                              "renderscript kernel coordinate",
                              eCommandRequiresProcess | eCommandProcessMustBeLaunched | eCommandProcessMustBePaused)
    {
    }

    ~CommandObjectRenderScriptRuntimeKernelCoordinate() override = default;

    bool
    DoExecute(Args &command, CommandReturnObject &result) override
    {
        RSCoordinate coord{}; // Zero initialize array
        bool success = RenderScriptRuntime::GetKernelCoordinate(coord, m_exe_ctx.GetThreadPtr());
        Stream &stream = result.GetOutputStream();

        if (success)
        {
            stream.Printf("Coordinate: (%" PRIu32 ", %" PRIu32 ", %" PRIu32 ")", coord[0], coord[1], coord[2]);
            stream.EOL();
            result.SetStatus(eReturnStatusSuccessFinishResult);
        }
        else
        {
            stream.Printf("Error: Coordinate could not be found.");
            stream.EOL();
            result.SetStatus(eReturnStatusFailed);
        }
        return true;
    }
};

class CommandObjectRenderScriptRuntimeKernelBreakpoint : public CommandObjectMultiword
{
public:
    CommandObjectRenderScriptRuntimeKernelBreakpoint(CommandInterpreter &interpreter)
        : CommandObjectMultiword(interpreter, "renderscript kernel", "Commands that generate breakpoints on renderscript kernels.",
                                 nullptr)
    {
        LoadSubCommand("set", CommandObjectSP(new CommandObjectRenderScriptRuntimeKernelBreakpointSet(interpreter)));
        LoadSubCommand("all", CommandObjectSP(new CommandObjectRenderScriptRuntimeKernelBreakpointAll(interpreter)));
    }

    ~CommandObjectRenderScriptRuntimeKernelBreakpoint() override = default;
};

class CommandObjectRenderScriptRuntimeKernel : public CommandObjectMultiword
{
public:
    CommandObjectRenderScriptRuntimeKernel(CommandInterpreter &interpreter)
        : CommandObjectMultiword(interpreter, "renderscript kernel", "Commands that deal with renderscript kernels.",
                                 NULL)
    {
        LoadSubCommand("list", CommandObjectSP(new CommandObjectRenderScriptRuntimeKernelList(interpreter)));
        LoadSubCommand("breakpoint", CommandObjectSP(new CommandObjectRenderScriptRuntimeKernelBreakpoint(interpreter)));
        LoadSubCommand("coordinate", CommandObjectSP(new CommandObjectRenderScriptRuntimeKernelCoordinate(interpreter)));
    }

    ~CommandObjectRenderScriptRuntimeKernel() override = default;
};

class CommandObjectRenderScriptRuntimeContextDump : public CommandObjectParsed
{
public:
    CommandObjectRenderScriptRuntimeContextDump(CommandInterpreter &interpreter)
        : CommandObjectParsed(interpreter, "renderscript context dump",
                              "Dumps renderscript context information.", "renderscript context dump",
                              eCommandRequiresProcess | eCommandProcessMustBeLaunched)
    {
    }

    ~CommandObjectRenderScriptRuntimeContextDump() override = default;

    bool
    DoExecute(Args &command, CommandReturnObject &result) override
    {
        RenderScriptRuntime *runtime =
            (RenderScriptRuntime *)m_exe_ctx.GetProcessPtr()->GetLanguageRuntime(eLanguageTypeExtRenderScript);
        runtime->DumpContexts(result.GetOutputStream());
        result.SetStatus(eReturnStatusSuccessFinishResult);
        return true;
    }
};

class CommandObjectRenderScriptRuntimeContext : public CommandObjectMultiword
{
public:
    CommandObjectRenderScriptRuntimeContext(CommandInterpreter &interpreter)
        : CommandObjectMultiword(interpreter, "renderscript context", "Commands that deal with renderscript contexts.",
                                 NULL)
    {
        LoadSubCommand("dump", CommandObjectSP(new CommandObjectRenderScriptRuntimeContextDump(interpreter)));
    }

    ~CommandObjectRenderScriptRuntimeContext() override = default;
};

class CommandObjectRenderScriptRuntimeAllocationDump : public CommandObjectParsed
{
public:
    CommandObjectRenderScriptRuntimeAllocationDump(CommandInterpreter &interpreter)
        : CommandObjectParsed(interpreter, "renderscript allocation dump",
                              "Displays the contents of a particular allocation", "renderscript allocation dump <ID>",
                              eCommandRequiresProcess | eCommandProcessMustBeLaunched), m_options(interpreter)
    {
    }

    ~CommandObjectRenderScriptRuntimeAllocationDump() override = default;

    Options*
    GetOptions() override
    {
        return &m_options;
    }

    class CommandOptions : public Options
    {
    public:
        CommandOptions(CommandInterpreter &interpreter) : Options(interpreter)
        {
        }

        ~CommandOptions() override = default;

        Error
        SetOptionValue(uint32_t option_idx, const char *option_arg) override
        {
            Error error;
            const int short_option = m_getopt_table[option_idx].val;

            switch (short_option)
            {
                case 'f':
                    m_outfile.SetFile(option_arg, true);
                    if (m_outfile.Exists())
                    {
                        m_outfile.Clear();
                        error.SetErrorStringWithFormat("file already exists: '%s'", option_arg);
                    }
                    break;
                default:
                    error.SetErrorStringWithFormat("unrecognized option '%c'", short_option);
                    break;
            }
            return error;
        }

        void
        OptionParsingStarting() override
        {
            m_outfile.Clear();
        }

        const OptionDefinition*
        GetDefinitions() override
        {
            return g_option_table;
        }

        static OptionDefinition g_option_table[];
        FileSpec m_outfile;
    };

    bool
    DoExecute(Args &command, CommandReturnObject &result) override
    {
        const size_t argc = command.GetArgumentCount();
        if (argc < 1)
        {
            result.AppendErrorWithFormat("'%s' takes 1 argument, an allocation ID. As well as an optional -f argument",
                                         m_cmd_name.c_str());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        RenderScriptRuntime *runtime =
          static_cast<RenderScriptRuntime *>(m_exe_ctx.GetProcessPtr()->GetLanguageRuntime(eLanguageTypeExtRenderScript));

        const char* id_cstr = command.GetArgumentAtIndex(0);
        bool convert_complete = false;
        const uint32_t id = StringConvert::ToUInt32(id_cstr, UINT32_MAX, 0, &convert_complete);
        if (!convert_complete)
        {
            result.AppendErrorWithFormat("invalid allocation id argument '%s'", id_cstr);
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        Stream* output_strm = nullptr;
        StreamFile outfile_stream;
        const FileSpec &outfile_spec = m_options.m_outfile; // Dump allocation to file instead
        if (outfile_spec)
        {
            // Open output file
            char path[256];
            outfile_spec.GetPath(path, sizeof(path));
            if (outfile_stream.GetFile().Open(path, File::eOpenOptionWrite | File::eOpenOptionCanCreate).Success())
            {
                output_strm = &outfile_stream;
                result.GetOutputStream().Printf("Results written to '%s'", path);
                result.GetOutputStream().EOL();
            }
            else
            {
                result.AppendErrorWithFormat("Couldn't open file '%s'", path);
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
        }
        else
            output_strm = &result.GetOutputStream();

        assert(output_strm != nullptr);
        bool success = runtime->DumpAllocation(*output_strm, m_exe_ctx.GetFramePtr(), id);

        if (success)
            result.SetStatus(eReturnStatusSuccessFinishResult);
        else
            result.SetStatus(eReturnStatusFailed);

        return true;
    }

private:
    CommandOptions m_options;
};

OptionDefinition
CommandObjectRenderScriptRuntimeAllocationDump::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_1, false, "file", 'f', OptionParser::eRequiredArgument, NULL, NULL, 0, eArgTypeFilename,
      "Print results to specified file instead of command line."},
    { 0, false, NULL, 0, 0, NULL, NULL, 0, eArgTypeNone, NULL }
};

class CommandObjectRenderScriptRuntimeAllocationList : public CommandObjectParsed
{
public:
    CommandObjectRenderScriptRuntimeAllocationList(CommandInterpreter &interpreter)
        : CommandObjectParsed(interpreter, "renderscript allocation list",
                              "List renderscript allocations and their information.", "renderscript allocation list",
                              eCommandRequiresProcess | eCommandProcessMustBeLaunched), m_options(interpreter)
    {
    }

    ~CommandObjectRenderScriptRuntimeAllocationList() override = default;

    Options*
    GetOptions() override
    {
        return &m_options;
    }

    class CommandOptions : public Options
    {
    public:
        CommandOptions(CommandInterpreter &interpreter) : Options(interpreter), m_refresh(false)
        {
        }

        ~CommandOptions() override = default;

        Error
        SetOptionValue(uint32_t option_idx, const char *option_arg) override
        {
            Error error;
            const int short_option = m_getopt_table[option_idx].val;

            switch (short_option)
            {
                case 'r':
                    m_refresh = true;
                    break;
                default:
                    error.SetErrorStringWithFormat("unrecognized option '%c'", short_option);
                    break;
            }
            return error;
        }

        void
        OptionParsingStarting() override
        {
            m_refresh = false;
        }

        const OptionDefinition*
        GetDefinitions() override
        {
            return g_option_table;
        }

        static OptionDefinition g_option_table[];
        bool m_refresh;
    };

    bool
    DoExecute(Args &command, CommandReturnObject &result) override
    {
        RenderScriptRuntime *runtime =
          static_cast<RenderScriptRuntime *>(m_exe_ctx.GetProcessPtr()->GetLanguageRuntime(eLanguageTypeExtRenderScript));
        runtime->ListAllocations(result.GetOutputStream(), m_exe_ctx.GetFramePtr(), m_options.m_refresh);
        result.SetStatus(eReturnStatusSuccessFinishResult);
        return true;
    }

private:
    CommandOptions m_options;
};

OptionDefinition
CommandObjectRenderScriptRuntimeAllocationList::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_1, false, "refresh", 'r', OptionParser::eNoArgument, NULL, NULL, 0, eArgTypeNone,
      "Recompute allocation details."},
    { 0, false, NULL, 0, 0, NULL, NULL, 0, eArgTypeNone, NULL }
};

class CommandObjectRenderScriptRuntimeAllocationLoad : public CommandObjectParsed
{
public:
    CommandObjectRenderScriptRuntimeAllocationLoad(CommandInterpreter &interpreter)
        : CommandObjectParsed(interpreter, "renderscript allocation load",
                              "Loads renderscript allocation contents from a file.", "renderscript allocation load <ID> <filename>",
                              eCommandRequiresProcess | eCommandProcessMustBeLaunched)
    {
    }

    ~CommandObjectRenderScriptRuntimeAllocationLoad() override = default;

    bool
    DoExecute(Args &command, CommandReturnObject &result) override
    {
        const size_t argc = command.GetArgumentCount();
        if (argc != 2)
        {
            result.AppendErrorWithFormat("'%s' takes 2 arguments, an allocation ID and filename to read from.", m_cmd_name.c_str());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        RenderScriptRuntime *runtime =
          static_cast<RenderScriptRuntime *>(m_exe_ctx.GetProcessPtr()->GetLanguageRuntime(eLanguageTypeExtRenderScript));

        const char* id_cstr = command.GetArgumentAtIndex(0);
        bool convert_complete = false;
        const uint32_t id = StringConvert::ToUInt32(id_cstr, UINT32_MAX, 0, &convert_complete);
        if (!convert_complete)
        {
            result.AppendErrorWithFormat ("invalid allocation id argument '%s'", id_cstr);
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const char* filename = command.GetArgumentAtIndex(1);
        bool success = runtime->LoadAllocation(result.GetOutputStream(), id, filename, m_exe_ctx.GetFramePtr());

        if (success)
            result.SetStatus(eReturnStatusSuccessFinishResult);
        else
            result.SetStatus(eReturnStatusFailed);

        return true;
    }
};

class CommandObjectRenderScriptRuntimeAllocationSave : public CommandObjectParsed
{
public:
    CommandObjectRenderScriptRuntimeAllocationSave(CommandInterpreter &interpreter)
        : CommandObjectParsed(interpreter, "renderscript allocation save",
                              "Write renderscript allocation contents to a file.", "renderscript allocation save <ID> <filename>",
                              eCommandRequiresProcess | eCommandProcessMustBeLaunched)
    {
    }

    ~CommandObjectRenderScriptRuntimeAllocationSave() override = default;

    bool
    DoExecute(Args &command, CommandReturnObject &result) override
    {
        const size_t argc = command.GetArgumentCount();
        if (argc != 2)
        {
            result.AppendErrorWithFormat("'%s' takes 2 arguments, an allocation ID and filename to read from.", m_cmd_name.c_str());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        RenderScriptRuntime *runtime =
          static_cast<RenderScriptRuntime *>(m_exe_ctx.GetProcessPtr()->GetLanguageRuntime(eLanguageTypeExtRenderScript));

        const char* id_cstr = command.GetArgumentAtIndex(0);
        bool convert_complete = false;
        const uint32_t id = StringConvert::ToUInt32(id_cstr, UINT32_MAX, 0, &convert_complete);
        if (!convert_complete)
        {
            result.AppendErrorWithFormat ("invalid allocation id argument '%s'", id_cstr);
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const char* filename = command.GetArgumentAtIndex(1);
        bool success = runtime->SaveAllocation(result.GetOutputStream(), id, filename, m_exe_ctx.GetFramePtr());

        if (success)
            result.SetStatus(eReturnStatusSuccessFinishResult);
        else
            result.SetStatus(eReturnStatusFailed);

        return true;
    }
};

class CommandObjectRenderScriptRuntimeAllocation : public CommandObjectMultiword
{
public:
    CommandObjectRenderScriptRuntimeAllocation(CommandInterpreter &interpreter)
        : CommandObjectMultiword(interpreter, "renderscript allocation", "Commands that deal with renderscript allocations.",
                                 NULL)
    {
        LoadSubCommand("list", CommandObjectSP(new CommandObjectRenderScriptRuntimeAllocationList(interpreter)));
        LoadSubCommand("dump", CommandObjectSP(new CommandObjectRenderScriptRuntimeAllocationDump(interpreter)));
        LoadSubCommand("save", CommandObjectSP(new CommandObjectRenderScriptRuntimeAllocationSave(interpreter)));
        LoadSubCommand("load", CommandObjectSP(new CommandObjectRenderScriptRuntimeAllocationLoad(interpreter)));
    }

    ~CommandObjectRenderScriptRuntimeAllocation() override = default;
};

class CommandObjectRenderScriptRuntimeStatus : public CommandObjectParsed
{
public:
    CommandObjectRenderScriptRuntimeStatus(CommandInterpreter &interpreter)
        : CommandObjectParsed(interpreter, "renderscript status",
                              "Displays current renderscript runtime status.", "renderscript status",
                              eCommandRequiresProcess | eCommandProcessMustBeLaunched)
    {
    }

    ~CommandObjectRenderScriptRuntimeStatus() override = default;

    bool
    DoExecute(Args &command, CommandReturnObject &result) override
    {
        RenderScriptRuntime *runtime =
            (RenderScriptRuntime *)m_exe_ctx.GetProcessPtr()->GetLanguageRuntime(eLanguageTypeExtRenderScript);
        runtime->Status(result.GetOutputStream());
        result.SetStatus(eReturnStatusSuccessFinishResult);
        return true;
    }
};

class CommandObjectRenderScriptRuntime : public CommandObjectMultiword
{
public:
    CommandObjectRenderScriptRuntime(CommandInterpreter &interpreter)
        : CommandObjectMultiword(interpreter, "renderscript", "A set of commands for operating on renderscript.",
                                 "renderscript <subcommand> [<subcommand-options>]")
    {
        LoadSubCommand("module", CommandObjectSP(new CommandObjectRenderScriptRuntimeModule(interpreter)));
        LoadSubCommand("status", CommandObjectSP(new CommandObjectRenderScriptRuntimeStatus(interpreter)));
        LoadSubCommand("kernel", CommandObjectSP(new CommandObjectRenderScriptRuntimeKernel(interpreter)));
        LoadSubCommand("context", CommandObjectSP(new CommandObjectRenderScriptRuntimeContext(interpreter)));
        LoadSubCommand("allocation", CommandObjectSP(new CommandObjectRenderScriptRuntimeAllocation(interpreter)));
    }

    ~CommandObjectRenderScriptRuntime() override = default;
};

void
RenderScriptRuntime::Initiate()
{
    assert(!m_initiated);
}

RenderScriptRuntime::RenderScriptRuntime(Process *process)
    : lldb_private::CPPLanguageRuntime(process), m_initiated(false), m_debuggerPresentFlagged(false),
      m_breakAllKernels(false)
{
    ModulesDidLoad(process->GetTarget().GetImages());
}

lldb::CommandObjectSP
RenderScriptRuntime::GetCommandObject(lldb_private::CommandInterpreter& interpreter)
{
    static CommandObjectSP command_object;
    if(!command_object)
    {
        command_object.reset(new CommandObjectRenderScriptRuntime(interpreter));
    }
    return command_object;
}

RenderScriptRuntime::~RenderScriptRuntime() = default;
