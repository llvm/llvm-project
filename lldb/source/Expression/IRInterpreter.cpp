//===-- IRInterpreter.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/DataEncoder.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Expression/ClangExpressionDeclMap.h"
#include "lldb/Expression/ClangExpressionVariable.h"
#include "lldb/Expression/IRForTarget.h"
#include "lldb/Expression/IRInterpreter.h"

#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/DataLayout.h"

#include <map>

using namespace llvm;

IRInterpreter::IRInterpreter(lldb_private::ClangExpressionDeclMap &decl_map,
                                           lldb_private::Stream *error_stream) :
    m_decl_map(decl_map),
    m_error_stream(error_stream)
{
    
}

IRInterpreter::~IRInterpreter()
{
    
}

static std::string 
PrintValue(const Value *value, bool truncate = false)
{
    std::string s;
    raw_string_ostream rso(s);
    value->print(rso);
    rso.flush();
    if (truncate)
        s.resize(s.length() - 1);
    
    size_t offset;
    while ((offset = s.find('\n')) != s.npos)
        s.erase(offset, 1);
    while (s[0] == ' ' || s[0] == '\t')
        s.erase(0, 1);
        
    return s;
}

static std::string
PrintType(const Type *type, bool truncate = false)
{
    std::string s;
    raw_string_ostream rso(s);
    type->print(rso);
    rso.flush();
    if (truncate)
        s.resize(s.length() - 1);
    return s;
}

typedef STD_SHARED_PTR(lldb_private::DataEncoder) DataEncoderSP;
typedef STD_SHARED_PTR(lldb_private::DataExtractor) DataExtractorSP;

class Memory
{
public:
    typedef uint32_t                    index_t;
    
    struct Allocation
    {
        // m_virtual_address is always the address of the variable in the virtual memory
        // space provided by Memory.
        //
        // m_origin is always non-NULL and describes the source of the data (possibly
        // m_data if this allocation is the authoritative source).
        //
        // Possible value configurations:
        //
        // Allocation type  getValueType()          getContextType()            m_origin->GetScalar()       m_data
        // =========================================================================================================================
        // FileAddress      eValueTypeFileAddress   eContextTypeInvalid         A location in a binary      NULL
        //                                                                      image
        //                                                      
        // LoadAddress      eValueTypeLoadAddress   eContextTypeInvalid         A location in the target's  NULL
        //                                                                      virtual memory
        //
        // Alloca           eValueTypeHostAddress   eContextTypeInvalid         == m_data->GetBytes()       Deleted at end of 
        //                                                                                                  execution
        //
        // PersistentVar    eValueTypeHostAddress   eContextTypeClangType       A persistent variable's     NULL
        //                                                                      location in LLDB's memory
        //
        // Register         [ignored]               eContextTypeRegister        [ignored]                   Flushed to the register
        //                                                                                                  at the end of execution
        
        lldb::addr_t        m_virtual_address;
        size_t              m_extent;
        lldb_private::Value m_origin;
        lldb::DataBufferSP  m_data;
        
        Allocation (lldb::addr_t virtual_address,
                    size_t extent,
                    lldb::DataBufferSP data) :
            m_virtual_address(virtual_address),
            m_extent(extent),
            m_data(data)
        {
        }
        
        Allocation (const Allocation &allocation) :
            m_virtual_address(allocation.m_virtual_address),
            m_extent(allocation.m_extent),
            m_origin(allocation.m_origin),
            m_data(allocation.m_data)
        {
        }
    };
    
    typedef STD_SHARED_PTR(Allocation)  AllocationSP;
    
    struct Region
    {
        AllocationSP m_allocation;
        uint64_t m_base;
        uint64_t m_extent;
        
        Region () :
            m_allocation(),
            m_base(0),
            m_extent(0)
        {
        }
        
        Region (AllocationSP allocation, uint64_t base, uint64_t extent) :
            m_allocation(allocation),
            m_base(base),
            m_extent(extent)
        {
        }
        
        Region (const Region &region) :
            m_allocation(region.m_allocation),
            m_base(region.m_base),
            m_extent(region.m_extent)
        {
        }
        
        bool IsValid ()
        {
            return (bool) m_allocation;
        }
        
        bool IsInvalid ()
        {
            return !m_allocation;
        }
    };
    
    typedef std::vector <AllocationSP>          MemoryMap;

private:
    lldb::addr_t        m_addr_base;
    lldb::addr_t        m_addr_max;
    MemoryMap           m_memory;
    lldb::ByteOrder     m_byte_order;
    lldb::addr_t        m_addr_byte_size;
    DataLayout         &m_target_data;
    
    lldb_private::ClangExpressionDeclMap   &m_decl_map;
    
    MemoryMap::iterator LookupInternal (lldb::addr_t addr)
    {
        for (MemoryMap::iterator i = m_memory.begin(), e = m_memory.end();
             i != e;
             ++i)
        {
            if ((*i)->m_virtual_address <= addr &&
                (*i)->m_virtual_address + (*i)->m_extent > addr)
                return i;
        }
        
        return m_memory.end();
    }
    
public:
    Memory (DataLayout &target_data,
            lldb_private::ClangExpressionDeclMap &decl_map,
            lldb::addr_t alloc_start,
            lldb::addr_t alloc_max) :
        m_addr_base(alloc_start),
        m_addr_max(alloc_max),
        m_target_data(target_data),
        m_decl_map(decl_map)
    {
        m_byte_order = (target_data.isLittleEndian() ? lldb::eByteOrderLittle : lldb::eByteOrderBig);
        m_addr_byte_size = (target_data.getPointerSize(0));
    }
    
    Region Malloc (size_t size, size_t align)
    {
        lldb::DataBufferSP data(new lldb_private::DataBufferHeap(size, 0));
        
        if (data)
        {
            index_t index = m_memory.size();
            
            const size_t mask = (align - 1);
            
            m_addr_base += mask;
            m_addr_base &= ~mask;
            
            if (m_addr_base + size < m_addr_base ||
                m_addr_base + size > m_addr_max)
                return Region();
            
            uint64_t base = m_addr_base;
                        
            m_memory.push_back(AllocationSP(new Allocation(base, size, data)));
            
            m_addr_base += size;
            
            AllocationSP alloc = m_memory[index];
            
            alloc->m_origin.GetScalar() = (unsigned long long)data->GetBytes();
            alloc->m_origin.SetContext(lldb_private::Value::eContextTypeInvalid, NULL);
            alloc->m_origin.SetValueType(lldb_private::Value::eValueTypeHostAddress);
            
            return Region(alloc, base, size);
        }
        
        return Region();
    }
    
    Region Malloc (Type *type)
    {
        return Malloc (m_target_data.getTypeAllocSize(type),
                       m_target_data.getPrefTypeAlignment(type));
    }
    
    Region Place (Type *type, lldb::addr_t base, lldb_private::Value &value)
    {
        index_t index = m_memory.size();
        size_t size = m_target_data.getTypeAllocSize(type);
        
        m_memory.push_back(AllocationSP(new Allocation(base, size, lldb::DataBufferSP())));
        
        AllocationSP alloc = m_memory[index];
        
        alloc->m_origin = value;
        
        return Region(alloc, base, size);
    }
    
    void Free (lldb::addr_t addr)
    {
        MemoryMap::iterator i = LookupInternal (addr);
        
        if (i != m_memory.end())
            m_memory.erase(i);
    }
    
    Region Lookup (lldb::addr_t addr, Type *type)
    {
        MemoryMap::iterator i = LookupInternal(addr);
        
        if (i == m_memory.end() || !type->isSized())
            return Region();
    
        size_t size = m_target_data.getTypeStoreSize(type);
                
        return Region(*i, addr, size);
    }
        
    DataEncoderSP GetEncoder (Region region)
    {
        if (region.m_allocation->m_origin.GetValueType() != lldb_private::Value::eValueTypeHostAddress)
            return DataEncoderSP();
        
        lldb::DataBufferSP buffer = region.m_allocation->m_data;
        
        if (!buffer)
            return DataEncoderSP();
        
        size_t base_offset = (size_t)(region.m_base - region.m_allocation->m_virtual_address);
                
        return DataEncoderSP(new lldb_private::DataEncoder(buffer->GetBytes() + base_offset, region.m_extent, m_byte_order, m_addr_byte_size));
    }
    
    DataExtractorSP GetExtractor (Region region)
    {
        if (region.m_allocation->m_origin.GetValueType() != lldb_private::Value::eValueTypeHostAddress)
            return DataExtractorSP();
        
        lldb::DataBufferSP buffer = region.m_allocation->m_data;
        size_t base_offset = (size_t)(region.m_base - region.m_allocation->m_virtual_address);

        if (buffer)
            return DataExtractorSP(new lldb_private::DataExtractor(buffer->GetBytes() + base_offset, region.m_extent, m_byte_order, m_addr_byte_size));
        else
            return DataExtractorSP(new lldb_private::DataExtractor((uint8_t*)region.m_allocation->m_origin.GetScalar().ULongLong() + base_offset, region.m_extent, m_byte_order, m_addr_byte_size));
    }
    
    lldb_private::Value GetAccessTarget(lldb::addr_t addr)
    {
        MemoryMap::iterator i = LookupInternal(addr);
        
        if (i == m_memory.end())
            return lldb_private::Value();
        
        lldb_private::Value target = (*i)->m_origin;
        
        if (target.GetContextType() == lldb_private::Value::eContextTypeRegisterInfo)
        {
            target.SetContext(lldb_private::Value::eContextTypeInvalid, NULL);
            target.SetValueType(lldb_private::Value::eValueTypeHostAddress);
            target.GetScalar() = (unsigned long long)(*i)->m_data->GetBytes();
        }
        
        target.GetScalar() += (addr - (*i)->m_virtual_address);
        
        return target;
    }
    
    bool Write (lldb::addr_t addr, const uint8_t *data, size_t length)
    {
        lldb_private::Value target = GetAccessTarget(addr);
        
        return m_decl_map.WriteTarget(target, data, length);
    }
    
    bool Read (uint8_t *data, lldb::addr_t addr, size_t length)
    {
        lldb_private::Value source = GetAccessTarget(addr);
        
        return m_decl_map.ReadTarget(data, source, length);
    }
    
    bool WriteToRawPtr (lldb::addr_t addr, const uint8_t *data, size_t length)
    {
        lldb_private::Value target = m_decl_map.WrapBareAddress(addr);
        
        return m_decl_map.WriteTarget(target, data, length);
    }
    
    bool ReadFromRawPtr (uint8_t *data, lldb::addr_t addr, size_t length)
    {
        lldb_private::Value source = m_decl_map.WrapBareAddress(addr);
        
        return m_decl_map.ReadTarget(data, source, length);
    }
    
    std::string PrintData (lldb::addr_t addr, size_t length)
    {
        lldb_private::Value target = GetAccessTarget(addr);
        
        lldb_private::DataBufferHeap buf(length, 0);
        
        if (!m_decl_map.ReadTarget(buf.GetBytes(), target, length))
            return std::string("<couldn't read data>");
        
        lldb_private::StreamString ss;
        
        for (size_t i = 0; i < length; i++)
        {
            if ((!(i & 0xf)) && i)
                ss.Printf("%02hhx - ", buf.GetBytes()[i]);
            else
                ss.Printf("%02hhx ", buf.GetBytes()[i]);
        }
        
        return ss.GetString();
    }
    
    std::string SummarizeRegion (Region &region)
    {
        lldb_private::StreamString ss;

        lldb_private::Value base = GetAccessTarget(region.m_base);
        
        ss.Printf("%llx [%s - %s %llx]",
                  region.m_base,
                  lldb_private::Value::GetValueTypeAsCString(base.GetValueType()),
                  lldb_private::Value::GetContextTypeAsCString(base.GetContextType()),
                  base.GetScalar().ULongLong());
        
        ss.Printf(" %s", PrintData(region.m_base, region.m_extent).c_str());
        
        return ss.GetString();
    }
};

class InterpreterStackFrame
{
public:
    typedef std::map <const Value*, Memory::Region> ValueMap;

    ValueMap                                m_values;
    Memory                                 &m_memory;
    DataLayout                             &m_target_data;
    lldb_private::ClangExpressionDeclMap   &m_decl_map;
    const BasicBlock                       *m_bb;
    BasicBlock::const_iterator              m_ii;
    BasicBlock::const_iterator              m_ie;
    
    lldb::ByteOrder                         m_byte_order;
    size_t                                  m_addr_byte_size;
    
    InterpreterStackFrame (DataLayout &target_data,
                           Memory &memory,
                           lldb_private::ClangExpressionDeclMap &decl_map) :
        m_memory (memory),
        m_target_data (target_data),
        m_decl_map (decl_map)
    {
        m_byte_order = (target_data.isLittleEndian() ? lldb::eByteOrderLittle : lldb::eByteOrderBig);
        m_addr_byte_size = (target_data.getPointerSize(0));
    }
    
    void Jump (const BasicBlock *bb)
    {
        m_bb = bb;
        m_ii = m_bb->begin();
        m_ie = m_bb->end();
    }
    
    bool Cache (Memory::AllocationSP allocation, Type *type)
    {
        if (allocation->m_origin.GetContextType() != lldb_private::Value::eContextTypeRegisterInfo)
            return false;
        
        return m_decl_map.ReadTarget(allocation->m_data->GetBytes(), allocation->m_origin, allocation->m_data->GetByteSize());
    }
    
    std::string SummarizeValue (const Value *value)
    {
        lldb_private::StreamString ss;

        ss.Printf("%s", PrintValue(value).c_str());
        
        ValueMap::iterator i = m_values.find(value);

        if (i != m_values.end())
        {
            Memory::Region region = i->second;
            
            ss.Printf(" %s", m_memory.SummarizeRegion(region).c_str());
        }
        
        return ss.GetString();
    }
    
    bool AssignToMatchType (lldb_private::Scalar &scalar, uint64_t u64value, Type *type)
    {
        size_t type_size = m_target_data.getTypeStoreSize(type);

        switch (type_size)
        {
        case 1:
            scalar = (uint8_t)u64value;
            break;
        case 2:
            scalar = (uint16_t)u64value;
            break;
        case 4:
            scalar = (uint32_t)u64value;
            break;
        case 8:
            scalar = (uint64_t)u64value;
            break;
        default:
            return false;
        }
        
        return true;
    }
    
    bool EvaluateValue (lldb_private::Scalar &scalar, const Value *value, Module &module)
    {
        const Constant *constant = dyn_cast<Constant>(value);
        
        if (constant)
        {
            if (const ConstantInt *constant_int = dyn_cast<ConstantInt>(constant))
            {                
                return AssignToMatchType(scalar, constant_int->getLimitedValue(), value->getType());
            }
        }
        else
        {
            Memory::Region region = ResolveValue(value, module);
            DataExtractorSP value_extractor = m_memory.GetExtractor(region);
            
            if (!value_extractor)
                return false;
            
            size_t value_size = m_target_data.getTypeStoreSize(value->getType());
                        
            uint32_t offset = 0;
            uint64_t u64value = value_extractor->GetMaxU64(&offset, value_size);
                    
            return AssignToMatchType(scalar, u64value, value->getType());
        }
        
        return false;
    }
    
    bool AssignValue (const Value *value, lldb_private::Scalar &scalar, Module &module)
    {
        Memory::Region region = ResolveValue (value, module);
    
        lldb_private::Scalar cast_scalar;
        
        if (!AssignToMatchType(cast_scalar, scalar.GetRawBits64(0), value->getType()))
            return false;
        
        lldb_private::DataBufferHeap buf(cast_scalar.GetByteSize(), 0);
        
        lldb_private::Error err;
        
        if (!cast_scalar.GetAsMemoryData(buf.GetBytes(), buf.GetByteSize(), m_byte_order, err))
            return false;
        
        DataEncoderSP region_encoder = m_memory.GetEncoder(region);
        
        memcpy(region_encoder->GetDataStart(), buf.GetBytes(), buf.GetByteSize());
        
        return true;
    }
    
    bool ResolveConstantValue (APInt &value, const Constant *constant)
    {
        if (const ConstantInt *constant_int = dyn_cast<ConstantInt>(constant))
        {
            value = constant_int->getValue();
            return true;
        }
        else if (const ConstantFP *constant_fp = dyn_cast<ConstantFP>(constant))
        {
            value = constant_fp->getValueAPF().bitcastToAPInt();
            return true;
        }
        else if (const ConstantExpr *constant_expr = dyn_cast<ConstantExpr>(constant))
        {
            switch (constant_expr->getOpcode())
            {
                default:
                    return false;
                case Instruction::IntToPtr:
                case Instruction::BitCast:
                    return ResolveConstantValue(value, constant_expr->getOperand(0));
                case Instruction::GetElementPtr:
                {
                    ConstantExpr::const_op_iterator op_cursor = constant_expr->op_begin();
                    ConstantExpr::const_op_iterator op_end = constant_expr->op_end();
                                    
                    Constant *base = dyn_cast<Constant>(*op_cursor);
                    
                    if (!base)
                        return false;
                    
                    if (!ResolveConstantValue(value, base))
                        return false;
                    
                    op_cursor++;
                    
                    if (op_cursor == op_end)
                        return true; // no offset to apply!
                    
                    SmallVector <Value *, 8> indices (op_cursor, op_end);
                    
                    uint64_t offset = m_target_data.getIndexedOffset(base->getType(), indices);
                    
                    const bool is_signed = true;
                    value += APInt(value.getBitWidth(), offset, is_signed);
                    
                    return true;
                }
            }
        }
        
        return false;
    }
    
    bool ResolveConstant (Memory::Region &region, const Constant *constant)
    {
        APInt resolved_value;
        
        if (!ResolveConstantValue(resolved_value, constant))
            return false;
        
        const uint64_t *raw_data = resolved_value.getRawData();
            
        size_t constant_size = m_target_data.getTypeStoreSize(constant->getType());
        return m_memory.Write(region.m_base, (const uint8_t*)raw_data, constant_size);
    }
        
    Memory::Region ResolveValue (const Value *value, Module &module)
    {
        ValueMap::iterator i = m_values.find(value);
        
        if (i != m_values.end())
            return i->second;
        
        const GlobalValue *global_value = dyn_cast<GlobalValue>(value);
        
        // If the variable is indirected through the argument
        // array then we need to build an extra level of indirection
        // for it.  This is the default; only magic arguments like
        // "this", "self", and "_cmd" are direct.
        bool indirect_variable = true; 
        
        // Attempt to resolve the value using the program's data.
        // If it is, the values to be created are:
        //
        // data_region - a region of memory in which the variable's data resides.
        // ref_region - a region of memory in which its address (i.e., &var) resides.
        //   In the JIT case, this region would be a member of the struct passed in.
        // pointer_region - a region of memory in which the address of the pointer
        //   resides.  This is an IR-level variable.
        do
        {
            lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

            lldb_private::Value resolved_value;
            lldb_private::ClangExpressionVariable::FlagType flags = 0;
            
            if (global_value)
            {            
                clang::NamedDecl *decl = IRForTarget::DeclForGlobal(global_value, &module);

                if (!decl)
                    break;
                
                if (isa<clang::FunctionDecl>(decl))
                {
                    if (log)
                        log->Printf("The interpreter does not handle function pointers at the moment");
                    
                    return Memory::Region();
                }
                
                resolved_value = m_decl_map.LookupDecl(decl, flags);
            }
            else
            {
                // Special-case "this", "self", and "_cmd"
                
                std::string name_str = value->getName().str();
                
                if (name_str == "this" ||
                    name_str == "self" ||
                    name_str == "_cmd")
                    resolved_value = m_decl_map.GetSpecialValue(lldb_private::ConstString(name_str.c_str()));
                
                indirect_variable = false;
            }
            
            if (resolved_value.GetScalar().GetType() != lldb_private::Scalar::e_void)
            {
                if (resolved_value.GetContextType() == lldb_private::Value::eContextTypeRegisterInfo)
                {
                    bool bare_register = (flags & lldb_private::ClangExpressionVariable::EVBareRegister);

                    if (bare_register)
                        indirect_variable = false;
                    
                    lldb_private::RegisterInfo *reg_info = resolved_value.GetRegisterInfo();
                    Memory::Region data_region = (reg_info->encoding == lldb::eEncodingVector) ?
                        m_memory.Malloc(reg_info->byte_size, m_target_data.getPrefTypeAlignment(value->getType())) :
                        m_memory.Malloc(value->getType());

                    data_region.m_allocation->m_origin = resolved_value;
                    Memory::Region ref_region = m_memory.Malloc(value->getType());
                    Memory::Region pointer_region;
                    
                    if (indirect_variable)
                        pointer_region = m_memory.Malloc(value->getType());
                    
                    if (!Cache(data_region.m_allocation, value->getType()))
                        return Memory::Region();

                    if (ref_region.IsInvalid())
                        return Memory::Region();
                    
                    if (pointer_region.IsInvalid() && indirect_variable)
                        return Memory::Region();
                    
                    DataEncoderSP ref_encoder = m_memory.GetEncoder(ref_region);
                    
                    if (ref_encoder->PutAddress(0, data_region.m_base) == UINT32_MAX)
                        return Memory::Region();
                    
                    if (log)
                    {
                        log->Printf("Made an allocation for register variable %s", PrintValue(value).c_str());
                        log->Printf("  Data contents  : %s", m_memory.PrintData(data_region.m_base, data_region.m_extent).c_str());
                        log->Printf("  Data region    : %llx", (unsigned long long)data_region.m_base);
                        log->Printf("  Ref region     : %llx", (unsigned long long)ref_region.m_base);
                        if (indirect_variable)
                            log->Printf("  Pointer region : %llx", (unsigned long long)pointer_region.m_base);
                    }
                    
                    if (indirect_variable)
                    {
                        DataEncoderSP pointer_encoder = m_memory.GetEncoder(pointer_region);
                        
                        if (pointer_encoder->PutAddress(0, ref_region.m_base) == UINT32_MAX)
                            return Memory::Region();
                        
                        m_values[value] = pointer_region;
                        return pointer_region;
                    }
                    else
                    {
                        m_values[value] = ref_region;
                        return ref_region;
                    }
                }
                else
                {
                    Memory::Region data_region = m_memory.Place(value->getType(), resolved_value.GetScalar().ULongLong(), resolved_value);
                    Memory::Region ref_region = m_memory.Malloc(value->getType());
                    Memory::Region pointer_region;
                    
                    if (indirect_variable)
                        pointer_region = m_memory.Malloc(value->getType());
                           
                    if (ref_region.IsInvalid())
                        return Memory::Region();
                    
                    if (pointer_region.IsInvalid() && indirect_variable)
                        return Memory::Region();
                    
                    DataEncoderSP ref_encoder = m_memory.GetEncoder(ref_region);
                    
                    if (ref_encoder->PutAddress(0, data_region.m_base) == UINT32_MAX)
                        return Memory::Region();
                    
                    if (indirect_variable)
                    {
                        DataEncoderSP pointer_encoder = m_memory.GetEncoder(pointer_region);
                    
                        if (pointer_encoder->PutAddress(0, ref_region.m_base) == UINT32_MAX)
                            return Memory::Region();
                        
                        m_values[value] = pointer_region;
                    }
                    
                    if (log)
                    {
                        log->Printf("Made an allocation for %s", PrintValue(value).c_str());
                        log->Printf("  Data contents  : %s", m_memory.PrintData(data_region.m_base, data_region.m_extent).c_str());
                        log->Printf("  Data region    : %llx", (unsigned long long)data_region.m_base);
                        log->Printf("  Ref region     : %llx", (unsigned long long)ref_region.m_base);
                        if (indirect_variable)
                            log->Printf("  Pointer region : %llx", (unsigned long long)pointer_region.m_base);
                    }
                    
                    if (indirect_variable)
                        return pointer_region;
                    else 
                        return ref_region;
                }
            }
        }
        while(0);
        
        // Fall back and allocate space [allocation type Alloca]
        
        Type *type = value->getType();
        
        lldb::ValueSP backing_value(new lldb_private::Value);
                
        Memory::Region data_region = m_memory.Malloc(type);
        data_region.m_allocation->m_origin.GetScalar() = (unsigned long long)data_region.m_allocation->m_data->GetBytes();
        data_region.m_allocation->m_origin.SetContext(lldb_private::Value::eContextTypeInvalid, NULL);
        data_region.m_allocation->m_origin.SetValueType(lldb_private::Value::eValueTypeHostAddress);
        
        const Constant *constant = dyn_cast<Constant>(value);
        
        do
        {
            if (!constant)
                break;
            
            if (!ResolveConstant (data_region, constant))
                return Memory::Region();
        }
        while(0);
        
        m_values[value] = data_region;
        return data_region;
    }
    
    bool ConstructResult (lldb::ClangExpressionVariableSP &result,
                          const GlobalValue *result_value,
                          const lldb_private::ConstString &result_name,
                          lldb_private::TypeFromParser result_type,
                          Module &module)
    {
        // The result_value resolves to P, a pointer to a region R containing the result data.
        // If the result variable is a reference, the region R contains a pointer to the result R_final in the original process.
        
        if (!result_value)
            return true; // There was no slot for a result – the expression doesn't return one.
        
        ValueMap::iterator i = m_values.find(result_value);

        if (i == m_values.end())
            return false; // There was a slot for the result, but we didn't write into it.
        
        Memory::Region P = i->second;
        DataExtractorSP P_extractor = m_memory.GetExtractor(P);
        
        if (!P_extractor)
            return false;
        
        Type *pointer_ty = result_value->getType();
        PointerType *pointer_ptr_ty = dyn_cast<PointerType>(pointer_ty);
        if (!pointer_ptr_ty)
            return false;
        Type *R_ty = pointer_ptr_ty->getElementType();
                
        uint32_t offset = 0;
        lldb::addr_t pointer = P_extractor->GetAddress(&offset);
        
        Memory::Region R = m_memory.Lookup(pointer, R_ty);
        
        if (R.m_allocation->m_origin.GetValueType() != lldb_private::Value::eValueTypeHostAddress ||
            !R.m_allocation->m_data)
            return false;
        
        lldb_private::Value base;
        
        bool transient = false;
        bool maybe_make_load = false;
        
        if (m_decl_map.ResultIsReference(result_name))
        {
            PointerType *R_ptr_ty = dyn_cast<PointerType>(R_ty);           
            if (!R_ptr_ty)
                return false;
            Type *R_final_ty = R_ptr_ty->getElementType();
            
            DataExtractorSP R_extractor = m_memory.GetExtractor(R);
            
            if (!R_extractor)
                return false;
            
            offset = 0;
            lldb::addr_t R_pointer = R_extractor->GetAddress(&offset);
            
            Memory::Region R_final = m_memory.Lookup(R_pointer, R_final_ty);
            
            if (R_final.m_allocation)
            {            
                if (R_final.m_allocation->m_data)
                    transient = true; // this is a stack allocation
            
                base = R_final.m_allocation->m_origin;
                base.GetScalar() += (R_final.m_base - R_final.m_allocation->m_virtual_address);
            }
            else
            {
                // We got a bare pointer.  We are going to treat it as a load address
                // or a file address, letting decl_map make the choice based on whether
                // or not a process exists.
                
                base.SetContext(lldb_private::Value::eContextTypeInvalid, NULL);
                base.SetValueType(lldb_private::Value::eValueTypeFileAddress);
                base.GetScalar() = (unsigned long long)R_pointer;
                maybe_make_load = true;
            }
        }
        else
        {
            base.SetContext(lldb_private::Value::eContextTypeInvalid, NULL);
            base.SetValueType(lldb_private::Value::eValueTypeHostAddress);
            base.GetScalar() = (unsigned long long)R.m_allocation->m_data->GetBytes() + (R.m_base - R.m_allocation->m_virtual_address);
        }                     
                        
        return m_decl_map.CompleteResultVariable (result, base, result_name, result_type, transient, maybe_make_load);
    }
};

bool
IRInterpreter::maybeRunOnFunction (lldb::ClangExpressionVariableSP &result,
                                   const lldb_private::ConstString &result_name,
                                   lldb_private::TypeFromParser result_type,
                                   Function &llvm_function,
                                   Module &llvm_module,
                                   lldb_private::Error &err)
{
    if (supportsFunction (llvm_function, err))
        return runOnFunction(result,
                             result_name, 
                             result_type, 
                             llvm_function,
                             llvm_module,
                             err);
    else
        return false;
}

static const char *unsupported_opcode_error         = "Interpreter doesn't handle one of the expression's opcodes";
static const char *interpreter_initialization_error = "Interpreter couldn't be initialized";
static const char *interpreter_internal_error       = "Interpreter encountered an internal error";
static const char *bad_value_error                  = "Interpreter couldn't resolve a value during execution";
static const char *memory_allocation_error          = "Interpreter couldn't allocate memory";
static const char *memory_write_error               = "Interpreter couldn't write to memory";
static const char *memory_read_error                = "Interpreter couldn't read from memory";
static const char *infinite_loop_error              = "Interpreter ran for too many cycles";
static const char *bad_result_error                 = "Result of expression is in bad memory";

bool
IRInterpreter::supportsFunction (Function &llvm_function, 
                                 lldb_private::Error &err)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    for (Function::iterator bbi = llvm_function.begin(), bbe = llvm_function.end();
         bbi != bbe;
         ++bbi)
    {
        for (BasicBlock::iterator ii = bbi->begin(), ie = bbi->end();
             ii != ie;
             ++ii)
        {
            switch (ii->getOpcode())
            {
            default:
                {
                    if (log)
                        log->Printf("Unsupported instruction: %s", PrintValue(ii).c_str());
                    err.SetErrorToGenericError();
                    err.SetErrorString(unsupported_opcode_error);
                    return false;
                }
            case Instruction::Add:
            case Instruction::Alloca:
            case Instruction::BitCast:
            case Instruction::Br:
            case Instruction::GetElementPtr:
                break;
            case Instruction::ICmp:
                {
                    ICmpInst *icmp_inst = dyn_cast<ICmpInst>(ii);
                    
                    if (!icmp_inst)
                    {
                        err.SetErrorToGenericError();
                        err.SetErrorString(interpreter_internal_error);
                        return false;
                    }
                    
                    switch (icmp_inst->getPredicate())
                    {
                    default:
                        {
                            if (log)
                                log->Printf("Unsupported ICmp predicate: %s", PrintValue(ii).c_str());
                            
                            err.SetErrorToGenericError();
                            err.SetErrorString(unsupported_opcode_error);
                            return false;
                        }
                    case CmpInst::ICMP_EQ:
                    case CmpInst::ICMP_NE:
                    case CmpInst::ICMP_UGT:
                    case CmpInst::ICMP_UGE:
                    case CmpInst::ICMP_ULT:
                    case CmpInst::ICMP_ULE:
                    case CmpInst::ICMP_SGT:
                    case CmpInst::ICMP_SGE:
                    case CmpInst::ICMP_SLT:
                    case CmpInst::ICMP_SLE:
                        break;
                    }
                }
                break;
            case Instruction::IntToPtr:
            case Instruction::Load:
            case Instruction::Mul:
            case Instruction::Ret:
            case Instruction::SDiv:
            case Instruction::Store:
            case Instruction::Sub:
            case Instruction::UDiv:
            case Instruction::ZExt:
                break;
            }
        }
    }
    
    return true;
}

bool 
IRInterpreter::runOnFunction (lldb::ClangExpressionVariableSP &result,
                              const lldb_private::ConstString &result_name,
                              lldb_private::TypeFromParser result_type,
                              Function &llvm_function,
                              Module &llvm_module,
                              lldb_private::Error &err)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    lldb_private::ClangExpressionDeclMap::TargetInfo target_info = m_decl_map.GetTargetInfo();
    
    if (!target_info.IsValid())
    {
        err.SetErrorToGenericError();
        err.SetErrorString(interpreter_initialization_error);
        return false;
    }
    
    lldb::addr_t alloc_min;
    lldb::addr_t alloc_max;
    
    switch (target_info.address_byte_size)
    {
    default:
        err.SetErrorToGenericError();
        err.SetErrorString(interpreter_initialization_error);
        return false;    
    case 4:
        alloc_min = 0x00001000llu;
        alloc_max = 0x0000ffffllu;
        break;
    case 8:
        alloc_min = 0x0000000000001000llu;
        alloc_max = 0x000000000000ffffllu;
        break;
    }
    
    DataLayout target_data(&llvm_module);
    if (target_data.getPointerSize(0) != target_info.address_byte_size)
    {
        err.SetErrorToGenericError();
        err.SetErrorString(interpreter_initialization_error);
        return false;
    }
    if (target_data.isLittleEndian() != (target_info.byte_order == lldb::eByteOrderLittle))
    {
        err.SetErrorToGenericError();
        err.SetErrorString(interpreter_initialization_error);
        return false;
    }
    
    Memory memory(target_data, m_decl_map, alloc_min, alloc_max);
    InterpreterStackFrame frame(target_data, memory, m_decl_map);

    uint32_t num_insts = 0;
    
    frame.Jump(llvm_function.begin());
    
    while (frame.m_ii != frame.m_ie && (++num_insts < 4096))
    {
        const Instruction *inst = frame.m_ii;
        
        if (log)
            log->Printf("Interpreting %s", PrintValue(inst).c_str());
        
        switch (inst->getOpcode())
        {
        default:
            break;
        case Instruction::Add:
        case Instruction::Sub:
        case Instruction::Mul:
        case Instruction::SDiv:
        case Instruction::UDiv:
            {
                const BinaryOperator *bin_op = dyn_cast<BinaryOperator>(inst);
                
                if (!bin_op)
                {
                    if (log)
                        log->Printf("getOpcode() returns %s, but instruction is not a BinaryOperator", inst->getOpcodeName());
                    err.SetErrorToGenericError();
                    err.SetErrorString(interpreter_internal_error);
                    return false;                
                }
                
                Value *lhs = inst->getOperand(0);
                Value *rhs = inst->getOperand(1);
                
                lldb_private::Scalar L;
                lldb_private::Scalar R;
                
                if (!frame.EvaluateValue(L, lhs, llvm_module))
                {
                    if (log)
                        log->Printf("Couldn't evaluate %s", PrintValue(lhs).c_str());
                    err.SetErrorToGenericError();
                    err.SetErrorString(bad_value_error);
                    return false;
                }
                
                if (!frame.EvaluateValue(R, rhs, llvm_module))
                {
                    if (log)
                        log->Printf("Couldn't evaluate %s", PrintValue(rhs).c_str());
                    err.SetErrorToGenericError();
                    err.SetErrorString(bad_value_error);                 
                    return false;
                }
                
                lldb_private::Scalar result;
                
                switch (inst->getOpcode())
                {
                default:
                    break;
                case Instruction::Add:
                    result = L + R;
                    break;
                case Instruction::Mul:
                    result = L * R;
                    break;
                case Instruction::Sub:
                    result = L - R;
                    break;
                case Instruction::SDiv:
                    result = L / R;
                    break;
                case Instruction::UDiv:
                    result = L.GetRawBits64(0) / R.GetRawBits64(1);
                    break;
                }
                                
                frame.AssignValue(inst, result, llvm_module);
                
                if (log)
                {
                    log->Printf("Interpreted a %s", inst->getOpcodeName());
                    log->Printf("  L : %s", frame.SummarizeValue(lhs).c_str());
                    log->Printf("  R : %s", frame.SummarizeValue(rhs).c_str());
                    log->Printf("  = : %s", frame.SummarizeValue(inst).c_str());
                }
            }
            break;
        case Instruction::Alloca:
            {
                const AllocaInst *alloca_inst = dyn_cast<AllocaInst>(inst);
                
                if (!alloca_inst)
                {
                    if (log)
                        log->Printf("getOpcode() returns Alloca, but instruction is not an AllocaInst");
                    err.SetErrorToGenericError();
                    err.SetErrorString(interpreter_internal_error);
                    return false;
                }
                
                if (alloca_inst->isArrayAllocation())
                {
                    if (log)
                        log->Printf("AllocaInsts are not handled if isArrayAllocation() is true");
                    err.SetErrorToGenericError();
                    err.SetErrorString(unsupported_opcode_error);
                    return false;
                }
                
                // The semantics of Alloca are:
                //   Create a region R of virtual memory of type T, backed by a data buffer
                //   Create a region P of virtual memory of type T*, backed by a data buffer
                //   Write the virtual address of R into P
                
                Type *T = alloca_inst->getAllocatedType();
                Type *Tptr = alloca_inst->getType();
                
                Memory::Region R = memory.Malloc(T);
                
                if (R.IsInvalid())
                {
                    if (log)
                        log->Printf("Couldn't allocate memory for an AllocaInst");
                    err.SetErrorToGenericError();
                    err.SetErrorString(memory_allocation_error);
                    return false;
                }
                
                Memory::Region P = memory.Malloc(Tptr);
                
                if (P.IsInvalid())
                {
                    if (log)
                        log->Printf("Couldn't allocate the result pointer for an AllocaInst");
                    err.SetErrorToGenericError();
                    err.SetErrorString(memory_allocation_error);
                    return false;
                }
                
                DataEncoderSP P_encoder = memory.GetEncoder(P);
                
                if (P_encoder->PutAddress(0, R.m_base) == UINT32_MAX)
                {
                    if (log)
                        log->Printf("Couldn't write the result pointer for an AllocaInst");
                    err.SetErrorToGenericError();
                    err.SetErrorString(memory_write_error);
                    return false;
                }
                
                frame.m_values[alloca_inst] = P;
                
                if (log)
                {
                    log->Printf("Interpreted an AllocaInst");
                    log->Printf("  R : %s", memory.SummarizeRegion(R).c_str());
                    log->Printf("  P : %s", frame.SummarizeValue(alloca_inst).c_str());
                }
            }
            break;
        case Instruction::BitCast:
        case Instruction::ZExt:
            {
                const CastInst *cast_inst = dyn_cast<CastInst>(inst);
                
                if (!cast_inst)
                {
                    if (log)
                        log->Printf("getOpcode() returns %s, but instruction is not a BitCastInst", cast_inst->getOpcodeName());
                    err.SetErrorToGenericError();
                    err.SetErrorString(interpreter_internal_error);
                    return false;
                }
                
                Value *source = cast_inst->getOperand(0);
                
                lldb_private::Scalar S;
                
                if (!frame.EvaluateValue(S, source, llvm_module))
                {
                    if (log)
                        log->Printf("Couldn't evaluate %s", PrintValue(source).c_str());
                    err.SetErrorToGenericError();
                    err.SetErrorString(bad_value_error);
                    return false;
                }
                
                frame.AssignValue(inst, S, llvm_module);
            }
            break;
        case Instruction::Br:
            {
                const BranchInst *br_inst = dyn_cast<BranchInst>(inst);
                
                if (!br_inst)
                {
                    if (log)
                        log->Printf("getOpcode() returns Br, but instruction is not a BranchInst");
                    err.SetErrorToGenericError();
                    err.SetErrorString(interpreter_internal_error);
                    return false;
                }
                
                if (br_inst->isConditional())
                {
                    Value *condition = br_inst->getCondition();
                    
                    lldb_private::Scalar C;
                    
                    if (!frame.EvaluateValue(C, condition, llvm_module))
                    {
                        if (log)
                            log->Printf("Couldn't evaluate %s", PrintValue(condition).c_str());
                        err.SetErrorToGenericError();
                        err.SetErrorString(bad_value_error);
                        return false;
                    }
                
                    if (C.GetRawBits64(0))
                        frame.Jump(br_inst->getSuccessor(0));
                    else
                        frame.Jump(br_inst->getSuccessor(1));
                    
                    if (log)
                    {
                        log->Printf("Interpreted a BrInst with a condition");
                        log->Printf("  cond : %s", frame.SummarizeValue(condition).c_str());
                    }
                }
                else
                {
                    frame.Jump(br_inst->getSuccessor(0));
                    
                    if (log)
                    {
                        log->Printf("Interpreted a BrInst with no condition");
                    }
                }
            }
            continue;
        case Instruction::GetElementPtr:
            {
                const GetElementPtrInst *gep_inst = dyn_cast<GetElementPtrInst>(inst);
                
                if (!gep_inst)
                {
                    if (log)
                        log->Printf("getOpcode() returns GetElementPtr, but instruction is not a GetElementPtrInst");
                    err.SetErrorToGenericError();
                    err.SetErrorString(interpreter_internal_error);
                    return false;
                }
             
                const Value *pointer_operand = gep_inst->getPointerOperand();
                Type *pointer_type = pointer_operand->getType();
                
                lldb_private::Scalar P;
                
                if (!frame.EvaluateValue(P, pointer_operand, llvm_module))
                {
                    if (log)
                        log->Printf("Couldn't evaluate %s", PrintValue(pointer_operand).c_str());
                    err.SetErrorToGenericError();
                    err.SetErrorString(bad_value_error);
                    return false;
                }
                    
                typedef SmallVector <Value *, 8> IndexVector;
                typedef IndexVector::iterator IndexIterator;
                
                SmallVector <Value *, 8> indices (gep_inst->idx_begin(),
                                                  gep_inst->idx_end());
                
                SmallVector <Value *, 8> const_indices;
                
                for (IndexIterator ii = indices.begin(), ie = indices.end();
                     ii != ie;
                     ++ii)
                {
                    ConstantInt *constant_index = dyn_cast<ConstantInt>(*ii);
                    
                    if (!constant_index)
                    {
                        lldb_private::Scalar I;
                        
                        if (!frame.EvaluateValue(I, *ii, llvm_module))
                        {
                            if (log)
                                log->Printf("Couldn't evaluate %s", PrintValue(*ii).c_str());
                            err.SetErrorToGenericError();
                            err.SetErrorString(bad_value_error);
                            return false;
                        }
                        
                        if (log)
                            log->Printf("Evaluated constant index %s as %llu", PrintValue(*ii).c_str(), I.ULongLong(LLDB_INVALID_ADDRESS));
                        
                        constant_index = cast<ConstantInt>(ConstantInt::get((*ii)->getType(), I.ULongLong(LLDB_INVALID_ADDRESS)));
                    }
                    
                    const_indices.push_back(constant_index);
                }
                
                uint64_t offset = target_data.getIndexedOffset(pointer_type, const_indices);
                
                lldb_private::Scalar Poffset = P + offset;
                
                frame.AssignValue(inst, Poffset, llvm_module);
                
                if (log)
                {
                    log->Printf("Interpreted a GetElementPtrInst");
                    log->Printf("  P       : %s", frame.SummarizeValue(pointer_operand).c_str());
                    log->Printf("  Poffset : %s", frame.SummarizeValue(inst).c_str());
                }
            }
            break;
        case Instruction::ICmp:
            {
                const ICmpInst *icmp_inst = dyn_cast<ICmpInst>(inst);
                
                if (!icmp_inst)
                {
                    if (log)
                        log->Printf("getOpcode() returns ICmp, but instruction is not an ICmpInst");
                    err.SetErrorToGenericError();
                    err.SetErrorString(interpreter_internal_error);
                    return false;
                }
                
                CmpInst::Predicate predicate = icmp_inst->getPredicate();
                
                Value *lhs = inst->getOperand(0);
                Value *rhs = inst->getOperand(1);
                
                lldb_private::Scalar L;
                lldb_private::Scalar R;
                
                if (!frame.EvaluateValue(L, lhs, llvm_module))
                {
                    if (log)
                        log->Printf("Couldn't evaluate %s", PrintValue(lhs).c_str());
                    err.SetErrorToGenericError();
                    err.SetErrorString(bad_value_error);
                    return false;
                }
                
                if (!frame.EvaluateValue(R, rhs, llvm_module))
                {
                    if (log)
                        log->Printf("Couldn't evaluate %s", PrintValue(rhs).c_str());
                    err.SetErrorToGenericError();
                    err.SetErrorString(bad_value_error);
                    return false;
                }
                
                lldb_private::Scalar result;

                switch (predicate)
                {
                default:
                    return false;
                case CmpInst::ICMP_EQ:
                    result = (L == R);
                    break;
                case CmpInst::ICMP_NE:
                    result = (L != R);
                    break;    
                case CmpInst::ICMP_UGT:
                    result = (L.GetRawBits64(0) > R.GetRawBits64(0));
                    break;
                case CmpInst::ICMP_UGE:
                    result = (L.GetRawBits64(0) >= R.GetRawBits64(0));
                    break;
                case CmpInst::ICMP_ULT:
                    result = (L.GetRawBits64(0) < R.GetRawBits64(0));
                    break;
                case CmpInst::ICMP_ULE:
                    result = (L.GetRawBits64(0) <= R.GetRawBits64(0));
                    break;
                case CmpInst::ICMP_SGT:
                    result = (L > R);
                    break;
                case CmpInst::ICMP_SGE:
                    result = (L >= R);
                    break;
                case CmpInst::ICMP_SLT:
                    result = (L < R);
                    break;
                case CmpInst::ICMP_SLE:
                    result = (L <= R);
                    break;
                }
                
                frame.AssignValue(inst, result, llvm_module);
                
                if (log)
                {
                    log->Printf("Interpreted an ICmpInst");
                    log->Printf("  L : %s", frame.SummarizeValue(lhs).c_str());
                    log->Printf("  R : %s", frame.SummarizeValue(rhs).c_str());
                    log->Printf("  = : %s", frame.SummarizeValue(inst).c_str());
                }
            }
            break;
        case Instruction::IntToPtr:
            {
                const IntToPtrInst *int_to_ptr_inst = dyn_cast<IntToPtrInst>(inst);
                
                if (!int_to_ptr_inst)
                {
                    if (log)
                        log->Printf("getOpcode() returns IntToPtr, but instruction is not an IntToPtrInst");
                    err.SetErrorToGenericError();
                    err.SetErrorString(interpreter_internal_error);
                    return false;
                }
                
                Value *src_operand = int_to_ptr_inst->getOperand(0);
                
                lldb_private::Scalar I;
                
                if (!frame.EvaluateValue(I, src_operand, llvm_module))
                {
                    if (log)
                        log->Printf("Couldn't evaluate %s", PrintValue(src_operand).c_str());
                    err.SetErrorToGenericError();
                    err.SetErrorString(bad_value_error);
                    return false;
                }
                
                frame.AssignValue(inst, I, llvm_module);
                
                if (log)
                {
                    log->Printf("Interpreted an IntToPtr");
                    log->Printf("  Src : %s", frame.SummarizeValue(src_operand).c_str());
                    log->Printf("  =   : %s", frame.SummarizeValue(inst).c_str()); 
                }
            }
            break;
        case Instruction::Load:
            {
                const LoadInst *load_inst = dyn_cast<LoadInst>(inst);
                
                if (!load_inst)
                {
                    if (log)
                        log->Printf("getOpcode() returns Load, but instruction is not a LoadInst");
                    err.SetErrorToGenericError();
                    err.SetErrorString(interpreter_internal_error);
                    return false;
                }
                
                // The semantics of Load are:
                //   Create a region D that will contain the loaded data
                //   Resolve the region P containing a pointer
                //   Dereference P to get the region R that the data should be loaded from
                //   Transfer a unit of type type(D) from R to D
                                
                const Value *pointer_operand = load_inst->getPointerOperand();
                
                Type *pointer_ty = pointer_operand->getType();
                PointerType *pointer_ptr_ty = dyn_cast<PointerType>(pointer_ty);
                if (!pointer_ptr_ty)
                {
                    if (log)
                        log->Printf("getPointerOperand()->getType() is not a PointerType");
                    err.SetErrorToGenericError();
                    err.SetErrorString(interpreter_internal_error);
                    return false;
                }
                Type *target_ty = pointer_ptr_ty->getElementType();
                
                Memory::Region D = frame.ResolveValue(load_inst, llvm_module);
                Memory::Region P = frame.ResolveValue(pointer_operand, llvm_module);
                
                if (D.IsInvalid())
                {
                    if (log)
                        log->Printf("LoadInst's value doesn't resolve to anything");
                    err.SetErrorToGenericError();
                    err.SetErrorString(bad_value_error);
                    return false;
                }
                
                if (P.IsInvalid())
                {
                    if (log)
                        log->Printf("LoadInst's pointer doesn't resolve to anything");
                    err.SetErrorToGenericError();
                    err.SetErrorString(bad_value_error);
                    return false;
                }
                
                DataExtractorSP P_extractor(memory.GetExtractor(P));
                DataEncoderSP D_encoder(memory.GetEncoder(D));

                uint32_t offset = 0;
                lldb::addr_t pointer = P_extractor->GetAddress(&offset);
                
                Memory::Region R = memory.Lookup(pointer, target_ty);
                
                if (R.IsValid())
                {
                    if (!memory.Read(D_encoder->GetDataStart(), R.m_base, target_data.getTypeStoreSize(target_ty)))
                    {
                        if (log)
                            log->Printf("Couldn't read from a region on behalf of a LoadInst");
                        err.SetErrorToGenericError();
                        err.SetErrorString(memory_read_error);
                        return false;
                    }
                }
                else
                {
                    if (!memory.ReadFromRawPtr(D_encoder->GetDataStart(), pointer, target_data.getTypeStoreSize(target_ty)))
                    {
                        if (log)
                            log->Printf("Couldn't read from a raw pointer on behalf of a LoadInst");
                        err.SetErrorToGenericError();
                        err.SetErrorString(memory_read_error);
                        return false;
                    }
                }
                
                if (log)
                {
                    log->Printf("Interpreted a LoadInst");
                    log->Printf("  P : %s", frame.SummarizeValue(pointer_operand).c_str());
                    if (R.IsValid())
                        log->Printf("  R : %s", memory.SummarizeRegion(R).c_str());
                    else
                        log->Printf("  R : raw pointer 0x%llx", (unsigned long long)pointer);
                    log->Printf("  D : %s", frame.SummarizeValue(load_inst).c_str());
                }
            }
            break;
        case Instruction::Ret:
            {
                if (result_name.IsEmpty())
                    return true;
                
                GlobalValue *result_value = llvm_module.getNamedValue(result_name.GetCString());
                
                if (!frame.ConstructResult(result, result_value, result_name, result_type, llvm_module))
                {
                    if (log)
                        log->Printf("Couldn't construct the expression's result");
                    err.SetErrorToGenericError();
                    err.SetErrorString(bad_result_error);
                    return false;
                }
                
                return true;
            }
        case Instruction::Store:
            {
                const StoreInst *store_inst = dyn_cast<StoreInst>(inst);
                
                if (!store_inst)
                {
                    if (log)
                        log->Printf("getOpcode() returns Store, but instruction is not a StoreInst");
                    err.SetErrorToGenericError();
                    err.SetErrorString(interpreter_internal_error);
                    return false;
                }
                
                // The semantics of Store are:
                //   Resolve the region D containing the data to be stored
                //   Resolve the region P containing a pointer
                //   Dereference P to get the region R that the data should be stored in
                //   Transfer a unit of type type(D) from D to R
                
                const Value *value_operand = store_inst->getValueOperand();
                const Value *pointer_operand = store_inst->getPointerOperand();
                
                Type *pointer_ty = pointer_operand->getType();
                PointerType *pointer_ptr_ty = dyn_cast<PointerType>(pointer_ty);
                if (!pointer_ptr_ty)
                    return false;
                Type *target_ty = pointer_ptr_ty->getElementType();
                
                Memory::Region D = frame.ResolveValue(value_operand, llvm_module);
                Memory::Region P = frame.ResolveValue(pointer_operand, llvm_module);
                
                if (D.IsInvalid())
                {
                    if (log)
                        log->Printf("StoreInst's value doesn't resolve to anything");
                    err.SetErrorToGenericError();
                    err.SetErrorString(bad_value_error);
                    return false;
                }
                
                if (P.IsInvalid())
                {
                    if (log)
                        log->Printf("StoreInst's pointer doesn't resolve to anything");
                    err.SetErrorToGenericError();
                    err.SetErrorString(bad_value_error);
                    return false;
                }
                
                DataExtractorSP P_extractor(memory.GetExtractor(P));
                DataExtractorSP D_extractor(memory.GetExtractor(D));

                if (!P_extractor || !D_extractor)
                    return false;
                
                uint32_t offset = 0;
                lldb::addr_t pointer = P_extractor->GetAddress(&offset);
                
                Memory::Region R = memory.Lookup(pointer, target_ty);
                
                if (R.IsValid())
                {
                    if (!memory.Write(R.m_base, D_extractor->GetDataStart(), target_data.getTypeStoreSize(target_ty)))
                    {
                        if (log)
                            log->Printf("Couldn't write to a region on behalf of a LoadInst");
                        err.SetErrorToGenericError();
                        err.SetErrorString(memory_write_error);
                        return false;
                    }
                }
                else
                {
                    if (!memory.WriteToRawPtr(pointer, D_extractor->GetDataStart(), target_data.getTypeStoreSize(target_ty)))
                    {
                        if (log)
                            log->Printf("Couldn't write to a raw pointer on behalf of a LoadInst");
                        err.SetErrorToGenericError();
                        err.SetErrorString(memory_write_error);
                        return false;
                    }
                }
                
                
                if (log)
                {
                    log->Printf("Interpreted a StoreInst");
                    log->Printf("  D : %s", frame.SummarizeValue(value_operand).c_str());
                    log->Printf("  P : %s", frame.SummarizeValue(pointer_operand).c_str());
                    log->Printf("  R : %s", memory.SummarizeRegion(R).c_str());
                }
            }
            break;
        }
        
        ++frame.m_ii;
    }
    
    if (num_insts >= 4096)
    {
        err.SetErrorToGenericError();
        err.SetErrorString(infinite_loop_error);
        return false;
    }
        
    return false; 
}
