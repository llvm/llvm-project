//===-- IRInterpreter.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Expression/IRMemoryMap.h"
#include "lldb/Expression/IRInterpreter.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

using namespace llvm;

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

class InterpreterStackFrame
{
public:
    typedef std::map <const Value*, lldb::addr_t> ValueMap;
    
    ValueMap                                m_values;
    DataLayout                             &m_target_data;
    lldb_private::IRMemoryMap              &m_memory_map;
    const BasicBlock                       *m_bb;
    BasicBlock::const_iterator              m_ii;
    BasicBlock::const_iterator              m_ie;
    
    lldb::addr_t                            m_frame_process_address;
    size_t                                  m_frame_size;
    lldb::addr_t                            m_stack_pointer;
    
    lldb::ByteOrder                         m_byte_order;
    size_t                                  m_addr_byte_size;
    
    InterpreterStackFrame (DataLayout &target_data,
                           lldb_private::IRMemoryMap &memory_map,
                           lldb::addr_t stack_frame_bottom,
                           lldb::addr_t stack_frame_top) :
        m_target_data (target_data),
        m_memory_map (memory_map)
    {
        m_byte_order = (target_data.isLittleEndian() ? lldb::eByteOrderLittle : lldb::eByteOrderBig);
        m_addr_byte_size = (target_data.getPointerSize(0));
                        
        m_frame_process_address = stack_frame_bottom;
        m_frame_size = stack_frame_top - stack_frame_bottom;
        m_stack_pointer = stack_frame_top;
    }
    
    ~InterpreterStackFrame ()
    {
    }
    
    void Jump (const BasicBlock *bb)
    {
        m_bb = bb;
        m_ii = m_bb->begin();
        m_ie = m_bb->end();
    }
    
    std::string SummarizeValue (const Value *value)
    {
        lldb_private::StreamString ss;

        ss.Printf("%s", PrintValue(value).c_str());
        
        ValueMap::iterator i = m_values.find(value);

        if (i != m_values.end())
        {
            lldb::addr_t addr = i->second;
            
            ss.Printf(" 0x%llx", (unsigned long long)addr);
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
            lldb::addr_t process_address = ResolveValue(value, module);
            size_t value_size = m_target_data.getTypeStoreSize(value->getType());
        
            lldb_private::DataExtractor value_extractor;
            lldb_private::Error extract_error;
            
            m_memory_map.GetMemoryData(value_extractor, process_address, value_size, extract_error);
            
            if (!extract_error.Success())
                return false;
            
            lldb::offset_t offset = 0;
            if (value_size <= 8)
            {
                uint64_t u64value = value_extractor.GetMaxU64(&offset, value_size);
                return AssignToMatchType(scalar, u64value, value->getType());
            }
        }
        
        return false;
    }
    
    bool AssignValue (const Value *value, lldb_private::Scalar &scalar, Module &module)
    {
        lldb::addr_t process_address = ResolveValue (value, module);
        
        if (process_address == LLDB_INVALID_ADDRESS)
            return false;
    
        lldb_private::Scalar cast_scalar;
        
        if (!AssignToMatchType(cast_scalar, scalar.GetRawBits64(0), value->getType()))
            return false;
                
        size_t value_byte_size = m_target_data.getTypeStoreSize(value->getType());
        
        lldb_private::DataBufferHeap buf(value_byte_size, 0);
        
        lldb_private::Error get_data_error;
        
        if (!cast_scalar.GetAsMemoryData(buf.GetBytes(), buf.GetByteSize(), m_byte_order, get_data_error))
            return false;
        
        lldb_private::Error write_error;
        
        m_memory_map.WriteMemory(process_address, buf.GetBytes(), buf.GetByteSize(), write_error);
        
        return write_error.Success();
    }
    
    bool ResolveConstantValue (APInt &value, const Constant *constant)
    {
        switch (constant->getValueID())
        {
        default:
            break;
        case Value::ConstantIntVal:
            if (const ConstantInt *constant_int = dyn_cast<ConstantInt>(constant))
            {
                value = constant_int->getValue();
                return true;
            }
            break;
        case Value::ConstantFPVal:
            if (const ConstantFP *constant_fp = dyn_cast<ConstantFP>(constant))
            {
                value = constant_fp->getValueAPF().bitcastToAPInt();
                return true;
            }
            break;
        case Value::ConstantExprVal:
            if (const ConstantExpr *constant_expr = dyn_cast<ConstantExpr>(constant))
            {
                switch (constant_expr->getOpcode())
                {
                    default:
                        return false;
                    case Instruction::IntToPtr:
                    case Instruction::PtrToInt:
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
            break;
        case Value::ConstantPointerNullVal:
            if (isa<ConstantPointerNull>(constant))
            {
                value = APInt(m_target_data.getPointerSizeInBits(), 0);
                return true;
            }
            break;
        }
        return false;
    }
    
    bool MakeArgument(const Argument *value, uint64_t address)
    {
        lldb::addr_t data_address = Malloc(value->getType());
        
        if (data_address == LLDB_INVALID_ADDRESS)
            return false;
        
        lldb_private::Error write_error;
        
        m_memory_map.WritePointerToMemory(data_address, address, write_error);
        
        if (!write_error.Success())
        {
            lldb_private::Error free_error;
            m_memory_map.Free(data_address, free_error);
            return false;
        }
        
        m_values[value] = data_address;
        
        lldb_private::Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

        if (log)
        {
            log->Printf("Made an allocation for argument %s", PrintValue(value).c_str());
            log->Printf("  Data region    : %llx", (unsigned long long)address);
            log->Printf("  Ref region     : %llx", (unsigned long long)data_address);
        }
        
        return true;
    }
    
    bool ResolveConstant (lldb::addr_t process_address, const Constant *constant)
    {
        APInt resolved_value;
        
        if (!ResolveConstantValue(resolved_value, constant))
            return false;
        
        const uint64_t *raw_data = resolved_value.getRawData();
            
        size_t constant_size = m_target_data.getTypeStoreSize(constant->getType());
        
        lldb_private::Error write_error;
        
        m_memory_map.WriteMemory(process_address, (uint8_t*)raw_data, constant_size, write_error);
        
        return write_error.Success();
    }
    
    lldb::addr_t Malloc (size_t size, uint8_t byte_alignment)
    {
        lldb::addr_t ret = m_stack_pointer;
        
        ret -= size;
        ret -= (ret % byte_alignment);
        
        if (ret < m_frame_process_address)
            return LLDB_INVALID_ADDRESS;
        
        m_stack_pointer = ret;
        return ret;
    }
        
    lldb::addr_t MallocPointer ()
    {
        return Malloc(m_target_data.getPointerSize(), m_target_data.getPointerPrefAlignment());
    }
    
    lldb::addr_t Malloc (llvm::Type *type)
    {
        lldb_private::Error alloc_error;
        
        return Malloc(m_target_data.getTypeAllocSize(type), m_target_data.getPrefTypeAlignment(type));
    }
    
    std::string PrintData (lldb::addr_t addr, llvm::Type *type)
    {
        size_t length = m_target_data.getTypeStoreSize(type);
        
        lldb_private::DataBufferHeap buf(length, 0);
        
        lldb_private::Error read_error;
        
        m_memory_map.ReadMemory(buf.GetBytes(), addr, length, read_error);
        
        if (!read_error.Success())
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
    
    lldb::addr_t ResolveValue (const Value *value, Module &module)
    {
        ValueMap::iterator i = m_values.find(value);
        
        if (i != m_values.end())
            return i->second;
        
        // Fall back and allocate space [allocation type Alloca]
        
        lldb::addr_t data_address = Malloc(value->getType());
        
        if (const Constant *constant = dyn_cast<Constant>(value))
        {
            if (!ResolveConstant (data_address, constant))
            {
                lldb_private::Error free_error;
                m_memory_map.Free(data_address, free_error);
                return LLDB_INVALID_ADDRESS;
            }
        }
        
        m_values[value] = data_address;
        return data_address;
    }
};

static const char *unsupported_opcode_error         = "Interpreter doesn't handle one of the expression's opcodes";
static const char *unsupported_operand_error        = "Interpreter doesn't handle one of the expression's operands";
//static const char *interpreter_initialization_error = "Interpreter couldn't be initialized";
static const char *interpreter_internal_error       = "Interpreter encountered an internal error";
static const char *bad_value_error                  = "Interpreter couldn't resolve a value during execution";
static const char *memory_allocation_error          = "Interpreter couldn't allocate memory";
static const char *memory_write_error               = "Interpreter couldn't write to memory";
static const char *memory_read_error                = "Interpreter couldn't read from memory";
static const char *infinite_loop_error              = "Interpreter ran for too many cycles";
//static const char *bad_result_error                 = "Result of expression is in bad memory";

bool
IRInterpreter::CanInterpret (llvm::Module &module,
                             llvm::Function &function,
                             lldb_private::Error &error)
{
    lldb_private::Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    for (Function::iterator bbi = function.begin(), bbe = function.end();
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
                    error.SetErrorToGenericError();
                    error.SetErrorString(unsupported_opcode_error);
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
                        error.SetErrorToGenericError();
                        error.SetErrorString(interpreter_internal_error);
                        return false;
                    }
                    
                    switch (icmp_inst->getPredicate())
                    {
                    default:
                    {
                        if (log)
                            log->Printf("Unsupported ICmp predicate: %s", PrintValue(ii).c_str());
                        
                        error.SetErrorToGenericError();
                        error.SetErrorString(unsupported_opcode_error);
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
            case Instruction::And:
            case Instruction::AShr:
            case Instruction::IntToPtr:
            case Instruction::PtrToInt:
            case Instruction::Load:
            case Instruction::LShr:
            case Instruction::Mul:
            case Instruction::Or:
            case Instruction::Ret:
            case Instruction::SDiv:
            case Instruction::Shl:
            case Instruction::SRem:
            case Instruction::Store:
            case Instruction::Sub:
            case Instruction::UDiv:
            case Instruction::URem:
            case Instruction::Xor:
            case Instruction::ZExt:
                break;
            }
            
            for (int oi = 0, oe = ii->getNumOperands();
                 oi != oe;
                 ++oi)
            {
                Value *operand = ii->getOperand(oi);
                Type *operand_type = operand->getType();
                
                switch (operand_type->getTypeID())
                {
                default:
                    break;
                case Type::VectorTyID:
                    {
                        if (log)
                            log->Printf("Unsupported operand type: %s", PrintType(operand_type).c_str());
                        error.SetErrorString(unsupported_operand_error);
                        return false;
                    }
                }
            }
        }
        
    }
    
    return true;}

bool
IRInterpreter::Interpret (llvm::Module &module,
                          llvm::Function &function,
                          llvm::ArrayRef<lldb::addr_t> args,
                          lldb_private::IRMemoryMap &memory_map,
                          lldb_private::Error &error,
                          lldb::addr_t stack_frame_bottom,
                          lldb::addr_t stack_frame_top)
{
    lldb_private::Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    if (log)
    {
        std::string s;
        raw_string_ostream oss(s);
        
        module.print(oss, NULL);
        
        oss.flush();
        
        log->Printf("Module as passed in to IRInterpreter::Interpret: \n\"%s\"", s.c_str());
    }
    
    DataLayout data_layout(&module);
    
    InterpreterStackFrame frame(data_layout, memory_map, stack_frame_bottom, stack_frame_top);
    
    if (frame.m_frame_process_address == LLDB_INVALID_ADDRESS)
    {
        error.SetErrorString("Couldn't allocate stack frame");
    }
    
    int arg_index = 0;
    
    for (llvm::Function::arg_iterator ai = function.arg_begin(), ae = function.arg_end();
         ai != ae;
         ++ai, ++arg_index)
    {
        if (args.size() < arg_index)
        {
            error.SetErrorString ("Not enough arguments passed in to function");
            return false;
        }
        
        lldb::addr_t ptr = args[arg_index];

        frame.MakeArgument(ai, ptr);
    }
    
    uint32_t num_insts = 0;
    
    frame.Jump(function.begin());
    
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
            case Instruction::SRem:
            case Instruction::URem:
            case Instruction::Shl:
            case Instruction::LShr:
            case Instruction::AShr:
            case Instruction::And:
            case Instruction::Or:
            case Instruction::Xor:
            {
                const BinaryOperator *bin_op = dyn_cast<BinaryOperator>(inst);
                
                if (!bin_op)
                {
                    if (log)
                        log->Printf("getOpcode() returns %s, but instruction is not a BinaryOperator", inst->getOpcodeName());
                    error.SetErrorToGenericError();
                    error.SetErrorString(interpreter_internal_error);
                    return false;
                }
                
                Value *lhs = inst->getOperand(0);
                Value *rhs = inst->getOperand(1);
                
                lldb_private::Scalar L;
                lldb_private::Scalar R;
                
                if (!frame.EvaluateValue(L, lhs, module))
                {
                    if (log)
                        log->Printf("Couldn't evaluate %s", PrintValue(lhs).c_str());
                    error.SetErrorToGenericError();
                    error.SetErrorString(bad_value_error);
                    return false;
                }
                
                if (!frame.EvaluateValue(R, rhs, module))
                {
                    if (log)
                        log->Printf("Couldn't evaluate %s", PrintValue(rhs).c_str());
                    error.SetErrorToGenericError();
                    error.SetErrorString(bad_value_error);
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
                    case Instruction::SRem:
                        result = L % R;
                        break;
                    case Instruction::URem:
                        result = L.GetRawBits64(0) % R.GetRawBits64(1);
                        break;
                    case Instruction::Shl:
                        result = L << R;
                        break;
                    case Instruction::AShr:
                        result = L >> R;
                        break;
                    case Instruction::LShr:
                        result = L;
                        result.ShiftRightLogical(R);
                        break;
                    case Instruction::And:
                        result = L & R;
                        break;
                    case Instruction::Or:
                        result = L | R;
                        break;
                    case Instruction::Xor:
                        result = L ^ R;
                        break;
                }
                
                frame.AssignValue(inst, result, module);
                
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
                    error.SetErrorToGenericError();
                    error.SetErrorString(interpreter_internal_error);
                    return false;
                }
                
                if (alloca_inst->isArrayAllocation())
                {
                    if (log)
                        log->Printf("AllocaInsts are not handled if isArrayAllocation() is true");
                    error.SetErrorToGenericError();
                    error.SetErrorString(unsupported_opcode_error);
                    return false;
                }
                
                // The semantics of Alloca are:
                //   Create a region R of virtual memory of type T, backed by a data buffer
                //   Create a region P of virtual memory of type T*, backed by a data buffer
                //   Write the virtual address of R into P
                
                Type *T = alloca_inst->getAllocatedType();
                Type *Tptr = alloca_inst->getType();
                
                lldb::addr_t R = frame.Malloc(T);
                
                if (R == LLDB_INVALID_ADDRESS)
                {
                    if (log)
                        log->Printf("Couldn't allocate memory for an AllocaInst");
                    error.SetErrorToGenericError();
                    error.SetErrorString(memory_allocation_error);
                    return false;
                }
                
                lldb::addr_t P = frame.Malloc(Tptr);
                
                if (P == LLDB_INVALID_ADDRESS)
                {
                    if (log)
                        log->Printf("Couldn't allocate the result pointer for an AllocaInst");
                    error.SetErrorToGenericError();
                    error.SetErrorString(memory_allocation_error);
                    return false;
                }
                
                lldb_private::Error write_error;
                
                memory_map.WritePointerToMemory(P, R, write_error);
                
                if (!write_error.Success())
                {
                    if (log)
                        log->Printf("Couldn't write the result pointer for an AllocaInst");
                    error.SetErrorToGenericError();
                    error.SetErrorString(memory_write_error);
                    lldb_private::Error free_error;
                    memory_map.Free(P, free_error);
                    memory_map.Free(R, free_error);
                    return false;
                }
                
                frame.m_values[alloca_inst] = P;
                
                if (log)
                {
                    log->Printf("Interpreted an AllocaInst");
                    log->Printf("  R : 0x%llx", R);
                    log->Printf("  P : 0x%llx", P);
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
                    error.SetErrorToGenericError();
                    error.SetErrorString(interpreter_internal_error);
                    return false;
                }
                
                Value *source = cast_inst->getOperand(0);
                
                lldb_private::Scalar S;
                
                if (!frame.EvaluateValue(S, source, module))
                {
                    if (log)
                        log->Printf("Couldn't evaluate %s", PrintValue(source).c_str());
                    error.SetErrorToGenericError();
                    error.SetErrorString(bad_value_error);
                    return false;
                }
                
                frame.AssignValue(inst, S, module);
            }
                break;
            case Instruction::Br:
            {
                const BranchInst *br_inst = dyn_cast<BranchInst>(inst);
                
                if (!br_inst)
                {
                    if (log)
                        log->Printf("getOpcode() returns Br, but instruction is not a BranchInst");
                    error.SetErrorToGenericError();
                    error.SetErrorString(interpreter_internal_error);
                    return false;
                }
                
                if (br_inst->isConditional())
                {
                    Value *condition = br_inst->getCondition();
                    
                    lldb_private::Scalar C;
                    
                    if (!frame.EvaluateValue(C, condition, module))
                    {
                        if (log)
                            log->Printf("Couldn't evaluate %s", PrintValue(condition).c_str());
                        error.SetErrorToGenericError();
                        error.SetErrorString(bad_value_error);
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
                    error.SetErrorToGenericError();
                    error.SetErrorString(interpreter_internal_error);
                    return false;
                }
                
                const Value *pointer_operand = gep_inst->getPointerOperand();
                Type *pointer_type = pointer_operand->getType();
                
                lldb_private::Scalar P;
                
                if (!frame.EvaluateValue(P, pointer_operand, module))
                {
                    if (log)
                        log->Printf("Couldn't evaluate %s", PrintValue(pointer_operand).c_str());
                    error.SetErrorToGenericError();
                    error.SetErrorString(bad_value_error);
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
                        
                        if (!frame.EvaluateValue(I, *ii, module))
                        {
                            if (log)
                                log->Printf("Couldn't evaluate %s", PrintValue(*ii).c_str());
                            error.SetErrorToGenericError();
                            error.SetErrorString(bad_value_error);
                            return false;
                        }
                        
                        if (log)
                            log->Printf("Evaluated constant index %s as %llu", PrintValue(*ii).c_str(), I.ULongLong(LLDB_INVALID_ADDRESS));
                        
                        constant_index = cast<ConstantInt>(ConstantInt::get((*ii)->getType(), I.ULongLong(LLDB_INVALID_ADDRESS)));
                    }
                    
                    const_indices.push_back(constant_index);
                }
                
                uint64_t offset = data_layout.getIndexedOffset(pointer_type, const_indices);
                
                lldb_private::Scalar Poffset = P + offset;
                
                frame.AssignValue(inst, Poffset, module);
                
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
                    error.SetErrorToGenericError();
                    error.SetErrorString(interpreter_internal_error);
                    return false;
                }
                
                CmpInst::Predicate predicate = icmp_inst->getPredicate();
                
                Value *lhs = inst->getOperand(0);
                Value *rhs = inst->getOperand(1);
                
                lldb_private::Scalar L;
                lldb_private::Scalar R;
                
                if (!frame.EvaluateValue(L, lhs, module))
                {
                    if (log)
                        log->Printf("Couldn't evaluate %s", PrintValue(lhs).c_str());
                    error.SetErrorToGenericError();
                    error.SetErrorString(bad_value_error);
                    return false;
                }
                
                if (!frame.EvaluateValue(R, rhs, module))
                {
                    if (log)
                        log->Printf("Couldn't evaluate %s", PrintValue(rhs).c_str());
                    error.SetErrorToGenericError();
                    error.SetErrorString(bad_value_error);
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
                
                frame.AssignValue(inst, result, module);
                
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
                    error.SetErrorToGenericError();
                    error.SetErrorString(interpreter_internal_error);
                    return false;
                }
                
                Value *src_operand = int_to_ptr_inst->getOperand(0);
                
                lldb_private::Scalar I;
                
                if (!frame.EvaluateValue(I, src_operand, module))
                {
                    if (log)
                        log->Printf("Couldn't evaluate %s", PrintValue(src_operand).c_str());
                    error.SetErrorToGenericError();
                    error.SetErrorString(bad_value_error);
                    return false;
                }
                
                frame.AssignValue(inst, I, module);
                
                if (log)
                {
                    log->Printf("Interpreted an IntToPtr");
                    log->Printf("  Src : %s", frame.SummarizeValue(src_operand).c_str());
                    log->Printf("  =   : %s", frame.SummarizeValue(inst).c_str());
                }
            }
                break;
            case Instruction::PtrToInt:
            {
                const PtrToIntInst *ptr_to_int_inst = dyn_cast<PtrToIntInst>(inst);
                
                if (!ptr_to_int_inst)
                {
                    if (log)
                        log->Printf("getOpcode() returns PtrToInt, but instruction is not an PtrToIntInst");
                    error.SetErrorToGenericError();
                    error.SetErrorString(interpreter_internal_error);
                    return false;
                }
                
                Value *src_operand = ptr_to_int_inst->getOperand(0);
                
                lldb_private::Scalar I;
                
                if (!frame.EvaluateValue(I, src_operand, module))
                {
                    if (log)
                        log->Printf("Couldn't evaluate %s", PrintValue(src_operand).c_str());
                    error.SetErrorToGenericError();
                    error.SetErrorString(bad_value_error);
                    return false;
                }
                
                frame.AssignValue(inst, I, module);
                
                if (log)
                {
                    log->Printf("Interpreted a PtrToInt");
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
                    error.SetErrorToGenericError();
                    error.SetErrorString(interpreter_internal_error);
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
                    error.SetErrorToGenericError();
                    error.SetErrorString(interpreter_internal_error);
                    return false;
                }
                Type *target_ty = pointer_ptr_ty->getElementType();
                
                lldb::addr_t D = frame.ResolveValue(load_inst, module);
                lldb::addr_t P = frame.ResolveValue(pointer_operand, module);
                
                if (D == LLDB_INVALID_ADDRESS)
                {
                    if (log)
                        log->Printf("LoadInst's value doesn't resolve to anything");
                    error.SetErrorToGenericError();
                    error.SetErrorString(bad_value_error);
                    return false;
                }
                
                if (P == LLDB_INVALID_ADDRESS)
                {
                    if (log)
                        log->Printf("LoadInst's pointer doesn't resolve to anything");
                    error.SetErrorToGenericError();
                    error.SetErrorString(bad_value_error);
                    return false;
                }
                
                lldb::addr_t R;
                lldb_private::Error read_error;
                memory_map.ReadPointerFromMemory(&R, P, read_error);
                
                if (!read_error.Success())
                {
                    if (log)
                        log->Printf("Couldn't read the address to be loaded for a LoadInst");
                    error.SetErrorToGenericError();
                    error.SetErrorString(memory_read_error);
                    return false;
                }
                
                size_t target_size = data_layout.getTypeStoreSize(target_ty);
                lldb_private::DataBufferHeap buffer(target_size, 0);
                
                read_error.Clear();
                memory_map.ReadMemory(buffer.GetBytes(), R, buffer.GetByteSize(), read_error);
                if (!read_error.Success())
                {
                    if (log)
                        log->Printf("Couldn't read from a region on behalf of a LoadInst");
                    error.SetErrorToGenericError();
                    error.SetErrorString(memory_read_error);
                    return false;
                }
                
                lldb_private::Error write_error;
                memory_map.WriteMemory(D, buffer.GetBytes(), buffer.GetByteSize(), write_error);
                if (!write_error.Success())
                {
                    if (log)
                        log->Printf("Couldn't write to a region on behalf of a LoadInst");
                    error.SetErrorToGenericError();
                    error.SetErrorString(memory_read_error);
                    return false;
                }
                
                if (log)
                {
                    log->Printf("Interpreted a LoadInst");
                    log->Printf("  P : 0x%llx", P);
                    log->Printf("  R : 0x%llx", R);
                    log->Printf("  D : 0x%llx", D);
                }
            }
                break;
            case Instruction::Ret:
            {
                return true;
            }
            case Instruction::Store:
            {
                const StoreInst *store_inst = dyn_cast<StoreInst>(inst);
                
                if (!store_inst)
                {
                    if (log)
                        log->Printf("getOpcode() returns Store, but instruction is not a StoreInst");
                    error.SetErrorToGenericError();
                    error.SetErrorString(interpreter_internal_error);
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
                
                lldb::addr_t D = frame.ResolveValue(value_operand, module);
                lldb::addr_t P = frame.ResolveValue(pointer_operand, module);
                
                if (D == LLDB_INVALID_ADDRESS)
                {
                    if (log)
                        log->Printf("StoreInst's value doesn't resolve to anything");
                    error.SetErrorToGenericError();
                    error.SetErrorString(bad_value_error);
                    return false;
                }
                
                if (P == LLDB_INVALID_ADDRESS)
                {
                    if (log)
                        log->Printf("StoreInst's pointer doesn't resolve to anything");
                    error.SetErrorToGenericError();
                    error.SetErrorString(bad_value_error);
                    return false;
                }
                
                lldb::addr_t R;
                lldb_private::Error read_error;
                memory_map.ReadPointerFromMemory(&R, P, read_error);
                
                if (!read_error.Success())
                {
                    if (log)
                        log->Printf("Couldn't read the address to be loaded for a LoadInst");
                    error.SetErrorToGenericError();
                    error.SetErrorString(memory_read_error);
                    return false;
                }
                
                size_t target_size = data_layout.getTypeStoreSize(target_ty);
                lldb_private::DataBufferHeap buffer(target_size, 0);
                
                read_error.Clear();
                memory_map.ReadMemory(buffer.GetBytes(), D, buffer.GetByteSize(), read_error);
                if (!read_error.Success())
                {
                    if (log)
                        log->Printf("Couldn't read from a region on behalf of a StoreInst");
                    error.SetErrorToGenericError();
                    error.SetErrorString(memory_read_error);
                    return false;
                }
                
                lldb_private::Error write_error;
                memory_map.WriteMemory(R, buffer.GetBytes(), buffer.GetByteSize(), write_error);
                if (!write_error.Success())
                {
                    if (log)
                        log->Printf("Couldn't write to a region on behalf of a StoreInst");
                    error.SetErrorToGenericError();
                    error.SetErrorString(memory_write_error);
                    return false;
                }
                
                if (log)
                {
                    log->Printf("Interpreted a StoreInst");
                    log->Printf("  D : 0x%llx", D);
                    log->Printf("  P : 0x%llx", P);
                    log->Printf("  R : 0x%llx", R);
                }
            }
                break;
        }
        
        ++frame.m_ii;
    }
    
    if (num_insts >= 4096)
    {
        error.SetErrorToGenericError();
        error.SetErrorString(infinite_loop_error);
        return false;
    }
    
    return false;
}
