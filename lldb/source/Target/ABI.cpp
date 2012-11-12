//===-- ABI.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ABI.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

ABISP
ABI::FindPlugin (const ArchSpec &arch)
{
    ABISP abi_sp;
    ABICreateInstance create_callback;

    for (uint32_t idx = 0;
         (create_callback = PluginManager::GetABICreateCallbackAtIndex(idx)) != NULL;
         ++idx)
    {
        abi_sp = create_callback(arch);

        if (abi_sp)
            return abi_sp;
    }
    abi_sp.reset();
    return abi_sp;
}

//----------------------------------------------------------------------
// Constructor
//----------------------------------------------------------------------
ABI::ABI()
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ABI::~ABI()
{
}


bool
ABI::GetRegisterInfoByName (const ConstString &name, RegisterInfo &info)
{
    uint32_t count = 0;
    const RegisterInfo *register_info_array = GetRegisterInfoArray (count);
    if (register_info_array)
    {
        const char *unique_name_cstr = name.GetCString();
        uint32_t i;
        for (i=0; i<count; ++i)
        {
            if (register_info_array[i].name == unique_name_cstr)
            {
                info = register_info_array[i];
                return true;
            }
        }
        for (i=0; i<count; ++i)
        {
            if (register_info_array[i].alt_name == unique_name_cstr)
            {
                info = register_info_array[i];
                return true;
            }
        }
    }
    return false;
}

bool
ABI::GetRegisterInfoByKind (RegisterKind reg_kind, uint32_t reg_num, RegisterInfo &info)
{
    if (reg_kind < eRegisterKindGCC || reg_kind >= kNumRegisterKinds)
        return false;
        
    uint32_t count = 0;
    const RegisterInfo *register_info_array = GetRegisterInfoArray (count);
    if (register_info_array)
    {
        for (uint32_t i=0; i<count; ++i)
        {
            if (register_info_array[i].kinds[reg_kind] == reg_num)
            {
                info = register_info_array[i];
                return true;
            }
        }
    }
    return false;
}

ValueObjectSP
ABI::GetReturnValueObject (Thread &thread,
                           ClangASTType &ast_type,
                           bool persistent) const
{
    if (!ast_type.IsValid())
        return ValueObjectSP();
        
    ValueObjectSP return_valobj_sp;
        
    return_valobj_sp = GetReturnValueObjectImpl(thread, ast_type);
    if (!return_valobj_sp)
        return return_valobj_sp;
    
    // Now turn this into a persistent variable.
    // FIXME: This code is duplicated from Target::EvaluateExpression, and it is used in similar form in a couple
    // of other places.  Figure out the correct Create function to do all this work.
    
    if (persistent)
    {
        ClangPersistentVariables& persistent_variables = thread.CalculateTarget()->GetPersistentVariables();
        ConstString persistent_variable_name (persistent_variables.GetNextPersistentVariableName());

        lldb::ValueObjectSP const_valobj_sp;
        
        // Check in case our value is already a constant value
        if (return_valobj_sp->GetIsConstant())
        {
            const_valobj_sp = return_valobj_sp;
            const_valobj_sp->SetName (persistent_variable_name);
        }
        else
            const_valobj_sp = return_valobj_sp->CreateConstantValue (persistent_variable_name);

        lldb::ValueObjectSP live_valobj_sp = return_valobj_sp;
        
        return_valobj_sp = const_valobj_sp;

        ClangExpressionVariableSP clang_expr_variable_sp(persistent_variables.CreatePersistentVariable(return_valobj_sp));
               
        assert (clang_expr_variable_sp.get());
        
        // Set flags and live data as appropriate

        const Value &result_value = live_valobj_sp->GetValue();
        
        switch (result_value.GetValueType())
        {
        case Value::eValueTypeHostAddress:
        case Value::eValueTypeFileAddress:
            // we don't do anything with these for now
            break;
        case Value::eValueTypeScalar:
        case Value::eValueTypeVector:
            clang_expr_variable_sp->m_flags |= ClangExpressionVariable::EVIsFreezeDried;
            clang_expr_variable_sp->m_flags |= ClangExpressionVariable::EVIsLLDBAllocated;
            clang_expr_variable_sp->m_flags |= ClangExpressionVariable::EVNeedsAllocation;
            break;
        case Value::eValueTypeLoadAddress:
            clang_expr_variable_sp->m_live_sp = live_valobj_sp;
            clang_expr_variable_sp->m_flags |= ClangExpressionVariable::EVIsProgramReference;
            break;
        }
        
        return_valobj_sp = clang_expr_variable_sp->GetValueObject();
    }
    return return_valobj_sp;
}


