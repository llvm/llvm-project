//===-- ClangExpressionVariable.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/ClangExpressionVariable.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "clang/AST/ASTContext.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"

using namespace lldb_private;
using namespace clang;

ClangExpressionVariable::ClangExpressionVariable(ExecutionContextScope *exe_scope, lldb::ByteOrder byte_order, uint32_t addr_byte_size) :
    m_parser_vars(),
    m_jit_vars (),
    m_flags (EVNone),
    m_frozen_sp (ValueObjectConstResult::Create (exe_scope, byte_order, addr_byte_size))
{
}

ClangExpressionVariable::ClangExpressionVariable (const lldb::ValueObjectSP &valobj_sp) :
    m_parser_vars(),
    m_jit_vars (),
    m_flags (EVNone),
    m_frozen_sp (valobj_sp)
{
}

//----------------------------------------------------------------------
/// Return the variable's size in bytes
//----------------------------------------------------------------------
size_t 
ClangExpressionVariable::GetByteSize ()
{
    return m_frozen_sp->GetByteSize();
}    

const ConstString &
ClangExpressionVariable::GetName ()
{
    return m_frozen_sp->GetName();
}    

lldb::ValueObjectSP
ClangExpressionVariable::GetValueObject()
{
    return m_frozen_sp;
}

RegisterInfo *
ClangExpressionVariable::GetRegisterInfo()
{
    return m_frozen_sp->GetValue().GetRegisterInfo();
}

void
ClangExpressionVariable::SetRegisterInfo (const RegisterInfo *reg_info)
{
    return m_frozen_sp->GetValue().SetContext (Value::eContextTypeRegisterInfo, const_cast<RegisterInfo *>(reg_info));
}

lldb::clang_type_t
ClangExpressionVariable::GetClangType()
{
    return m_frozen_sp->GetClangType();
}    

void
ClangExpressionVariable::SetClangType(lldb::clang_type_t clang_type)
{
    m_frozen_sp->GetValue().SetContext(Value::eContextTypeClangType, clang_type);
}    

clang::ASTContext *
ClangExpressionVariable::GetClangAST()
{
    return m_frozen_sp->GetClangAST();
}    

void
ClangExpressionVariable::SetClangAST (clang::ASTContext *ast)
{
    m_frozen_sp->SetClangAST (ast);
}

TypeFromUser
ClangExpressionVariable::GetTypeFromUser()
{
    TypeFromUser tfu (m_frozen_sp->GetClangType(), m_frozen_sp->GetClangAST());
    return tfu;
}    

uint8_t *
ClangExpressionVariable::GetValueBytes()
{
    const size_t byte_size = m_frozen_sp->GetByteSize();
    if (byte_size > 0)
    {
        if (m_frozen_sp->GetDataExtractor().GetByteSize() < byte_size)
        {
            m_frozen_sp->GetValue().ResizeData(byte_size);
            m_frozen_sp->GetValue().GetData (m_frozen_sp->GetDataExtractor());
        }
        return const_cast<uint8_t *>(m_frozen_sp->GetDataExtractor().GetDataStart());
    }
    return NULL;
}

void
ClangExpressionVariable::SetName (const ConstString &name)
{
    m_frozen_sp->SetName (name);
}

void
ClangExpressionVariable::ValueUpdated ()
{
    m_frozen_sp->ValueUpdated ();
}

void
ClangExpressionVariable::TransferAddress (bool force)
{
    if (m_live_sp.get() == NULL)
        return;

    if (m_frozen_sp.get() == NULL)
        return;
    
    if (force || (m_frozen_sp->GetLiveAddress() == LLDB_INVALID_ADDRESS))
        m_frozen_sp->SetLiveAddress(m_live_sp->GetLiveAddress());
}
