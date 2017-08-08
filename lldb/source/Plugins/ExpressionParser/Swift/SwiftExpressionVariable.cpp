//===-- SwiftExpressionVariable.cpp -----------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "SwiftExpressionVariable.h"

#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Stream.h"
#include "clang/AST/ASTContext.h"

using namespace lldb_private;

SwiftExpressionVariable::SwiftExpressionVariable(
    ExecutionContextScope *exe_scope, lldb::ByteOrder byte_order,
    uint32_t addr_byte_size)
    : ExpressionVariable(LLVMCastKind::eKindSwift) {
  m_swift_flags = EVSNone;
  m_frozen_sp =
      ValueObjectConstResult::Create(exe_scope, byte_order, addr_byte_size);
}

SwiftExpressionVariable::SwiftExpressionVariable(
    const lldb::ValueObjectSP &valobj_sp)
    : ExpressionVariable(LLVMCastKind::eKindSwift) {
  m_swift_flags = EVSNone;
  m_frozen_sp = valobj_sp;
}

SwiftExpressionVariable::SwiftExpressionVariable(
    ExecutionContextScope *exe_scope, const ConstString &name,
    const TypeFromUser &type, lldb::ByteOrder byte_order,
    uint32_t addr_byte_size)
    : ExpressionVariable(LLVMCastKind::eKindSwift) {
  m_swift_flags = EVSNone;
  m_frozen_sp =
      ValueObjectConstResult::Create(exe_scope, byte_order, addr_byte_size);
  SetName(name);
  SetCompilerType(type);
}
