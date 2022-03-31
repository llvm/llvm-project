//===- CodeComplete.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/PDLL/Parser/CodeComplete.h"
#include "mlir/Tools/PDLL/AST/Types.h"

using namespace mlir;
using namespace mlir::pdll;

//===----------------------------------------------------------------------===//
// CodeCompleteContext
//===----------------------------------------------------------------------===//

CodeCompleteContext::~CodeCompleteContext() = default;

void CodeCompleteContext::codeCompleteTupleMemberAccess(
    ast::TupleType tupleType) {}
void CodeCompleteContext::codeCompleteOperationMemberAccess(
    ast::OperationType opType) {}

void CodeCompleteContext::codeCompleteConstraintName(
    ast::Type currentType, bool allowNonCoreConstraints,
    bool allowInlineTypeConstraints, const ast::DeclScope *scope) {}
