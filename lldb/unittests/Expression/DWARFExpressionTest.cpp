//===-- DWARFExpressionTest.cpp ----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;

static llvm::Expected<Scalar> Evaluate(llvm::ArrayRef<uint8_t> expr) {
  DataExtractor extractor(expr.data(), expr.size(), lldb::eByteOrderLittle,
                          /*addr_size*/ 4);

  Value result;
  Status status;
  if (!DWARFExpression::Evaluate(
          /*exe_ctx*/ nullptr, /*reg_ctx*/ nullptr, /*opcode_ctx*/ nullptr,
          extractor, /*dwarf_cu*/ nullptr, /*offset*/ 0, expr.size(),
          lldb::eRegisterKindLLDB, /*initial_value_ptr*/ nullptr,
          /*object_address_ptr*/ nullptr, result, &status))
    return status.ToError();

  return result.GetScalar();
}

TEST(DWARFExpression, DW_OP_pick) {
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_lit1, DW_OP_lit0, DW_OP_pick, 0}),
                       llvm::HasValue(0));
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_lit1, DW_OP_lit0, DW_OP_pick, 1}),
                       llvm::HasValue(1));
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_lit1, DW_OP_lit0, DW_OP_pick, 2}),
                       llvm::Failed());
}
