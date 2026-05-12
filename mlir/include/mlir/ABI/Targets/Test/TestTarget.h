//===- TestTarget.h - Predictable test ABI target --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the test ABI target, a predictable, dialect-agnostic
// classifier used to exercise the MLIR ABIRewriteContext infrastructure
// without depending on any real ABI.  See TestTarget.cpp for the rules
// and the rationale.
//
// It also declares parseClassificationAttr, the helper used by the
// classification-injection driver: tests can attach an arbitrary
// FunctionClassification to a function via a plain mlir::DictionaryAttr,
// and the rewriter pass reads it back through this parser.  This lets
// tests verify rewriter output against any classification (including
// shapes the test target itself doesn't produce) without needing a real
// ABIInfo.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ABI_TARGETS_TEST_TESTTARGET_H
#define MLIR_ABI_TARGETS_TEST_TESTTARGET_H

#include "mlir/ABI/ABIRewriteContext.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/Support/Error.h"

namespace mlir {
namespace abi {
namespace test {

/// Classify a function signature using the test target's predictable rules.
///
/// The rules approximate x86_64 SysV thresholds for reviewer familiarity
/// (see TestTarget.cpp for the full list) but are not a substitute for
/// testing against a real ABIInfo.  Real-ABI-shaped tests should use the
/// classification-injection driver via `parseClassificationAttr` below.
///
/// \param argTypes   Argument types of the function.
/// \param returnType Return type of the function.
/// \param dl         DataLayout used for size and alignment queries.
FunctionClassification classify(ArrayRef<Type> argTypes, Type returnType,
                                const DataLayout &dl);

/// Parse a `FunctionClassification` from a plain MLIR DictionaryAttr.
///
/// Schema (all keys are required unless marked optional):
///
///   {
///     return = { kind = "<kind>", ...per-kind keys... },
///     args   = [ { kind = "<kind>", ...per-kind keys... }, ... ]
///   }
///
/// Per-arg/return dictionary keys:
///   kind: StringAttr.  One of "direct", "extend", "indirect",
///         "ignore", "expand".
///
/// For kind = "direct" (all optional):
///   coerced_type:  TypeAttr.  ABI-coerced type, if different from the
///                  original.
///   can_flatten:   BoolAttr.  Defaults to true.
///
/// For kind = "extend" (coerced_type required, sign_extend optional):
///   coerced_type:  TypeAttr.  Required; the extended integer type.
///   sign_extend:   BoolAttr.  Defaults to false (zero-extend).
///
/// For kind = "indirect" (indirect_align required, byval optional):
///   indirect_align: IntegerAttr.  Required; alignment of the pointed-to
///                   object in bytes.
///   byval:          BoolAttr.  Defaults to true.
///
/// For kind = "ignore" / "expand": no extra keys.
///
/// Future schema additions tracked in projects/daily_log.md (Step 0c
/// field-mapping table).  When we add new fields to ArgClassification
/// (e.g. direct_offset, extend_kind tristate, indirect_addr_space,
/// indirect_realign), the corresponding optional keys go here.
///
/// Unknown keys cause a parse error (no silent ignore — keeps schema
/// honest as it grows).
///
/// \param attr   The dictionary attribute to parse.
/// \param emitError  Diagnostic sink for parse errors.
/// \returns The parsed classification, or std::nullopt on error.
std::optional<FunctionClassification>
parseClassificationAttr(DictionaryAttr attr,
                        function_ref<InFlightDiagnostic()> emitError);

} // namespace test
} // namespace abi
} // namespace mlir

#endif // MLIR_ABI_TARGETS_TEST_TESTTARGET_H
