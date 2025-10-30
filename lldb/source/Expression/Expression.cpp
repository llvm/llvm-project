//===-- Expression.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/Expression.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/Target.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

using namespace lldb_private;

Expression::Expression(Target &target)
    : m_target_wp(target.shared_from_this()),
      m_jit_start_addr(LLDB_INVALID_ADDRESS),
      m_jit_end_addr(LLDB_INVALID_ADDRESS) {
  // Can't make any kind of expression without a target.
  assert(m_target_wp.lock());
}

Expression::Expression(ExecutionContextScope &exe_scope)
    : m_target_wp(exe_scope.CalculateTarget()),
      m_jit_start_addr(LLDB_INVALID_ADDRESS),
      m_jit_end_addr(LLDB_INVALID_ADDRESS) {
  assert(m_target_wp.lock());
}

llvm::Expected<FunctionCallLabel>
lldb_private::FunctionCallLabel::fromString(llvm::StringRef label) {
  llvm::SmallVector<llvm::StringRef, 5> components;
  label.split(components, ":", /*MaxSplit=*/4);

  if (components.size() != 5)
    return llvm::createStringError("malformed function call label.");

  if (components[0] != FunctionCallLabelPrefix)
    return llvm::createStringError(llvm::formatv(
        "expected function call label prefix '{0}' but found '{1}' instead.",
        FunctionCallLabelPrefix, components[0]));

  llvm::StringRef discriminator = components[1];
  llvm::StringRef module_label = components[2];
  llvm::StringRef die_label = components[3];
  llvm::StringRef lookup_name = components[4];

  lldb::user_id_t module_id = 0;
  if (!llvm::to_integer(module_label, module_id))
    return llvm::createStringError(
        llvm::formatv("failed to parse module ID from '{0}'.", module_label));

  lldb::user_id_t die_id;
  if (!llvm::to_integer(die_label, die_id))
    return llvm::createStringError(
        llvm::formatv("failed to parse symbol ID from '{0}'.", die_label));

  return FunctionCallLabel{/*.discriminator=*/discriminator,
                           /*.module_id=*/module_id,
                           /*.symbol_id=*/die_id,
                           /*.lookup_name=*/lookup_name};
}

std::string lldb_private::FunctionCallLabel::toString() const {
  return llvm::formatv("{0}:{1}:{2:x}:{3:x}:{4}", FunctionCallLabelPrefix,
                       discriminator, module_id, symbol_id, lookup_name)
      .str();
}

void llvm::format_provider<FunctionCallLabel>::format(
    const FunctionCallLabel &label, raw_ostream &OS, StringRef Style) {
  OS << llvm::formatv("FunctionCallLabel{ discriminator: {0}, module_id: "
                      "{1:x}, symbol_id: {2:x}, "
                      "lookup_name: {3} }",
                      label.discriminator, label.module_id, label.symbol_id,
                      label.lookup_name);
}
