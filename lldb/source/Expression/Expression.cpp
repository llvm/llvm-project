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

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
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

llvm::Expected<llvm::SmallVector<llvm::StringRef, 3>>
lldb_private::splitFunctionCallLabel(llvm::StringRef label) {
  if (!label.consume_front(FunctionCallLabelPrefix))
    return llvm::createStringError(
        "expected function call label prefix not found in %s", label.data());
  if (!label.consume_front(":"))
    return llvm::createStringError(
        "incorrect format: expected ':' as the first character.");

  llvm::SmallVector<llvm::StringRef, 3> components;
  label.split(components, ":");

  if (components.size() != 3)
    return llvm::createStringError(
        "incorrect format: too many label subcomponents.");

  return components;
}

llvm::Expected<FunctionCallLabel>
lldb_private::FunctionCallLabel::fromString(llvm::StringRef label) {
  auto components_or_err = splitFunctionCallLabel(label);
  if (!components_or_err)
    return llvm::joinErrors(
        llvm::createStringError("Failed to decode function call label"),
        components_or_err.takeError());

  const auto &components = *components_or_err;

  llvm::StringRef module_label = components[1];
  llvm::StringRef die_label = components[2];

  lldb::user_id_t module_id = 0;
  if (module_label.consumeInteger(0, module_id))
    return llvm::createStringError(
        llvm::formatv("failed to parse module ID from '{0}'.", components[1]));

  lldb::user_id_t die_id;
  if (die_label.consumeInteger(/*Radix=*/0, die_id))
    return llvm::createStringError(
        llvm::formatv("failed to parse DIE ID from '{0}'.", components[2]));

  return FunctionCallLabel{/*.lookup_name=*/components[0],
                           /*.module_id=*/module_id, /*.symbol_id=*/die_id};
}

std::string lldb_private::FunctionCallLabel::toString() const {
  return llvm::formatv("{0}:{1}:{2:x}:{3:x}", FunctionCallLabelPrefix,
                       lookup_name, module_id, symbol_id)
      .str();
}
