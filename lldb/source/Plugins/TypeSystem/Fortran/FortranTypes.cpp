//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FortranTypes.h"

using namespace lldb_private;
using namespace lldb_private::plugin::fortran;

char FortranType::ID;

FortranType::~FortranType() = default;

static ConstString CreateFortranFunctionSignature(
    ConstString func_name,
    const llvm::SmallVectorImpl<CompilerType> &parameters,
    const llvm::SmallVectorImpl<llvm::StringRef> &parameter_names,
    CompilerType return_type) {
  std::string name_buffer;
  llvm::raw_string_ostream name_stream(name_buffer);

  if (return_type.IsValid())
    name_stream << return_type.GetTypeName().AsCString("") << " ";

  name_stream << func_name.AsCString("<unnamed function>") << "(";

  for (size_t idx = 0; idx < parameters.size(); idx++) {
    if (idx != 0)
      name_stream << ", ";
    name_stream << parameters[idx].GetTypeName().AsCString("") << " ";
    name_stream << parameter_names[idx].str();
  }
  name_stream << ")";
  name_stream.flush();
  return ConstString(name_buffer);
}

FortranFunction::FortranFunction(
    ConstString func_name,
    const llvm::SmallVectorImpl<CompilerType> &parameters,
    const llvm::SmallVectorImpl<llvm::StringRef> &parameter_names,
    CompilerType return_type)
    : FortranType(FortranType::KIND_FUNCTION, 0, func_name),
      m_parameters(parameters.begin(), parameters.end()),
      m_return_type(return_type),
      m_function_signature(CreateFortranFunctionSignature(
          func_name, parameters, parameter_names, return_type)) {}
