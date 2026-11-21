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
char FortranFunction::ID;

FortranType::~FortranType() = default;

static ConstString
CreateFuncSignature(ConstString func_name,
                    const llvm::SmallVectorImpl<CompilerType> &param_types,
                    const llvm::SmallVectorImpl<llvm::StringRef> &param_names,
                    CompilerType return_type) {
  std::string buf;
  llvm::raw_string_ostream ss(buf);

  if (return_type.IsValid())
    ss << return_type.GetTypeName().AsCString("") << " ";

  ss << func_name.AsCString("<unnamed function>") << "(";

  for (size_t i = 0; i < param_types.size(); i++) {
    if (i)
      ss << ", ";
    ss << param_types[i].GetTypeName().AsCString("") << " ";
    ss << param_names[i];
  }
  ss << ")";
  ss.flush();
  return ConstString(buf);
}

FortranFunction::FortranFunction(
    ConstString func_name,
    const llvm::SmallVectorImpl<CompilerType> &param_types,
    const llvm::SmallVectorImpl<llvm::StringRef> &param_names,
    CompilerType return_type)
    : FortranType(FortranType::KIND_FUNCTION, 0, func_name),
      m_param_types(param_types.begin(), param_types.end()),
      m_return_type(return_type),
      m_func_signature(CreateFuncSignature(func_name, param_types, param_names,
                                           return_type)) {}

FortranFunction *FortranFunction::GetFunctionType(void *type) {
  if (!type)
    return nullptr;

  FortranType *base_type = static_cast<FortranType *>(type);
  if (base_type->GetKind() != FortranType::KIND_FUNCTION)
    return nullptr;

  return llvm::cast<FortranFunction>(base_type);
}
