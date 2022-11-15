//===-- Coroutines.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Coroutines.h"

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

static ValueObjectSP GetCoroFramePtrFromHandle(ValueObject &valobj) {
  ValueObjectSP valobj_sp(valobj.GetNonSyntheticValue());
  if (!valobj_sp)
    return nullptr;

  // We expect a single pointer in the `coroutine_handle` class.
  // We don't care about its name.
  if (valobj_sp->GetNumChildren() != 1)
    return nullptr;
  ValueObjectSP ptr_sp(valobj_sp->GetChildAtIndex(0, true));
  if (!ptr_sp)
    return nullptr;
  if (!ptr_sp->GetCompilerType().IsPointerType())
    return nullptr;

  return ptr_sp;
}

static CompilerType GetCoroutineFrameType(TypeSystemClang &ast_ctx,
                                          CompilerType promise_type) {
  CompilerType void_type = ast_ctx.GetBasicType(lldb::eBasicTypeVoid);
  CompilerType coro_func_type = ast_ctx.CreateFunctionType(
      /*result_type=*/void_type, /*args=*/&void_type, /*num_args=*/1,
      /*is_variadic=*/false, /*qualifiers=*/0);
  CompilerType coro_abi_type;
  if (promise_type.IsVoidType()) {
    coro_abi_type = ast_ctx.CreateStructForIdentifier(
        ConstString(), {{"resume", coro_func_type.GetPointerType()},
                        {"destroy", coro_func_type.GetPointerType()}});
  } else {
    coro_abi_type = ast_ctx.CreateStructForIdentifier(
        ConstString(), {{"resume", coro_func_type.GetPointerType()},
                        {"destroy", coro_func_type.GetPointerType()},
                        {"promise", promise_type}});
  }
  return coro_abi_type;
}

bool lldb_private::formatters::StdlibCoroutineHandleSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  ValueObjectSP ptr_sp(GetCoroFramePtrFromHandle(valobj));
  if (!ptr_sp)
    return false;

  stream.Printf("coro frame = 0x%" PRIx64, ptr_sp->GetValueAsUnsigned(0));
  return true;
}

lldb_private::formatters::StdlibCoroutineHandleSyntheticFrontEnd::
    StdlibCoroutineHandleSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp) {
  if (valobj_sp)
    Update();
}

lldb_private::formatters::StdlibCoroutineHandleSyntheticFrontEnd::
    ~StdlibCoroutineHandleSyntheticFrontEnd() = default;

size_t lldb_private::formatters::StdlibCoroutineHandleSyntheticFrontEnd::
    CalculateNumChildren() {
  if (!m_frame_ptr_sp)
    return 0;

  return m_frame_ptr_sp->GetNumChildren();
}

lldb::ValueObjectSP lldb_private::formatters::
    StdlibCoroutineHandleSyntheticFrontEnd::GetChildAtIndex(size_t idx) {
  if (!m_frame_ptr_sp)
    return lldb::ValueObjectSP();

  return m_frame_ptr_sp->GetChildAtIndex(idx, true);
}

bool lldb_private::formatters::StdlibCoroutineHandleSyntheticFrontEnd::
    Update() {
  m_frame_ptr_sp.reset();

  ValueObjectSP valobj_sp = m_backend.GetSP();
  if (!valobj_sp)
    return false;

  ValueObjectSP ptr_sp(GetCoroFramePtrFromHandle(m_backend));
  if (!ptr_sp)
    return false;

  auto ts = valobj_sp->GetCompilerType().GetTypeSystem();
  auto ast_ctx = ts.dyn_cast_or_null<TypeSystemClang>();
  if (!ast_ctx)
    return false;

  CompilerType promise_type(
      valobj_sp->GetCompilerType().GetTypeTemplateArgument(0));
  if (!promise_type)
    return false;
  CompilerType coro_frame_type = GetCoroutineFrameType(*ast_ctx, promise_type);

  m_frame_ptr_sp = ptr_sp->Cast(coro_frame_type.GetPointerType());

  return false;
}

bool lldb_private::formatters::StdlibCoroutineHandleSyntheticFrontEnd::
    MightHaveChildren() {
  return true;
}

size_t StdlibCoroutineHandleSyntheticFrontEnd::GetIndexOfChildWithName(
    ConstString name) {
  if (!m_frame_ptr_sp)
    return UINT32_MAX;

  return m_frame_ptr_sp->GetIndexOfChildWithName(name);
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::StdlibCoroutineHandleSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  return (valobj_sp ? new StdlibCoroutineHandleSyntheticFrontEnd(valobj_sp)
                    : nullptr);
}
