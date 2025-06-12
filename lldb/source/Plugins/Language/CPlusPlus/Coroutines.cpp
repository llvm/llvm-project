//===-- Coroutines.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Coroutines.h"

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/VariableList.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

static lldb::addr_t GetCoroFramePtrFromHandle(ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return LLDB_INVALID_ADDRESS;

  // We expect a single pointer in the `coroutine_handle` class.
  // We don't care about its name.
  if (valobj_sp->GetNumChildrenIgnoringErrors() != 1)
    return LLDB_INVALID_ADDRESS;
  ValueObjectSP ptr_sp(valobj_sp->GetChildAtIndex(0));
  if (!ptr_sp)
    return LLDB_INVALID_ADDRESS;
  if (!ptr_sp->GetCompilerType().IsPointerType())
    return LLDB_INVALID_ADDRESS;

  auto [frame_ptr_addr, addr_type] = ptr_sp->GetPointerValue();
  if (!frame_ptr_addr || frame_ptr_addr == LLDB_INVALID_ADDRESS)
    return LLDB_INVALID_ADDRESS;
  lldbassert(addr_type == AddressType::eAddressTypeLoad);
  if (addr_type != AddressType::eAddressTypeLoad)
    return LLDB_INVALID_ADDRESS;

  return frame_ptr_addr;
}

static Function *ExtractDestroyFunction(lldb::TargetSP target_sp,
                                        lldb::addr_t frame_ptr_addr) {
  lldb::ProcessSP process_sp = target_sp->GetProcessSP();
  auto ptr_size = process_sp->GetAddressByteSize();

  Status error;
  auto destroy_func_ptr_addr = frame_ptr_addr + ptr_size;
  lldb::addr_t destroy_func_addr =
      process_sp->ReadPointerFromMemory(destroy_func_ptr_addr, error);
  if (error.Fail())
    return nullptr;

  Address destroy_func_address;
  if (!target_sp->ResolveLoadAddress(destroy_func_addr, destroy_func_address))
    return nullptr;

  return destroy_func_address.CalculateSymbolContextFunction();
}

// clang generates aritifical `__promise` and `__coro_frame` variables inside
// the destroy function. Look for those variables and extract their type.
static CompilerType InferArtificialCoroType(Function *destroy_func,
                                            ConstString var_name) {
  if (!destroy_func)
    return {};

  Block &block = destroy_func->GetBlock(true);
  auto variable_list = block.GetBlockVariableList(true);

  auto var = variable_list->FindVariable(var_name);
  if (!var)
    return {};
  if (!var->IsArtificial())
    return {};

  Type *promise_type = var->GetType();
  if (!promise_type)
    return {};
  return promise_type->GetForwardCompilerType();
}

bool lldb_private::formatters::StdlibCoroutineHandleSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  lldb::addr_t frame_ptr_addr =
      GetCoroFramePtrFromHandle(valobj.GetNonSyntheticValue());
  if (frame_ptr_addr == LLDB_INVALID_ADDRESS)
    return false;

  if (frame_ptr_addr == 0) {
    stream << "nullptr";
  } else {
    stream.Printf("coro frame = 0x%" PRIx64, frame_ptr_addr);
  }

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

llvm::Expected<uint32_t> lldb_private::formatters::
    StdlibCoroutineHandleSyntheticFrontEnd::CalculateNumChildren() {
  return m_children.size();
}

lldb::ValueObjectSP lldb_private::formatters::
    StdlibCoroutineHandleSyntheticFrontEnd::GetChildAtIndex(uint32_t idx) {
  return idx < m_children.size() ? m_children[idx] : lldb::ValueObjectSP();
}

lldb::ChildCacheState
lldb_private::formatters::StdlibCoroutineHandleSyntheticFrontEnd::Update() {
  m_children.clear();

  ValueObjectSP valobj_sp = m_backend.GetNonSyntheticValue();
  if (!valobj_sp)
    return lldb::ChildCacheState::eRefetch;

  lldb::addr_t frame_ptr_addr = GetCoroFramePtrFromHandle(valobj_sp);
  if (frame_ptr_addr == 0 || frame_ptr_addr == LLDB_INVALID_ADDRESS)
    return lldb::ChildCacheState::eRefetch;

  lldb::TargetSP target_sp = m_backend.GetTargetSP();
  auto &exe_ctx = m_backend.GetExecutionContextRef();
  lldb::ProcessSP process_sp = target_sp->GetProcessSP();
  auto ptr_size = process_sp->GetAddressByteSize();
  auto ast_ctx = valobj_sp->GetCompilerType().GetTypeSystem<TypeSystemClang>();
  if (!ast_ctx)
    return lldb::ChildCacheState::eRefetch;

  // Determine the coroutine frame type and the promise type. Fall back
  // to `void`, since even the pointer itself might be useful, even if the
  // type inference failed.
  Function *destroy_func = ExtractDestroyFunction(target_sp, frame_ptr_addr);
  CompilerType void_type = ast_ctx->GetBasicType(lldb::eBasicTypeVoid);
  CompilerType promise_type;
  if (CompilerType template_arg =
          valobj_sp->GetCompilerType().GetTypeTemplateArgument(0))
    promise_type = std::move(template_arg);
  if (promise_type.IsVoidType()) {
    // Try to infer the promise_type if it was type-erased
    if (destroy_func) {
      if (CompilerType inferred_type =
              InferArtificialCoroType(destroy_func, ConstString("__promise"))) {
        promise_type = inferred_type;
      }
    }
  }
  CompilerType coro_frame_type =
      InferArtificialCoroType(destroy_func, ConstString("__coro_frame"));
  if (!coro_frame_type)
    coro_frame_type = void_type;

  // Create the `resume` and `destroy` children.
  std::array<CompilerType, 1> args{coro_frame_type};
  CompilerType coro_func_type = ast_ctx->CreateFunctionType(
      /*result_type=*/void_type, args,
      /*is_variadic=*/false, /*qualifiers=*/0);
  CompilerType coro_func_ptr_type = coro_func_type.GetPointerType();
  ValueObjectSP resume_ptr_sp = CreateValueObjectFromAddress(
      "resume", frame_ptr_addr + 0 * ptr_size, exe_ctx, coro_func_ptr_type);
  assert(resume_ptr_sp);
  m_children.push_back(std::move(resume_ptr_sp));
  ValueObjectSP destroy_ptr_sp = CreateValueObjectFromAddress(
      "destroy", frame_ptr_addr + 1 * ptr_size, exe_ctx, coro_func_ptr_type);
  assert(destroy_ptr_sp);
  m_children.push_back(std::move(destroy_ptr_sp));

  // Add promise and coro_frame
  // Add the `promise` and `coro_frame` member. We intentionally add them as
  // pointer types instead of a value type, and don't automatically dereference
  // those pointers. We do so to avoid potential very deep recursion in case
  // there is a cycle formed between `std::coroutine_handle`s and their
  // promises.
  ValueObjectSP promise_ptr_sp = CreateValueObjectFromAddress(
      "promise", frame_ptr_addr + 2 * ptr_size, exe_ctx,
      promise_type.GetPointerType(), /*do_deref=*/false);
  m_children.push_back(std::move(promise_ptr_sp));
  ValueObjectSP coroframe_ptr_sp = CreateValueObjectFromAddress(
      "coro_frame", frame_ptr_addr, exe_ctx, coro_frame_type.GetPointerType(),
      /*do_deref=*/false);
  m_children.push_back(std::move(coroframe_ptr_sp));

  return lldb::ChildCacheState::eRefetch;
}

llvm::Expected<size_t>
StdlibCoroutineHandleSyntheticFrontEnd::GetIndexOfChildWithName(
    ConstString name) {
  for (const auto &[idx, child_sp] : llvm::enumerate(m_children)) {
    if (child_sp->GetName() == name)
      return idx;
  }

  return llvm::createStringError("Type has no child named '%s'",
                                 name.AsCString());
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::StdlibCoroutineHandleSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  return (valobj_sp ? new StdlibCoroutineHandleSyntheticFrontEnd(valobj_sp)
                    : nullptr);
}
