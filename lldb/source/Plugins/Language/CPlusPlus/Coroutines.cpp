//===-- Coroutines.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Coroutines.h"

#include "Plugins/ExpressionParser/Clang/ClangASTImporter.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/VariableList.h"

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

static Function *ExtractFunction(ValueObjectSP &frame_ptr_sp, int offset) {
  lldb::TargetSP target_sp = frame_ptr_sp->GetTargetSP();
  lldb::ProcessSP process_sp = frame_ptr_sp->GetProcessSP();
  auto ptr_size = process_sp->GetAddressByteSize();

  AddressType addr_type;
  lldb::addr_t frame_ptr_addr = frame_ptr_sp->GetPointerValue(&addr_type);
  if (!frame_ptr_addr || frame_ptr_addr == LLDB_INVALID_ADDRESS)
    return nullptr;
  lldbassert(addr_type == AddressType::eAddressTypeLoad);

  Status error;
  auto func_ptr_addr = frame_ptr_addr + offset * ptr_size;
  lldb::addr_t func_addr =
      process_sp->ReadPointerFromMemory(func_ptr_addr, error);
  if (error.Fail())
    return nullptr;

  Address func_address;
  if (!target_sp->ResolveLoadAddress(func_addr, func_address))
    return nullptr;

  return func_address.CalculateSymbolContextFunction();
}

static Function *ExtractResumeFunction(ValueObjectSP &frame_ptr_sp) {
  return ExtractFunction(frame_ptr_sp, 0);
}

static Function *ExtractDestroyFunction(ValueObjectSP &frame_ptr_sp) {
  return ExtractFunction(frame_ptr_sp, 1);
}

static bool IsNoopCoroFunction(Function *f) {
  if (!f)
    return false;

  // clang's `__builtin_coro_noop` gets lowered to
  // `_NoopCoro_ResumeDestroy`. This is used by libc++
  // on clang.
  auto mangledName = f->GetMangled().GetMangledName();
  if (mangledName == "__NoopCoro_ResumeDestroy")
    return true;

  // libc++ uses the following name as a fallback on
  // compilers without `__builtin_coro_noop`.
  auto name = f->GetNameNoArguments();
  static RegularExpression libcxxRegex(
      "^std::coroutine_handle<std::noop_coroutine_promise>::"
      "__noop_coroutine_frame_ty_::__dummy_resume_destroy_func$");
  lldbassert(libcxxRegex.IsValid());
  if (libcxxRegex.Execute(name.GetStringRef()))
    return true;
  static RegularExpression libcxxRegexAbiNS(
      "^std::__[[:alnum:]]+::coroutine_handle<std::__[[:alnum:]]+::"
      "noop_coroutine_promise>::__noop_coroutine_frame_ty_::"
      "__dummy_resume_destroy_func$");
  lldbassert(libcxxRegexAbiNS.IsValid());
  if (libcxxRegexAbiNS.Execute(name.GetStringRef()))
    return true;

  // libstdc++ uses the following name on both gcc and clang.
  static RegularExpression libstdcppRegex(
      "^std::__[[:alnum:]]+::coroutine_handle<std::__[[:alnum:]]+::"
      "noop_coroutine_promise>::__frame::__dummy_resume_destroy$");
  lldbassert(libstdcppRegex.IsValid());
  if (libstdcppRegex.Execute(name.GetStringRef()))
    return true;

  return false;
}

static CompilerType InferPromiseType(Function &destroy_func) {
  Block &block = destroy_func.GetBlock(true);
  auto variable_list = block.GetBlockVariableList(true);

  // clang generates an artificial `__promise` variable inside the
  // `destroy` function. Look for it.
  auto promise_var = variable_list->FindVariable(ConstString("__promise"));
  if (!promise_var)
    return {};
  if (!promise_var->IsArtificial())
    return {};

  Type *promise_type = promise_var->GetType();
  if (!promise_type)
    return {};
  return promise_type->GetForwardCompilerType();
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

  if (!ptr_sp->GetValueAsUnsigned(0)) {
    stream << "nullptr";
    return true;
  }
  if (IsNoopCoroFunction(ExtractResumeFunction(ptr_sp)) &&
      IsNoopCoroFunction(ExtractDestroyFunction(ptr_sp))) {
    stream << "noop_coroutine";
    return true;
  }

  stream.Printf("coro frame = 0x%" PRIx64, ptr_sp->GetValueAsUnsigned(0));
  return true;
}

lldb_private::formatters::StdlibCoroutineHandleSyntheticFrontEnd::
    StdlibCoroutineHandleSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp),
      m_ast_importer(std::make_unique<ClangASTImporter>()) {
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

  Function *resume_func = ExtractResumeFunction(ptr_sp);
  Function *destroy_func = ExtractDestroyFunction(ptr_sp);

  if (IsNoopCoroFunction(resume_func) && IsNoopCoroFunction(destroy_func)) {
    // For `std::noop_coroutine()`, we don't want to display any child nodes.
    return false;
  }

  // Get the `promise_type` from the template argument
  CompilerType promise_type(
      valobj_sp->GetCompilerType().GetTypeTemplateArgument(0));
  if (!promise_type)
    return false;

  // Try to infer the promise_type if it was type-erased
  auto ts = valobj_sp->GetCompilerType().GetTypeSystem();
  auto ast_ctx = ts.dyn_cast_or_null<TypeSystemClang>();
  if (!ast_ctx)
    return false;
  if (promise_type.IsVoidType() && destroy_func) {
    if (CompilerType inferred_type = InferPromiseType(*destroy_func)) {
      // Copy the type over to the correct `TypeSystemClang` instance
      promise_type = m_ast_importer->CopyType(*ast_ctx, inferred_type);
    }
  }

  // Build the coroutine frame type
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
