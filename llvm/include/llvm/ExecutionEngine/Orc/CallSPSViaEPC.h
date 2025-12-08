//===---- CallSPSViaEPC.h - EPCCalls using SPS serialization ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// EPCCalls using SimplePackedSerialization.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_CALLSPSVIAEPC_H
#define LLVM_EXECUTIONENGINE_ORC_CALLSPSVIAEPC_H

#include "llvm/ExecutionEngine/Orc/CallViaEPC.h"
#include "llvm/ExecutionEngine/Orc/CallableTraitsHelper.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"

namespace llvm::orc {

namespace detail {
template <typename SPSRetT, typename... SPSArgTs>
struct SPSCallSerializationImpl {
  using RetSerialization = shared::SPSArgList<SPSRetT>;
  using ArgSerialization = shared::SPSArgList<SPSArgTs...>;

  template <typename... ArgTs>
  Expected<shared::WrapperFunctionResult> serialize(ArgTs &&...Args) {
    auto Buffer = shared::WrapperFunctionResult::allocate(
        ArgSerialization::size(Args...));
    shared::SPSOutputBuffer OB(Buffer.data(), Buffer.size());
    if (!ArgSerialization::serialize(OB, Args...))
      return make_error<StringError>("Could not serialize arguments",
                                     inconvertibleErrorCode());
    return std::move(Buffer);
  }
};

template <typename SPSSig>
struct SPSCallSerialization
    : public CallableTraitsHelper<detail::SPSCallSerializationImpl, SPSSig> {};

} // namespace detail

/// SPS serialization for non-void calls.
template <typename SPSSig>
struct SPSCallSerializer : public detail::SPSCallSerialization<SPSSig> {

  template <typename RetT>
  Expected<RetT> deserialize(shared::WrapperFunctionResult ResultBytes) {
    using RetDeserialization =
        typename detail::SPSCallSerialization<SPSSig>::RetSerialization;
    shared::SPSInputBuffer IB(ResultBytes.data(), ResultBytes.size());
    RetT ReturnValue;
    if (!RetDeserialization::deserialize(IB, ReturnValue))
      return make_error<StringError>("Could not deserialize return value",
                                     inconvertibleErrorCode());
    return ReturnValue;
  }
};

/// SPS serialization for void calls.
template <typename... SPSArgTs>
struct SPSCallSerializer<void(SPSArgTs...)>
    : public detail::SPSCallSerialization<void(SPSArgTs...)> {
  template <typename RetT>
  std::enable_if_t<std::is_void_v<RetT>, Error>
  deserialize(shared::WrapperFunctionResult ResultBytes) {
    if (!ResultBytes.empty())
      return make_error<StringError>("Could not deserialize return value",
                                     inconvertibleErrorCode());
    return Error::success();
  }
};

template <typename SPSSig>
class SPSEPCCaller : public EPCCaller<SPSCallSerializer<SPSSig>> {
public:
  SPSEPCCaller(ExecutorProcessControl &EPC)
      : EPCCaller<SPSCallSerializer<SPSSig>>(EPC, SPSCallSerializer<SPSSig>()) {
  }
};

template <typename SPSSig>
class SPSEPCCall : public EPCCall<SPSCallSerializer<SPSSig>> {
public:
  SPSEPCCall(ExecutorProcessControl &EPC, ExecutorSymbolDef Fn)
      : EPCCall<SPSCallSerializer<SPSSig>>(EPC, SPSCallSerializer<SPSSig>(),
                                           std::move(Fn)) {}
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_CALLSPSVIAEPC_H
