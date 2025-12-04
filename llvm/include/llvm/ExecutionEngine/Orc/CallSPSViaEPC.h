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
template <typename RetT, typename... ArgTs> struct SPSCallSerializationImpl {
  using RetSerialization = shared::SPSArgList<RetT>;
  using ArgSerialization = shared::SPSArgList<ArgTs...>;
};
} // namespace detail

template <typename SPSSig>
struct SPSCallSerialization
    : public CallableTraitsHelper<detail::SPSCallSerializationImpl, SPSSig> {};

template <typename SPSSig> class SPSCallSerializer {
public:
  template <typename... ArgTs>
  Expected<shared::WrapperFunctionResult> serialize(ArgTs &&...Args) {
    using ArgSerialization =
        typename SPSCallSerialization<SPSSig>::ArgSerialization;
    auto Buffer = shared::WrapperFunctionResult::allocate(
        ArgSerialization::size(Args...));
    shared::SPSOutputBuffer OB(Buffer.data(), Buffer.size());
    if (!ArgSerialization::serialize(OB, Args...))
      return make_error<StringError>("Could not serialize arguments",
                                     inconvertibleErrorCode());
    return std::move(Buffer);
  }

  template <typename RetT>
  Expected<RetT> deserialize(shared::WrapperFunctionResult ResultBytes) {
    using RetDeserialization =
        typename SPSCallSerialization<SPSSig>::RetSerialization;
    shared::SPSInputBuffer IB(ResultBytes.data(), ResultBytes.size());
    RetT ReturnValue;
    if (!RetDeserialization::deserialize(IB, ReturnValue))
      return make_error<StringError>("Could not deserialize return value",
                                     inconvertibleErrorCode());
    return ReturnValue;
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
