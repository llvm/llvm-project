//===- EPCGenericMemoryAccessSPS.cpp - SPS memory access ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/EPCGenericMemoryAccessSPS.h"

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/LookupAndApply.h"
#include "llvm/ExecutionEngine/Orc/RecordProxy.h"

namespace llvm::orc::sps {

Expected<std::unique_ptr<EPCGenericMemoryAccess>>
createEPCGenericMemoryAccess(JITDylib &JD) {
  auto &ES = JD.getExecutionSession();
  EPCGenericMemoryAccess::Funcs Fns;
  // The proxies resolve to the specs' default controller-interface names.
  if (auto Err = lookupAndApply(
          JD, {recordProxy<MemWriteUInt8sProxySpec>(&Fns.WriteUInt8s),
               recordProxy<MemWriteUInt16sProxySpec>(&Fns.WriteUInt16s),
               recordProxy<MemWriteUInt32sProxySpec>(&Fns.WriteUInt32s),
               recordProxy<MemWriteUInt64sProxySpec>(&Fns.WriteUInt64s),
               recordProxy<MemWritePointersProxySpec>(&Fns.WritePointers),
               recordProxy<MemWriteBuffersProxySpec>(&Fns.WriteBuffers),
               recordProxy<MemReadUInt8sProxySpec>(&Fns.ReadUInt8s),
               recordProxy<MemReadUInt16sProxySpec>(&Fns.ReadUInt16s),
               recordProxy<MemReadUInt32sProxySpec>(&Fns.ReadUInt32s),
               recordProxy<MemReadUInt64sProxySpec>(&Fns.ReadUInt64s),
               recordProxy<MemReadPointersProxySpec>(&Fns.ReadPointers),
               recordProxy<MemReadBuffersProxySpec>(&Fns.ReadBuffers),
               recordProxy<MemReadStringsProxySpec>(&Fns.ReadStrings)}))
    return std::move(Err);
  return std::make_unique<EPCGenericMemoryAccess>(ES, std::move(Fns));
}

Expected<std::unique_ptr<EPCGenericMemoryAccess>>
createEPCGenericMemoryAccess(ExecutionSession &ES) {
  return createEPCGenericMemoryAccess(ES.getBootstrapJITDylib());
}

} // namespace llvm::orc::sps
