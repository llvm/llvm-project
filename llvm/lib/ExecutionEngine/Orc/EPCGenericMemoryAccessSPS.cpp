//===- EPCGenericMemoryAccessSPS.cpp - SPS memory access ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/EPCGenericMemoryAccessSPS.h"

#include "llvm/ExecutionEngine/Orc/Core.h"

namespace llvm::orc::sps {

Expected<std::unique_ptr<EPCGenericMemoryAccess>>
createEPCGenericMemoryAccess(JITDylib &JD) {
  auto &ES = JD.getExecutionSession();
  EPCGenericMemoryAccess::Funcs Fns;
  // The proxies resolve to the specs' default controller-interface names.
  if (auto Err =
          buildProxies(JD, proxyInit<MemWriteUInt8sProxySpec>(&Fns.WriteUInt8s),
                       proxyInit<MemWriteUInt16sProxySpec>(&Fns.WriteUInt16s),
                       proxyInit<MemWriteUInt32sProxySpec>(&Fns.WriteUInt32s),
                       proxyInit<MemWriteUInt64sProxySpec>(&Fns.WriteUInt64s),
                       proxyInit<MemWritePointersProxySpec>(&Fns.WritePointers),
                       proxyInit<MemWriteBuffersProxySpec>(&Fns.WriteBuffers),
                       proxyInit<MemReadUInt8sProxySpec>(&Fns.ReadUInt8s),
                       proxyInit<MemReadUInt16sProxySpec>(&Fns.ReadUInt16s),
                       proxyInit<MemReadUInt32sProxySpec>(&Fns.ReadUInt32s),
                       proxyInit<MemReadUInt64sProxySpec>(&Fns.ReadUInt64s),
                       proxyInit<MemReadPointersProxySpec>(&Fns.ReadPointers),
                       proxyInit<MemReadBuffersProxySpec>(&Fns.ReadBuffers),
                       proxyInit<MemReadStringsProxySpec>(&Fns.ReadStrings)))
    return std::move(Err);
  return std::make_unique<EPCGenericMemoryAccess>(ES, std::move(Fns));
}

Expected<std::unique_ptr<EPCGenericMemoryAccess>>
createEPCGenericMemoryAccess(ExecutionSession &ES) {
  return createEPCGenericMemoryAccess(ES.getBootstrapJITDylib());
}

} // namespace llvm::orc::sps
