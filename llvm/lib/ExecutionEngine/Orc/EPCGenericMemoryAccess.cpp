//===----- EPCGenericMemoryAccess.cpp - Generic EPC MemoryAccess impl -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/EPCGenericMemoryAccess.h"
#include "llvm/ExecutionEngine/Orc/LookupAndRecordAddrs.h"

namespace llvm {
namespace orc {

/// Create from a ExecutorProcessControl instance.
Expected<std::unique_ptr<EPCGenericMemoryAccess>>
EPCGenericMemoryAccess::CreateUsingOrcRTFuncs(ExecutorProcessControl &EPC) {

  auto H = EPC.loadDylib("");
  if (!H)
    return H.takeError();

  StringRef GlobalPrefix = "";
  if (EPC.getTargetTriple().isOSBinFormatMachO())
    GlobalPrefix = "_";

  FuncAddrs FAs;
  if (auto Err = lookupAndRecordAddrs(
          EPC, *H,
          {{EPC.intern((GlobalPrefix + "__orc_rt_write_uint8s_wrapper").str()),
            &FAs.WriteUInt8s},
           {EPC.intern((GlobalPrefix + "__orc_rt_write_uint16s_wrapper").str()),
            &FAs.WriteUInt16s},
           {EPC.intern((GlobalPrefix + "__orc_rt_write_uint32s_wrapper").str()),
            &FAs.WriteUInt32s},
           {EPC.intern((GlobalPrefix + "__orc_rt_write_uint64s_wrapper").str()),
            &FAs.WriteUInt64s},
           {EPC.intern((GlobalPrefix + "__orc_rt_write_buffers_wrapper").str()),
            &FAs.WriteBuffers}}))
    return std::move(Err);

  return std::make_unique<EPCGenericMemoryAccess>(EPC, std::move(FAs));
}

} // end namespace orc
} // end namespace llvm
