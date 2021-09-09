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
EPCGenericMemoryAccess::CreateUsingOrcRTFuncs(ExecutionSession &ES,
                                              JITDylib &OrcRuntimeJD) {

  StringRef GlobalPrefix = "";
  if (ES.getExecutorProcessControl().getTargetTriple().isOSBinFormatMachO())
    GlobalPrefix = "_";

  FuncAddrs FAs;
  if (auto Err = lookupAndRecordAddrs(
          ES, LookupKind::Static, makeJITDylibSearchOrder(&OrcRuntimeJD),
          {{ES.intern((GlobalPrefix + "__orc_rt_write_uint8s_wrapper").str()),
            &FAs.WriteUInt8s},
           {ES.intern((GlobalPrefix + "__orc_rt_write_uint16s_wrapper").str()),
            &FAs.WriteUInt16s},
           {ES.intern((GlobalPrefix + "__orc_rt_write_uint32s_wrapper").str()),
            &FAs.WriteUInt32s},
           {ES.intern((GlobalPrefix + "__orc_rt_write_uint64s_wrapper").str()),
            &FAs.WriteUInt64s},
           {ES.intern((GlobalPrefix + "__orc_rt_write_buffers_wrapper").str()),
            &FAs.WriteBuffers}}))
    return std::move(Err);

  return std::make_unique<EPCGenericMemoryAccess>(
      ES.getExecutorProcessControl(), FAs);
}

} // end namespace orc
} // end namespace llvm
