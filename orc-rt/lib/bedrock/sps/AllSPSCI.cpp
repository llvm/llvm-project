//===- AllSPSCI.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of sps_ci::addAll.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/sps/AllSPSCI.h"
#include "orc-rt/bedrock/sps/CallSPSCI.h"
#include "orc-rt/bedrock/sps/GDBJITRegistrarSPSCI.h"
#include "orc-rt/bedrock/sps/MemoryAccessSPSCI.h"
#include "orc-rt/bedrock/sps/NativeDylibManagerSPSCI.h"
#include "orc-rt/bedrock/sps/SimpleNativeMemoryMapSPSCI.h"
#include "orc-rt/bedrock/sps/StandaloneMachOUnwindInfoRegistrarSPSCI.h"

namespace orc_rt::sps_ci {

Error addAll(SimpleSymbolTable &ST) {
  using AdderFn = Error (*)(SimpleSymbolTable &);
  AdderFn Adders[] = {addCall,
                      addGDBJITRegistrar,
                      addMemoryAccess,
                      addNativeDylibManager,
                      addSimpleNativeMemoryMap,
                      addStandaloneMachOUnwindInfoRegistrar};

  for (auto *Adder : Adders)
    if (auto Err = Adder(ST))
      return Err;

  return Error::success();
}

} // namespace orc_rt::sps_ci
