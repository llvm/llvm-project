//===- HandleLookupInJITDylibSPS.h - SPS spec for JD lookups ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS HandlerSpec for the handler created by createLookupInJITDylibHandler.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_HANDLELOOKUPINJITDYLIBSPS_H
#define LLVM_EXECUTIONENGINE_ORC_HANDLELOOKUPINJITDYLIBSPS_H

#include "llvm/ExecutionEngine/Orc/HandleLookupInJITDylib.h"
#include "llvm/ExecutionEngine/Orc/SPSHandlerSpec.h"
#include "llvm/ExecutionEngine/Orc/Shared/JITDylibLookupFlags.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"
#include "llvm/ExecutionEngine/Orc/Shared/SymbolLookupSetSPS.h"
#include "llvm/ExecutionEngine/Orc/Shared/SymbolNameSpec.h"

namespace llvm::orc::sps {

/// SPS spec for the handler created by createLookupInJITDylibHandler.
struct HandleLookupInJITDylibSpec
    : HandlerSpec<shared::SPSExpected<
          shared::SPSSequence<shared::SPSOptional<shared::SPSExecutorAddr>>>(
          shared::SPSExecutorAddr, shared::SPSJITDylibLookupFlags,
          shared::SPSSymbolLookupSet)> {
  static constexpr SymbolNameSpec Name =
      SymbolNameSpec::c("orc_rt_tag_sps_lookupInJITDylib");
};

} // namespace llvm::orc::sps

#endif // LLVM_EXECUTIONENGINE_ORC_HANDLELOOKUPINJITDYLIBSPS_H
