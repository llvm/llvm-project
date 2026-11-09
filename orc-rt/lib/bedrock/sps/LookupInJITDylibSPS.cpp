//===- LookupInJITDylibSPS.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Resolve symbols via JITDylib lookup in the controller.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/sps/LookupInJITDylibSPS.h"

#include "orc-rt-c/bedrock/ControllerHandlerTag.h"
#include "orc-rt-internal/support/StringExtras.h"
#include "orc-rt/bedrock/Session.h"
#include "orc-rt/support/sps/SPSJITDylibLookupFlags.h"
#include "orc-rt/support/sps/SPSSymbolLookupSet.h"
#include "orc-rt/support/sps/SPSWrapperFunction.h"

ORC_RT_C_DEFINE_CONTROLLER_HANDLER_TAG(orc_rt_tag_sps_lookupInJITDylib);

namespace orc_rt::sps {

void lookupInJITDylib(
    move_only_function<void(Expected<SymbolLookupResult>)> OnComplete,
    Session &S, const void *JITDylibKeyAddr, JITDylibLookupFlags JDLookupFlags,
    const SymbolLookupSet &LS) noexcept {
  using SPSSig = SPSExpected<SPSSymbolLookupResult>(
      SPSExecutorAddr, SPSJITDylibLookupFlags, SPSSymbolLookupSet);
  SPSWrapperFunction<SPSSig>::call(
      S.controllerCaller(&orc_rt_tag_sps_lookupInJITDylib),
      [OnComplete = std::move(OnComplete), NumSymbols = LS.size()](
          Expected<Expected<SymbolLookupResult>> Result) mutable {
        // The outer Expected reports failure to call the controller, the inner
        // one failure of the lookup itself. Clients see either as an error.
        if (!Result)
          return OnComplete(Result.takeError());

        // Clients index the result by lookup-set position, so a result of the
        // wrong length from the controller must not reach them.
        if (*Result && (*Result)->size() != NumSymbols)
          return OnComplete(make_error<StringError>(
              (StringOutputStream()
               << "lookupInJITDylib: expected " << NumSymbols
               << " results from controller, got " << (*Result)->size())
                  .str()));
        OnComplete(std::move(*Result));
      },
      JITDylibKeyAddr, JDLookupFlags, LS);
}

} // namespace orc_rt::sps
