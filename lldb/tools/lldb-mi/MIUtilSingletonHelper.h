//===-- MIUtilSingletonHelper.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// In house headers:
#include "MICmnResources.h"
#include "MIUtilString.h"

namespace MI {

//++
//============================================================================
// Details: Short cut helper function to simplify repeated initialisation of
//          MI components (singletons) required by a client module.
// Type:    Template method.
// Args:    vErrorResrcId   - (R)  The string resource ID error message
// identifier to place in errMsg.
//          vwrbOk          - (RW) On input True = Try to initialize MI driver
//          module.
//                                 On output True = MI driver module initialise
//                                 successfully.
//          vwrErrMsg       - (W)  MI driver module initialise error description
//          on failure.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
//--
template <typename T>
bool ModuleInit(const MIint vErrorResrcId, bool &vwrbOk,
                CMIUtilString &vwrErrMsg) {
  if (vwrbOk && !T::Instance().Initialize()) {
    vwrbOk = MIstatus::failure;
    vwrErrMsg = CMIUtilString::Format(
        MIRSRC(vErrorResrcId), T::Instance().GetErrorDescription().c_str());
  }

  return vwrbOk;
}

//++
//============================================================================
// Details: Short cut helper function to simplify repeated shutdown of
//          MI components (singletons) required by a client module.
// Type:    Template method.
// Args:    vErrorResrcId   - (R)  The string resource ID error message
// identifier
//                                 to place in errMsg.
//          vwrbOk          - (W)  If not already false make false on module
//                                 shutdown failure.
//          vwrErrMsg       - (RW) Append to existing error description string
//          MI
//                                 driver module initialise error description on
//                                 failure.
// Return:  True - Module shutdown succeeded.
//          False - Module shutdown failed.
//--
template <typename T>
bool ModuleShutdown(const MIint vErrorResrcId, bool &vwrbOk,
                    CMIUtilString &vwrErrMsg) {
  bool bOk = MIstatus::success;

  if (!T::Instance().Shutdown()) {
    const bool bMoreThanOneError(!vwrErrMsg.empty());
    bOk = MIstatus::failure;
    if (bMoreThanOneError)
      vwrErrMsg += ", ";
    vwrErrMsg += CMIUtilString::Format(
        MIRSRC(vErrorResrcId), T::Instance().GetErrorDescription().c_str());
  }

  vwrbOk = bOk ? vwrbOk : MIstatus::failure;

  return bOk;
}

} // namespace MI
