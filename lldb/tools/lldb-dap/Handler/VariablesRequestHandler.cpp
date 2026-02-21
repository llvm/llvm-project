//===-- VariablesRequestHandler.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "DAPError.h"
#include "EventHelper.h"
#include "Handler/RequestHandler.h"
#include "Protocol/DAPTypes.h"
#include "Protocol/ProtocolRequests.h"
#include "Variables.h"
#include "llvm/Support/ErrorExtras.h"

using namespace llvm;
using namespace lldb_dap::protocol;

namespace lldb_dap {

/// Retrieves all child variables for the given variable reference.
///
/// A filter can be used to limit the fetched children to either named or
/// indexed children.
Expected<VariablesResponseBody>
VariablesRequestHandler::Run(const VariablesArguments &arguments) const {
  const var_ref_t var_ref = arguments.variablesReference;
  if (var_ref.Kind() == eReferenceKindInvalid)
    return llvm::make_error<DAPError>(
        llvm::formatv("invalid variablesReference: {}.", var_ref.AsUInt32()),
        /*error_code=*/llvm::inconvertibleErrorCode(), /*show_user=*/false);

  VariableStore *store = dap.reference_storage.GetVariableStore(var_ref);
  if (!store)
    return llvm::make_error<DAPError>(
        llvm::formatv("invalid variablesReference: {}.", var_ref.AsUInt32()),
        /*error_code=*/llvm::inconvertibleErrorCode(), /*show_user=*/false);

  return VariablesResponseBody{
      store->GetVariables(dap.reference_storage, dap.configuration, arguments)};
}

} // namespace lldb_dap
