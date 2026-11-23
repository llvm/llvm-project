//===- RelayoutOpInterface.td ----- Intrerface Declaration -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_IR_RELAYOUTOPINTERFACE
#define LINALG_IR_RELAYOUTOPINTERFACE

include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/IR/OpBase.td"

def LinalgRelayoutOpInterface : OpInterface<"RelayoutOpInterface"> {
  let description = [{
    A Linalg relayout-op is either linalg.pack or linalg.unpack.

    While we could extend this interface with methods from Linalg_RelayoutOp,
    this is currently not needed and left as a TODO.
  }];
  let cppNamespace = "::mlir::linalg";
}

#endif // LINALG_IR_RELAYOUTOPINTERFACE
