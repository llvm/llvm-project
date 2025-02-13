//===-- RISCVSchedGeneric.td - Generic In-order Processor --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Adjust the default cost model to enable all scheduling heuristics, not just
// latency.
//
// In particular, this enables register pressure heuristics which are very
// important for vector code with high LMULs, and have little negative impact
// on other configurations.
def GenericModel : SchedMachineModel {
  let MicroOpBufferSize = 1;
  let CompleteModel = 0;
}
