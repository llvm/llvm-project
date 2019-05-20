//=- ARMScheduleM3.td - ARM Cortex-M3 Scheduling Definitions -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for the ARM Cortex-M3 processor.
//
//===----------------------------------------------------------------------===//

def CortexM3Model : SchedMachineModel {
  let IssueWidth        = 1; // Only IT can be dual-issued, so assume single-issue
  let MicroOpBufferSize = 0; // In-order
  let LoadLatency       = 2; // Latency when not pipelined, not pc-relative
  let MispredictPenalty = 2; // Best case branch taken cost

  let CompleteModel = 0;
}
