//===-- SuperHSchedule.td - SuperH Scheduling Definitions ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains SuperH scheduler definitions.
///
//===----------------------------------------------------------------------===//

/// SuperH Base Scheduling Model.
class SuperHSchedModel : SchedMachineModel {
  let LoadLatency = 4;  // Word (Rn)
  let HighLatency = 16; // Long ABS
  let PostRAScheduler = 0;
  let CompleteModel = 0;
}

def GenericSuperHModel : SuperHSchedModel;
