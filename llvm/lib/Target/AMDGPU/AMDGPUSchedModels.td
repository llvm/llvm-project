//===-- AMDGPUSchedModels.td - SI Scheduling definitions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MachineModel definitions for Southern Islands (SI)
//
//===----------------------------------------------------------------------===//

class SISchedMachineModel : SchedMachineModel {
  let CompleteModel = 1;
  // MicroOpBufferSize = 1 means that instructions will always be added
  // the ready queue when they become available.  This exposes them
  // to the register pressure analysis.
  let MicroOpBufferSize = 1;
  let IssueWidth = 1;
  let PostRAScheduler = 1;

  // FIXME:Approximate 2 * branch cost.  Try to hack around bad
  // early-ifcvt heuristics. These need improvement to avoid the OOE
  // heuristics.
  int MispredictPenalty = 20;
}

def SIFullSpeedModel : SISchedMachineModel;
def SIQuarterSpeedModel : SISchedMachineModel;
def SIDPFullSpeedModel : SISchedMachineModel;
def SIDPGFX942FullSpeedModel : SISchedMachineModel;
def SIDPGFX950FullSpeedModel : SISchedMachineModel;
def GFX10SpeedModel : SISchedMachineModel;
def GFX11SpeedModel : SISchedMachineModel;
def GFX12SpeedModel : SISchedMachineModel;
def GFX1250SpeedModel : SISchedMachineModel;
def GFX1251SpeedModel : SISchedMachineModel;
def GFX125xGenericSpeedModel : SISchedMachineModel;
