//=-- ConnexHazardRecognizer.h - Define frame lowering for Connex -- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///
//===----------------------------------------------------------------------===//

/* Inspired from llvm/lib/Target/PowerPC/PPCHazardRecognizer.h:
   /// PPCDispatchGroupSBHazardRecognizer - This class implements a
   ///      scoreboard-based
   /// hazard recognizer for PPC ooo processors with dispatch-group hazards.
*/

#ifndef LLVM_LIB_TARGET_CONNEX_HAZARDRECOGNIZER_H
#define LLVM_LIB_TARGET_CONNEX_HAZARDRECOGNIZER_H

#include "ConnexInstrInfo.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/CodeGen/ScoreboardHazardRecognizer.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"

namespace llvm {

/* NOTE: ScheduleHazardRecognizer is basically an "interface"
 * (almost abstract, i.e. almost no functionality implemented) class, so better
 * stick with ScoreboardHazardRecognizer if its functionality is OK for me:
 */

/* We choose to inherit the ScoreboardHazardRecognizer because only this
 * performs out-of-order scheduling, and NOT ScheduleHazardRecognizer.
 */
class ConnexDispatchGroupSBHazardRecognizer
    : public ScoreboardHazardRecognizer {
  const ScheduleDAG *DAG;
  bool isDataHazard(SUnit *SU);

public:
  ConnexDispatchGroupSBHazardRecognizer(const InstrItineraryData *ItinData,
                                        const ScheduleDAG *DAG_)
      : ScoreboardHazardRecognizer(ItinData, DAG_), DAG(DAG_) {}

  HazardType getHazardType(SUnit *SU, int Stalls) override;

  unsigned PreEmitNoops(SUnit *SU) override;
  void EmitInstruction(SUnit *SU) override;
};

} // End namespace llvm

#endif
