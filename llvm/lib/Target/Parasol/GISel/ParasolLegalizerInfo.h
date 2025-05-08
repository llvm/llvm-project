//===- ParasolLegalizerInfo.h ----------------------------------------*- C++
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares the targeting of the Machinelegalizer class for Parasol
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Parasol_GISEL_ParasolMACHINELEGALIZER_H
#define LLVM_LIB_TARGET_Parasol_GISEL_ParasolMACHINELEGALIZER_H

#include "ParasolSubtarget.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"

namespace llvm {

/// This class provides the information for the Parasol target legalizer for
/// GlobalISel.
class ParasolLegalizerInfo : public LegalizerInfo {
public:
  ParasolLegalizerInfo(const ParasolSubtarget &ST);

private:
  const ParasolSubtarget *ST;
};
} // namespace llvm
#endif
