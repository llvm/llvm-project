//===- llvm/MC/DXContainerPSVInfo.h - DXContainer PSVInfo -*- C++ -------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_DXCONTAINERPSVINFO_H
#define LLVM_MC_DXCONTAINERPSVINFO_H

#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/TargetParser/Triple.h"

#include <numeric>
#include <stdint.h>
#include <vector>

namespace llvm {

class raw_ostream;

namespace mcdxbc {
// This data structure is a helper for reading and writing PSV RuntimeInfo data.
// It is implemented in the BinaryFormat library so that it can be used by both
// the MC layer and Object tools.
// This structure is used to represent the extracted data in an inspectable and
// modifiable format, and can be used to serialize the data back into valid PSV
// RuntimeInfo.
struct PSVRuntimeInfo {
  dxbc::PSV::v2::RuntimeInfo BaseData;
  std::vector<dxbc::PSV::v2::ResourceBindInfo> Resources;

  // Serialize PSVInfo into the provided raw_ostream. The version field
  // specifies the data version to encode, the default value specifies encoding
  // the highest supported version.
  void write(raw_ostream &OS,
             uint32_t Version = std::numeric_limits<uint32_t>::max()) const;

  void swapBytes(Triple::EnvironmentType Stage) {
    BaseData.swapBytes();
    BaseData.swapBytes(Stage);
    for (auto Res : Resources)
      Res.swapBytes();
  }
};

} // namespace mcdxbc
} // namespace llvm

#endif // LLVM_MC_DXCONTAINERPSVINFO_H
