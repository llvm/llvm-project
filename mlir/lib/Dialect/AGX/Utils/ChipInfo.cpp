//===- ChipInfo.cpp - Capabilities of AGX GPUs ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AGX/Utils/ChipInfo.h"

using namespace mlir;
using namespace mlir::agx;

FailureOr<ChipInfo> ChipInfo::get(StringRef chip, StringRef features) {
  if (chip == "generic")
    chip = "agx-g13g";
  unsigned gen;
  if (!chip.consume_front("agx-g") || chip.size() != 3 ||
      chip.take_front(2).getAsInteger(10, gen) || gen < 13 || gen > 17)
    return failure();
  ChipInfo info;
  info.generation = gen;
  info.variant = chip.back();
  switch (gen) {
  case 13:
    // Mesa's threadgroup memory budget; no generation-specific measurements.
    info.threadgroupMemoryBytes = 32768;
    break;
  case 14:
    // Measured on G14C (M2 Max, Asahi Linux): AGX Performance Simulator and
    // Apple-M2-Playground novel-results 2026-10-11-fmadd-lane-offset,
    // barrier-0x68-vs-0x78, occupancy. The block barrier orders memory
    // within its block and completes for partial blocks and divergent callers
    // (ptx-re-blockers B16). The lane-offset fmadd waits for recorded
    // coverage of the forms its lowering emits (B18) and stays off until then.
    info.threadgroupMemoryBytes = 65536;
    info.blockBarrier = true;
    break;
  case 15:
    info.dynamicCaching = true;
    break;
  case 16:
  case 17:
    // Apple9: native 8x8x8 matrix multiply-accumulate with an A * B - C mode
    // (agx-re EXP-0022, EXP-0147). The LLVM AGX backend has no Apple9 matrix
    // instruction yet, so the emulated lowering is used until it does.
    info.dynamicCaching = true;
    info.maxGPRs = gen == 17 && info.variant == 'g' ? 126 : 96;
    info.threadgroupMemoryBytes = 32768;
    break;
  }

  SmallVector<StringRef> list;
  features.split(list, ',', -1, /*KeepEmpty=*/false);
  for (StringRef f : list) {
    f = f.trim();
    if (f == "+mesa-isa")
      info.mesaISA = true;
    else if (f == "-mesa-isa")
      info.mesaISA = false;
  }
  if (info.mesaISA) {
    // Mesa compiles for G13 and G14 only, and never emits the lane-offset
    // fmadd, the block barrier or a matrix instruction.
    if (gen != 13 && gen != 14)
      return failure();
    info.laneShiftFMA = false;
    info.blockBarrier = false;
    info.matrixUnit = MatrixUnit::Emulated;
    info.matrixNegatedAccumulate = false;
  }
  return info;
}
