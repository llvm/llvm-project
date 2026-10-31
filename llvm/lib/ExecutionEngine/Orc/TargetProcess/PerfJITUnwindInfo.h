//===- PerfJITUnwindInfo.h - Prepare CFI for perf --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_EXECUTIONENGINE_ORC_TARGETPROCESS_PERFJITUNWINDINFO_H
#define LLVM_LIB_EXECUTIONENGINE_ORC_TARGETPROCESS_PERFJITUNWINDINFO_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <string>

namespace llvm::orc {

/// Prepare profiler-only copies of common little-endian ELF64 CFI: DWARF32,
/// CIE v1/v3, zR/zPLR and PC-relative signed 32-bit FDE addresses. Unsupported
/// metadata is optional: callers must still publish ordinary code loads.
/// The input must remain alive until all calls to emitEHFrame have completed.
class LLVM_ABI PerfJITUnwindInfo {
public:
  static Expected<PerfJITUnwindInfo> create(StringRef Frame, uint64_t FrameAddr,
                                            uint8_t AddressSize,
                                            bool IsLittleEndian);

  /// Emit the intersecting FDEs and their CIEs for one synthetic perf ELF.
  Expected<std::string> emitEHFrame(uint64_t CodeAddr, uint64_t CodeSize) const;

private:
  PerfJITUnwindInfo() = default;
  struct CIE {
    StringRef Parameters, Instructions;
    uint8_t Version;
    bool HasLSDA;
  };
  struct FDE {
    uint64_t InitialLocation, Range;
    StringRef Instructions;
    size_t CIEIndex;
  };
  SmallVector<CIE, 1> CIEs;
  SmallVector<FDE, 4> FDEs;
};

} // namespace llvm::orc

#endif
