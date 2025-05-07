//===- IndexedMemProfData.h - MemProf format support ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MemProf data is serialized in writeMemProf provided in this header file.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/MemProf.h"

namespace llvm {

// Write the MemProf data to OS.
Error writeMemProf(ProfOStream &OS, memprof::IndexedMemProfData &MemProfData,
                   memprof::IndexedVersion MemProfVersionRequested,
                   bool MemProfFullSchema);

} // namespace llvm
