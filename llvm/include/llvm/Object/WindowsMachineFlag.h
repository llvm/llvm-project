//===- WindowsMachineFlag.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions for implementing the /machine: flag.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLDRIVERS_MACHINEFLAG_MACHINEFLAG_H
#define LLVM_TOOLDRIVERS_MACHINEFLAG_MACHINEFLAG_H

namespace llvm {

class StringRef;
namespace COFF {
enum MachineTypes : unsigned;
}

// Returns a user-readable string for ARMNT, ARM64, AMD64, I386.
// Other MachineTypes values must not be passed in.
StringRef machineToStr(COFF::MachineTypes MT);

// Maps /machine: arguments to a MachineTypes value.
// Only returns ARMNT, ARM64, AMD64, I386, or IMAGE_FILE_MACHINE_UNKNOWN.
COFF::MachineTypes getMachineType(StringRef S);

}

#endif
