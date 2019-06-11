//===- llvm-lib/LibDriver.h - lib.exe-compatible driver ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines an interface to a lib.exe-compatible driver that also understands
// bitcode files. Used by llvm-lib and lld-link /lib.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLDRIVERS_LLVM_LIB_LIBDRIVER_H
#define LLVM_TOOLDRIVERS_LLVM_LIB_LIBDRIVER_H

namespace llvm {
template <typename T> class ArrayRef;

int libDriverMain(ArrayRef<const char *> ARgs);


class StringRef;
namespace COFF {
enum MachineTypes : unsigned;
}

// Returns a user-readable string for ARMNT, ARM64, AMD64, I386.
// Other MachineTypes values must not be pased in.
StringRef machineToStr(COFF::MachineTypes MT);

// Maps /machine: arguments to a MachineTypes value.
// Only returns ARMNT, ARM64, AMD64, I386, or IMAGE_FILE_MACHINE_UNKNOWN.
COFF::MachineTypes getMachineType(StringRef S);

}

#endif
