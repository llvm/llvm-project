//===-- llvm/BinaryFormat/GOFF.h - GOFF definitions --------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header contains common, non-processor-specific data structures and
// constants for the GOFF file format.
//
// GOFF specifics can be found in MVS Program Management: Advanced Facilities
//===----------------------------------------------------------------------===//

#ifndef LLVM_BINARYFORMAT_GOFF_H
#define LLVM_BINARYFORMAT_GOFF_H

namespace llvm {

namespace GOFF {

// \brief Subsections of the primary C_CODE section in the object file.
enum SubsectionKind : uint8_t {
  SK_PPA1 = 2,
};

} // end namespace GOFF

} // end namespace llvm

#endif // LLVM_BINARYFORMAT_GOFF_H
