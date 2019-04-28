//===-- llvm/BinaryFormat/XCOFF.h - The XCOFF file format -------*- C++/-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines manifest constants for the XCOFF object file format.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BINARYFORMAT_XCOFF_H
#define LLVM_BINARYFORMAT_XCOFF_H

namespace llvm {
namespace XCOFF {

// Constants used in the XCOFF definition.
enum { SectionNameSize = 8 };

// Flags for defining the section type. Used for the s_flags field of
// the section header structure. Defined in the system header `scnhdr.h`.
enum SectionTypeFlags {
  STYP_PAD = 0x0008,
  STYP_DWARF = 0x0010,
  STYP_TEXT = 0x0020,
  STYP_DATA = 0x0040,
  STYP_BSS = 0x0080,
  STYP_EXCEPT = 0x0100,
  STYP_INFO = 0x0200,
  STYP_TDATA = 0x0400,
  STYP_TBSS = 0x0800,
  STYP_LOADER = 0x1000,
  STYP_DEBUG = 0x2000,
  STYP_TYPCHK = 0x4000,
  STYP_OVRFLO = 0x8000
};

} // end namespace XCOFF
} // end namespace llvm

#endif
