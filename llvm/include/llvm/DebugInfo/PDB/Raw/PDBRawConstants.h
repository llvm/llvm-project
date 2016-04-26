//===- PDBRawConstants.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_PDBRAWCONSTANTS_H
#define LLVM_DEBUGINFO_PDB_RAW_PDBRAWCONSTANTS_H

#include <stdint.h>

namespace llvm {

enum PdbRaw_ImplVer : uint32_t {
  VC2 = 19941610,
  VC4 = 19950623,
  VC41 = 19950814,
  VC50 = 19960307,
  VC98 = 19970604,
  VC70Dep = 19990604, // deprecated
  VC70 = 20000404,
  VC80 = 20030901,
  VC110 = 20091201,
  VC140 = 20140508,
};
}

#endif
