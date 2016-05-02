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
namespace pdb {
enum PdbRaw_ImplVer : uint32_t {
  PdbImplVC2 = 19941610,
  PdbImplVC4 = 19950623,
  PdbImplVC41 = 19950814,
  PdbImplVC50 = 19960307,
  PdbImplVC98 = 19970604,
  PdbImplVC70Dep = 19990604, // deprecated
  PdbImplVC70 = 20000404,
  PdbImplVC80 = 20030901,
  PdbImplVC110 = 20091201,
  PdbImplVC140 = 20140508,
};

enum PdbRaw_DbiVer : uint32_t {
  PdbDbiVC41 = 930803,
  PdbDbiV50 = 19960307,
  PdbDbiV60 = 19970606,
  PdbDbiV70 = 19990903,
  PdbDbiV110 = 20091201
};

enum SpecialStream : uint32_t {
  StreamPDB = 1,
  StreamTPI = 2,
  StreamDBI = 3,
  StreamIPI = 4,
};
}
}

#endif
