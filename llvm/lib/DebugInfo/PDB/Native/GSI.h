//===- GSI.h - Common Declarations for GlobalsStream and PublicsStream ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The data structures defined in this file are based on the reference
// implementation which is available at
// https://github.com/Microsoft/microsoft-pdb/blob/master/PDB/dbi/gsi.h
//
// When you are reading the reference source code, you'd find the
// information below useful.
//
//  - ppdb1->m_fMinimalDbgInfo seems to be always true.
//  - SMALLBUCKETS macro is defined.
//
// The reference doesn't compile, so I learned just by reading code.
// It's not guaranteed to be correct.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_PDB_RAW_GSI_H
#define LLVM_LIB_DEBUGINFO_PDB_RAW_GSI_H

#include "llvm/DebugInfo/MSF/BinaryStreamArray.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"

#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

namespace llvm {

namespace msf {
class StreamReader;
}

namespace pdb {

/// From https://github.com/Microsoft/microsoft-pdb/blob/master/PDB/dbi/gsi.cpp
static const unsigned IPHR_HASH = 4096;

/// Header of the hash tables found in the globals and publics sections.
/// Based on GSIHashHeader in
/// https://github.com/Microsoft/microsoft-pdb/blob/master/PDB/dbi/gsi.h
struct GSIHashHeader {
  enum : unsigned {
    HdrSignature = ~0U,
    HdrVersion = 0xeffe0000 + 19990810,
  };
  support::ulittle32_t VerSignature;
  support::ulittle32_t VerHdr;
  support::ulittle32_t HrSize;
  support::ulittle32_t NumBuckets;
};

Error readGSIHashBuckets(
    msf::FixedStreamArray<support::ulittle32_t> &HashBuckets,
    const GSIHashHeader *HashHdr, msf::StreamReader &Reader);
Error readGSIHashHeader(const GSIHashHeader *&HashHdr,
                        msf::StreamReader &Reader);
Error readGSIHashRecords(msf::FixedStreamArray<PSHashRecord> &HashRecords,
                         const GSIHashHeader *HashHdr,
                         msf::StreamReader &Reader);
}
}

#endif
