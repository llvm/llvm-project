#ifndef MEMPROF_DATA_INC
#define MEMPROF_DATA_INC
/*===-- MemProfData.inc - MemProf profiling runtime structures -*- C++ -*-=== *\
|*
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
|* See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
\*===----------------------------------------------------------------------===*/
/*
 * This is the main file that defines all the data structure, signature,
 * constant literals that are shared across profiling runtime library,
 * and host tools (reader/writer).
 *
 * This file has two identical copies. The primary copy lives in LLVM and
 * the other one sits in compiler-rt/include/profile directory. To make changes
 * in this file, first modify the primary copy and copy it over to compiler-rt.
 * Testing of any change in this file can start only after the two copies are
 * synced up.
 *
\*===----------------------------------------------------------------------===*/


#ifdef _MSC_VER
#define PACKED(__decl__) __pragma(pack(push,1)) __decl__ __pragma(pack(pop))
#else
#define PACKED(__decl__) __decl__ __attribute__((__packed__))
#endif

// A 64-bit magic number to uniquely identify the raw binary memprof profile file.
#define MEMPROF_RAW_MAGIC_64                                                                        \
  ((uint64_t)255 << 56 | (uint64_t)'m' << 48 | (uint64_t)'p' << 40 | (uint64_t)'r' << 32 |          \
   (uint64_t)'o' << 24 | (uint64_t)'f' << 16 | (uint64_t)'r' << 8 | (uint64_t)129)

// The version number of the raw binary format.
#define MEMPROF_RAW_VERSION 1ULL

namespace llvm {
namespace memprof {
// A struct describing the header used for the raw binary memprof profile format.
PACKED(struct Header {
  uint64_t Magic;
  uint64_t Version;
  uint64_t TotalSize;
  uint64_t SegmentOffset;
  uint64_t MIBOffset;
  uint64_t StackOffset;
});

// A struct describing the information necessary to describe a /proc/maps
// segment entry for a particular binary/library identified by its build id.
PACKED(struct SegmentEntry {
  uint64_t Start;
  uint64_t End;
  uint64_t Offset;
  uint8_t BuildId[32];
});
} // namespace memprof
} // namespace llvm

#endif
