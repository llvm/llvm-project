//===- MSFStreamLayout.h - Describes the layout of a stream -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_MSF_MSFSTREAMLAYOUT_H
#define LLVM_DEBUGINFO_MSF_MSFSTREAMLAYOUT_H

#include "llvm/Support/Endian.h"

#include <cstdint>
#include <vector>

namespace llvm {
namespace msf {

/// \brief Describes the layout of a stream in an MSF layout.  A "stream" here
/// is defined as any logical unit of data which may be arranged inside the MSF
/// file as a sequence of (possibly discontiguous) blocks.  When we want to read
/// from a particular MSF Stream, we fill out a stream layout structure and the
/// reader uses it to determine which blocks in the underlying MSF file contain
/// the data, so that it can be pieced together in the right order.
class MSFStreamLayout {
public:
  uint32_t Length;
  std::vector<support::ulittle32_t> Blocks;
};
} // namespace msf
} // namespace llvm

#endif // LLVM_DEBUGINFO_MSF_MSFSTREAMLAYOUT_H
