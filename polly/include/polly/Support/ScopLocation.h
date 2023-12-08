//=== ScopLocation.h -- Debug location helper for ScopDetection -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper function for extracting region debug information.
//
//===----------------------------------------------------------------------===//
//
#ifndef POLLY_SCOP_LOCATION_H
#define POLLY_SCOP_LOCATION_H

#include <string>

namespace llvm {
class Region;
} // namespace llvm

namespace polly {

/// Get the location of a region from the debug info.
///
/// @param R The region to get debug info for.
/// @param LineBegin The first line in the region.
/// @param LineEnd The last line in the region.
/// @param FileName The filename where the region was defined.
void getDebugLocation(const llvm::Region *R, unsigned &LineBegin,
                      unsigned &LineEnd, std::string &FileName);
} // namespace polly

#endif // POLLY_SCOP_LOCATION_H
