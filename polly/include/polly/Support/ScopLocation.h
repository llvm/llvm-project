//=== ScopLocation.h -- Debug location helper for ScopDetection -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
