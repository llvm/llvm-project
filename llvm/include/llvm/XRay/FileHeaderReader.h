//===- FileHeaderReader.h - XRay Trace File Header Reading Function -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares functions that can load an XRay log header from various
// sources.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_XRAY_FILEHEADERREADER_H
#define LLVM_XRAY_FILEHEADERREADER_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Error.h"
#include "llvm/XRay/XRayRecord.h"
#include <cstdint>

namespace llvm::xray {

/// Convenience function for loading the file header given a data extractor at a
/// specified offset.
LLVM_ABI Expected<XRayFileHeader>
readBinaryFormatHeader(DataExtractor &HeaderExtractor, uint64_t &OffsetPtr);

} // namespace llvm::xray

#endif // LLVM_XRAY_FILEHEADERREADER_H
