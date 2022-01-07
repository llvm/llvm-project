#ifndef LLVM_PROFILEDATA_RAWMEMPROFREADER_H_
#define LLVM_PROFILEDATA_RAWMEMPROFREADER_H_
//===- MemProfReader.h - Instrumented memory profiling reader ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for reading MemProf profiling data.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {
namespace memprof {

class RawMemProfReader {
public:
  RawMemProfReader(std::unique_ptr<MemoryBuffer> DataBuffer)
      : DataBuffer(std::move(DataBuffer)) {}
  // Prints the contents of the profile in YAML format.
  void printYAML(raw_ostream &OS);

  // Return true if the \p DataBuffer starts with magic bytes indicating it is
  // a raw binary memprof profile.
  static bool hasFormat(const MemoryBuffer &DataBuffer);

  // Create a RawMemProfReader after sanity checking the contents of the file at
  // \p Path.
  static Expected<std::unique_ptr<RawMemProfReader>> create(const Twine &Path);

private:
  // Prints aggregate counts for each raw profile parsed from the DataBuffer in
  // YAML format.
  void printSummaries(raw_ostream &OS) const;

  std::unique_ptr<MemoryBuffer> DataBuffer;
};

} // namespace memprof
} // namespace llvm

#endif // LLVM_PROFILEDATA_RAWMEMPROFREADER_H_
