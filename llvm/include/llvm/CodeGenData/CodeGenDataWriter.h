//===- CodeGenDataWriter.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing codegen data.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGENDATA_CODEGENDATAWRITER_H
#define LLVM_CODEGENDATA_CODEGENDATAWRITER_H

#include "llvm/CodeGenData/CodeGenData.h"
#include "llvm/CodeGenData/OutlinedHashTreeRecord.h"
#include "llvm/Support/Error.h"

namespace llvm {

class CGDataOStream;

class CodeGenDataWriter {
  /// The outlined hash tree to be written.
  OutlinedHashTreeRecord HashTreeRecord;

  /// A bit mask describing the kind of the codegen data.
  CGDataKind DataKind = CGDataKind::Unknown;

public:
  CodeGenDataWriter() = default;
  ~CodeGenDataWriter() = default;

  /// Add the outlined hash tree record. The input Record is released.
  void addRecord(OutlinedHashTreeRecord &Record);

  /// Write the codegen data to \c OS
  Error write(raw_fd_ostream &OS);

  /// Write the codegen data in text format to \c OS
  Error writeText(raw_fd_ostream &OS);

  /// Return the attributes of the current CGData.
  CGDataKind getCGDataKind() const { return DataKind; }

  /// Return true if the header indicates the data has an outlined hash tree.
  bool hasOutlinedHashTree() const {
    return static_cast<uint32_t>(DataKind) &
           static_cast<uint32_t>(CGDataKind::FunctionOutlinedHashTree);
  }

private:
  /// The offset of the outlined hash tree in the file.
  uint64_t OutlinedHashTreeOffset;

  /// Write the codegen data header to \c COS
  Error writeHeader(CGDataOStream &COS);

  /// Write the codegen data header in text to \c OS
  Error writeHeaderText(raw_fd_ostream &OS);

  Error writeImpl(CGDataOStream &COS);
};

} // end namespace llvm

#endif // LLVM_CODEGENDATA_CODEGENDATAWRITER_H
