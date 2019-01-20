//===- CoverageMappingWriter.h - Code coverage mapping writer ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing coverage mapping data for
// instrumentation based coverage.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_COVERAGE_COVERAGEMAPPINGWRITER_H
#define LLVM_PROFILEDATA_COVERAGE_COVERAGEMAPPINGWRITER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ProfileData/Coverage/CoverageMapping.h"

namespace llvm {

class raw_ostream;

namespace coverage {

/// Writer of the filenames section for the instrumentation
/// based code coverage.
class CoverageFilenamesSectionWriter {
  ArrayRef<StringRef> Filenames;

public:
  CoverageFilenamesSectionWriter(ArrayRef<StringRef> Filenames)
      : Filenames(Filenames) {}

  /// Write encoded filenames to the given output stream.
  void write(raw_ostream &OS);
};

/// Writer for instrumentation based coverage mapping data.
class CoverageMappingWriter {
  ArrayRef<unsigned> VirtualFileMapping;
  ArrayRef<CounterExpression> Expressions;
  MutableArrayRef<CounterMappingRegion> MappingRegions;

public:
  CoverageMappingWriter(ArrayRef<unsigned> VirtualFileMapping,
                        ArrayRef<CounterExpression> Expressions,
                        MutableArrayRef<CounterMappingRegion> MappingRegions)
      : VirtualFileMapping(VirtualFileMapping), Expressions(Expressions),
        MappingRegions(MappingRegions) {}

  /// Write encoded coverage mapping data to the given output stream.
  void write(raw_ostream &OS);
};

} // end namespace coverage

} // end namespace llvm

#endif // LLVM_PROFILEDATA_COVERAGE_COVERAGEMAPPINGWRITER_H
