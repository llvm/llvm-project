//===-- YAMLRemarkSerializer.h - YAML Remark serialization ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides an interface for serializing remarks to YAML.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_REMARKS_YAML_REMARK_SERIALIZER_H
#define LLVM_REMARKS_YAML_REMARK_SERIALIZER_H

#include "llvm/Remarks/RemarkSerializer.h"
#include "llvm/Support/YAMLTraits.h"

namespace llvm {
namespace remarks {

/// Serialize the remarks to YAML. One remark entry looks like this:
/// --- !<TYPE>
/// Pass:            <PASSNAME>
/// Name:            <REMARKNAME>
/// DebugLoc:        { File: <SOURCEFILENAME>, Line: <SOURCELINE>,
///                    Column: <SOURCECOLUMN> }
/// Function:        <FUNCTIONNAME>
/// Args:
///   - <KEY>: <VALUE>
///     DebugLoc:        { File: <FILE>, Line: <LINE>, Column: <COL> }
/// ...
struct YAMLSerializer : public Serializer {
  /// The YAML streamer.
  yaml::Output YAMLOutput;

  YAMLSerializer(raw_ostream &OS,
                 UseStringTable UseStringTable = remarks::UseStringTable::No);

  /// Emit a remark to the stream.
  void emit(const Remark &Remark) override;
};

} // end namespace remarks
} // end namespace llvm

#endif /* LLVM_REMARKS_REMARK_SERIALIZER_H */
