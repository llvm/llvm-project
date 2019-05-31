//===-- RemarkSerializer.h - Remark serialization interface -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides an interface for serializing remarks to different formats.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_REMARKS_REMARK_SERIALIZER_H
#define LLVM_REMARKS_REMARK_SERIALIZER_H

#include "llvm/Remarks/Remark.h"
#include "llvm/Remarks/RemarkStringTable.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace remarks {

/// This is the base class for a remark serializer.
/// It includes support for using a string table while emitting.
struct Serializer {
  /// The open raw_ostream that the remark diagnostics are emitted to.
  raw_ostream &OS;
  /// The string table containing all the unique strings used in the output.
  /// The table can be serialized to be consumed after the compilation.
  Optional<StringTable> StrTab;

  Serializer(raw_ostream &OS) : OS(OS), StrTab() {}

  /// This is just an interface.
  virtual ~Serializer() = default;
  virtual void emit(const Remark &Remark) = 0;
};

/// Wether the serializer should use a string table while emitting.
enum class UseStringTable { No, Yes };

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
