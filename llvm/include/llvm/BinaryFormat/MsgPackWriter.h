//===- MsgPackWriter.h - Simple MsgPack writer ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
///  \file
///  This file contains a MessagePack writer.
///
///  See https://github.com/msgpack/msgpack/blob/master/spec.md for the full
///  specification.
///
///  Typical usage:
///  \code
///  raw_ostream output = GetOutputStream();
///  msgpack::Writer MPWriter(output);
///  MPWriter.writeNil();
///  MPWriter.write(false);
///  MPWriter.write("string");
///  // ...
///  \endcode
///
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_BINARYFORMAT_MSGPACKWRITER_H
#define LLVM_BINARYFORMAT_MSGPACKWRITER_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/MemoryBufferRef.h"

namespace llvm {

class raw_ostream;

namespace msgpack {

/// Writes MessagePack objects to an output stream, one at a time.
class Writer {
public:
  /// Construct a writer, optionally enabling "Compatibility Mode" as defined
  /// in the MessagePack specification.
  ///
  /// When in \p Compatible mode, the writer will write \c Str16 formats
  /// instead of \c Str8 formats, and will refuse to write any \c Bin formats.
  ///
  /// \param OS stream to output MessagePack objects to.
  /// \param Compatible when set, write in "Compatibility Mode".
  LLVM_ABI Writer(raw_ostream &OS, bool Compatible = false);

  Writer(const Writer &) = delete;
  Writer &operator=(const Writer &) = delete;

  /// Write a \em Nil to the output stream.
  ///
  /// The output will be the \em nil format.
  LLVM_ABI void writeNil();

  /// Write a \em Boolean to the output stream.
  ///
  /// The output will be a \em bool format.
  LLVM_ABI void write(bool b);

  /// Write a signed integer to the output stream.
  ///
  /// The output will be in the smallest possible \em int format.
  ///
  /// The format chosen may be for an unsigned integer.
  LLVM_ABI void write(int64_t i);

  /// Write an unsigned integer to the output stream.
  ///
  /// The output will be in the smallest possible \em int format.
  LLVM_ABI void write(uint64_t u);

  /// Write a floating point number to the output stream.
  ///
  /// The output will be in the smallest possible \em float format.
  LLVM_ABI void write(double d);

  /// Write a string to the output stream.
  ///
  /// The output will be in the smallest possible \em str format.
  LLVM_ABI void write(StringRef s);

  /// Write a memory buffer to the output stream.
  ///
  /// The output will be in the smallest possible \em bin format.
  ///
  /// \warning Do not use this overload if in \c Compatible mode.
  LLVM_ABI void write(MemoryBufferRef Buffer);

  /// Write the header for an \em Array of the given size.
  ///
  /// The output will be in the smallest possible \em array format.
  //
  /// The header contains an identifier for the \em array format used, as well
  /// as an encoding of the size of the array.
  ///
  /// N.B. The caller must subsequently call \c Write an additional \p Size
  /// times to complete the array.
  LLVM_ABI void writeArraySize(uint32_t Size);

  /// Write the header for a \em Map of the given size.
  ///
  /// The output will be in the smallest possible \em map format.
  //
  /// The header contains an identifier for the \em map format used, as well
  /// as an encoding of the size of the map.
  ///
  /// N.B. The caller must subsequently call \c Write and additional \c Size*2
  /// times to complete the map. Each even numbered call to \c Write defines a
  /// new key, and each odd numbered call defines the previous key's value.
  LLVM_ABI void writeMapSize(uint32_t Size);

  /// Write a typed memory buffer (an extension type) to the output stream.
  ///
  /// The output will be in the smallest possible \em ext format.
  LLVM_ABI void writeExt(int8_t Type, MemoryBufferRef Buffer);

private:
  support::endian::Writer EW;
  bool Compatible;
};

} // end namespace msgpack
} // end namespace llvm

#endif // LLVM_BINARYFORMAT_MSGPACKWRITER_H
