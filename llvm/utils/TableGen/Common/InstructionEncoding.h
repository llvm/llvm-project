//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_COMMON_INSTRUCTIONENCODING_H
#define LLVM_UTILS_TABLEGEN_COMMON_INSTRUCTIONENCODING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/KnownBits.h"
#include <optional>
#include <string>
#include <vector>

namespace llvm {

class BitsInit;
class CodeGenInstruction;
class CodeGenTarget;
class Record;
class RecordVal;
class VarLenInst;

// Represents a span of bits in the instruction encoding that's based on a span
// of bits in an operand's encoding.
//
// Width is the width of the span.
// Base is the starting position of that span in the instruction encoding.
// Offset if the starting position of that span in the operand's encoding.
// That is, bits {Base + Width - 1, Base} in the instruction encoding form
// bits {Offset + Width - 1, Offset} in the operands encoding.
struct EncodingField {
  unsigned Base, Width, Offset;
  EncodingField(unsigned B, unsigned W, unsigned O)
      : Base(B), Width(W), Offset(O) {}
};

struct OperandInfo {
  StringRef Name;
  bool HasNoEncoding = false;
  std::vector<EncodingField> Fields;
  std::string Decoder;
  bool HasCompleteDecoder;
  std::optional<uint64_t> InitValue;

  OperandInfo(std::string D, bool HCD) : Decoder(D), HasCompleteDecoder(HCD) {}

  void addField(unsigned Base, unsigned Width, unsigned Offset) {
    Fields.emplace_back(Base, Width, Offset);
  }

  ArrayRef<EncodingField> fields() const { return Fields; }
};

/// Represents a parsed InstructionEncoding record or a record derived from it.
class InstructionEncoding {
  /// The Record this encoding originates from.
  const Record *EncodingDef;

  /// The instruction this encoding is for.
  const CodeGenInstruction *Inst;

  /// The name of this encoding (for debugging purposes).
  std::string Name;

  /// The namespace in which this encoding exists.
  StringRef DecoderNamespace;

  /// Known bits of this encoding. This is the value of the `Inst` field
  /// with any variable references replaced with '?'.
  KnownBits InstBits;

  /// Mask of bits that should be considered unknown during decoding.
  /// This is the value of the `SoftFail` field.
  APInt SoftFailMask;

  /// The name of the function to use for decoding. May be an empty string,
  /// meaning the decoder is generated.
  StringRef DecoderMethod;

  /// Whether the custom decoding function always succeeds. If a custom decoder
  /// function is specified, the value is taken from the target description,
  /// otherwise it is inferred.
  bool HasCompleteDecoder;

  /// Information about the operands' contribution to this encoding.
  SmallVector<OperandInfo, 16> Operands;

public:
  InstructionEncoding(const Record *EncodingDef,
                      const CodeGenInstruction *Inst);

  /// Returns the Record this encoding originates from.
  const Record *getRecord() const { return EncodingDef; }

  /// Returns the instruction this encoding is for.
  const CodeGenInstruction *getInstruction() const { return Inst; }

  /// Returns the name of this encoding, for debugging purposes.
  StringRef getName() const { return Name; }

  /// Returns the namespace in which this encoding exists.
  StringRef getDecoderNamespace() const { return DecoderNamespace; }

  /// Returns the size of this encoding, in bits.
  unsigned getBitWidth() const { return InstBits.getBitWidth(); }

  /// Returns the known bits of this encoding.
  const KnownBits &getInstBits() const { return InstBits; }

  /// Returns a mask of bits that should be considered unknown during decoding.
  const APInt &getSoftFailMask() const { return SoftFailMask; }

  /// Returns the known bits of this encoding that must match for
  /// successful decoding.
  KnownBits getMandatoryBits() const {
    KnownBits EncodingBits = InstBits;
    // Mark all bits that are allowed to change according to SoftFail mask
    // as unknown.
    EncodingBits.Zero &= ~SoftFailMask;
    EncodingBits.One &= ~SoftFailMask;
    return EncodingBits;
  }

  /// Returns the name of the function to use for decoding, or an empty string
  /// if the decoder is generated.
  StringRef getDecoderMethod() const { return DecoderMethod; }

  /// Returns whether the decoder (either generated or specified by the user)
  /// always succeeds.
  bool hasCompleteDecoder() const { return HasCompleteDecoder; }

  /// Returns information about the operands' contribution to this encoding.
  ArrayRef<OperandInfo> getOperands() const { return Operands; }

  /// \returns the effective value of the DecoderMethod field. If DecoderMethod
  /// is an explictly set value, return false for second.
  static std::pair<std::string, bool>
  findOperandDecoderMethod(const Record *Record);

  static OperandInfo getOpInfo(const Record *TypeRecord);

private:
  void parseVarLenEncoding(const VarLenInst &VLI);
  void parseFixedLenEncoding(const BitsInit &RecordInstBits);

  void parseVarLenOperands(const VarLenInst &VLI);
  void parseFixedLenOperands(const BitsInit &Bits);
};

} // namespace llvm

#endif // LLVM_UTILS_TABLEGEN_COMMON_INSTRUCTIONENCODING_H
