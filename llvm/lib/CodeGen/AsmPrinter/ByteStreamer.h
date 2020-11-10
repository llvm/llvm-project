//===-- llvm/CodeGen/ByteStreamer.h - ByteStreamer class --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a class that can take bytes that would normally be
// streamed via the AsmPrinter.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_BYTESTREAMER_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_BYTESTREAMER_H

#include "DIEHash.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/LEB128.h"
#include <string>

namespace llvm {
class ByteStreamer {
 protected:
  ~ByteStreamer() = default;
  ByteStreamer(const ByteStreamer&) = default;
  ByteStreamer() = default;

 public:
  // For now we're just handling the calls we need for dwarf emission/hashing.
  virtual void emitInt8(uint8_t Byte, const Twine &Comment = "") = 0;
  virtual void emitSLEB128(uint64_t DWord, const Twine &Comment = "") = 0;
  virtual void emitULEB128(uint64_t DWord, const Twine &Comment = "",
                           unsigned PadTo = 0) = 0;
};

class APByteStreamer final : public ByteStreamer {
private:
  AsmPrinter &AP;

public:
  APByteStreamer(AsmPrinter &Asm) : AP(Asm) {}
  void emitInt8(uint8_t Byte, const Twine &Comment) override {
    AP.OutStreamer->AddComment(Comment);
    AP.emitInt8(Byte);
  }
  void emitSLEB128(uint64_t DWord, const Twine &Comment) override {
    AP.OutStreamer->AddComment(Comment);
    AP.emitSLEB128(DWord);
  }
  void emitULEB128(uint64_t DWord, const Twine &Comment,
                   unsigned PadTo) override {
    AP.OutStreamer->AddComment(Comment);
    AP.emitULEB128(DWord, nullptr, PadTo);
  }
};

class HashingByteStreamer final : public ByteStreamer {
 private:
  DIEHash &Hash;
 public:
 HashingByteStreamer(DIEHash &H) : Hash(H) {}
  void emitInt8(uint8_t Byte, const Twine &Comment) override {
    Hash.update(Byte);
  }
  void emitSLEB128(uint64_t DWord, const Twine &Comment) override {
    Hash.addSLEB128(DWord);
  }
  void emitULEB128(uint64_t DWord, const Twine &Comment,
                   unsigned PadTo) override {
    Hash.addULEB128(DWord);
  }
};

class BufferByteStreamer final : public ByteStreamer {
private:
  SmallVectorImpl<char> &Buffer;
  std::vector<std::string> &Comments;

public:
  /// Only verbose textual output needs comments.  This will be set to
  /// true for that case, and false otherwise.  If false, comments passed in to
  /// the emit methods will be ignored.
  const bool GenerateComments;

  BufferByteStreamer(SmallVectorImpl<char> &Buffer,
                     std::vector<std::string> &Comments, bool GenerateComments)
      : Buffer(Buffer), Comments(Comments), GenerateComments(GenerateComments) {
  }
  void emitInt8(uint8_t Byte, const Twine &Comment) override {
    Buffer.push_back(Byte);
    if (GenerateComments)
      Comments.push_back(Comment.str());
  }
  void emitSLEB128(uint64_t DWord, const Twine &Comment) override {
    raw_svector_ostream OSE(Buffer);
    unsigned Length = encodeSLEB128(DWord, OSE);
    if (GenerateComments) {
      Comments.push_back(Comment.str());
      // Add some empty comments to keep the Buffer and Comments vectors aligned
      // with each other.
      for (size_t i = 1; i < Length; ++i)
        Comments.push_back("");

    }
  }
  void emitULEB128(uint64_t DWord, const Twine &Comment,
                   unsigned PadTo) override {
    raw_svector_ostream OSE(Buffer);
    unsigned Length = encodeULEB128(DWord, OSE, PadTo);
    if (GenerateComments) {
      Comments.push_back(Comment.str());
      // Add some empty comments to keep the Buffer and Comments vectors aligned
      // with each other.
      for (size_t i = 1; i < Length; ++i)
        Comments.push_back("");

    }
  }
};

}

#endif
