//===-- llvm/Bitcode/ReaderWriter.h - Bitcode reader/writers ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines interfaces to read and write LLVM bitcode files/streams.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BITCODE_H
#define LLVM_BITCODE_H

#include <string>

namespace llvm {
  class BitstreamWriter;
  class MemoryBuffer;
  class DataStreamer;
  class LLVMContext;
  class Module;
  class ModulePass;
  class raw_ostream;

  /// getLazyBitcodeModule - Read the header of the specified bitcode buffer
  /// and prepare for lazy deserialization of function bodies.  If successful,
  /// this takes ownership of 'buffer' and returns a non-null pointer.  On
  /// error, this returns null, *does not* take ownership of Buffer, and fills
  /// in *ErrMsg with an error description if ErrMsg is non-null.
  Module *getLazyBitcodeModule(MemoryBuffer *Buffer,
                               LLVMContext &Context,
                               std::string *ErrMsg = 0);

  /// getStreamedBitcodeModule - Read the header of the specified stream
  /// and prepare for lazy deserialization and streaming of function bodies.
  /// On error, this returns null, and fills in *ErrMsg with an error
  /// description if ErrMsg is non-null.
  Module *getStreamedBitcodeModule(const std::string &name,
                                   DataStreamer *streamer,
                                   LLVMContext &Context,
                                   std::string *ErrMsg = 0);

  /// getBitcodeTargetTriple - Read the header of the specified bitcode
  /// buffer and extract just the triple information. If successful,
  /// this returns a string and *does not* take ownership
  /// of 'buffer'. On error, this returns "", and fills in *ErrMsg
  /// if ErrMsg is non-null.
  std::string getBitcodeTargetTriple(MemoryBuffer *Buffer,
                                     LLVMContext &Context,
                                     std::string *ErrMsg = 0);

  /// ParseBitcodeFile - Read the specified bitcode file, returning the module.
  /// If an error occurs, this returns null and fills in *ErrMsg if it is
  /// non-null.  This method *never* takes ownership of Buffer.
  Module *ParseBitcodeFile(MemoryBuffer *Buffer, LLVMContext &Context,
                           std::string *ErrMsg = 0);

  /// WriteBitcodeToFile - Write the specified module to the specified
  /// raw output stream.  For streams where it matters, the given stream
  /// should be in "binary" mode.
  void WriteBitcodeToFile(const Module *M, raw_ostream &Out);

  /// createBitcodeWriterPass - Create and return a pass that writes the module
  /// to the specified ostream.
  ModulePass *createBitcodeWriterPass(raw_ostream &Str);


  /// isBitcodeWrapper - Return true if the given bytes are the magic bytes
  /// for an LLVM IR bitcode wrapper.
  ///
  inline bool isBitcodeWrapper(const unsigned char *BufPtr,
                               const unsigned char *BufEnd) {
    // See if you can find the hidden message in the magic bytes :-).
    // (Hint: it's a little-endian encoding.)
    return BufPtr != BufEnd &&
           BufPtr[0] == 0xDE &&
           BufPtr[1] == 0xC0 &&
           BufPtr[2] == 0x17 &&
           BufPtr[3] == 0x0B;
  }

  /// isRawBitcode - Return true if the given bytes are the magic bytes for
  /// raw LLVM IR bitcode (without a wrapper).
  ///
  inline bool isRawBitcode(const unsigned char *BufPtr,
                           const unsigned char *BufEnd) {
    // These bytes sort of have a hidden message, but it's not in
    // little-endian this time, and it's a little redundant.
    return BufPtr != BufEnd &&
           BufPtr[0] == 'B' &&
           BufPtr[1] == 'C' &&
           BufPtr[2] == 0xc0 &&
           BufPtr[3] == 0xde;
  }

  /// isBitcode - Return true if the given bytes are the magic bytes for
  /// LLVM IR bitcode, either with or without a wrapper.
  ///
  inline bool isBitcode(const unsigned char *BufPtr,
                        const unsigned char *BufEnd) {
    return isBitcodeWrapper(BufPtr, BufEnd) ||
           isRawBitcode(BufPtr, BufEnd);
  }

  /// SkipBitcodeWrapperHeader - Some systems wrap bc files with a special
  /// header for padding or other reasons.  The format of this header is:
  ///
  /// struct bc_header {
  ///   uint32_t Magic;         // 0x0B17C0DE
  ///   uint32_t Version;       // Version, currently always 0.
  ///   uint32_t BitcodeOffset; // Offset to traditional bitcode file.
  ///   uint32_t BitcodeSize;   // Size of traditional bitcode file.
  ///   ... potentially other gunk ...
  /// };
  ///
  /// This function is called when we find a file with a matching magic number.
  /// In this case, skip down to the subsection of the file that is actually a
  /// BC file.
  /// If 'VerifyBufferSize' is true, check that the buffer is large enough to
  /// contain the whole bitcode file.
  inline bool SkipBitcodeWrapperHeader(const unsigned char *&BufPtr,
                                       const unsigned char *&BufEnd,
                                       bool VerifyBufferSize) {
    enum {
      KnownHeaderSize = 4*4,  // Size of header we read.
      OffsetField = 2*4,      // Offset in bytes to Offset field.
      SizeField = 3*4         // Offset in bytes to Size field.
    };

    // Must contain the header!
    if (BufEnd-BufPtr < KnownHeaderSize) return true;

    unsigned Offset = ( BufPtr[OffsetField  ]        |
                       (BufPtr[OffsetField+1] << 8)  |
                       (BufPtr[OffsetField+2] << 16) |
                       (BufPtr[OffsetField+3] << 24));
    unsigned Size   = ( BufPtr[SizeField    ]        |
                       (BufPtr[SizeField  +1] << 8)  |
                       (BufPtr[SizeField  +2] << 16) |
                       (BufPtr[SizeField  +3] << 24));

    // Verify that Offset+Size fits in the file.
    if (VerifyBufferSize && Offset+Size > unsigned(BufEnd-BufPtr))
      return true;
    BufPtr += Offset;
    BufEnd = BufPtr+Size;
    return false;
  }
} // End llvm namespace

#endif
