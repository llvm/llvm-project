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

#include <iosfwd>
#include <string>

namespace llvm {
  class Module;
  class ModuleProvider;
  class MemoryBuffer;
  class ModulePass;
  class BitstreamWriter;
  class raw_ostream;
  
  /// getBitcodeModuleProvider - Read the header of the specified bitcode buffer
  /// and prepare for lazy deserialization of function bodies.  If successful,
  /// this takes ownership of 'buffer' and returns a non-null pointer.  On
  /// error, this returns null, *does not* take ownership of Buffer, and fills
  /// in *ErrMsg with an error description if ErrMsg is non-null.
  ModuleProvider *getBitcodeModuleProvider(MemoryBuffer *Buffer,
                                           std::string *ErrMsg = 0);

  /// ParseBitcodeFile - Read the specified bitcode file, returning the module.
  /// If an error occurs, this returns null and fills in *ErrMsg if it is
  /// non-null.  This method *never* takes ownership of Buffer.
  Module *ParseBitcodeFile(MemoryBuffer *Buffer, std::string *ErrMsg = 0);
  
  /// WriteBitcodeToFile - Write the specified module to the specified output
  /// stream.
  void WriteBitcodeToFile(const Module *M, std::ostream &Out);
  
  /// WriteBitcodeToFile - Write the specified module to the specified
  /// raw output stream.
  void WriteBitcodeToFile(const Module *M, raw_ostream &Out);

  /// WriteBitcodeToStream - Write the specified module to the specified
  /// raw output stream.
  void WriteBitcodeToStream(const Module *M, BitstreamWriter &Stream);

  /// CreateBitcodeWriterPass - Create and return a pass that writes the module
  /// to the specified ostream.
  ModulePass *CreateBitcodeWriterPass(std::ostream &Str);

  /// createBitcodeWriterPass - Create and return a pass that writes the module
  /// to the specified ostream.
  ModulePass *createBitcodeWriterPass(raw_ostream &Str);
} // End llvm namespace

#endif
