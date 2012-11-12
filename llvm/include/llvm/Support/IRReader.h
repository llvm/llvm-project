//===---- llvm/Support/IRReader.h - Reader for LLVM IR files ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines functions for reading LLVM IR. They support both
// Bitcode and Assembly, automatically detecting the input format.
//
// These functions must be defined in a header file in order to avoid
// library dependencies, since they reference both Bitcode and Assembly
// functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_IRREADER_H
#define LLVM_SUPPORT_IRREADER_H

#include "llvm/ADT/OwningPtr.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/system_error.h"

namespace llvm {

  /// If the given MemoryBuffer holds a bitcode image, return a Module for it
  /// which does lazy deserialization of function bodies.  Otherwise, attempt to
  /// parse it as LLVM Assembly and return a fully populated Module. This
  /// function *always* takes ownership of the given MemoryBuffer.
  inline Module *getLazyIRModule(MemoryBuffer *Buffer,
                                 SMDiagnostic &Err,
                                 LLVMContext &Context) {
    if (isBitcode((const unsigned char *)Buffer->getBufferStart(),
                  (const unsigned char *)Buffer->getBufferEnd())) {
      std::string ErrMsg;
      Module *M = getLazyBitcodeModule(Buffer, Context, &ErrMsg);
      if (M == 0) {
        Err = SMDiagnostic(Buffer->getBufferIdentifier(), SourceMgr::DK_Error,
                           ErrMsg);
        // ParseBitcodeFile does not take ownership of the Buffer in the
        // case of an error.
        delete Buffer;
      }
      return M;
    }

    return ParseAssembly(Buffer, 0, Err, Context);
  }

  /// If the given file holds a bitcode image, return a Module
  /// for it which does lazy deserialization of function bodies.  Otherwise,
  /// attempt to parse it as LLVM Assembly and return a fully populated
  /// Module.
  inline Module *getLazyIRFileModule(const std::string &Filename,
                                     SMDiagnostic &Err,
                                     LLVMContext &Context) {
    OwningPtr<MemoryBuffer> File;
    if (error_code ec = MemoryBuffer::getFileOrSTDIN(Filename.c_str(), File)) {
      Err = SMDiagnostic(Filename, SourceMgr::DK_Error,
                         "Could not open input file: " + ec.message());
      return 0;
    }

    return getLazyIRModule(File.take(), Err, Context);
  }

  /// If the given MemoryBuffer holds a bitcode image, return a Module
  /// for it.  Otherwise, attempt to parse it as LLVM Assembly and return
  /// a Module for it. This function *always* takes ownership of the given
  /// MemoryBuffer.
  inline Module *ParseIR(MemoryBuffer *Buffer,
                         SMDiagnostic &Err,
                         LLVMContext &Context) {
    if (isBitcode((const unsigned char *)Buffer->getBufferStart(),
                  (const unsigned char *)Buffer->getBufferEnd())) {
      std::string ErrMsg;
      Module *M = ParseBitcodeFile(Buffer, Context, &ErrMsg);
      if (M == 0)
        Err = SMDiagnostic(Buffer->getBufferIdentifier(), SourceMgr::DK_Error,
                           ErrMsg);
      // ParseBitcodeFile does not take ownership of the Buffer.
      delete Buffer;
      return M;
    }

    return ParseAssembly(Buffer, 0, Err, Context);
  }

  /// If the given file holds a bitcode image, return a Module for it.
  /// Otherwise, attempt to parse it as LLVM Assembly and return a Module
  /// for it.
  inline Module *ParseIRFile(const std::string &Filename,
                             SMDiagnostic &Err,
                             LLVMContext &Context) {
    OwningPtr<MemoryBuffer> File;
    if (error_code ec = MemoryBuffer::getFileOrSTDIN(Filename.c_str(), File)) {
      Err = SMDiagnostic(Filename, SourceMgr::DK_Error,
                         "Could not open input file: " + ec.message());
      return 0;
    }

    return ParseIR(File.take(), Err, Context);
  }

}

#endif
