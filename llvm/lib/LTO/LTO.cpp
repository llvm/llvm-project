//===-LTO.cpp - LLVM Link Time Optimizer ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements functions and classes used to support LTO.
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/LTO.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

// Simple helper to load a module from bitcode
std::unique_ptr<Module> loadModuleFromBuffer(const MemoryBufferRef &Buffer,
                                             LLVMContext &Context, bool Lazy) {
  SMDiagnostic Err;
  ErrorOr<std::unique_ptr<Module>> ModuleOrErr(nullptr);
  if (Lazy) {
    ModuleOrErr =
        getLazyBitcodeModule(MemoryBuffer::getMemBuffer(Buffer, false), Context,
                             /* ShouldLazyLoadMetadata */ Lazy);
  } else {
    ModuleOrErr = parseBitcodeFile(Buffer, Context);
  }
  if (std::error_code EC = ModuleOrErr.getError()) {
    Err = SMDiagnostic(Buffer.getBufferIdentifier(), SourceMgr::DK_Error,
                       EC.message());
    Err.print("ThinLTO", errs());
    report_fatal_error("Can't load module, abort.");
  }
  return std::move(ModuleOrErr.get());
}
}
