//===-- LLVMTargetMachineC.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVM-C part of TargetMachine.h that directly
// depends on the CodeGen library.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Core.h"
#include "llvm-c/TargetMachine.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

static TargetMachine *unwrap(LLVMTargetMachineRef P) {
  return reinterpret_cast<TargetMachine *>(P);
}

static Target *unwrap(LLVMTargetRef P) { return reinterpret_cast<Target *>(P); }

static LLVMTargetMachineRef wrap(const TargetMachine *P) {
  return reinterpret_cast<LLVMTargetMachineRef>(const_cast<TargetMachine *>(P));
}

static LLVMTargetRef wrap(const Target *P) {
  return reinterpret_cast<LLVMTargetRef>(const_cast<Target *>(P));
}

static LLVMBool LLVMTargetMachineEmit(LLVMTargetMachineRef T, LLVMModuleRef M,
                                      raw_pwrite_stream &OS,
                                      LLVMCodeGenFileType codegen,
                                      char **ErrorMessage) {
  TargetMachine *TM = unwrap(T);
  Module *Mod = unwrap(M);

  legacy::PassManager pass;
  MachineModuleInfo MMI(static_cast<LLVMTargetMachine *>(TM));

  std::string error;

  Mod->setDataLayout(TM->createDataLayout());

  CodeGenFileType ft;
  switch (codegen) {
  case LLVMAssemblyFile:
    ft = CodeGenFileType::AssemblyFile;
    break;
  default:
    ft = CodeGenFileType::ObjectFile;
    break;
  }
  if (TM->addPassesToEmitFile(pass, MMI, OS, nullptr, ft)) {
    error = "TargetMachine can't emit a file of this type";
    *ErrorMessage = strdup(error.c_str());
    return true;
  }

  pass.run(*Mod);

  OS.flush();
  return false;
}

LLVMBool LLVMTargetMachineEmitToFile(LLVMTargetMachineRef T, LLVMModuleRef M,
                                     const char *Filename,
                                     LLVMCodeGenFileType codegen,
                                     char **ErrorMessage) {
  std::error_code EC;
  raw_fd_ostream dest(Filename, EC, sys::fs::OF_None);
  if (EC) {
    *ErrorMessage = strdup(EC.message().c_str());
    return true;
  }
  bool Result = LLVMTargetMachineEmit(T, M, dest, codegen, ErrorMessage);
  dest.flush();
  return Result;
}

LLVMBool LLVMTargetMachineEmitToMemoryBuffer(LLVMTargetMachineRef T,
                                             LLVMModuleRef M,
                                             LLVMCodeGenFileType codegen,
                                             char **ErrorMessage,
                                             LLVMMemoryBufferRef *OutMemBuf) {
  SmallString<0> CodeString;
  raw_svector_ostream OStream(CodeString);
  bool Result = LLVMTargetMachineEmit(T, M, OStream, codegen, ErrorMessage);

  StringRef Data = OStream.str();
  *OutMemBuf =
      LLVMCreateMemoryBufferWithMemoryRangeCopy(Data.data(), Data.size(), "");
  return Result;
}
