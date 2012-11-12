//===-- BitWriter.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/BitWriter.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;


/*===-- Operations on modules ---------------------------------------------===*/

int LLVMWriteBitcodeToFile(LLVMModuleRef M, const char *Path) {
  std::string ErrorInfo;
  raw_fd_ostream OS(Path, ErrorInfo,
                    raw_fd_ostream::F_Binary);
  
  if (!ErrorInfo.empty())
    return -1;
  
  WriteBitcodeToFile(unwrap(M), OS);
  return 0;
}

int LLVMWriteBitcodeToFD(LLVMModuleRef M, int FD, int ShouldClose,
                         int Unbuffered) {
  raw_fd_ostream OS(FD, ShouldClose, Unbuffered);
  
  WriteBitcodeToFile(unwrap(M), OS);
  return 0;
}

int LLVMWriteBitcodeToFileHandle(LLVMModuleRef M, int FileHandle) {
  return LLVMWriteBitcodeToFD(M, FileHandle, true, false);
}
