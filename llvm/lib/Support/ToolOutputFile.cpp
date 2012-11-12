//===--- ToolOutputFile.cpp - Implement the tool_output_file class --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the tool_output_file class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/Signals.h"
using namespace llvm;

tool_output_file::CleanupInstaller::CleanupInstaller(const char *filename)
  : Filename(filename), Keep(false) {
  // Arrange for the file to be deleted if the process is killed.
  if (Filename != "-")
    sys::RemoveFileOnSignal(sys::Path(Filename));
}

tool_output_file::CleanupInstaller::~CleanupInstaller() {
  // Delete the file if the client hasn't told us not to.
  if (!Keep && Filename != "-")
    sys::Path(Filename).eraseFromDisk();

  // Ok, the file is successfully written and closed, or deleted. There's no
  // further need to clean it up on signals.
  if (Filename != "-")
    sys::DontRemoveFileOnSignal(sys::Path(Filename));
}

tool_output_file::tool_output_file(const char *filename, std::string &ErrorInfo,
                                   unsigned Flags)
  : Installer(filename),
    OS(filename, ErrorInfo, Flags) {
  // If open fails, no cleanup is needed.
  if (!ErrorInfo.empty())
    Installer.Keep = true;
}
