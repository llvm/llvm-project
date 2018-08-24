//===-- LLDBUtils.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDBVSCODE_LLDBUTILS_H_
#define LLDBVSCODE_LLDBUTILS_H_

#include "VSCodeForward.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

namespace lldb_vscode {

///----------------------------------------------------------------------
/// Run a list of LLDB commands in the LLDB command interpreter.
///
/// All output from every command, including the prompt + the command
/// is placed into the "strm" argument.
///
/// @param[in] prefix
///     A string that will be printed into \a strm prior to emitting
///     the prmopt + command and command output. Can be NULL.
///
/// @param[in] commands
///     An array of LLDB commands to execute.
///
/// @param[in] strm
///     The stream that will receive the prefix, prompt + command and
///     all command output.
//----------------------------------------------------------------------
void RunLLDBCommands(llvm::StringRef prefix,
                     const llvm::ArrayRef<std::string> &commands,
                     llvm::raw_ostream &strm);

///----------------------------------------------------------------------
/// Run a list of LLDB commands in the LLDB command interpreter.
///
/// All output from every command, including the prompt + the command
/// is returned in the std::string return value.
///
/// @param[in] prefix
///     A string that will be printed into \a strm prior to emitting
///     the prmopt + command and command output. Can be NULL.
///
/// @param[in] commands
///     An array of LLDB commands to execute.
///
/// @return
///     A std::string that contains the prefix and all commands and
///     command output
//----------------------------------------------------------------------
std::string RunLLDBCommands(llvm::StringRef prefix,
                            const llvm::ArrayRef<std::string> &commands);

///----------------------------------------------------------------------
/// Check if a thread has a stop reason.
///
/// @param[in] thread
///     The LLDB thread object to check
///
/// @return
///     \b True if the thread has a valid stop reason, \b false
///     otherwise.
//----------------------------------------------------------------------
bool ThreadHasStopReason(lldb::SBThread &thread);

///----------------------------------------------------------------------
/// Given a LLDB frame, make a frame ID that is unique to a specific
/// thread and frame.
///
/// VSCode requires a Stackframe "id" to be unique, so we use the frame
/// index in the lower 32 bits and the thread index ID in the upper 32
/// bits.
///
/// @param[in] frame
///     The LLDB stack frame object generate the ID for
///
/// @return
///     A unique integer that allows us to easily find the right
///     stack frame within a thread on subsequent VS code requests.
//----------------------------------------------------------------------
int64_t MakeVSCodeFrameID(lldb::SBFrame &frame);

} // namespace lldb_vscode

#endif
