//===-- LLDBUtils.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

/// Run a list of LLDB commands in the LLDB command interpreter.
///
/// All output from every command, including the prompt + the command
/// is placed into the "strm" argument.
///
/// \param[in] prefix
///     A string that will be printed into \a strm prior to emitting
///     the prompt + command and command output. Can be NULL.
///
/// \param[in] commands
///     An array of LLDB commands to execute.
///
/// \param[in] strm
///     The stream that will receive the prefix, prompt + command and
///     all command output.
void RunLLDBCommands(llvm::StringRef prefix,
                     const llvm::ArrayRef<std::string> &commands,
                     llvm::raw_ostream &strm);

/// Run a list of LLDB commands in the LLDB command interpreter.
///
/// All output from every command, including the prompt + the command
/// is returned in the std::string return value.
///
/// \param[in] prefix
///     A string that will be printed into \a strm prior to emitting
///     the prompt + command and command output. Can be NULL.
///
/// \param[in] commands
///     An array of LLDB commands to execute.
///
/// \return
///     A std::string that contains the prefix and all commands and
///     command output
std::string RunLLDBCommands(llvm::StringRef prefix,
                            const llvm::ArrayRef<std::string> &commands);

/// Check if a thread has a stop reason.
///
/// \param[in] thread
///     The LLDB thread object to check
///
/// \return
///     \b True if the thread has a valid stop reason, \b false
///     otherwise.
bool ThreadHasStopReason(lldb::SBThread &thread);

/// Given a LLDB frame, make a frame ID that is unique to a specific
/// thread and frame.
///
/// VSCode requires a Stackframe "id" to be unique, so we use the frame
/// index in the lower 32 bits and the thread index ID in the upper 32
/// bits.
///
/// \param[in] frame
///     The LLDB stack frame object generate the ID for
///
/// \return
///     A unique integer that allows us to easily find the right
///     stack frame within a thread on subsequent VS code requests.
int64_t MakeVSCodeFrameID(lldb::SBFrame &frame);

/// Given a VSCode frame ID, convert to a LLDB thread index id.
///
/// VSCode requires a Stackframe "id" to be unique, so we use the frame
/// index in the lower THREAD_INDEX_SHIFT bits and the thread index ID in
/// the upper 32 - THREAD_INDEX_SHIFT bits.
///
/// \param[in] dap_frame_id
///     The VSCode frame ID to convert to a thread index ID.
///
/// \return
///     The LLDB thread index ID.
uint32_t GetLLDBThreadIndexID(uint64_t dap_frame_id);

/// Given a VSCode frame ID, convert to a LLDB frame ID.
///
/// VSCode requires a Stackframe "id" to be unique, so we use the frame
/// index in the lower THREAD_INDEX_SHIFT bits and the thread index ID in
/// the upper 32 - THREAD_INDEX_SHIFT bits.
///
/// \param[in] dap_frame_id
///     The VSCode frame ID to convert to a frame ID.
///
/// \return
///     The LLDB frame index ID.
uint32_t GetLLDBFrameID(uint64_t dap_frame_id);

/// Given a LLDB breakpoint, make a breakpoint ID that is unique to a
/// specific breakpoint and breakpoint location.
///
/// VSCode requires a Breakpoint "id" to be unique, so we use the
/// breakpoint ID in the lower BREAKPOINT_ID_SHIFT bits and the
/// breakpoint location ID in the upper BREAKPOINT_ID_SHIFT bits.
///
/// \param[in] frame
///     The LLDB stack frame object generate the ID for
///
/// \return
///     A unique integer that allows us to easily find the right
///     stack frame within a thread on subsequent VS code requests.
int64_t MakeVSCodeBreakpointID(lldb::SBBreakpointLocation &bp_loc);

/// Given a VSCode breakpoint ID, convert to a LLDB breakpoint ID.
///
/// VSCode requires a Breakpoint "id" to be unique, so we use the
/// breakpoint ID in the lower BREAKPOINT_ID_SHIFT bits and the
/// breakpoint location ID in the upper BREAKPOINT_ID_SHIFT bits.
///
/// \param[in] dap_breakpoint_id
///     The VSCode breakpoint ID to convert to an LLDB breakpoint ID.
///
/// \return
///     The LLDB breakpoint ID.
uint32_t GetLLDBBreakpointID(uint64_t dap_breakpoint_id);

/// Given a VSCode breakpoint ID, convert to a LLDB breakpoint location ID.
///
/// VSCode requires a Breakpoint "id" to be unique, so we use the
/// breakpoint ID in the lower BREAKPOINT_ID_SHIFT bits and the
/// breakpoint location ID in the upper BREAKPOINT_ID_SHIFT bits.
///
/// \param[in] dap_breakpoint_id
///     The VSCode frame ID to convert to a breakpoint location ID.
///
/// \return
///     The LLDB breakpoint location ID.
uint32_t GetLLDBBreakpointLocationID(uint64_t dap_breakpoint_id);
} // namespace lldb_vscode

#endif
