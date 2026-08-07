//===-- JSONUtils.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_JSONUTILS_H
#define LLDB_TOOLS_LLDB_DAP_JSONUTILS_H

#include "DAPForward.h"
#include "Protocol/ProtocolRequests.h"
#include "lldb/API/SBType.h"
#include "lldb/API/SBValue.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lldb_dap {

/// Emplace a StringRef in a json::Object after ensuring that the
/// string is valid UTF8. If not, first call llvm::json::fixUTF8
/// before emplacing.
///
/// \param[in] obj
///     A JSON object that we will attempt to emplace the value in
///
/// \param[in] key
///     The key to use when emplacing the value
///
/// \param[in] str
///     The string to emplace
void EmplaceSafeString(llvm::json::Object &obj, llvm::StringRef key,
                       llvm::StringRef str);

/// Extract the integer value for the specified key from the specified object
/// and return it as the specified integer type T.
///
/// \param[in] obj
///     A JSON object that we will attempt to extract the value from
///
/// \param[in] key
///     The key to use when extracting the value
///
/// \return
///     The integer value for the specified \a key, or std::nullopt if there is
///     no key that matches or if the value is not an integer.
/// @{
template <typename T>
std::optional<T> GetInteger(const llvm::json::Object &obj,
                            llvm::StringRef key) {
  return obj.getInteger(key);
}
/// @}

/// Encodes a memory reference
std::string EncodeMemoryReference(lldb::addr_t addr);

/// Decodes a memory reference
std::optional<lldb::addr_t>
DecodeMemoryReference(llvm::StringRef memoryReference);

/// Decodes a memory reference from the given json value.
///
/// \param[in] v
///    A JSON value that we expected to contain the memory reference.
///
/// \param[in] key
///    The key of the memory reference.
///
/// \param[out] out
///    The memory address, if successfully decoded.
///
/// \param[in] path
///    The path for reporting errors.
///
/// \param[in] required
///    Indicates if the key is required to be present, otherwise report an error
///    if the key is missing.
///
/// \param[in] allow_empty
///    Interpret empty string as a valid value, don't report an error (see
///    VS Code issue https://github.com/microsoft/vscode/issues/270593).
///
/// \return
///    Returns \b true if the address was decoded successfully.
bool DecodeMemoryReference(const llvm::json::Value &v, llvm::StringLiteral key,
                           lldb::addr_t &out, llvm::json::Path path,
                           bool required, bool allow_empty = false);

/// Create a "Event" JSON object using \a event_name as the event name
///
/// \param[in] event_name
///     The string value to use for the "event" key in the JSON object.
///
/// \return
///     A "Event" JSON object with that follows the formal JSON
///     definition outlined by Microsoft.
llvm::json::Object CreateEventObject(const llvm::StringRef event_name);

/// \return
///     The variable name of \a value or a default placeholder.
llvm::StringRef GetNonNullVariableName(lldb::SBValue &value);

/// VSCode can't display two variables with the same name, so we need to
/// distinguish them by using a suffix.
///
/// If the source and line information is present, we use it as the suffix.
/// Otherwise, we fallback to the variable address or register location.
std::string CreateUniqueVariableNameForDisplay(lldb::SBValue &v,
                                               bool is_name_duplicated);

/// Helper struct that parses the metadata of an \a lldb::SBValue and produces
/// a canonical set of properties that can be sent to DAP clients.
struct VariableDescription {
  // The error message if SBValue.GetValue() fails.
  std::optional<std::string> error;
  // The display description to show on the IDE.
  std::string display_value;
  // The display name to show on the IDE.
  std::string name;
  // The variable path for this variable.
  std::string evaluate_name;
  // The output of SBValue.GetValue() if it doesn't fail. It might be empty.
  llvm::StringRef value;
  // The summary string of this variable. It might be empty.
  llvm::StringRef summary;
  // The auto summary if using `enableAutoVariableSummaries`.
  std::optional<std::string> auto_summary;
  // The type of this variable.
  lldb::SBType type_obj;
  // The display type name of this variable.
  llvm::StringRef display_type_name;
  /// The SBValue for this variable.
  lldb::SBValue val;

  VariableDescription(lldb::SBValue v, bool auto_variable_summaries,
                      bool format_hex = false, bool is_name_duplicated = false,
                      std::optional<llvm::StringRef> custom_name = {});

  /// Returns a description of the value appropriate for the specified context.
  std::string GetResult(protocol::EvaluateContext context);
};

/// Does the given variable have an associated value location?
bool ValuePointsToCode(lldb::SBValue v);

/// Pack a location into a single integer which we can send via
/// the debug adapter protocol.
int64_t PackLocation(int64_t var_ref, bool is_value_location);

/// Reverse of `PackLocation`
std::pair<int64_t, bool> UnpackLocation(int64_t location_id);

/// Create a runInTerminal reverse request object
///
/// \param[in] program
///     Path to the program to run in the terminal.
///
/// \param[in] args
///     The arguments for the program.
///
/// \param[in] env
///     The environment variables to set in the terminal.
///
/// \param[in] cwd
///     The working directory for the run in terminal request.
///
/// \param[in] comm_file
///     The fifo file used to communicate the with the target launcher.
///
/// \param[in] debugger_pid
///     The PID of the lldb-dap instance that will attach to the target. The
///     launcher uses it on Linux tell the kernel that it should allow the
///     debugger process to attach.
///
/// \param[in] stdio
///     An array of file paths for redirecting the program's standard IO
///     streams.
///
/// \param[in] external
///     If set to true, the program will run in an external terminal window
///     instead of IDE's integrated terminal.
///
/// \return
///     A "runInTerminal" JSON object that follows the specification outlined by
///     Microsoft.
llvm::json::Object CreateRunInTerminalReverseRequest(
    llvm::StringRef program, const std::vector<protocol::String> &args,
    const llvm::StringMap<protocol::String> &env, llvm::StringRef cwd,
    llvm::StringRef comm_file, lldb::pid_t debugger_pid,
    const std::vector<std::optional<protocol::String>> &stdio, bool external);

/// Create a "Terminated" JSON object that contains statistics
///
/// \return
///     A body JSON object with debug info and breakpoint info
llvm::json::Object CreateTerminatedEventObject(lldb::SBTarget &target);

/// Create a "Initialized" JSON object that contains statistics
///
/// \return
///     A body JSON object with debug info
llvm::json::Object CreateInitializedEventObject(lldb::SBTarget &target);

/// Convert a given JSON object to a string.
std::string JSONToString(const llvm::json::Value &json);

} // namespace lldb_dap

#endif
