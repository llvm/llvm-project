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
#include "Protocol/ProtocolTypes.h"
#include "lldb/API/SBCompileUnit.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBFormat.h"
#include "lldb/API/SBLineEntry.h"
#include "lldb/API/SBType.h"
#include "lldb/API/SBValue.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lldb_dap {

/// Emplace a StringRef in a json::Object after enusring that the
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

/// Extract simple values as a string.
///
/// \param[in] value
///     A JSON value to extract the string from.
///
/// \return
///     A llvm::StringRef that contains the string value, or an empty
///     string if \a value isn't a string.
llvm::StringRef GetAsString(const llvm::json::Value &value);

/// Extract the string value for the specified key from the
/// specified object.
///
/// \param[in] obj
///     A JSON object that we will attempt to extract the value from
///
/// \param[in] key
///     The key to use when extracting the value
///
/// \return
///     A llvm::StringRef that contains the string value for the
///     specified \a key, or \a std::nullopt if there is no key that
///     matches or if the value is not a string.
std::optional<llvm::StringRef> GetString(const llvm::json::Object &obj,
                                         llvm::StringRef key);
std::optional<llvm::StringRef> GetString(const llvm::json::Object *obj,
                                         llvm::StringRef key);

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

template <typename T>
std::optional<T> GetInteger(const llvm::json::Object *obj,
                            llvm::StringRef key) {
  if (obj != nullptr)
    return GetInteger<T>(*obj, key);
  return std::nullopt;
}
/// @}

/// Extract the boolean value for the specified key from the
/// specified object.
///
/// \param[in] obj
///     A JSON object that we will attempt to extract the value from
///
/// \param[in] key
///     The key to use when extracting the value
///
/// \return
///     The boolean value for the specified \a key, or std::nullopt
///     if there is no key that matches or if the value is not a
///     boolean value of an integer.
/// @{
std::optional<bool> GetBoolean(const llvm::json::Object &obj,
                               llvm::StringRef key);
std::optional<bool> GetBoolean(const llvm::json::Object *obj,
                               llvm::StringRef key);
/// @}

/// Check if the specified key exists in the specified object.
///
/// \param[in] obj
///     A JSON object that we will attempt to extract the value from
///
/// \param[in] key
///     The key to check for
///
/// \return
///     \b True if the key exists in the \a obj, \b False otherwise.
bool ObjectContainsKey(const llvm::json::Object &obj, llvm::StringRef key);

/// Encodes a memory reference
std::string EncodeMemoryReference(lldb::addr_t addr);

/// Decodes a memory reference
std::optional<lldb::addr_t>
DecodeMemoryReference(llvm::StringRef memoryReference);

/// Extract an array of strings for the specified key from an object.
///
/// String values in the array will be extracted without any quotes
/// around them. Numbers and Booleans will be converted into
/// strings. Any NULL, array or objects values in the array will be
/// ignored.
///
/// \param[in] obj
///     A JSON object that we will attempt to extract the array from
///
/// \param[in] key
///     The key to use when extracting the value
///
/// \return
///     An array of string values for the specified \a key, or
///     \a fail_value if there is no key that matches or if the
///     value is not an array or all items in the array are not
///     strings, numbers or booleans.
std::vector<std::string> GetStrings(const llvm::json::Object *obj,
                                    llvm::StringRef key);

/// Extract an object of key value strings for the specified key from an object.
///
/// String values in the object will be extracted without any quotes
/// around them. Numbers and Booleans will be converted into
/// strings. Any NULL, array or objects values in the array will be
/// ignored.
///
/// \param[in] obj
///     A JSON object that we will attempt to extract the array from
///
/// \param[in] key
///     The key to use when extracting the value
///
/// \return
///     An object of key value strings for the specified \a key, or
///     \a fail_value if there is no key that matches or if the
///     value is not an object or key and values in the object are not
///     strings, numbers or booleans.
std::unordered_map<std::string, std::string>
GetStringMap(const llvm::json::Object &obj, llvm::StringRef key);

/// Fill a response object given the request object.
///
/// The \a response object will get its "type" set to "response",
/// the "seq" set to zero, "response_seq" set to the "seq" value from
/// \a request, "command" set to the "command" from \a request,
/// and "success" set to true.
///
/// \param[in] request
///     The request object received from a call to DAP::ReadJSON().
///
/// \param[in,out] response
///     An empty llvm::json::Object object that will be filled
///     in as noted in description.
void FillResponse(const llvm::json::Object &request,
                  llvm::json::Object &response);

/// Converts a LLDB module to a VS Code DAP module for use in "modules" events.
///
/// \param[in] target
///     A LLDB target object to convert into a JSON value.
///
/// \param[in] module
///     A LLDB module object to convert into a JSON value
///
/// \param[in] id_only
///     Only include the module ID in the JSON value. This is used when sending
///     a "removed" module event.
///
/// \return
///     A "Module" JSON object with that follows the formal JSON
///     definition outlined by Microsoft.
llvm::json::Value CreateModule(lldb::SBTarget &target, lldb::SBModule &module,
                               bool id_only = false);

/// Create a "Event" JSON object using \a event_name as the event name
///
/// \param[in] event_name
///     The string value to use for the "event" key in the JSON object.
///
/// \return
///     A "Event" JSON object with that follows the formal JSON
///     definition outlined by Microsoft.
llvm::json::Object CreateEventObject(const llvm::StringRef event_name);

/// Create a "ExceptionBreakpointsFilter" JSON object as described in
/// the debug adapter definition.
///
/// \param[in] bp
///     The exception breakpoint object to use
///
/// \return
///     A "ExceptionBreakpointsFilter" JSON object with that follows
///     the formal JSON definition outlined by Microsoft.
protocol::ExceptionBreakpointsFilter
CreateExceptionBreakpointFilter(const ExceptionBreakpoint &bp);

/// Create a "Source" JSON object as described in the debug adapter definition.
///
/// \param[in] file
///     The SBFileSpec to use when populating out the "Source" object
///
/// \return
///     A "Source" JSON object that follows the formal JSON
///     definition outlined by Microsoft.
protocol::Source CreateSource(const lldb::SBFileSpec &file);

/// Create a "Source" JSON object as described in the debug adapter definition.
///
/// \param[in] line_entry
///     The LLDB line table to use when populating out the "Source"
///     object
///
/// \return
///     A "Source" JSON object that follows the formal JSON
///     definition outlined by Microsoft.
protocol::Source CreateSource(const lldb::SBLineEntry &line_entry);

/// Create a "Source" object for a given source path.
///
/// \param[in] source_path
///     The path to the source to use when creating the "Source" object.
///
/// \return
///     A "Source" JSON object that follows the formal JSON
///     definition outlined by Microsoft.
protocol::Source CreateSource(llvm::StringRef source_path);

/// Create a "Source" object for a given frame, using its assembly for source.
///
/// \param[in] target
///     The relevant target.
///
/// \param[in] address
///     The address to use when creating the "Source" object.
///
/// \return
///     A "Source" JSON object that follows the formal JSON
///     definition outlined by Microsoft.
protocol::Source CreateAssemblySource(const lldb::SBTarget &target,
                                      lldb::SBAddress &address);

/// Return true if the given line entry should be displayed as assembly.
///
/// \param[in] line_entry
///     The LLDB line entry to check.
///
/// \param[in] stop_disassembly_display
///     The value of the "stop-disassembly-display" setting.
///
/// \return
///     True if the line entry should be displayed as assembly, false
///     otherwise.
bool ShouldDisplayAssemblySource(
    const lldb::SBLineEntry &line_entry,
    lldb::StopDisassemblyType stop_disassembly_display);

/// Create a "StackFrame" object for a LLDB frame object.
///
/// This function will fill in the following keys in the returned
/// object:
///   "id" - the stack frame ID as an integer
///   "name" - the function name as a string
///   "source" - source file information as a "Source" DAP object
///   "line" - the source file line number as an integer
///   "column" - the source file column number as an integer
///
/// \param[in] frame
///     The LLDB stack frame to use when populating out the "StackFrame"
///     object.
///
/// \param[in] format
///     The LLDB format to use when populating out the "StackFrame"
///     object.
///
/// \param[in] stop_disassembly_display
///     The value of the "stop-disassembly-display" setting.
///
/// \return
///     A "StackFrame" JSON object with that follows the formal JSON
///     definition outlined by Microsoft.
llvm::json::Value CreateStackFrame(lldb::SBFrame &frame, lldb::SBFormat &format,
                                   lldb::StopDisassemblyType);

/// Create a "StackFrame" label object for a LLDB thread.
///
/// This function will fill in the following keys in the returned
/// object:
///   "id" - the thread ID as an integer
///   "name" - the thread name as a string which combines the LLDB
///            thread index ID along with the string name of the thread
///            from the OS if it has a name.
///   "presentationHint" - "label"
///
/// \param[in] thread
///     The LLDB thread to use when populating out the "Thread"
///     object.
///
/// \param[in] format
///     The configured formatter for the DAP session.
///
/// \return
///     A "StackFrame" JSON object with that follows the formal JSON
///     definition outlined by Microsoft.
llvm::json::Value CreateExtendedStackFrameLabel(lldb::SBThread &thread,
                                                lldb::SBFormat &format);

/// Create a "Thread" object for a LLDB thread object.
///
/// This function will fill in the following keys in the returned
/// object:
///   "id" - the thread ID as an integer
///   "name" - the thread name as a string which combines the LLDB
///            thread index ID along with the string name of the thread
///            from the OS if it has a name.
///
/// \param[in] thread
///     The LLDB thread to use when populating out the "Thread"
///     object.
///
/// \param[in] format
///     The LLDB format to use when populating out the "Thread"
///     object.
///
/// \return
///     A "Thread" JSON object with that follows the formal JSON
///     definition outlined by Microsoft.
llvm::json::Value CreateThread(lldb::SBThread &thread, lldb::SBFormat &format);

llvm::json::Array GetThreads(lldb::SBProcess process, lldb::SBFormat &format);

/// Create a "StoppedEvent" object for a LLDB thread object.
///
/// This function will fill in the following keys in the returned
/// object's "body" object:
///   "reason" - With a valid stop reason enumeration string value
///              that Microsoft specifies
///   "threadId" - The thread ID as an integer
///   "description" - a stop description (like "breakpoint 12.3") as a
///                   string
///   "preserveFocusHint" - a boolean value that states if this thread
///                         should keep the focus in the GUI.
///   "allThreadsStopped" - set to True to indicate that all threads
///                         stop when any thread stops.
///
/// \param[in] dap
///     The DAP session associated with the stopped thread.
///
/// \param[in] thread
///     The LLDB thread to use when populating out the "StoppedEvent"
///     object.
///
/// \param[in] stop_id
///     The stop id for this event.
///
/// \return
///     A "StoppedEvent" JSON object with that follows the formal JSON
///     definition outlined by Microsoft.
llvm::json::Value CreateThreadStopped(DAP &dap, lldb::SBThread &thread,
                                      uint32_t stop_id);

/// \return
///     The variable name of \a value or a default placeholder.
const char *GetNonNullVariableName(lldb::SBValue &value);

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
  std::string value;
  // The summary string of this variable. It might be empty.
  std::string summary;
  // The auto summary if using `enableAutoVariableSummaries`.
  std::optional<std::string> auto_summary;
  // The type of this variable.
  lldb::SBType type_obj;
  // The display type name of this variable.
  std::string display_type_name;
  /// The SBValue for this variable.
  lldb::SBValue v;

  VariableDescription(lldb::SBValue v, bool auto_variable_summaries,
                      bool format_hex = false, bool is_name_duplicated = false,
                      std::optional<std::string> custom_name = {});

  /// Create a JSON object that represents these extensions to the DAP variable
  /// response.
  llvm::json::Object GetVariableExtensionsJSON();

  /// Returns a description of the value appropriate for the specified context.
  std::string GetResult(llvm::StringRef context);
};

/// Does the given variable have an associated value location?
bool ValuePointsToCode(lldb::SBValue v);

/// Pack a location into a single integer which we can send via
/// the debug adapter protocol.
int64_t PackLocation(int64_t var_ref, bool is_value_location);

/// Reverse of `PackLocation`
std::pair<int64_t, bool> UnpackLocation(int64_t location_id);

/// Create a "Variable" object for a LLDB thread object.
///
/// This function will fill in the following keys in the returned
/// object:
///   "name" - the name of the variable
///   "value" - the value of the variable as a string
///   "type" - the typename of the variable as a string
///   "id" - a unique identifier for a value in case there are multiple
///          variables with the same name. Other parts of the DAP
///          protocol refer to values by name so this can help
///          disambiguate such cases if a IDE passes this "id" value
///          back down.
///   "variablesReference" - Zero if the variable has no children,
///          non-zero integer otherwise which can be used to expand
///          the variable.
///   "evaluateName" - The name of the variable to use in expressions
///                    as a string.
///
/// \param[in] v
///     The LLDB value to use when populating out the "Variable"
///     object.
///
/// \param[in] var_ref
///     The variable reference. Used to identify the value, e.g.
///     in the `variablesReference` or `declarationLocationReference`
///     properties.
///
/// \param[in] format_hex
///     If set to true the variable will be formatted as hex in
///     the "value" key value pair for the value of the variable.
///
/// \param[in] auto_variable_summaries
///     IF set to true the variable will create an automatic variable summary.
///
/// \param[in] is_name_duplicated
///     Whether the same variable name appears multiple times within the same
///     context (e.g. locals). This can happen due to shadowed variables in
///     nested blocks.
///
///     As VSCode doesn't render two of more variables with the same name, we
///     apply a suffix to distinguish duplicated variables.
///
/// \param[in] custom_name
///     A provided custom name that is used instead of the SBValue's when
///     creating the JSON representation.
///
/// \return
///     A "Variable" JSON object with that follows the formal JSON
///     definition outlined by Microsoft.
llvm::json::Value CreateVariable(lldb::SBValue v, int64_t var_ref,
                                 bool format_hex, bool auto_variable_summaries,
                                 bool synthetic_child_debugging,
                                 bool is_name_duplicated = false,
                                 std::optional<std::string> custom_name = {});

llvm::json::Value CreateCompileUnit(lldb::SBCompileUnit &unit);

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
/// \return
///     A "runInTerminal" JSON object that follows the specification outlined by
///     Microsoft.
llvm::json::Object CreateRunInTerminalReverseRequest(
    llvm::StringRef program, const std::vector<std::string> &args,
    const llvm::StringMap<std::string> &env, llvm::StringRef cwd,
    llvm::StringRef comm_file, lldb::pid_t debugger_pid);

/// Create a "Terminated" JSON object that contains statistics
///
/// \return
///     A body JSON object with debug info and breakpoint info
llvm::json::Object CreateTerminatedEventObject(lldb::SBTarget &target);

/// Convert a given JSON object to a string.
std::string JSONToString(const llvm::json::Value &json);

} // namespace lldb_dap

#endif
