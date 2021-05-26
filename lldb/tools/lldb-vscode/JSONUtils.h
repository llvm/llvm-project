//===-- JSONUtils.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_VSCODE_JSONUTILS_H
#define LLDB_TOOLS_LLDB_VSCODE_JSONUTILS_H

#include "VSCodeForward.h"
#include "lldb/API/SBModule.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <cstdint>

namespace lldb_vscode {

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
///     specified \a key, or an empty string if there is no key that
///     matches or if the value is not a string.
llvm::StringRef GetString(const llvm::json::Object &obj, llvm::StringRef key);
llvm::StringRef GetString(const llvm::json::Object *obj, llvm::StringRef key);

/// Extract the unsigned integer value for the specified key from
/// the specified object.
///
/// \param[in] obj
///     A JSON object that we will attempt to extract the value from
///
/// \param[in] key
///     The key to use when extracting the value
///
/// \return
///     The unsigned integer value for the specified \a key, or
///     \a fail_value  if there is no key that matches or if the
///     value is not an integer.
uint64_t GetUnsigned(const llvm::json::Object &obj, llvm::StringRef key,
                     uint64_t fail_value);
uint64_t GetUnsigned(const llvm::json::Object *obj, llvm::StringRef key,
                     uint64_t fail_value);

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
///     The boolean value for the specified \a key, or \a fail_value
///     if there is no key that matches or if the value is not a
///     boolean value of an integer.
bool GetBoolean(const llvm::json::Object &obj, llvm::StringRef key,
                bool fail_value);
bool GetBoolean(const llvm::json::Object *obj, llvm::StringRef key,
                bool fail_value);

/// Extract the signed integer for the specified key from the
/// specified object.
///
/// \param[in] obj
///     A JSON object that we will attempt to extract the value from
///
/// \param[in] key
///     The key to use when extracting the value
///
/// \return
///     The signed integer value for the specified \a key, or
///     \a fail_value if there is no key that matches or if the
///     value is not an integer.
int64_t GetSigned(const llvm::json::Object &obj, llvm::StringRef key,
                  int64_t fail_value);
int64_t GetSigned(const llvm::json::Object *obj, llvm::StringRef key,
                  int64_t fail_value);

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

/// Fill a response object given the request object.
///
/// The \a response object will get its "type" set to "response",
/// the "seq" set to zero, "response_seq" set to the "seq" value from
/// \a request, "command" set to the "command" from \a request,
/// and "success" set to true.
///
/// \param[in] request
///     The request object received from a call to VSCode::ReadJSON().
///
/// \param[in,out] response
///     An empty llvm::json::Object object that will be filled
///     in as noted in description.
void FillResponse(const llvm::json::Object &request,
                  llvm::json::Object &response);

/// Emplace the string value from an SBValue into the supplied object
/// using \a key as the key that will contain the value.
///
/// The value is what we will display in VS Code. Some SBValue objects
/// can have a value and/or a summary. If a value has both, we
/// combine the value and the summary into one string. If we only have a
/// value or summary, then that is considered the value. If there is
/// no value and no summary then the value is the type name followed by
/// the address of the type if it has an address.
///
///
/// \param[in] v
///     A lldb::SBValue object to extract the string value from
///
///
/// \param[in] object
///     The object to place the value object into
///
///
/// \param[in] key
///     The key name to use when inserting the value object we create
void SetValueForKey(lldb::SBValue &v, llvm::json::Object &object,
                    llvm::StringRef key);

/// Converts \a bp to a JSON value and appends the first valid location to the
/// \a breakpoints array.
///
/// \param[in] bp
///     A LLDB breakpoint object which will get the first valid location
///     extracted and converted into a JSON object in the \a breakpoints array
///
/// \param[in] breakpoints
///     A JSON array that will get a llvm::json::Value for \a bp
///     appended to it.
///
/// \param[in] request_path
///     An optional source path to use when creating the "Source" object of this
///     breakpoint. If not specified, the "Source" object is created from the
///     breakpoint's address' LineEntry. It is useful to ensure the same source
///     paths provided by the setBreakpoints request are returned to the IDE.
///
/// \param[in] request_line
///     An optional line to use when creating the "Breakpoint" object to append.
///     It is used if the breakpoint has no valid locations.
///     It is useful to ensure the same line
///     provided by the setBreakpoints request are returned to the IDE as a
///     fallback.
void AppendBreakpoint(lldb::SBBreakpoint &bp, llvm::json::Array &breakpoints,
                      llvm::Optional<llvm::StringRef> request_path = llvm::None,
                      llvm::Optional<uint32_t> request_line = llvm::None);

/// Converts breakpoint location to a Visual Studio Code "Breakpoint"
///
/// \param[in] bp
///     A LLDB breakpoint object to convert into a JSON value
///
/// \param[in] request_path
///     An optional source path to use when creating the "Source" object of this
///     breakpoint. If not specified, the "Source" object is created from the
///     breakpoint's address' LineEntry. It is useful to ensure the same source
///     paths provided by the setBreakpoints request are returned to the IDE.
///
/// \param[in] request_line
///     An optional line to use when creating the resulting "Breakpoint" object.
///     It is used if the breakpoint has no valid locations.
///     It is useful to ensure the same line
///     provided by the setBreakpoints request are returned to the IDE as a
///     fallback.
///
/// \return
///     A "Breakpoint" JSON object with that follows the formal JSON
///     definition outlined by Microsoft.
llvm::json::Value
CreateBreakpoint(lldb::SBBreakpoint &bp,
                 llvm::Optional<llvm::StringRef> request_path = llvm::None,
                 llvm::Optional<uint32_t> request_line = llvm::None);

/// Converts a LLDB module to a VS Code DAP module for use in "modules" events.
///
/// \param[in] module
///     A LLDB module object to convert into a JSON value
///
/// \return
///     A "Module" JSON object with that follows the formal JSON
///     definition outlined by Microsoft.
llvm::json::Value CreateModule(lldb::SBModule &module);

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
/// the Visual Studio Code debug adaptor definition.
///
/// \param[in] bp
///     The exception breakpoint object to use
///
/// \return
///     A "ExceptionBreakpointsFilter" JSON object with that follows
///     the formal JSON definition outlined by Microsoft.
llvm::json::Value
CreateExceptionBreakpointFilter(const ExceptionBreakpoint &bp);

/// Create a "Scope" JSON object as described in the Visual Studio Code
/// debug adaptor definition.
///
/// \param[in] name
///     The value to place into the "name" key
//
/// \param[in] variablesReference
///     The value to place into the "variablesReference" key
//
/// \param[in] namedVariables
///     The value to place into the "namedVariables" key
//
/// \param[in] expensive
///     The value to place into the "expensive" key
///
/// \return
///     A "Scope" JSON object with that follows the formal JSON
///     definition outlined by Microsoft.
llvm::json::Value CreateScope(const llvm::StringRef name,
                              int64_t variablesReference,
                              int64_t namedVariables, bool expensive);

/// Create a "Source" JSON object as described in the Visual Studio Code
/// debug adaptor definition.
///
/// \param[in] line_entry
///     The LLDB line table to use when populating out the "Source"
///     object
///
/// \return
///     A "Source" JSON object with that follows the formal JSON
///     definition outlined by Microsoft.
llvm::json::Value CreateSource(lldb::SBLineEntry &line_entry);

/// Create a "Source" object for a given source path.
///
/// \param[in] source_path
///     The path to the source to use when creating the "Source" object.
///
/// \return
///     A "Source" JSON object that follows the formal JSON
///     definition outlined by Microsoft.
llvm::json::Value CreateSource(llvm::StringRef source_path);

/// Create a "Source" object for a given frame.
///
/// When there is no source file information for a stack frame, we will
/// create disassembly for a function and store a permanent
/// "sourceReference" that contains the textual disassembly for a
/// function along with address to line information. The "Source" object
/// that is created will contain a "sourceReference" that the VSCode
/// protocol can later fetch as text in order to display disassembly.
/// The PC will be extracted from the frame and the disassembly line
/// within the source referred to by "sourceReference" will be filled
/// in.
///
/// \param[in] frame
///     The LLDB stack frame to use when populating out the "Source"
///     object.
///
/// \param[out] disasm_line
///     The line within the "sourceReference" file that the PC from
///     \a frame matches.
///
/// \return
///     A "Source" JSON object with that follows the formal JSON
///     definition outlined by Microsoft.
llvm::json::Value CreateSource(lldb::SBFrame &frame, int64_t &disasm_line);

/// Create a "StackFrame" object for a LLDB frame object.
///
/// This function will fill in the following keys in the returned
/// object:
///   "id" - the stack frame ID as an integer
///   "name" - the function name as a string
///   "source" - source file information as a "Source" VSCode object
///   "line" - the source file line number as an integer
///   "column" - the source file column number as an integer
///
/// \param[in] frame
///     The LLDB stack frame to use when populating out the "StackFrame"
///     object.
///
/// \return
///     A "StackFrame" JSON object with that follows the formal JSON
///     definition outlined by Microsoft.
llvm::json::Value CreateStackFrame(lldb::SBFrame &frame);

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
/// \return
///     A "Thread" JSON object with that follows the formal JSON
///     definition outlined by Microsoft.
llvm::json::Value CreateThread(lldb::SBThread &thread);

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
/// \param[in] thread
///     The LLDB thread to use when populating out the "StoppedEvent"
///     object.
///
/// \return
///     A "StoppedEvent" JSON object with that follows the formal JSON
///     definition outlined by Microsoft.
llvm::json::Value CreateThreadStopped(lldb::SBThread &thread, uint32_t stop_id);

/// \return
///     The variable name of \a value or a default placeholder.
const char *GetNonNullVariableName(lldb::SBValue value);

/// VSCode can't display two variables with the same name, so we need to
/// distinguish them by using a suffix.
///
/// If the source and line information is present, we use it as the suffix.
/// Otherwise, we fallback to the variable address or register location.
std::string CreateUniqueVariableNameForDisplay(lldb::SBValue v,
                                               bool is_name_duplicated);

/// Create a "Variable" object for a LLDB thread object.
///
/// This function will fill in the following keys in the returned
/// object:
///   "name" - the name of the variable
///   "value" - the value of the variable as a string
///   "type" - the typename of the variable as a string
///   "id" - a unique identifier for a value in case there are multiple
///          variables with the same name. Other parts of the VSCode
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
/// \param[in] variablesReference
///     The variable reference. Zero if this value isn't structured
///     and has no children, non-zero if it does have children and
///     might be asked to expand itself.
///
/// \param[in] varID
///     A unique variable identifier to help in properly identifying
///     variables with the same name. This is an extension to the
///     VS protocol.
///
/// \param[in] format_hex
///     It set to true the variable will be formatted as hex in
///     the "value" key value pair for the value of the variable.
///
/// \param[in] is_name_duplicated
///     Whether the same variable name appears multiple times within the same
///     context (e.g. locals). This can happen due to shadowed variables in
///     nested blocks.
///
///     As VSCode doesn't render two of more variables with the same name, we
///     apply a suffix to distinguish duplicated variables.
///
/// \return
///     A "Variable" JSON object with that follows the formal JSON
///     definition outlined by Microsoft.
llvm::json::Value CreateVariable(lldb::SBValue v, int64_t variablesReference,
                                 int64_t varID, bool format_hex,
                                 bool is_name_duplicated = false);

llvm::json::Value CreateCompileUnit(lldb::SBCompileUnit unit);

/// Create a runInTerminal reverse request object
///
/// \param[in] launch_request
///     The original launch_request object whose fields are used to construct
///     the reverse request object.
///
/// \param[in] debug_adaptor_path
///     Path to the current debug adaptor. It will be used to delegate the
///     launch of the target.
///
/// \param[in] comm_file
///     The fifo file used to communicate the with the target launcher.
///
/// \return
///     A "runInTerminal" JSON object that follows the specification outlined by
///     Microsoft.
llvm::json::Object
CreateRunInTerminalReverseRequest(const llvm::json::Object &launch_request,
                                  llvm::StringRef debug_adaptor_path,
                                  llvm::StringRef comm_file);

/// Convert a given JSON object to a string.
std::string JSONToString(const llvm::json::Value &json);

} // namespace lldb_vscode

#endif
