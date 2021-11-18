//===-- ScriptInterpreterPython.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_SWIGPYTHONBRIDGE_H
#define LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_SWIGPYTHONBRIDGE_H

#include <string>

#include "lldb/Host/Config.h"

#if LLDB_ENABLE_PYTHON

// LLDB Python header must be included first
#include "lldb-python.h"

#include "lldb/lldb-forward.h"
#include "lldb/lldb-types.h"
#include "llvm/Support/Error.h"

namespace lldb_private {

// GetPythonValueFormatString provides a system independent type safe way to
// convert a variable's type into a python value format. Python value formats
// are defined in terms of builtin C types and could change from system to as
// the underlying typedef for uint* types, size_t, off_t and other values
// change.

template <typename T> const char *GetPythonValueFormatString(T t);
template <> const char *GetPythonValueFormatString(char *);
template <> const char *GetPythonValueFormatString(char);
template <> const char *GetPythonValueFormatString(unsigned char);
template <> const char *GetPythonValueFormatString(short);
template <> const char *GetPythonValueFormatString(unsigned short);
template <> const char *GetPythonValueFormatString(int);
template <> const char *GetPythonValueFormatString(unsigned int);
template <> const char *GetPythonValueFormatString(long);
template <> const char *GetPythonValueFormatString(unsigned long);
template <> const char *GetPythonValueFormatString(long long);
template <> const char *GetPythonValueFormatString(unsigned long long);
template <> const char *GetPythonValueFormatString(float t);
template <> const char *GetPythonValueFormatString(double t);

void *LLDBSWIGPython_CastPyObjectToSBData(PyObject *data);
void *LLDBSWIGPython_CastPyObjectToSBError(PyObject *data);
void *LLDBSWIGPython_CastPyObjectToSBValue(PyObject *data);
void *LLDBSWIGPython_CastPyObjectToSBMemoryRegionInfo(PyObject *data);

// These prototypes are the Pythonic implementations of the required callbacks.
// Although these are scripting-language specific, their definition depends on
// the public API.

void *LLDBSwigPythonCreateScriptedProcess(const char *python_class_name,
                                          const char *session_dictionary_name,
                                          const lldb::TargetSP &target_sp,
                                          const StructuredDataImpl &args_impl,
                                          std::string &error_string);

void *LLDBSwigPythonCreateScriptedThread(const char *python_class_name,
                                         const char *session_dictionary_name,
                                         const lldb::ProcessSP &process_sp,
                                         const StructuredDataImpl &args_impl,
                                         std::string &error_string);

llvm::Expected<bool> LLDBSwigPythonBreakpointCallbackFunction(
    const char *python_function_name, const char *session_dictionary_name,
    const lldb::StackFrameSP &sb_frame,
    const lldb::BreakpointLocationSP &sb_bp_loc,
    const lldb_private::StructuredDataImpl &args_impl);

bool LLDBSwigPythonWatchpointCallbackFunction(
    const char *python_function_name, const char *session_dictionary_name,
    const lldb::StackFrameSP &sb_frame, const lldb::WatchpointSP &sb_wp);

bool LLDBSwigPythonCallTypeScript(const char *python_function_name,
                                  const void *session_dictionary,
                                  const lldb::ValueObjectSP &valobj_sp,
                                  void **pyfunct_wrapper,
                                  const lldb::TypeSummaryOptionsSP &options_sp,
                                  std::string &retval);

void *
LLDBSwigPythonCreateSyntheticProvider(const char *python_class_name,
                                      const char *session_dictionary_name,
                                      const lldb::ValueObjectSP &valobj_sp);

void *LLDBSwigPythonCreateCommandObject(const char *python_class_name,
                                        const char *session_dictionary_name,
                                        lldb::DebuggerSP debugger_sp);

void *LLDBSwigPythonCreateScriptedThreadPlan(
    const char *python_class_name, const char *session_dictionary_name,
    const StructuredDataImpl &args_data, std::string &error_string,
    const lldb::ThreadPlanSP &thread_plan_sp);

bool LLDBSWIGPythonCallThreadPlan(void *implementor, const char *method_name,
                                  lldb_private::Event *event_sp,
                                  bool &got_error);

void *LLDBSwigPythonCreateScriptedBreakpointResolver(
    const char *python_class_name, const char *session_dictionary_name,
    const StructuredDataImpl &args, const lldb::BreakpointSP &bkpt_sp);

unsigned int
LLDBSwigPythonCallBreakpointResolver(void *implementor, const char *method_name,
                                     lldb_private::SymbolContext *sym_ctx);

void *LLDBSwigPythonCreateScriptedStopHook(lldb::TargetSP target_sp,
                                           const char *python_class_name,
                                           const char *session_dictionary_name,
                                           const StructuredDataImpl &args,
                                           lldb_private::Status &error);

bool LLDBSwigPythonStopHookCallHandleStop(void *implementor,
                                          lldb::ExecutionContextRefSP exc_ctx,
                                          lldb::StreamSP stream);

size_t LLDBSwigPython_CalculateNumChildren(PyObject *implementor, uint32_t max);

PyObject *LLDBSwigPython_GetChildAtIndex(PyObject *implementor, uint32_t idx);

int LLDBSwigPython_GetIndexOfChildWithName(PyObject *implementor,
                                           const char *child_name);

lldb::ValueObjectSP LLDBSWIGPython_GetValueObjectSPFromSBValue(void *data);

bool LLDBSwigPython_UpdateSynthProviderInstance(PyObject *implementor);

bool LLDBSwigPython_MightHaveChildrenSynthProviderInstance(
    PyObject *implementor);

PyObject *LLDBSwigPython_GetValueSynthProviderInstance(PyObject *implementor);

bool LLDBSwigPythonCallCommand(const char *python_function_name,
                               const char *session_dictionary_name,
                               lldb::DebuggerSP debugger, const char *args,
                               lldb_private::CommandReturnObject &cmd_retobj,
                               lldb::ExecutionContextRefSP exe_ctx_ref_sp);

bool LLDBSwigPythonCallCommandObject(
    PyObject *implementor, lldb::DebuggerSP debugger, const char *args,
    lldb_private::CommandReturnObject &cmd_retobj,
    lldb::ExecutionContextRefSP exe_ctx_ref_sp);

bool LLDBSwigPythonCallModuleInit(const char *python_module_name,
                                  const char *session_dictionary_name,
                                  lldb::DebuggerSP debugger);

void *LLDBSWIGPythonCreateOSPlugin(const char *python_class_name,
                                   const char *session_dictionary_name,
                                   const lldb::ProcessSP &process_sp);

void *LLDBSWIGPython_CreateFrameRecognizer(const char *python_class_name,
                                           const char *session_dictionary_name);

PyObject *
LLDBSwigPython_GetRecognizedArguments(PyObject *implementor,
                                      const lldb::StackFrameSP &frame_sp);

bool LLDBSWIGPythonRunScriptKeywordProcess(const char *python_function_name,
                                           const char *session_dictionary_name,
                                           const lldb::ProcessSP &process,
                                           std::string &output);

llvm::Optional<std::string>
LLDBSWIGPythonRunScriptKeywordThread(const char *python_function_name,
                                     const char *session_dictionary_name,
                                     lldb::ThreadSP thread);

bool LLDBSWIGPythonRunScriptKeywordTarget(const char *python_function_name,
                                          const char *session_dictionary_name,
                                          const lldb::TargetSP &target,
                                          std::string &output);

llvm::Optional<std::string>
LLDBSWIGPythonRunScriptKeywordFrame(const char *python_function_name,
                                    const char *session_dictionary_name,
                                    lldb::StackFrameSP frame);

bool LLDBSWIGPythonRunScriptKeywordValue(const char *python_function_name,
                                         const char *session_dictionary_name,
                                         const lldb::ValueObjectSP &value,
                                         std::string &output);

void *LLDBSWIGPython_GetDynamicSetting(void *module, const char *setting,
                                       const lldb::TargetSP &target_sp);

} // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
#endif // LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_SWIGPYTHONBRIDGE_H
