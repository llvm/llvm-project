//===-- SWIGJavaScriptBridge.h --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_JAVASCRIPT_SWIGJAVASCRIPTBRIDGE_H
#define LLVM_LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_JAVASCRIPT_SWIGJAVASCRIPTBRIDGE_H

#include "lldb/lldb-forward.h"
#include "llvm/Support/Error.h"

namespace lldb_private {
class StructuredDataImpl;
} // namespace lldb_private

namespace v8 {
class Isolate;
} // namespace v8

// This will be implemented by SWIG-generated code
extern "C" {
void init_lldb(v8::Isolate *isolate);
}

namespace javascript {

// Bridge functions for calling LLDB from JavaScript
// These will be generated/implemented by SWIG
namespace SWIGBridge {

// TODO: Implement bridge functions
// These are similar to LuaBridge functions but for JavaScript/V8

llvm::Expected<bool> LLDBSwigJavaScriptBreakpointCallbackFunction(
    v8::Isolate *isolate, lldb::StackFrameSP stop_frame_sp,
    lldb::BreakpointLocationSP bp_loc_sp,
    const lldb_private::StructuredDataImpl &extra_args_impl);

llvm::Expected<bool>
LLDBSwigJavaScriptWatchpointCallbackFunction(v8::Isolate *isolate,
                                             lldb::StackFrameSP stop_frame_sp,
                                             lldb::WatchpointSP wp_sp);

} // namespace SWIGBridge

} // namespace javascript

#endif // LLVM_LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_JAVASCRIPT_SWIGJAVASCRIPTBRIDGE_H
