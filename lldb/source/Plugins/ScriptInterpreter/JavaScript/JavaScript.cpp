//===-- JavaScript.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JavaScript.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Utility/FileSpec.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include <libplatform/libplatform.h>
#include <v8.h>

using namespace lldb_private;
using namespace lldb;

// SWIG-generated init function (SWIGV8_INIT is a macro that expands to
// lldb_initialize)
extern "C" void lldb_initialize(v8::Local<v8::Object> exports,
                                v8::Local<v8::Object> module);

// Static V8 platform (initialized once)
std::unique_ptr<v8::Platform> JavaScript::s_platform;
bool JavaScript::s_platform_initialized = false;

// Helper to format and write output
static void
WriteFormattedOutput(const v8::FunctionCallbackInfo<v8::Value> &args,
                     bool add_newline = true) {
  v8::Isolate *isolate = args.GetIsolate();
  v8::HandleScope handle_scope(isolate);

  v8::Local<v8::Context> context = isolate->GetCurrentContext();
  JavaScript *js_instance =
      static_cast<JavaScript *>(context->GetAlignedPointerFromEmbedderData(1));

  std::string output;
  for (int i = 0; i < args.Length(); i++) {
    if (i > 0)
      output += " ";
    v8::String::Utf8Value str(isolate, args[i]);
    output += *str;
  }
  if (add_newline)
    output += "\n";

  if (js_instance) {
    js_instance->WriteOutput(output);
  } else {
    printf("%s", output.c_str());
    fflush(stdout);
  }
}

// Console.log implementation
static void ConsoleLog(const v8::FunctionCallbackInfo<v8::Value> &args) {
  WriteFormattedOutput(args, true);
}

// Console.warn implementation
static void ConsoleWarn(const v8::FunctionCallbackInfo<v8::Value> &args) {
  WriteFormattedOutput(args, true);
}

// Console.error implementation
static void ConsoleError(const v8::FunctionCallbackInfo<v8::Value> &args) {
  WriteFormattedOutput(args, true);
}

void JavaScript::InitializePlatform() {
  if (s_platform_initialized)
    return;

  v8::V8::InitializeICUDefaultLocation("");
  v8::V8::InitializeExternalStartupData("");
  s_platform = v8::platform::NewDefaultPlatform();
  v8::V8::InitializePlatform(s_platform.get());
  v8::V8::Initialize();

  s_platform_initialized = true;
}

JavaScript::JavaScript(lldb::FileSP output_file)
    : m_stdout(stdout), m_stderr(stderr), m_output_file(output_file) {
  InitializePlatform();

  // Create isolate
  v8::Isolate::CreateParams create_params;
  create_params.array_buffer_allocator =
      v8::ArrayBuffer::Allocator::NewDefaultAllocator();
  m_isolate = v8::Isolate::New(create_params);

  // Create context
  v8::Isolate::Scope isolate_scope(m_isolate);
  v8::HandleScope handle_scope(m_isolate);

  v8::Local<v8::Context> context = v8::Context::New(m_isolate);
  m_context = new v8::Global<v8::Context>(m_isolate, context);

  // Initialize SWIG bindings
  v8::Context::Scope context_scope(context);
  v8::Local<v8::Object> lldb_module = v8::Object::New(m_isolate);
  v8::Local<v8::Object> empty_module = v8::Object::New(m_isolate);

  lldb_initialize(lldb_module, empty_module);

  context->Global()
      ->Set(context,
            v8::String::NewFromUtf8(m_isolate, "lldb").ToLocalChecked(),
            lldb_module)
      .Check();

  v8::Local<v8::Object> console_obj = v8::Object::New(m_isolate);
  console_obj
      ->Set(context, v8::String::NewFromUtf8(m_isolate, "log").ToLocalChecked(),
            v8::Function::New(context, ConsoleLog).ToLocalChecked())
      .Check();
  console_obj
      ->Set(context,
            v8::String::NewFromUtf8(m_isolate, "warn").ToLocalChecked(),
            v8::Function::New(context, ConsoleWarn).ToLocalChecked())
      .Check();
  console_obj
      ->Set(context,
            v8::String::NewFromUtf8(m_isolate, "error").ToLocalChecked(),
            v8::Function::New(context, ConsoleError).ToLocalChecked())
      .Check();
  context->Global()
      ->Set(context,
            v8::String::NewFromUtf8(m_isolate, "console").ToLocalChecked(),
            console_obj)
      .Check();

  context->SetAlignedPointerInEmbedderData(1, this);
}

JavaScript::~JavaScript() {
  // Clear all callbacks
  for (auto &pair : m_breakpoint_callbacks) {
    pair.second.Reset();
  }
  m_breakpoint_callbacks.clear();

  for (auto &pair : m_watchpoint_callbacks) {
    pair.second.Reset();
  }
  m_watchpoint_callbacks.clear();

  if (m_context) {
    m_context->Reset();
    delete m_context;
  }
  if (m_isolate) {
    m_isolate->Dispose();
  }
}

llvm::Error JavaScript::Run(llvm::StringRef code) {
  v8::Isolate::Scope isolate_scope(m_isolate);
  v8::HandleScope handle_scope(m_isolate);
  v8::Local<v8::Context> context = m_context->Get(m_isolate);
  v8::Context::Scope context_scope(context);

  v8::TryCatch try_catch(m_isolate);

  // Compile
  v8::Local<v8::String> source =
      v8::String::NewFromUtf8(m_isolate, code.data(),
                              v8::NewStringType::kNormal, code.size())
          .ToLocalChecked();

  v8::Local<v8::Script> script;
  if (!v8::Script::Compile(context, source).ToLocal(&script)) {
    v8::String::Utf8Value error(m_isolate, try_catch.Exception());
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("Compilation error: {0}\n", *error),
        llvm::inconvertibleErrorCode());
  }

  // Run
  v8::Local<v8::Value> result;
  if (!script->Run(context).ToLocal(&result)) {
    v8::String::Utf8Value error(m_isolate, try_catch.Exception());
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("Runtime error: {0}\n", *error),
        llvm::inconvertibleErrorCode());
  }

  // Print the result if it's not undefined (REPL behavior)
  if (!result->IsUndefined()) {
    v8::String::Utf8Value result_str(m_isolate, result);
    WriteOutput(std::string(*result_str) + "\n");
  }

  return llvm::Error::success();
}

llvm::Error JavaScript::LoadModule(llvm::StringRef filename) {
  const FileSpec file(filename);
  if (!FileSystem::Instance().Exists(file)) {
    return llvm::make_error<llvm::StringError>("File not found",
                                               llvm::inconvertibleErrorCode());
  }

  if (file.GetFileNameExtension() != ".js") {
    return llvm::make_error<llvm::StringError>(
        "Invalid extension (expected .js)", llvm::inconvertibleErrorCode());
  }

  // Read file using llvm MemoryBuffer
  auto buffer_or_error = llvm::MemoryBuffer::getFile(file.GetPath());
  if (!buffer_or_error) {
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("Failed to read file: {0}",
                      buffer_or_error.getError().message()),
        llvm::inconvertibleErrorCode());
  }

  std::unique_ptr<llvm::MemoryBuffer> buffer = std::move(*buffer_or_error);
  llvm::StringRef contents = buffer->getBuffer();

  if (contents.empty()) {
    return llvm::make_error<llvm::StringError>("Empty file",
                                               llvm::inconvertibleErrorCode());
  }

  return Run(contents);
}

llvm::Error JavaScript::CheckSyntax(llvm::StringRef code) {
  v8::Isolate::Scope isolate_scope(m_isolate);
  v8::HandleScope handle_scope(m_isolate);
  v8::Local<v8::Context> context = m_context->Get(m_isolate);
  v8::Context::Scope context_scope(context);

  v8::TryCatch try_catch(m_isolate);

  v8::Local<v8::String> source =
      v8::String::NewFromUtf8(m_isolate, code.data(),
                              v8::NewStringType::kNormal, code.size())
          .ToLocalChecked();

  v8::Local<v8::Script> script;
  if (!v8::Script::Compile(context, source).ToLocal(&script)) {
    v8::String::Utf8Value error(m_isolate, try_catch.Exception());
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("Syntax error: {0}\n", *error),
        llvm::inconvertibleErrorCode());
  }

  return llvm::Error::success();
}

llvm::Error JavaScript::ChangeIO(FILE *out, FILE *err) {
  m_stdout = out;
  m_stderr = err;
  return llvm::Error::success();
}

llvm::Error
JavaScript::RegisterBreakpointCallback(void *baton,
                                       const char *command_body_text) {
  v8::Isolate::Scope isolate_scope(m_isolate);
  v8::HandleScope handle_scope(m_isolate);
  v8::Local<v8::Context> context = m_context->Get(m_isolate);
  v8::Context::Scope context_scope(context);

  v8::TryCatch try_catch(m_isolate);

  v8::Local<v8::String> source =
      v8::String::NewFromUtf8(m_isolate, command_body_text,
                              v8::NewStringType::kNormal,
                              strlen(command_body_text))
          .ToLocalChecked();

  v8::Local<v8::Script> script;
  if (!v8::Script::Compile(context, source).ToLocal(&script)) {
    v8::String::Utf8Value error(m_isolate, try_catch.Exception());
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("Failed to compile callback: {0}", *error),
        llvm::inconvertibleErrorCode());
  }

  v8::Local<v8::Value> result;
  if (!script->Run(context).ToLocal(&result)) {
    v8::String::Utf8Value error(m_isolate, try_catch.Exception());
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("Failed to evaluate callback: {0}", *error),
        llvm::inconvertibleErrorCode());
  }

  if (!result->IsFunction()) {
    return llvm::make_error<llvm::StringError>(
        "Breakpoint callback must be a JavaScript function",
        llvm::inconvertibleErrorCode());
  }

  // Store the function in our map
  v8::Local<v8::Function> callback = result.As<v8::Function>();
  m_breakpoint_callbacks[baton] = v8::Global<v8::Function>(m_isolate, callback);

  return llvm::Error::success();
}

llvm::Expected<bool>
JavaScript::CallBreakpointCallback(void *baton,
                                   lldb::StackFrameSP stop_frame_sp,
                                   lldb::BreakpointLocationSP bp_loc_sp,
                                   StructuredData::ObjectSP extra_args_sp) {
  auto it = m_breakpoint_callbacks.find(baton);
  if (it == m_breakpoint_callbacks.end()) {
    return llvm::make_error<llvm::StringError>(
        "No callback registered for this baton",
        llvm::inconvertibleErrorCode());
  }

  v8::Isolate::Scope isolate_scope(m_isolate);
  v8::HandleScope handle_scope(m_isolate);
  v8::Local<v8::Context> context = m_context->Get(m_isolate);
  v8::Context::Scope context_scope(context);

  v8::TryCatch try_catch(m_isolate);

  v8::Local<v8::Function> callback = it->second.Get(m_isolate);

  v8::Local<v8::Value> args[2] = {v8::Null(m_isolate), v8::Null(m_isolate)};

  v8::Local<v8::Value> result;
  if (!callback->Call(context, context->Global(), 2, args).ToLocal(&result)) {
    v8::String::Utf8Value error(m_isolate, try_catch.Exception());
    WriteOutput(
        llvm::formatv("Breakpoint callback error: {0}\n", *error).str());
    return false;
  }

  bool should_stop = false;
  if (result->IsBoolean()) {
    should_stop = result->BooleanValue(m_isolate);
  }

  return should_stop;
}

llvm::Error
JavaScript::RegisterWatchpointCallback(void *baton,
                                       const char *command_body_text) {
  return llvm::make_error<llvm::StringError>(
      "Watchpoint callbacks not yet implemented",
      llvm::inconvertibleErrorCode());
}

llvm::Expected<bool> JavaScript::CallWatchpointCallback(
    void *baton, lldb::StackFrameSP stop_frame_sp, lldb::WatchpointSP wp_sp) {
  return false;
}

void JavaScript::SetOutputCallback(OutputCallback callback) {
  m_output_callback = callback;
}

void JavaScript::WriteOutput(const std::string &text) {
  if (m_output_callback) {
    m_output_callback(text);
  } else if (m_output_file && m_output_file->IsValid()) {
    m_output_file->Printf("%s", text.c_str());
    m_output_file->Flush();
  } else {
    printf("%s", text.c_str());
    fflush(stdout);
  }
}

void JavaScript::SetDebugger(lldb::DebuggerSP debugger_sp) {
  m_debugger = debugger_sp;

  if (!debugger_sp)
    return;

  v8::Isolate::Scope isolate_scope(m_isolate);
  v8::HandleScope handle_scope(m_isolate);
  v8::Local<v8::Context> context = m_context->Get(m_isolate);
  v8::Context::Scope context_scope(context);

  v8::Local<v8::Value> lldb_val;
  if (!context->Global()
           ->Get(context,
                 v8::String::NewFromUtf8(m_isolate, "lldb").ToLocalChecked())
           .ToLocal(&lldb_val) ||
      !lldb_val->IsObject())
    return;

  v8::Local<v8::Object> lldb_obj = lldb_val.As<v8::Object>();

  std::string js_code = llvm::formatv("lldb.SBDebugger.FindDebuggerWithID({0})",
                                      debugger_sp->GetID())
                            .str();

  v8::TryCatch try_catch(m_isolate);
  v8::Local<v8::String> source =
      v8::String::NewFromUtf8(m_isolate, js_code.c_str(),
                              v8::NewStringType::kNormal, js_code.length())
          .ToLocalChecked();

  v8::Local<v8::Script> script;
  if (!v8::Script::Compile(context, source).ToLocal(&script)) {
    lldb_obj
        ->Set(context,
              v8::String::NewFromUtf8(m_isolate, "debugger").ToLocalChecked(),
              v8::Null(m_isolate))
        .Check();
    return;
  }

  v8::Local<v8::Value> debugger_obj;
  if (!script->Run(context).ToLocal(&debugger_obj)) {
    lldb_obj
        ->Set(context,
              v8::String::NewFromUtf8(m_isolate, "debugger").ToLocalChecked(),
              v8::Null(m_isolate))
        .Check();
    return;
  }

  lldb_obj
      ->Set(context,
            v8::String::NewFromUtf8(m_isolate, "debugger").ToLocalChecked(),
            debugger_obj)
      .Check();
}
