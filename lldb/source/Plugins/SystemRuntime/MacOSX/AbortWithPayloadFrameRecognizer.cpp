//===-- AbortWithPayloadFrameRecognizer.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AbortWithPayloadFrameRecognizer.h"

#include "lldb/Core/Value.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/ValueObject/ValueObjectConstResult.h"

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"

using namespace lldb;
using namespace lldb_private;

namespace lldb_private {
void RegisterAbortWithPayloadFrameRecognizer(Process *process) {
  // There are two user-level API's that this recognizer captures,
  // abort_with_reason and abort_with_payload.  But they both call the private
  // __abort_with_payload, the abort_with_reason call fills in a null payload.
  static ConstString module_name("libsystem_kernel.dylib");
  static ConstString sym_name("__abort_with_payload");

  if (!process)
    return;

  process->GetTarget().GetFrameRecognizerManager().AddRecognizer(
      std::make_shared<AbortWithPayloadFrameRecognizer>(), module_name,
      sym_name, Mangled::NamePreference::ePreferDemangled,
      /*first_instruction_only*/ false);
}

RecognizedStackFrameSP
AbortWithPayloadFrameRecognizer::RecognizeFrame(lldb::StackFrameSP frame_sp) {
  // We have two jobs:
  // 1) to add the data passed to abort_with_payload to the
  //    ExtraCrashInformation dictionary.
  // 2) To make up faux arguments for this frame.
  static constexpr llvm::StringLiteral namespace_key("namespace");
  static constexpr llvm::StringLiteral code_key("code");
  static constexpr llvm::StringLiteral payload_addr_key("payload_addr");
  static constexpr llvm::StringLiteral payload_size_key("payload_size");
  static constexpr llvm::StringLiteral reason_key("reason");
  static constexpr llvm::StringLiteral flags_key("flags");
  static constexpr llvm::StringLiteral info_key("abort_with_payload");

  Log *log = GetLog(LLDBLog::SystemRuntime);
  
  if (!frame_sp) {
    LLDB_LOG(log, "abort_with_payload recognizer: invalid frame.");
    return {};
  }

  Thread *thread = frame_sp->GetThread().get();
  if (!thread) {
    LLDB_LOG(log, "abort_with_payload recognizer: invalid thread.");
    return {};
  }

  Process *process = thread->GetProcess().get();
  if (!thread) {
    LLDB_LOG(log, "abort_with_payload recognizer: invalid process.");
  }

  TypeSystemClangSP scratch_ts_sp =
      ScratchTypeSystemClang::GetForTarget(process->GetTarget());
  if (!scratch_ts_sp) {
    LLDB_LOG(log, "abort_with_payload recognizer: invalid scratch typesystem.");
    return {};
  }

  // The abort_with_payload signature is:
  // abort_with_payload(uint32_t reason_namespace, uint64_t reason_code,
  //                      void* payload, uint32_t payload_size,
  //                      const char* reason_string, uint64_t reason_flags);

  ValueList arg_values;
  Value input_value_32;
  Value input_value_64;
  Value input_value_void_ptr;
  Value input_value_char_ptr;

  CompilerType clang_void_ptr_type =
      scratch_ts_sp->GetBasicType(eBasicTypeVoid).GetPointerType();
  CompilerType clang_char_ptr_type =
      scratch_ts_sp->GetBasicType(eBasicTypeChar).GetPointerType();
  CompilerType clang_uint64_type =
      scratch_ts_sp->GetBuiltinTypeForEncodingAndBitSize(lldb::eEncodingUint,
                                                         64);
  CompilerType clang_uint32_type =
      scratch_ts_sp->GetBuiltinTypeForEncodingAndBitSize(lldb::eEncodingUint,
                                                         32);
  CompilerType clang_char_star_type =
      scratch_ts_sp->GetBuiltinTypeForEncodingAndBitSize(lldb::eEncodingUint,
                                                         64);

  input_value_32.SetValueType(Value::ValueType::Scalar);
  input_value_32.SetCompilerType(clang_uint32_type);
  input_value_64.SetValueType(Value::ValueType::Scalar);
  input_value_64.SetCompilerType(clang_uint64_type);
  input_value_void_ptr.SetValueType(Value::ValueType::Scalar);
  input_value_void_ptr.SetCompilerType(clang_void_ptr_type);
  input_value_char_ptr.SetValueType(Value::ValueType::Scalar);
  input_value_char_ptr.SetCompilerType(clang_char_ptr_type);

  arg_values.PushValue(input_value_32);
  arg_values.PushValue(input_value_64);
  arg_values.PushValue(input_value_void_ptr);
  arg_values.PushValue(input_value_32);
  arg_values.PushValue(input_value_char_ptr);
  arg_values.PushValue(input_value_64);

  lldb::ABISP abi_sp = process->GetABI();
  bool success = abi_sp->GetArgumentValues(*thread, arg_values);
  if (!success)
    return {};

  Value *cur_value;
  StackFrame *frame = frame_sp.get();
  ValueObjectListSP arguments_sp = ValueObjectListSP(new ValueObjectList());

  auto add_to_arguments = [&](llvm::StringRef name, Value *value,
                              bool dynamic) {
    ValueObjectSP cur_valobj_sp =
        ValueObjectConstResult::Create(frame, *value, ConstString(name));
    cur_valobj_sp = ValueObjectRecognizerSynthesizedValue::Create(
        *cur_valobj_sp, eValueTypeVariableArgument);
    ValueObjectSP dyn_valobj_sp;
    if (dynamic) {
      dyn_valobj_sp = cur_valobj_sp->GetDynamicValue(eDynamicDontRunTarget);
      if (dyn_valobj_sp)
        cur_valobj_sp = dyn_valobj_sp;
    }
    arguments_sp->Append(cur_valobj_sp);
  };

  // Decode the arg_values:

  uint32_t namespace_val = 0;
  cur_value = arg_values.GetValueAtIndex(0);
  add_to_arguments(namespace_key, cur_value, false);
  namespace_val = cur_value->GetScalar().UInt(namespace_val);

  uint32_t code_val = 0;
  cur_value = arg_values.GetValueAtIndex(1);
  add_to_arguments(code_key, cur_value, false);
  code_val = cur_value->GetScalar().UInt(code_val);

  lldb::addr_t payload_addr = LLDB_INVALID_ADDRESS;
  cur_value = arg_values.GetValueAtIndex(2);
  add_to_arguments(payload_addr_key, cur_value, true);
  payload_addr = cur_value->GetScalar().ULongLong(payload_addr);

  uint32_t payload_size = 0;
  cur_value = arg_values.GetValueAtIndex(3);
  add_to_arguments(payload_size_key, cur_value, false);
  payload_size = cur_value->GetScalar().UInt(payload_size);

  lldb::addr_t reason_addr = LLDB_INVALID_ADDRESS;
  cur_value = arg_values.GetValueAtIndex(4);
  add_to_arguments(reason_key, cur_value, false);
  reason_addr = cur_value->GetScalar().ULongLong(payload_addr);

  // For the reason string, we want the string not the address, so fetch that.
  std::string reason_string;
  Status error;
  process->ReadCStringFromMemory(reason_addr, reason_string, error);
  if (error.Fail()) {
    // Even if we couldn't read the string, return the other data.
    LLDB_LOG(log, "Couldn't fetch reason string: {0}.", error);
    reason_string = "<error fetching reason string>";
  }

  uint32_t flags_val = 0;
  cur_value = arg_values.GetValueAtIndex(5);
  add_to_arguments(flags_key, cur_value, false);
  flags_val = cur_value->GetScalar().UInt(flags_val);

  // Okay, we've gotten all the argument values, now put them in a
  // StructuredData, and add that to the Process ExtraCrashInformation:
  StructuredData::DictionarySP abort_dict_sp(new StructuredData::Dictionary());
  abort_dict_sp->AddIntegerItem(namespace_key, namespace_val);
  abort_dict_sp->AddIntegerItem(code_key, code_val);
  abort_dict_sp->AddIntegerItem(payload_addr_key, payload_addr);
  abort_dict_sp->AddIntegerItem(payload_size_key, payload_size);
  abort_dict_sp->AddStringItem(reason_key, reason_string);
  abort_dict_sp->AddIntegerItem(flags_key, flags_val);

  // This will overwrite the abort_with_payload information in the dictionary  
  // already.  But we can only crash on abort_with_payload once, so that 
  // shouldn't matter.
  process->GetExtendedCrashInfoDict()->AddItem(info_key, abort_dict_sp);

  return RecognizedStackFrameSP(
      new AbortWithPayloadRecognizedStackFrame(frame_sp, arguments_sp));
}

AbortWithPayloadRecognizedStackFrame::AbortWithPayloadRecognizedStackFrame(
    lldb::StackFrameSP &frame_sp, ValueObjectListSP &args_sp)
    : RecognizedStackFrame() {
  m_arguments = args_sp;
  m_stop_desc = "abort with payload or reason";
}

} // namespace lldb_private
