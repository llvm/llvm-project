//===-- SwiftArray.cpp ------------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See http://swift.org/LICENSE.txt for license information
// See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "SwiftArray.h"

#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/SwiftASTContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SwiftLanguageRuntime.h"
#include "lldb/Target/Target.h"

// FIXME: we should not need this
#include "Plugins/Language/ObjC/Cocoa.h"

#include "swift/AST/ASTContext.h"
#include "llvm/ADT/StringRef.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace lldb_private::formatters::swift;

size_t SwiftArrayNativeBufferHandler::GetCount() { return m_size; }

size_t SwiftArrayNativeBufferHandler::GetCapacity() { return m_capacity; }

lldb_private::CompilerType SwiftArrayNativeBufferHandler::GetElementType() {
  return m_elem_type;
}

ValueObjectSP SwiftArrayNativeBufferHandler::GetElementAtIndex(size_t idx) {
  if (idx >= m_size)
    return ValueObjectSP();

  lldb::addr_t child_location = m_first_elem_ptr + idx * m_element_stride;

  ProcessSP process_sp(m_exe_ctx_ref.GetProcessSP());
  if (!process_sp)
    return ValueObjectSP();

  DataBufferSP buffer(new DataBufferHeap(m_element_size, 0));
  Error error;
  if (process_sp->ReadMemory(child_location, buffer->GetBytes(), m_element_size,
                             error) != m_element_size ||
      error.Fail())
    return ValueObjectSP();
  DataExtractor data(buffer, process_sp->GetByteOrder(),
                     process_sp->GetAddressByteSize());
  StreamString name;
  name.Printf("[%zu]", idx);
  return ValueObject::CreateValueObjectFromData(name.GetData(), data,
                                                m_exe_ctx_ref, m_elem_type);
}

SwiftArrayNativeBufferHandler::SwiftArrayNativeBufferHandler(
    ValueObject &valobj, lldb::addr_t native_ptr, CompilerType elem_type)
    : m_metadata_ptr(LLDB_INVALID_ADDRESS),
      m_reserved_word(LLDB_INVALID_ADDRESS), m_size(0), m_capacity(0),
      m_first_elem_ptr(LLDB_INVALID_ADDRESS), m_elem_type(elem_type),
      m_element_size(elem_type.GetByteSize(nullptr)),
      m_element_stride(elem_type.GetByteStride()),
      m_exe_ctx_ref(valobj.GetExecutionContextRef()) {
  if (native_ptr == LLDB_INVALID_ADDRESS)
    return;
  if (native_ptr == 0) {
    // 0 is a valid value for the pointer here - it just means empty
    // never-written-to array
    m_metadata_ptr = 0;
    m_reserved_word = 0;
    m_size = m_capacity = 0;
    m_first_elem_ptr = 0;
    return;
  }
  ProcessSP process_sp(m_exe_ctx_ref.GetProcessSP());
  if (!process_sp)
    return;
  size_t ptr_size = process_sp->GetAddressByteSize();
  Error error;
  lldb::addr_t next_read = native_ptr;
  m_metadata_ptr = process_sp->ReadPointerFromMemory(next_read, error);
  if (error.Fail())
    return;
  next_read += ptr_size;
  m_reserved_word =
      process_sp->ReadUnsignedIntegerFromMemory(next_read, 8, 0, error);
  if (error.Fail())
    return;
  next_read += 8;
  m_size =
      process_sp->ReadUnsignedIntegerFromMemory(next_read, ptr_size, 0, error);
  if (error.Fail())
    return;
  next_read += ptr_size;
  m_capacity =
      process_sp->ReadUnsignedIntegerFromMemory(next_read, ptr_size, 0, error);
  if (error.Fail())
    return;
  next_read += ptr_size;
  m_first_elem_ptr = next_read;
}

bool SwiftArrayNativeBufferHandler::IsValid() {
  return m_metadata_ptr != LLDB_INVALID_ADDRESS &&
         m_first_elem_ptr != LLDB_INVALID_ADDRESS && m_capacity >= m_size &&
         m_elem_type.IsValid();
}

size_t SwiftArrayBridgedBufferHandler::GetCount() {
  return m_frontend->CalculateNumChildren();
}

size_t SwiftArrayBridgedBufferHandler::GetCapacity() { return GetCount(); }

lldb_private::CompilerType SwiftArrayBridgedBufferHandler::GetElementType() {
  return m_elem_type;
}

lldb::ValueObjectSP
SwiftArrayBridgedBufferHandler::GetElementAtIndex(size_t idx) {
  return m_frontend->GetChildAtIndex(idx);
}

SwiftArrayBridgedBufferHandler::SwiftArrayBridgedBufferHandler(
    ProcessSP process_sp, lldb::addr_t native_ptr)
    : SwiftArrayBufferHandler(), m_elem_type(), m_synth_array_sp(),
      m_frontend(nullptr) {
  m_elem_type =
      process_sp->GetTarget().GetScratchClangASTContext()->GetBasicType(
          lldb::eBasicTypeObjCID);
  InferiorSizedWord isw(native_ptr, *process_sp);
  m_synth_array_sp = ValueObjectConstResult::CreateValueObjectFromData(
      "_", isw.GetAsData(process_sp->GetByteOrder()), *process_sp, m_elem_type);
  if ((m_frontend = NSArraySyntheticFrontEndCreator(nullptr, m_synth_array_sp)))
    m_frontend->Update();
}

bool SwiftArrayBridgedBufferHandler::IsValid() {
  return m_synth_array_sp.get() != nullptr && m_frontend != nullptr;
}

size_t SwiftArraySliceBufferHandler::GetCount() { return m_size; }

size_t SwiftArraySliceBufferHandler::GetCapacity() {
  // Slices don't have a separate capacity - at least not in any obvious sense
  return m_size;
}

lldb_private::CompilerType SwiftArraySliceBufferHandler::GetElementType() {
  return m_elem_type;
}

lldb::ValueObjectSP
SwiftArraySliceBufferHandler::GetElementAtIndex(size_t idx) {
  if (idx >= m_size)
    return ValueObjectSP();

  const uint64_t effective_idx = idx + m_start_index;

  lldb::addr_t child_location =
      m_first_elem_ptr + effective_idx * m_element_stride;

  ProcessSP process_sp(m_exe_ctx_ref.GetProcessSP());
  if (!process_sp)
    return ValueObjectSP();

  DataBufferSP buffer(new DataBufferHeap(m_element_size, 0));
  Error error;
  if (process_sp->ReadMemory(child_location, buffer->GetBytes(), m_element_size,
                             error) != m_element_size ||
      error.Fail())
    return ValueObjectSP();
  DataExtractor data(buffer, process_sp->GetByteOrder(),
                     process_sp->GetAddressByteSize());
  StreamString name;
  name.Printf("[%" PRIu64 "]", effective_idx);
  return ValueObject::CreateValueObjectFromData(name.GetData(), data,
                                                m_exe_ctx_ref, m_elem_type);
}

// this gets passed the "buffer" element?
SwiftArraySliceBufferHandler::SwiftArraySliceBufferHandler(
    ValueObject &valobj, CompilerType elem_type)
    : m_size(0), m_first_elem_ptr(LLDB_INVALID_ADDRESS), m_elem_type(elem_type),
      m_element_size(elem_type.GetByteSize(nullptr)),
      m_element_stride(elem_type.GetByteStride()),
      m_exe_ctx_ref(valobj.GetExecutionContextRef()), m_native_buffer(false),
      m_start_index(0) {
  static ConstString g_start("subscriptBaseAddress");
  static ConstString g_value("_value");
  static ConstString g__rawValue("_rawValue");
  static ConstString g__countAndFlags("endIndexAndFlags");
  static ConstString g__startIndex("startIndex");

  ProcessSP process_sp(m_exe_ctx_ref.GetProcessSP());
  if (!process_sp)
    return;

  ValueObjectSP value_sp(valobj.GetChildAtNamePath({g_start, g__rawValue}));
  if (!value_sp)
    return;

  m_first_elem_ptr = value_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);

  ValueObjectSP _countAndFlags_sp(
      valobj.GetChildAtNamePath({g__countAndFlags, g_value}));

  if (!_countAndFlags_sp)
    return;

  ValueObjectSP startIndex_sp(
      valobj.GetChildAtNamePath({g__startIndex, g_value}));

  if (startIndex_sp)
    m_start_index = startIndex_sp->GetValueAsUnsigned(0);

  InferiorSizedWord isw(_countAndFlags_sp->GetValueAsUnsigned(0), *process_sp);

  m_size = (isw >> 1).GetValue() - m_start_index;

  m_native_buffer = !((isw & 1).IsZero());
}

bool SwiftArraySliceBufferHandler::IsValid() {
  return m_first_elem_ptr != LLDB_INVALID_ADDRESS && m_elem_type.IsValid();
}

size_t SwiftSyntheticFrontEndBufferHandler::GetCount() {
  return m_frontend->CalculateNumChildren();
}

size_t SwiftSyntheticFrontEndBufferHandler::GetCapacity() {
  return m_frontend->CalculateNumChildren();
}

lldb_private::CompilerType
SwiftSyntheticFrontEndBufferHandler::GetElementType() {
  // this doesn't make sense here - the synthetic children know best
  return CompilerType();
}

lldb::ValueObjectSP
SwiftSyntheticFrontEndBufferHandler::GetElementAtIndex(size_t idx) {
  return m_frontend->GetChildAtIndex(idx);
}

// this receives a pointer to the NSArray
SwiftSyntheticFrontEndBufferHandler::SwiftSyntheticFrontEndBufferHandler(
    ValueObjectSP valobj_sp)
    : m_valobj_sp(valobj_sp),
      m_frontend(NSArraySyntheticFrontEndCreator(nullptr, valobj_sp)) {
  // Cocoa NSArray frontends must be updated before use
  if (m_frontend)
    m_frontend->Update();
}

bool SwiftSyntheticFrontEndBufferHandler::IsValid() {
  return m_frontend.get() != nullptr;
}

std::unique_ptr<SwiftArrayBufferHandler>
SwiftArrayBufferHandler::CreateBufferHandler(ValueObject &valobj) {
  llvm::StringRef valobj_typename(
      valobj.GetCompilerType().GetTypeName().AsCString(""));

  if (valobj_typename.startswith("Swift._NSSwiftArray")) {
    CompilerType anyobject_type =
        valobj.GetTargetSP()->GetScratchClangASTContext()->GetBasicType(
            lldb::eBasicTypeObjCID);
    auto handler = std::unique_ptr<SwiftArrayBufferHandler>(
        new SwiftArrayNativeBufferHandler(valobj, valobj.GetPointerValue(),
                                          anyobject_type));
    if (handler && handler->IsValid())
      return handler;
    return nullptr;
  }

  if (valobj_typename.startswith("_TtCs23_ContiguousArrayStorage") ||
      valobj_typename.startswith("Swift._ContiguousArrayStorage")) {
    CompilerType anyobject_type =
        valobj.GetTargetSP()->GetScratchClangASTContext()->GetBasicType(
            lldb::eBasicTypeObjCID);
    auto handler = std::unique_ptr<SwiftArrayBufferHandler>(
        new SwiftArrayNativeBufferHandler(
            valobj, valobj.GetValueAsUnsigned(LLDB_INVALID_ADDRESS),
            anyobject_type));
    if (handler && handler->IsValid())
      return handler;
    return nullptr;
  }

  if (valobj_typename.startswith("_TtCs21_SwiftDeferredNSArray")) {
    ProcessSP process_sp(valobj.GetProcessSP());
    if (!process_sp)
      return nullptr;
    Error error;

    lldb::addr_t buffer_ptr = valobj.GetValueAsUnsigned(LLDB_INVALID_ADDRESS) +
                              3 * process_sp->GetAddressByteSize();
    buffer_ptr = process_sp->ReadPointerFromMemory(buffer_ptr, error);
    if (error.Fail() || buffer_ptr == LLDB_INVALID_ADDRESS)
      return nullptr;

    lldb::addr_t argmetadata_ptr =
        process_sp->ReadPointerFromMemory(buffer_ptr, error);
    if (error.Fail() || argmetadata_ptr == LLDB_INVALID_ADDRESS)
      return nullptr;

    SwiftLanguageRuntime *swift_runtime = process_sp->GetSwiftLanguageRuntime();
    if (!swift_runtime)
      return nullptr;

    CompilerType argument_type;

    SwiftASTContext *swift_ast_ctx(llvm::dyn_cast_or_null<SwiftASTContext>(
        valobj.GetCompilerType().GetTypeSystem()));
    SwiftLanguageRuntime::MetadataPromiseSP promise_sp(
        swift_runtime->GetMetadataPromise(argmetadata_ptr, swift_ast_ctx));
    if (promise_sp) {
      if (CompilerType type = promise_sp->FulfillTypePromise()) {
        lldb::TemplateArgumentKind kind;
        argument_type = type.GetTemplateArgument(0, kind);
      }
    }

    if (!argument_type.IsValid())
      return nullptr;

    auto handler = std::unique_ptr<SwiftArrayBufferHandler>(
        new SwiftArrayNativeBufferHandler(valobj, buffer_ptr, argument_type));
    if (handler && handler->IsValid())
      return handler;
    return nullptr;
  }

  if (valobj_typename.startswith("Swift.NativeArray<")) {
    // Swift.NativeArray
    static ConstString g_buffer("_buffer");
    static ConstString g_base("base");
    static ConstString g_storage("storage");
    static ConstString g_some("Some");

    ValueObjectSP some_sp(valobj.GetNonSyntheticValue()->GetChildAtNamePath(
        {g_buffer, g_base, g_storage, g_some}));

    if (!some_sp)
      return nullptr;

    CompilerType elem_type(valobj.GetCompilerType().GetArrayElementType());

    auto handler = std::unique_ptr<SwiftArrayBufferHandler>(
        new SwiftArrayNativeBufferHandler(
            *some_sp, some_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS),
            elem_type));
    if (handler && handler->IsValid())
      return handler;
    return nullptr;
  } else if (valobj_typename.startswith("Swift.ArraySlice<")) {
    // Swift.ArraySlice
    static ConstString g_buffer("_buffer");

    ValueObjectSP buffer_sp(
        valobj.GetNonSyntheticValue()->GetChildAtNamePath({g_buffer}));
    if (!buffer_sp)
      return nullptr;

    CompilerType elem_type(valobj.GetCompilerType().GetArrayElementType());

    auto handler = std::unique_ptr<SwiftArrayBufferHandler>(
        new SwiftArraySliceBufferHandler(*buffer_sp, elem_type));
    if (handler && handler->IsValid())
      return handler;
    return nullptr;
  } else {
    // Swift.Array
    static ConstString g_buffer("_buffer");
    static ConstString g__storage("_storage");
    static ConstString g_rawValue("rawValue");

    static ConstString g___bufferPointer("__bufferPointer");
    static ConstString g__nativeBuffer("_nativeBuffer");

    ValueObjectSP buffer_sp(valobj.GetNonSyntheticValue()->GetChildAtNamePath(
        {g_buffer, g__storage, g_rawValue}));

    if (!buffer_sp)
      buffer_sp = valobj.GetNonSyntheticValue()->GetChildAtNamePath(
          {g_buffer, g___bufferPointer, g__nativeBuffer});

    if (!buffer_sp)
      return nullptr;

    lldb::addr_t storage_location =
        buffer_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);

    if (storage_location != LLDB_INVALID_ADDRESS) {
      ProcessSP process_sp(valobj.GetProcessSP());
      if (!process_sp)
        return nullptr;
      SwiftLanguageRuntime *swift_runtime =
          process_sp->GetSwiftLanguageRuntime();
      if (!swift_runtime)
        return nullptr;
      lldb::addr_t masked_storage_location =
          swift_runtime->MaskMaybeBridgedPointer(storage_location);

      std::unique_ptr<SwiftArrayBufferHandler> handler;
      if (masked_storage_location == storage_location) {
        CompilerType elem_type(valobj.GetCompilerType().GetArrayElementType());
        handler.reset(new SwiftArrayNativeBufferHandler(
            valobj, storage_location, elem_type));
      } else {
        handler.reset(new SwiftArrayBridgedBufferHandler(
            process_sp, masked_storage_location));
      }

      if (handler && handler->IsValid())
        return handler;
      return nullptr;
    } else {
      CompilerType elem_type(valobj.GetCompilerType().GetArrayElementType());
      return std::unique_ptr<SwiftArrayBufferHandler>(
          new SwiftArrayEmptyBufferHandler(elem_type));
    }
  }

  return nullptr;
}

bool lldb_private::formatters::swift::Array_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  auto handler = SwiftArrayBufferHandler::CreateBufferHandler(valobj);

  if (!handler)
    return false;

  auto count = handler->GetCount();

  stream.Printf("%zu value%s", count, (count == 1 ? "" : "s"));

  return true;
};

lldb_private::formatters::swift::ArraySyntheticFrontEnd::ArraySyntheticFrontEnd(
    lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp.get()), m_array_buffer() {
  if (valobj_sp)
    Update();
}

size_t lldb_private::formatters::swift::ArraySyntheticFrontEnd::
    CalculateNumChildren() {
  return m_array_buffer ? m_array_buffer->GetCount() : 0;
}

lldb::ValueObjectSP
lldb_private::formatters::swift::ArraySyntheticFrontEnd::GetChildAtIndex(
    size_t idx) {
  if (!m_array_buffer)
    return ValueObjectSP();

  lldb::ValueObjectSP child_sp = m_array_buffer->GetElementAtIndex(idx);
  if (child_sp)
    child_sp->SetSyntheticChildrenGenerated(true);

  return child_sp;
}

bool lldb_private::formatters::swift::ArraySyntheticFrontEnd::Update() {
  m_array_buffer = SwiftArrayBufferHandler::CreateBufferHandler(m_backend);
  return false;
}

bool lldb_private::formatters::swift::ArraySyntheticFrontEnd::IsValid() {
  if (m_array_buffer)
    return m_array_buffer->IsValid();
  return false;
}

bool lldb_private::formatters::swift::ArraySyntheticFrontEnd::
    MightHaveChildren() {
  return true;
}

size_t lldb_private::formatters::swift::ArraySyntheticFrontEnd::
    GetIndexOfChildWithName(const ConstString &name) {
  if (!m_array_buffer)
    return UINT32_MAX;
  const char *item_name = name.GetCString();
  uint32_t idx = ExtractIndexFromString(item_name);
  if (idx < UINT32_MAX && idx >= CalculateNumChildren())
    return UINT32_MAX;
  return idx;
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::swift::ArraySyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;

  ArraySyntheticFrontEnd *front_end = new ArraySyntheticFrontEnd(valobj_sp);
  if (front_end && front_end->IsValid())
    return front_end;
  return nullptr;
}
