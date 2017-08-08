//===-- SwiftArray.h --------------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftArray_h_
#define liblldb_SwiftArray_h_

#include "lldb/lldb-forward.h"

#include "lldb/Utility/ConstString.h"
#include "lldb/DataFormatters/FormatClasses.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Target/Target.h"

namespace lldb_private {
namespace formatters {
namespace swift {

// Some part of the buffer handling logic needs to be shared between summary and
// synthetic children
// If I was only making synthetic children, this would be best modelled as
// different FrontEnds
class SwiftArrayBufferHandler {
public:
  virtual size_t GetCount() = 0;

  virtual size_t GetCapacity() = 0;

  virtual lldb_private::CompilerType GetElementType() = 0;

  virtual lldb::ValueObjectSP GetElementAtIndex(size_t) = 0;

  static std::unique_ptr<SwiftArrayBufferHandler>
  CreateBufferHandler(ValueObject &valobj);

  virtual bool IsValid() = 0;

  virtual ~SwiftArrayBufferHandler() {}

protected:
  static bool DoesTypeEntailIndirectBuffer(const CompilerType &element_type);
};

class SwiftArrayEmptyBufferHandler : public SwiftArrayBufferHandler {
public:
  virtual size_t GetCount() { return 0; }

  virtual size_t GetCapacity() { return 0; }

  virtual lldb_private::CompilerType GetElementType() { return m_elem_type; }

  virtual lldb::ValueObjectSP GetElementAtIndex(size_t) {
    return lldb::ValueObjectSP();
  }

  virtual ~SwiftArrayEmptyBufferHandler() {}

  virtual bool IsValid() { return true; }

protected:
  SwiftArrayEmptyBufferHandler(CompilerType elem_type)
      : m_elem_type(elem_type) {}
  friend class SwiftArrayBufferHandler;

private:
  lldb_private::CompilerType m_elem_type;
};

class SwiftArrayNativeBufferHandler : public SwiftArrayBufferHandler {
public:
  virtual size_t GetCount();

  virtual size_t GetCapacity();

  virtual lldb_private::CompilerType GetElementType();

  virtual lldb::ValueObjectSP GetElementAtIndex(size_t);

  virtual bool IsValid();

  virtual ~SwiftArrayNativeBufferHandler() {}

protected:
  SwiftArrayNativeBufferHandler(ValueObject &valobj, lldb::addr_t native_ptr,
                                CompilerType elem_type);
  friend class SwiftArrayBufferHandler;

private:
  lldb::addr_t m_metadata_ptr;
  uint64_t m_reserved_word;
  lldb::addr_t m_size;
  lldb::addr_t m_capacity;
  lldb::addr_t m_first_elem_ptr;
  lldb_private::CompilerType m_elem_type;
  size_t m_element_size;
  size_t m_element_stride;
  lldb_private::ExecutionContextRef m_exe_ctx_ref;
};

class SwiftArrayBridgedBufferHandler : public SwiftArrayBufferHandler {
public:
  virtual size_t GetCount();

  virtual size_t GetCapacity();

  virtual lldb_private::CompilerType GetElementType();

  virtual lldb::ValueObjectSP GetElementAtIndex(size_t);

  virtual bool IsValid();

  virtual ~SwiftArrayBridgedBufferHandler() {}

protected:
  SwiftArrayBridgedBufferHandler(lldb::ProcessSP, lldb::addr_t);
  friend class SwiftArrayBufferHandler;

private:
  CompilerType m_elem_type;
  lldb::ValueObjectSP m_synth_array_sp;
  SyntheticChildrenFrontEnd *m_frontend;
};

class SwiftArraySliceBufferHandler : public SwiftArrayBufferHandler {
public:
  virtual size_t GetCount();

  virtual size_t GetCapacity();

  virtual lldb_private::CompilerType GetElementType();

  virtual lldb::ValueObjectSP GetElementAtIndex(size_t);

  virtual bool IsValid();

  virtual ~SwiftArraySliceBufferHandler() {}

protected:
  SwiftArraySliceBufferHandler(ValueObject &valobj, CompilerType elem_type);
  friend class SwiftArrayBufferHandler;

private:
  lldb::addr_t m_size;
  lldb::addr_t m_first_elem_ptr;
  lldb_private::CompilerType m_elem_type;
  size_t m_element_size;
  size_t m_element_stride;
  lldb_private::ExecutionContextRef m_exe_ctx_ref;
  bool m_native_buffer;
  uint64_t m_start_index;
};

class SwiftSyntheticFrontEndBufferHandler : public SwiftArrayBufferHandler {
public:
  virtual size_t GetCount();

  virtual size_t GetCapacity();

  virtual lldb_private::CompilerType GetElementType();

  virtual lldb::ValueObjectSP GetElementAtIndex(size_t);

  virtual bool IsValid();

  virtual ~SwiftSyntheticFrontEndBufferHandler() {}

protected:
  SwiftSyntheticFrontEndBufferHandler(lldb::ValueObjectSP valobj_sp);
  friend class SwiftArrayBufferHandler;

private:
  lldb::ValueObjectSP m_valobj_sp; // reader beware: this entails you must only
                                   // pass self-rooted valueobjects to this
                                   // class
  std::unique_ptr<SyntheticChildrenFrontEnd> m_frontend;
};

bool Array_SummaryProvider(ValueObject &valobj, Stream &stream,
                           const TypeSummaryOptions &options);

class ArraySyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  ArraySyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  virtual size_t CalculateNumChildren();

  virtual lldb::ValueObjectSP GetChildAtIndex(size_t idx);

  virtual bool Update();

  virtual bool MightHaveChildren();

  virtual size_t GetIndexOfChildWithName(const ConstString &name);

  virtual ~ArraySyntheticFrontEnd() = default;

  bool IsValid();

private:
  std::unique_ptr<SwiftArrayBufferHandler> m_array_buffer;
};

SyntheticChildrenFrontEnd *ArraySyntheticFrontEndCreator(CXXSyntheticChildren *,
                                                         lldb::ValueObjectSP);
}
}
}

#endif // liblldb_SwiftArray_h_
