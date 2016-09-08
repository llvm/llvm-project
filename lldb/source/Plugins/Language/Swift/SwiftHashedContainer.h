//===-- SwiftHashedContainer.h ----------------------------------*- C++ -*-===//
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

#ifndef liblldb_SwiftHashedContainer_h_
#define liblldb_SwiftHashedContainer_h_

#include "lldb/lldb-forward.h"

#include "lldb/Core/ConstString.h"
#include "lldb/DataFormatters/FormatClasses.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Target/Target.h"

#include <functional>

namespace lldb_private {
namespace formatters {
namespace swift {

// Some part of the buffer handling logic needs to be shared between summary and
// synthetic children
// If I was only making synthetic children, this would be best modelled as
// different FrontEnds
class SwiftHashedContainerBufferHandler {
public:
  enum class Kind { eDictionary, eSet };

  virtual Kind GetKind() = 0;

  virtual size_t GetCount() = 0;

  virtual lldb_private::CompilerType GetElementType() = 0;

  virtual lldb::ValueObjectSP GetElementAtIndex(size_t) = 0;

  typedef std::function<SwiftHashedContainerBufferHandler *(
      lldb::ValueObjectSP, CompilerType, CompilerType)>
      NativeCreatorFunction;

  typedef std::function<SwiftHashedContainerBufferHandler *(
      lldb::ValueObjectSP)>
      SyntheticCreatorFunction;

  static std::unique_ptr<SwiftHashedContainerBufferHandler>
  CreateBufferHandler(ValueObject &valobj, NativeCreatorFunction Native,
                      SyntheticCreatorFunction Synthetic, ConstString mangled,
                      ConstString demangled);

  virtual ~SwiftHashedContainerBufferHandler() {}

protected:
  virtual bool IsValid() = 0;

  static std::unique_ptr<SwiftHashedContainerBufferHandler>
  CreateBufferHandlerForNativeStorageOwner(ValueObject &valobj,
                                           lldb::addr_t storage_ptr,
                                           bool fail_on_no_children,
                                           NativeCreatorFunction Native);
};

class SwiftHashedContainerEmptyBufferHandler
    : public SwiftHashedContainerBufferHandler {
public:
  virtual size_t GetCount() { return 0; }

  virtual lldb_private::CompilerType GetElementType() { return m_elem_type; }

  virtual lldb::ValueObjectSP GetElementAtIndex(size_t) {
    return lldb::ValueObjectSP();
  }

  virtual ~SwiftHashedContainerEmptyBufferHandler() {}

protected:
  SwiftHashedContainerEmptyBufferHandler(CompilerType elem_type)
      : m_elem_type(elem_type) {}
  friend class SwiftHashedContainerBufferHandler;

  virtual bool IsValid() { return true; }

private:
  lldb_private::CompilerType m_elem_type;
};

class SwiftHashedContainerNativeBufferHandler
    : public SwiftHashedContainerBufferHandler {
public:
  virtual size_t GetCount();

  virtual lldb_private::CompilerType GetElementType();

  virtual lldb::ValueObjectSP GetElementAtIndex(size_t);

  virtual ~SwiftHashedContainerNativeBufferHandler() {}

protected:
  typedef uint64_t Index;
  typedef uint64_t Cell;

  SwiftHashedContainerNativeBufferHandler(lldb::ValueObjectSP nativeStorage_sp,
                                          CompilerType key_type,
                                          CompilerType value_type);
  friend class SwiftHashedContainerBufferHandler;

  virtual bool IsValid();

  bool ReadBitmaskAtIndex(Index);

  lldb::addr_t GetLocationOfKeyAtCell(Cell);

  lldb::addr_t GetLocationOfValueAtCell(Cell);

  bool GetDataForKeyAtCell(Cell, void *);

  bool GetDataForValueAtCell(Cell, void *);

private:
  ValueObject *m_nativeStorage;
  Process *m_process;
  uint32_t m_ptr_size;
  uint64_t m_count;
  uint64_t m_capacity;
  lldb::addr_t m_bitmask_ptr;
  lldb::addr_t m_keys_ptr;
  lldb::addr_t m_values_ptr;
  CompilerType m_element_type;
  uint64_t m_key_stride;
  uint64_t m_value_stride;
  std::map<lldb::addr_t, uint64_t> m_bitmask_cache;
};

class SwiftHashedContainerSyntheticFrontEndBufferHandler
    : public SwiftHashedContainerBufferHandler {
public:
  virtual size_t GetCount();

  virtual lldb_private::CompilerType GetElementType();

  virtual lldb::ValueObjectSP GetElementAtIndex(size_t);

  virtual ~SwiftHashedContainerSyntheticFrontEndBufferHandler() {}

protected:
  SwiftHashedContainerSyntheticFrontEndBufferHandler(
      lldb::ValueObjectSP valobj_sp);
  friend class SwiftHashedContainerBufferHandler;

  virtual bool IsValid();

private:
  lldb::ValueObjectSP m_valobj_sp; // reader beware: this entails you must only
                                   // pass self-rooted valueobjects to this
                                   // class
  std::unique_ptr<SyntheticChildrenFrontEnd> m_frontend;
};

class HashedContainerSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  HashedContainerSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  virtual size_t CalculateNumChildren();

  virtual lldb::ValueObjectSP GetChildAtIndex(size_t idx);

  virtual bool Update() = 0;

  virtual bool MightHaveChildren();

  virtual size_t GetIndexOfChildWithName(const ConstString &name);

  virtual ~HashedContainerSyntheticFrontEnd() = default;

protected:
  std::unique_ptr<SwiftHashedContainerBufferHandler> m_buffer;
};
}
}
}

#endif // liblldb_SwiftHashedContainer_h_
