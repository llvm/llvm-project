//===-- SwiftOptional.h -----------------------------------------*- C++ -*-===//
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

#ifndef liblldb_SwiftOptional_h_
#define liblldb_SwiftOptional_h_

#include "lldb/lldb-forward.h"

#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/TypeSynthetic.h"

#include <cstddef>

namespace lldb_private {
namespace formatters {
// ExtractSomeIfAny() can return EITHER a child member or some other long-lived
// ValueObject
// OR an entirely consed-up ValueObject
// The lifetime of these two is radically different, and there is no trivial way
// to do the right
// thing for both cases - except have a class that can wrap either and is safe
// to store and pass around
class PointerOrSP {
public:
  PointerOrSP(std::nullptr_t) : m_raw_ptr(nullptr), m_shared_ptr(nullptr) {}

  PointerOrSP(ValueObject *valobj) : m_raw_ptr(valobj), m_shared_ptr(nullptr) {}

  PointerOrSP(lldb::ValueObjectSP valobj_sp)
      : m_raw_ptr(nullptr), m_shared_ptr(valobj_sp) {}

  ValueObject *operator->() {
    if (m_shared_ptr)
      return m_shared_ptr.get();
    return m_raw_ptr;
  }

  ValueObject &operator*() { return *(this->operator->()); }

  operator ValueObject *() { return this->operator->(); }

  explicit operator bool() const {
    return (m_shared_ptr.get() != nullptr) || (m_raw_ptr != nullptr);
  }

  bool operator==(std::nullptr_t) const { return !(this->operator bool()); }

protected:
  ValueObject *m_raw_ptr;
  lldb::ValueObjectSP m_shared_ptr;
};

namespace swift {
struct SwiftOptionalSummaryProvider : public TypeSummaryImpl {
  SwiftOptionalSummaryProvider(const TypeSummaryImpl::Flags &flags)
      : TypeSummaryImpl(TypeSummaryImpl::Kind::eInternal,
                        TypeSummaryImpl::Flags()) {}

  virtual ~SwiftOptionalSummaryProvider() {}

  virtual bool FormatObject(ValueObject *valobj, std::string &dest,
                            const TypeSummaryOptions &options);

  virtual std::string GetDescription();

  virtual bool IsScripted() { return false; }

  virtual bool DoesPrintChildren(ValueObject *valobj) const;

  virtual bool DoesPrintValue(ValueObject *valobj) const;

private:
  DISALLOW_COPY_AND_ASSIGN(SwiftOptionalSummaryProvider);
};

bool SwiftOptional_SummaryProvider(ValueObject &valobj, Stream &stream);

class SwiftOptionalSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  SwiftOptionalSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  virtual size_t CalculateNumChildren();

  virtual lldb::ValueObjectSP GetChildAtIndex(size_t idx);

  virtual bool Update();

  virtual bool MightHaveChildren();

  virtual size_t GetIndexOfChildWithName(const ConstString &name);

  virtual lldb::ValueObjectSP GetSyntheticValue();

  virtual ~SwiftOptionalSyntheticFrontEnd() = default;

private:
  bool m_is_none;
  bool m_children;
  PointerOrSP m_some;

  bool IsEmpty() const;
};

SyntheticChildrenFrontEnd *
SwiftOptionalSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                      lldb::ValueObjectSP);
SyntheticChildrenFrontEnd *
SwiftUncheckedOptionalSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                               lldb::ValueObjectSP);
}
}
}

#endif // liblldb_SwiftOptional_h_
