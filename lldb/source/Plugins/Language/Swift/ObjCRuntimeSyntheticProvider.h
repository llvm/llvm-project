//===-- ObjCRuntimeSyntheticProvider.h --------------------------*- C++ -*-===//
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

#ifndef lldb_ObjCRuntimeSyntheticProvider_h
#define lldb_ObjCRuntimeSyntheticProvider_h

#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Target/ObjCLanguageRuntime.h"

namespace lldb_private {
class ObjCRuntimeSyntheticProvider : public SyntheticChildren {
  ObjCLanguageRuntime::ClassDescriptorSP m_descriptor_sp;

public:
  ObjCRuntimeSyntheticProvider(
      const SyntheticChildren::Flags &flags,
      ObjCLanguageRuntime::ClassDescriptorSP descriptor_sp)
      : SyntheticChildren(flags), m_descriptor_sp(descriptor_sp) {
    // no point in making one with no descriptor!
    assert(descriptor_sp.get());
  }

  bool IsScripted() { return false; }

  size_t GetNumIVars() { return m_descriptor_sp->GetNumIVars(); }

  ObjCLanguageRuntime::ClassDescriptor::iVarDescriptor
  GetIVarAtIndex(size_t idx) {
    return m_descriptor_sp->GetIVarAtIndex(idx);
  }

  std::string GetDescription();

  class FrontEnd : public SyntheticChildrenFrontEnd {
  private:
    ObjCRuntimeSyntheticProvider *m_provider;

  public:
    FrontEnd(ObjCRuntimeSyntheticProvider *prv, ValueObject &backend);

    virtual ~FrontEnd() {}

    virtual size_t CalculateNumChildren();

    virtual lldb::ValueObjectSP GetChildAtIndex(size_t idx);

    virtual bool Update() { return false; }

    virtual bool MightHaveChildren() { return true; }

    virtual size_t GetIndexOfChildWithName(const ConstString &name);

    typedef std::shared_ptr<SyntheticChildrenFrontEnd> SharedPointer;

  private:
    size_t GetNumBases();

    lldb::ValueObjectSP m_root_sp;

    DISALLOW_COPY_AND_ASSIGN(FrontEnd);
  };

  virtual SyntheticChildrenFrontEnd::AutoPointer
  GetFrontEnd(ValueObject &backend) {
    return SyntheticChildrenFrontEnd::AutoPointer(new FrontEnd(this, backend));
  }

private:
  DISALLOW_COPY_AND_ASSIGN(ObjCRuntimeSyntheticProvider);
};
}

#endif // lldb_ObjCRuntimeSyntheticProvider_h
