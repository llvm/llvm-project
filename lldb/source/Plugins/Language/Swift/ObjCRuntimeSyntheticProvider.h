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

#include "Plugins/LanguageRuntime/ObjC/ObjCLanguageRuntime.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/lldb-enumerations.h"

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

  bool IsScripted() override { return false; }

  size_t GetNumIVars() { return m_descriptor_sp->GetNumIVars(); }

  ObjCLanguageRuntime::ClassDescriptor::iVarDescriptor
  GetIVarAtIndex(size_t idx) {
    return m_descriptor_sp->GetIVarAtIndex(idx);
  }

  std::string GetDescription() override;

  class FrontEnd : public SyntheticChildrenFrontEnd {
  private:
    ObjCRuntimeSyntheticProvider *m_provider;

  public:
    FrontEnd(ObjCRuntimeSyntheticProvider *prv, ValueObject &backend);

    llvm::Expected<uint32_t> CalculateNumChildren() override;
    lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;
    lldb::ChildCacheState Update() override {
      return lldb::ChildCacheState::eRefetch;
    }
    bool MightHaveChildren() override { return true; }
    size_t GetIndexOfChildWithName(ConstString name) override;

    typedef std::shared_ptr<SyntheticChildrenFrontEnd> SharedPointer;

  private:
    size_t GetNumBases();

    lldb::ValueObjectSP m_root_sp;

    FrontEnd(const FrontEnd &) = delete;
    const FrontEnd &operator=(const FrontEnd &) = delete;
  };

  virtual SyntheticChildrenFrontEnd::AutoPointer
  GetFrontEnd(ValueObject &backend) override {
    return SyntheticChildrenFrontEnd::AutoPointer(new FrontEnd(this, backend));
  }

private:
  ObjCRuntimeSyntheticProvider(const ObjCRuntimeSyntheticProvider &) = delete;
  const ObjCRuntimeSyntheticProvider &
  operator=(const ObjCRuntimeSyntheticProvider &) = delete;
};
}

#endif // lldb_ObjCRuntimeSyntheticProvider_h
