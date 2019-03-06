//===-- ObjCRuntimeSyntheticProvider.cpp ------------------------*- C++ -*-===//
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

#include "ObjCRuntimeSyntheticProvider.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/DeclVendor.h"

#include "lldb/lldb-public.h"

using namespace lldb;
using namespace lldb_private;

std::string ObjCRuntimeSyntheticProvider::GetDescription() {
  StreamString sstr;
  sstr.Printf("%s%s%s Runtime-generated synthetic provider for %s {\n",
              Cascades() ? "" : " (not cascading)",
              SkipsPointers() ? " (skip pointers)" : "",
              SkipsReferences() ? " (skip references)" : "",
              m_descriptor_sp->GetClassName().AsCString("<unknown>"));

  return sstr.GetString();
}

size_t ObjCRuntimeSyntheticProvider::FrontEnd::GetNumBases() {
  return m_provider->m_descriptor_sp->GetSuperclass().get() ? 1 : 0;
}

size_t ObjCRuntimeSyntheticProvider::FrontEnd::CalculateNumChildren() {
  size_t ivars = m_provider->GetNumIVars();
  size_t bases = GetNumBases();
  return bases + ivars;
}

static lldb::ValueObjectSP GetSuitableRootObject(ValueObjectSP valobj_sp) {
  if (valobj_sp) {
    if (!valobj_sp->GetParent())
      return valobj_sp;
    if (valobj_sp->IsBaseClass()) {
      if (valobj_sp->GetParent()->IsBaseClass())
        return GetSuitableRootObject(valobj_sp->GetParent()->GetSP());
      return valobj_sp;
    }
  }
  return valobj_sp;
}

ObjCRuntimeSyntheticProvider::FrontEnd::FrontEnd(
    ObjCRuntimeSyntheticProvider *prv, ValueObject &backend)
    : SyntheticChildrenFrontEnd(backend), m_provider(prv),
      m_root_sp(::GetSuitableRootObject(backend.GetSP())) {}

lldb::ValueObjectSP
ObjCRuntimeSyntheticProvider::FrontEnd::GetChildAtIndex(size_t idx) {
  lldb::ValueObjectSP child_sp(nullptr);
  if (idx < CalculateNumChildren()) {
    if (GetNumBases() == 1) {
      if (idx == 0) {
        do {
          ProcessSP process_sp(m_backend.GetProcessSP());
          if (!process_sp)
            break;
          ObjCLanguageRuntime *runtime = process_sp->GetObjCLanguageRuntime();
          if (!runtime)
            break;
          DeclVendor *vendor = runtime->GetDeclVendor();
          if (!vendor)
            break;
          std::vector<clang::NamedDecl *> decls;
          auto descriptor_sp(m_provider->m_descriptor_sp);
          if (!descriptor_sp)
            break;
          descriptor_sp = descriptor_sp->GetSuperclass();
          if (!descriptor_sp)
            break;
          const bool append = false;
          const uint32_t max = 1;
          if (0 ==
              vendor->FindDecls(descriptor_sp->GetClassName(), append, max,
                                decls))
            break;
          const uint32_t offset = 0;
          const bool can_create = true;
          if (decls.empty())
            break;
          CompilerType type = ClangASTContext::GetTypeForDecl(decls[0]);
          if (!type.IsValid())
            break;
          child_sp = m_backend.GetSyntheticBase(offset, type, can_create);
        } while (false);
        return child_sp;
      } else
        --idx;
    }
    if (m_root_sp) {
      const auto &ivar_info(m_provider->GetIVarAtIndex(idx));
      const bool can_create = true;
      child_sp = m_root_sp->GetSyntheticChildAtOffset(
          ivar_info.m_offset, ivar_info.m_type, can_create);
      if (child_sp)
        child_sp->SetName(ivar_info.m_name);
    }
  }
  return child_sp;
}

size_t ObjCRuntimeSyntheticProvider::FrontEnd::GetIndexOfChildWithName(
    ConstString name) {
  for (size_t idx = 0; idx < CalculateNumChildren(); idx++) {
    const auto &ivar_info(m_provider->GetIVarAtIndex(idx));
    if (name == ivar_info.m_name)
      return idx + GetNumBases();
  }
  return UINT32_MAX;
}
