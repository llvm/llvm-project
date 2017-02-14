//===-- NSException.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "clang/AST/DeclCXX.h"

// Project includes
#include "Cocoa.h"

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/ProcessStructReader.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/Error.h"
#include "lldb/Utility/Stream.h"

#include "Plugins/Language/ObjC/NSString.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

bool lldb_private::formatters::NSException_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  ProcessSP process_sp(valobj.GetProcessSP());
  if (!process_sp)
    return false;

  lldb::addr_t ptr_value = LLDB_INVALID_ADDRESS;

  CompilerType valobj_type(valobj.GetCompilerType());
  Flags type_flags(valobj_type.GetTypeInfo());
  if (type_flags.AllClear(eTypeHasValue)) {
    if (valobj.IsBaseClass() && valobj.GetParent())
      ptr_value = valobj.GetParent()->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
  } else
    ptr_value = valobj.GetValueAsUnsigned(LLDB_INVALID_ADDRESS);

  if (ptr_value == LLDB_INVALID_ADDRESS)
    return false;
  size_t ptr_size = process_sp->GetAddressByteSize();
  lldb::addr_t name_location = ptr_value + 1 * ptr_size;
  lldb::addr_t reason_location = ptr_value + 2 * ptr_size;

  Error error;
  lldb::addr_t name = process_sp->ReadPointerFromMemory(name_location, error);
  if (error.Fail() || name == LLDB_INVALID_ADDRESS)
    return false;

  lldb::addr_t reason =
      process_sp->ReadPointerFromMemory(reason_location, error);
  if (error.Fail() || reason == LLDB_INVALID_ADDRESS)
    return false;

  InferiorSizedWord name_isw(name, *process_sp);
  InferiorSizedWord reason_isw(reason, *process_sp);

  CompilerType voidstar = process_sp->GetTarget()
                              .GetScratchClangASTContext()
                              ->GetBasicType(lldb::eBasicTypeVoid)
                              .GetPointerType();

  ValueObjectSP name_sp = ValueObject::CreateValueObjectFromData(
      "name_str", name_isw.GetAsData(process_sp->GetByteOrder()),
      valobj.GetExecutionContextRef(), voidstar);
  ValueObjectSP reason_sp = ValueObject::CreateValueObjectFromData(
      "reason_str", reason_isw.GetAsData(process_sp->GetByteOrder()),
      valobj.GetExecutionContextRef(), voidstar);

  if (!name_sp || !reason_sp)
    return false;

  StreamString name_str_summary;
  StreamString reason_str_summary;
  if (NSStringSummaryProvider(*name_sp, name_str_summary, options) &&
      NSStringSummaryProvider(*reason_sp, reason_str_summary, options) &&
      !name_str_summary.Empty() && !reason_str_summary.Empty()) {
    stream.Printf("name: %s - reason: %s", name_str_summary.GetData(),
                  reason_str_summary.GetData());
    return true;
  } else
    return false;
}

class NSExceptionSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  NSExceptionSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
      : SyntheticChildrenFrontEnd(*valobj_sp) {}

  ~NSExceptionSyntheticFrontEnd() override = default;
  // no need to delete m_child_ptr - it's kept alive by the cluster manager on
  // our behalf

  size_t CalculateNumChildren() override {
    if (m_child_ptr)
      return 1;
    if (m_child_sp)
      return 1;
    return 0;
  }

  lldb::ValueObjectSP GetChildAtIndex(size_t idx) override {
    if (idx != 0)
      return lldb::ValueObjectSP();

    if (m_child_ptr)
      return m_child_ptr->GetSP();
    return m_child_sp;
  }

  bool Update() override {
    m_child_ptr = nullptr;
    m_child_sp.reset();

    ProcessSP process_sp(m_backend.GetProcessSP());
    if (!process_sp)
      return false;

    lldb::addr_t userinfo_location = LLDB_INVALID_ADDRESS;

    CompilerType valobj_type(m_backend.GetCompilerType());
    Flags type_flags(valobj_type.GetTypeInfo());
    if (type_flags.AllClear(eTypeHasValue)) {
      if (m_backend.IsBaseClass() && m_backend.GetParent())
        userinfo_location =
            m_backend.GetParent()->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
    } else
      userinfo_location = m_backend.GetValueAsUnsigned(LLDB_INVALID_ADDRESS);

    if (userinfo_location == LLDB_INVALID_ADDRESS)
      return false;

    size_t ptr_size = process_sp->GetAddressByteSize();

    userinfo_location += 3 * ptr_size;
    Error error;
    lldb::addr_t userinfo =
        process_sp->ReadPointerFromMemory(userinfo_location, error);
    if (userinfo == LLDB_INVALID_ADDRESS || error.Fail())
      return false;
    InferiorSizedWord isw(userinfo, *process_sp);
    m_child_sp = CreateValueObjectFromData(
        "userInfo", isw.GetAsData(process_sp->GetByteOrder()),
        m_backend.GetExecutionContextRef(),
        process_sp->GetTarget().GetScratchClangASTContext()->GetBasicType(
            lldb::eBasicTypeObjCID));
    return false;
  }

  bool MightHaveChildren() override { return true; }

  size_t GetIndexOfChildWithName(const ConstString &name) override {
    static ConstString g___userInfo("userInfo");
    if (name == g___userInfo)
      return 0;
    return UINT32_MAX;
  }

private:
  // the child here can be "real" (i.e. an actual child of the root) or
  // synthetized from raw memory
  // if the former, I need to store a plain pointer to it - or else a loop of
  // references will cause this entire hierarchy of values to leak
  // if the latter, then I need to store a SharedPointer to it - so that it only
  // goes away when everyone else in the cluster goes away
  // oh joy!
  ValueObject *m_child_ptr;
  ValueObjectSP m_child_sp;
};

SyntheticChildrenFrontEnd *
lldb_private::formatters::NSExceptionSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  lldb::ProcessSP process_sp(valobj_sp->GetProcessSP());
  if (!process_sp)
    return nullptr;
  ObjCLanguageRuntime *runtime =
      (ObjCLanguageRuntime *)process_sp->GetLanguageRuntime(
          lldb::eLanguageTypeObjC);
  if (!runtime)
    return nullptr;

  ObjCLanguageRuntime::ClassDescriptorSP descriptor(
      runtime->GetClassDescriptor(*valobj_sp.get()));

  if (!descriptor.get() || !descriptor->IsValid())
    return nullptr;

  const char *class_name = descriptor->GetClassName().GetCString();

  if (!class_name || !*class_name)
    return nullptr;

  if (!strcmp(class_name, "NSException"))
    return (new NSExceptionSyntheticFrontEnd(valobj_sp));
  else if (!strcmp(class_name, "NSCFException"))
    return (new NSExceptionSyntheticFrontEnd(valobj_sp));
  else if (!strcmp(class_name, "__NSCFException"))
    return (new NSExceptionSyntheticFrontEnd(valobj_sp));

  return nullptr;
}
