//===-- SwiftUIFormatters.cpp -----------------------------------*- C++ -*-===//
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

#include "SwiftUIFormatters.h"

#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Log.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

using namespace lldb;
using namespace lldb_private;

namespace {

/// Synthetic children provider for SwiftUI.AtomicBuffer<T>.
///
/// Exposes two children:
///   [0] lock  - the lock that is the header of a ManagedBuffer
///   [1] value - generic value stored immediately after the lock
class AtomicBufferSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  AtomicBufferSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
      : SyntheticChildrenFrontEnd(*valobj_sp) {}

  llvm::Expected<uint32_t> CalculateNumChildren() override { return 2; }

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    switch (idx) {
    case 0:
      return m_lock_sp;
    case 1:
      return m_value_sp;
    default:
      return nullptr;
    }
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    if (name == "lock")
      return 0;
    if (name == "value")
      return 1;
    return llvm::createStringError(
        llvm::formatv("Type has no child named '{0}'", name));
  }

  lldb::ChildCacheState Update() override {
    m_lock_sp = nullptr;
    m_value_sp = nullptr;

    ValueObjectSP header_sp = m_backend.GetChildMemberWithName("header");
    if (!header_sp)
      return ChildCacheState::eRefetch;

    m_lock_sp = header_sp->Clone(ConstString("lock"));

    auto lock_type = m_lock_sp->GetCompilerType();
    auto value_type = m_backend.GetCompilerType().GetTypeTemplateArgument(0);
    if (!lock_type || !value_type) {
      LLDB_LOG(GetLog(LLDBLog::DataFormatters),
               "could not retrieve field types of {0}",
               m_backend.GetTypeName());
      return ChildCacheState::eRefetch;
    }

    auto *target = m_backend.GetTargetSP().get();

    // The generic value comes after the lock. To calculate the offset of the
    // value, the calculation needs the lock's size, and the value's alignment.

    auto lock_size_or_err = lock_type.GetByteSize(target);
    if (!lock_size_or_err) {
      LLDB_LOG_ERROR(
          GetLog(LLDBLog::DataFormatters), lock_size_or_err.takeError(),
          "could not determine size of {1}: {0}", m_lock_sp->GetTypeName());
      return ChildCacheState::eRefetch;
    }

    auto maybe_value_bit_align = value_type.GetTypeBitAlign(target);
    if (!maybe_value_bit_align) {
      LLDB_LOG(GetLog(LLDBLog::DataFormatters),
               "could not determine alignment of {0}",
               value_type.GetTypeName());
      return ChildCacheState::eRefetch;
    }

    auto lock_size = *lock_size_or_err;
    auto value_align = *maybe_value_bit_align / 8;
    uint32_t value_offset = llvm::alignTo(lock_size, value_align);
    m_value_sp = header_sp->GetSyntheticChildAtOffset(
        value_offset, value_type, true, ConstString("value"));

    return ChildCacheState::eReuse;
  }

private:
  lldb::ValueObjectSP m_lock_sp;
  lldb::ValueObjectSP m_value_sp;
};
} // namespace

SyntheticChildrenFrontEnd *
lldb_private::formatters::swift::AtomicBufferSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;
  return new AtomicBufferSyntheticFrontEnd(valobj_sp);
}

namespace {

/// Synthetic children provider for SwiftUI.State<T>.
///
/// Exposes one child:
///   [0] wrappedValue - the current value of the State
///
/// The source of the wrapped value depends on context:
///   - If _location is nil, or if AG::Graph::UpdateStack::update() is currently
///     executing, the value comes from the local _value storage.
///   - Otherwise the value comes from the graph node:
///     _location[0]._data.buffer.value.currentValue
class StateSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  StateSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
      : SyntheticChildrenFrontEnd(*valobj_sp) {}

  llvm::Expected<uint32_t> CalculateNumChildren() override { return 1; }

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    if (idx == 0)
      return m_wrapped_value_sp;
    return nullptr;
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    if (name == "wrappedValue")
      return 0;
    return llvm::createStringError(
        llvm::formatv("Type has no child named '{0}'", name));
  }

  lldb::ChildCacheState Update() override {
    m_wrapped_value_sp = nullptr;

    ValueObjectSP location_sp = m_backend.GetChildMemberWithName("_location");
    if (!location_sp)
      return ChildCacheState::eRefetch;

    ValueObjectSP location_synth_sp = location_sp->GetSyntheticValue();
    if (!location_synth_sp)
      return ChildCacheState::eRefetch;

    bool use_value_property = false;
    bool has_location = location_synth_sp->MightHaveChildren();
    if (!has_location) {
      use_value_property = true;
    } else {
      // This logic is a proxy for calling GraphHost.isUpdating. During a call
      // to AG::Graph::UpdateStack::update(), the State's active value will be
      // found in the _value property (not within _location).
      ThreadSP thread_sp = m_backend.GetThreadSP();
      if (thread_sp) {
        uint32_t num_frames = thread_sp->GetStackFrameCount();
        static constexpr uint32_t k_frame_search_limit = 50;
        uint32_t limit = std::min(num_frames, k_frame_search_limit);
        for (uint32_t i = 0; i < limit; ++i) {
          StackFrameSP frame_sp = thread_sp->GetStackFrameAtIndex(i);
          if (!frame_sp)
            continue;
          llvm::StringRef name = frame_sp->GetFunctionName();
          if (name == "AG::Graph::UpdateStack::update()") {
            use_value_property = true;
            break;
          }
        }
      }
    }

    ValueObjectSP value_sp;
    if (use_value_property) {
      value_sp = m_backend.GetChildMemberWithName("_value");
    } else {
      if (auto first_child_sp = location_synth_sp->GetChildAtIndex(0))
        value_sp = first_child_sp->GetValueForExpressionPath(
            "._data.buffer.value.currentValue");
    }

    if (value_sp) {
      m_wrapped_value_sp = value_sp->Clone(ConstString("wrappedValue"));
      return ChildCacheState::eReuse;
    }

    return ChildCacheState::eRefetch;
  }

private:
  lldb::ValueObjectSP m_wrapped_value_sp;
};
} // namespace

SyntheticChildrenFrontEnd *
lldb_private::formatters::swift::StateSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;
  return new StateSyntheticFrontEnd(valobj_sp);
}

void lldb_private::formatters::swift::LoadSwiftUIFormatters(
    lldb::TypeCategoryImplSP swift_category_sp) {
  if (!swift_category_sp)
    return;

  SyntheticChildren::Flags synth_flags;
  synth_flags.SetCascades(true);

  TypeSummaryImpl::Flags summary_flags;
  summary_flags.SetCascades(true)
      .SetDontShowValue(true)
      .SetShowMembersOneLiner(false)
      .SetHideItemNames(false);

  AddCXXSynthetic(swift_category_sp, AtomicBufferSyntheticFrontEndCreator,
                  "SwiftUI AtomicBuffer synthetic children",
                  ConstString("^SwiftUI(Core)?[.]AtomicBuffer<.+>$"),
                  synth_flags, true);

  AddCXXSynthetic(swift_category_sp, StateSyntheticFrontEndCreator,
                  "SwiftUI State synthetic children",
                  ConstString("^SwiftUI(Core)?[.]State<.+>$"), synth_flags,
                  true);

  AddStringSummary(swift_category_sp, "${var.wrappedValue}",
                   ConstString("^SwiftUI(Core)?[.]State<.+>$"), summary_flags,
                   true);
}
