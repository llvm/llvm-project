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

namespace swift {
struct SwiftOptionalSummaryProvider : public TypeSummaryImpl {
  SwiftOptionalSummaryProvider(const TypeSummaryImpl::Flags &flags)
      : TypeSummaryImpl(TypeSummaryImpl::Kind::eInternal, flags) {}

  bool FormatObject(ValueObject *valobj, std::string &dest,
                    const TypeSummaryOptions &options) override;
  std::string GetDescription() override;
  std::string GetName() override;
  bool DoesPrintChildren(ValueObject *valobj) const override;
  bool DoesPrintValue(ValueObject *valobj) const override;

private:
  SwiftOptionalSummaryProvider(const SwiftOptionalSummaryProvider &) = delete;
  const SwiftOptionalSummaryProvider &
  operator=(const SwiftOptionalSummaryProvider &) = delete;
};

bool SwiftOptional_SummaryProvider(ValueObject &valobj, Stream &stream);

class SwiftOptionalSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  SwiftOptionalSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  llvm::Expected<uint32_t> CalculateNumChildren() override;
  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;
  lldb::ChildCacheState Update() override;
  bool MightHaveChildren() override;
  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override;
  lldb::ValueObjectSP GetSyntheticValue() override;

private:
  bool m_is_none;
  bool m_children;
  lldb::ValueObjectSP m_some;

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
