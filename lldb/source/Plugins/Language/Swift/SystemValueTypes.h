//===-- SystemValueTypes.h ------------------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2026 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SystemValueTypes_h_
#define liblldb_SystemValueTypes_h_

#include "lldb/lldb-forward.h"
#include "lldb/DataFormatters/TypeSummary.h"

namespace lldb_private {
namespace formatters {
namespace swift {

struct FilePathSummaryProvider : public TypeSummaryImpl {
  FilePathSummaryProvider(const TypeSummaryImpl::Flags &flags)
      : TypeSummaryImpl(TypeSummaryImpl::Kind::eInternal, flags) {}

  bool FormatObject(ValueObject *valobj, std::string &dest,
                    const TypeSummaryOptions &options) override;
  std::string GetDescription() override;
  std::string GetName() override;
  bool DoesPrintChildren(ValueObject *valobj) const override;

private:
  FilePathSummaryProvider(const FilePathSummaryProvider &) = delete;
  const FilePathSummaryProvider &
  operator=(const FilePathSummaryProvider &) = delete;
};

bool SystemString_SummaryProvider(ValueObject &valobj, Stream &stream,
                                  const TypeSummaryOptions &options);
}
}
}

#endif // liblldb_SystemValueTypes_h_
