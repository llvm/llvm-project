//===-- SwiftOptionSet.h ----------------------------------------*- C++ -*-===//
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

#ifndef liblldb_SwiftOptionSet_h_
#define liblldb_SwiftOptionSet_h_

#include "lldb/lldb-forward.h"

#include "lldb/Utility/ConstString.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Symbol/CompilerType.h"

#include "llvm/ADT/APInt.h"
#include <optional>

#include <vector>

namespace lldb_private {
namespace formatters {
namespace swift {
struct SwiftOptionSetSummaryProvider : public TypeSummaryImpl {
  static bool WouldEvenConsiderFormatting(CompilerType);

  SwiftOptionSetSummaryProvider(CompilerType);
  bool FormatObject(ValueObject *valobj, std::string &dest,
                    const TypeSummaryOptions &options) override;
  std::string GetDescription() override;
  bool DoesPrintChildren(ValueObject *valobj) const override;

private:
  SwiftOptionSetSummaryProvider(const SwiftOptionSetSummaryProvider &) = delete;
  const SwiftOptionSetSummaryProvider &
  operator=(const SwiftOptionSetSummaryProvider &) = delete;

  void FillCasesIfNeeded(const ExecutionContext *);

  CompilerType m_type;

  typedef std::vector<std::pair<llvm::APInt, lldb_private::ConstString>>
      CasesVector;

  std::optional<CasesVector> m_cases;
};

bool SwiftOptionSet_SummaryProvider(ValueObject &valobj, Stream &stream);
}
}
}

#endif // liblldb_SwiftOptionSet_h_
