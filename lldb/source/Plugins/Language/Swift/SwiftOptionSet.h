//===-- SwiftOptionSet.h ----------------------------------------*- C++ -*-===//
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

#ifndef liblldb_SwiftOptionSet_h_
#define liblldb_SwiftOptionSet_h_

#include "lldb/lldb-forward.h"

#include "lldb/Core/ConstString.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Symbol/CompilerType.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Optional.h"

#include <vector>

namespace lldb_private {
namespace formatters {
namespace swift {
struct SwiftOptionSetSummaryProvider : public TypeSummaryImpl {
  static bool WouldEvenConsiderFormatting(CompilerType);

  SwiftOptionSetSummaryProvider(CompilerType);

  virtual ~SwiftOptionSetSummaryProvider() = default;

  virtual bool FormatObject(ValueObject *valobj, std::string &dest,
                            const TypeSummaryOptions &options);

  virtual std::string GetDescription();

  virtual bool IsScripted() { return false; }

  virtual bool DoesPrintChildren(ValueObject *valobj) const;

private:
  DISALLOW_COPY_AND_ASSIGN(SwiftOptionSetSummaryProvider);

  void FillCasesIfNeeded();

  CompilerType m_type;

  typedef std::vector<std::pair<llvm::APInt, lldb_private::ConstString>>
      CasesVector;

  llvm::Optional<CasesVector> m_cases;
};

bool SwiftOptionSet_SummaryProvider(ValueObject &valobj, Stream &stream);
}
}
}

#endif // liblldb_SwiftOptionSet_h_
