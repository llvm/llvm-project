//===-- GoFormatterFunctions.h-----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_GoFormatterFunctions_h_
#define liblldb_GoFormatterFunctions_h_

// C Includes
#include <stdint.h>
#include <time.h>

// C++ Includes
// Other libraries and framework includes
#include "clang/AST/ASTContext.h"

// Project includes
#include "lldb/lldb-forward.h"

#include "lldb/DataFormatters/FormatClasses.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ConstString.h"

namespace lldb_private {
namespace formatters {

bool GoStringSummaryProvider(ValueObject &valobj, Stream &stream,
                             const TypeSummaryOptions &options);

SyntheticChildrenFrontEnd *
GoSliceSyntheticFrontEndCreator(CXXSyntheticChildren *, lldb::ValueObjectSP);

} // namespace formatters
} // namespace lldb_private

#endif // liblldb_GoFormatterFunctions_h_
