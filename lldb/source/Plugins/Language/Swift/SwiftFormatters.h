//===-- SwiftFormatters.h ---------------------------------------*- C++ -*-===//
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

#ifndef liblldb_SwiftFormatters_h_
#define liblldb_SwiftFormatters_h_

#include <stdint.h>
#include "lldb/lldb-forward.h"

#include "lldb/Core/ConstString.h"
#include "lldb/DataFormatters/FormatClasses.h"
#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/Target/Target.h"

#include "FoundationValueTypes.h"
#include "SwiftArray.h"
#include "SwiftBasicTypes.h"
#include "SwiftDictionary.h"
#include "SwiftMetatype.h"
#include "SwiftOptional.h"
#include "SwiftOptionSet.h"
#include "SwiftSet.h"

namespace lldb_private {
    namespace formatters
    {
        namespace swift {
            bool
            Character_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
            
            bool
            UnicodeScalar_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
            
            bool
            StringCore_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);

            bool
            StringCore_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions&, StringPrinter::ReadStringAndDumpToStreamOptions);
            
            bool
            String_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
            
            bool
            String_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions&, StringPrinter::ReadStringAndDumpToStreamOptions);

            bool
            StaticString_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
            
            bool
            StaticString_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions&, StringPrinter::ReadStringAndDumpToStreamOptions);
            
            bool
            NSContiguousString_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
          
            bool
            Bool_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);

            bool
            DarwinBoolean_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
            
            bool
            Range_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);

            bool
            CountableRange_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);

            bool
            ClosedRange_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);

            bool
            CountableClosedRange_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);

            bool
            StridedRangeGenerator_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
            
            // TODO: this is a transient workaround for the fact that
            // ObjC types are totally opaque in Swift for LLDB
            bool
            BuiltinObjC_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);

            bool
            ObjC_Selector_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
            
            bool
            TypePreservingNSNumber_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
            
            SyntheticChildrenFrontEnd* EnumSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        }
    }
}

#endif // liblldb_SwiftFormatters_h_
