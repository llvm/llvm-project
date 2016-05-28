//===-- FoundationValueTypes.cpp ----------------------------------*- C++ -*-===//
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

#include "FoundationValueTypes.h"

#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/FormattersHelpers.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace lldb_private::formatters::swift;

bool
lldb_private::formatters::swift::Date_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options)
{
    static ConstString g__time("_time");
    
    ValueObjectSP time_sp(valobj.GetChildAtNamePath( {g__time} ));
    
    if (!time_sp)
        return false;
    
    DataExtractor data_extractor;
    Error error;
    if (!time_sp->GetData(data_extractor, error))
        return false;
    
    offset_t offset_ptr = 0;
    double date_value = data_extractor.GetDouble(&offset_ptr);

    if (date_value == -63114076800)
    {
        stream.Printf("0001-12-30 00:00:00 +0000");
        return true;
    }
    // this snippet of code assumes that time_t == seconds since Jan-1-1970
    // this is generally true and POSIXly happy, but might break if a library
    // vendor decides to get creative
    time_t epoch = GetOSXEpoch();
    epoch = epoch + (time_t)date_value;
    tm *tm_date = gmtime(&epoch);
    if (!tm_date)
        return false;
    std::string buffer(1024,0);
    if (strftime (&buffer[0], 1023, "%Z", tm_date) == 0)
        return false;
    stream.Printf("%04d-%02d-%02d %02d:%02d:%02d %s", tm_date->tm_year+1900, tm_date->tm_mon+1, tm_date->tm_mday, tm_date->tm_hour, tm_date->tm_min, tm_date->tm_sec, buffer.c_str());
    return true;
}

bool
lldb_private::formatters::swift::NotificationName_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options)
{
    static ConstString g__rawValue("_rawValue");
    
    ValueObjectSP underlying_name_sp(valobj.GetChildAtNamePath( {g__rawValue} ));
    
    if (!underlying_name_sp)
        return false;
    
    std::string summary;
    if (!underlying_name_sp->GetSummaryAsCString(summary, options))
        return false;
    
    stream.Printf("%s", summary.c_str());
    return true;
}

bool
lldb_private::formatters::swift::URL_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options)
{
    static ConstString g__url("_url");
    
    ValueObjectSP underlying_url_sp(valobj.GetChildAtNamePath( {g__url} ));
    
    if (!underlying_url_sp)
        return false;
    
    std::string summary;
    if (!underlying_url_sp->GetSummaryAsCString(summary, options))
        return false;
    
    stream.Printf("%s", summary.c_str());
    return true;
}

bool
lldb_private::formatters::swift::IndexPath_SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options)
{
    static ConstString g__indexes("_indexes");
    
    ValueObjectSP underlying_array_sp(valobj.GetChildAtNamePath( {g__indexes} ));
    
    if (!underlying_array_sp)
        return false;
    
    underlying_array_sp = underlying_array_sp->GetQualifiedRepresentationIfAvailable(lldb::eDynamicDontRunTarget, true);
    
    size_t num_children = underlying_array_sp->GetNumChildren();
    
    if (num_children == 1)
        stream.Printf("%s", "1 index");
    else
        stream.Printf("%zu indices", num_children);
    return true;
}
