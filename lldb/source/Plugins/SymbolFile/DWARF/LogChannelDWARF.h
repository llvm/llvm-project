//===-- LogChannelDWARF.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_LogChannelDWARF_h_
#define SymbolFileDWARF_LogChannelDWARF_h_

// C Includes
// C++ Includes
// Other libraries and framework includes

// Project includes
#include "lldb/Core/Log.h"

#define DWARF_LOG_VERBOSE           (1u << 0)
#define DWARF_LOG_DEBUG_INFO        (1u << 1)
#define DWARF_LOG_DEBUG_LINE        (1u << 2)
#define DWARF_LOG_DEBUG_PUBNAMES    (1u << 3)
#define DWARF_LOG_DEBUG_PUBTYPES    (1u << 4)
#define DWARF_LOG_DEBUG_ARANGES     (1u << 5)
#define DWARF_LOG_LOOKUPS           (1u << 6)
#define DWARF_LOG_TYPE_COMPLETION   (1u << 7)
#define DWARF_LOG_DEBUG_MAP         (1u << 8)
#define DWARF_LOG_ALL               (UINT32_MAX)
#define DWARF_LOG_DEFAULT           (DWARF_LOG_DEBUG_INFO)

class LogChannelDWARF : public lldb_private::LogChannel
{
public:
    LogChannelDWARF ();

    virtual
    ~LogChannelDWARF ();

    static void
    Initialize();

    static void
    Terminate();

    static lldb_private::ConstString
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    static lldb_private::LogChannel *
    CreateInstance ();

    virtual lldb_private::ConstString
    GetPluginName();

    virtual uint32_t
    GetPluginVersion();

    virtual void
    Disable (const char** categories, lldb_private::Stream *feedback_strm);

    void
    Delete ();

    virtual bool
    Enable (lldb::StreamSP &log_stream_sp,
            uint32_t log_options,
            lldb_private::Stream *feedback_strm,      // Feedback stream for argument errors etc
            const char **categories);    // The categories to enable within this logging stream, if empty, enable default set

    virtual void
    ListCategories (lldb_private::Stream *strm);

    static lldb_private::Log *
    GetLog ();

    static lldb_private::Log *
    GetLogIfAll (uint32_t mask);
    
    static lldb_private::Log *
    GetLogIfAny (uint32_t mask);
    
    static void
    LogIf (uint32_t mask, const char *format, ...);
};

#endif  // SymbolFileDWARF_LogChannelDWARF_h_
