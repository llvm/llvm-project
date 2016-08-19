//===-- LogMessageOsLog.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LogMessageOsLog.h"

#include "ActivityStore.h"
#include "ActivityStreamSPI.h"

namespace
{
    static os_log_copy_formatted_message_t s_log_copy_formatted_message;
}

void
LogMessageOsLog::SetFormatterFunction(os_log_copy_formatted_message_t
                                      format_func)
{
    s_log_copy_formatted_message = format_func;
}

LogMessageOsLog::LogMessageOsLog(const ActivityStore &activity_store,
                                 ActivityStreamEntry &entry) :
    LogMessage(),
    m_activity_store(activity_store),
    m_entry(entry),
    m_message()
{
}

bool
LogMessageOsLog::HasActivity() const
{
    return m_entry.activity_id != 0;
}

const char*
LogMessageOsLog::GetActivity() const
{
    return m_activity_store.GetActivityForID(m_entry.activity_id);
}

std::string
LogMessageOsLog::GetActivityChain() const
{
    return m_activity_store.GetActivityChainForID(m_entry.activity_id);
}

bool
LogMessageOsLog::HasCategory() const
{
    return m_entry.log_message.category &&
        (m_entry.log_message.category[0] != 0);
}

const char*
LogMessageOsLog::GetCategory() const
{
    return m_entry.log_message.category;
}

bool
LogMessageOsLog::HasSubsystem() const
{
    return m_entry.log_message.subsystem &&
        (m_entry.log_message.subsystem[0] != 0);
}

const char*
LogMessageOsLog::GetSubsystem() const
{
    return m_entry.log_message.subsystem;
}

const char*
LogMessageOsLog::GetMessage() const
{
    if (m_message.empty())
    {
        std::unique_ptr<char[]> formatted_message(
                            s_log_copy_formatted_message(&m_entry.log_message));
        if (formatted_message)
            m_message = formatted_message.get();
        // else
        //     TODO log
    }

    // This is safe to return as we're not modifying it once we've formatted it.
    return m_message.c_str();
}
