//===-- CommandHistory.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <inttypes.h>

#include "lldb/Interpreter/CommandHistory.h"
#include "lldb/Host/StringConvert.h"

using namespace lldb;
using namespace lldb_private;

CommandHistory::CommandHistory() : m_mutex(), m_history()
{}

CommandHistory::~CommandHistory ()
{}

size_t
CommandHistory::GetSize () const
{
    std::lock_guard<std::recursive_mutex> guard(m_mutex);
    return m_history.size();
}

bool
CommandHistory::IsEmpty () const
{
    std::lock_guard<std::recursive_mutex> guard(m_mutex);
    return m_history.empty();
}

const char*
CommandHistory::FindString (const char* input_str) const
{
    std::lock_guard<std::recursive_mutex> guard(m_mutex);
    if (!input_str)
        return nullptr;
    if (input_str[0] != g_repeat_char)
        return nullptr;
    if (input_str[1] == '-')
    {
        bool success;
        size_t idx = StringConvert::ToUInt32 (input_str+2, 0, 0, &success);
        if (!success)
            return nullptr;
        if (idx > m_history.size())
            return nullptr;
        idx = m_history.size() - idx;
        return m_history[idx].c_str();
        
    }
    else if (input_str[1] == g_repeat_char)
    {
        if (m_history.empty())
            return nullptr;
        else
            return m_history.back().c_str();
    }
    else
    {
        bool success;
        uint32_t idx = StringConvert::ToUInt32 (input_str+1, 0, 0, &success);
        if (!success)
            return nullptr;
        if (idx >= m_history.size())
            return nullptr;
        return m_history[idx].c_str();
    }
}

const char*
CommandHistory::GetStringAtIndex (size_t idx) const
{
    std::lock_guard<std::recursive_mutex> guard(m_mutex);
    if (idx < m_history.size())
        return m_history[idx].c_str();
    return nullptr;
}

const char*
CommandHistory::operator [] (size_t idx) const
{
    return GetStringAtIndex(idx);
}

const char*
CommandHistory::GetRecentmostString () const
{
    std::lock_guard<std::recursive_mutex> guard(m_mutex);
    if (m_history.empty())
        return nullptr;
    return m_history.back().c_str();
}

void
CommandHistory::AppendString (const std::string& str,
                              bool reject_if_dupe)
{
    std::lock_guard<std::recursive_mutex> guard(m_mutex);
    if (reject_if_dupe)
    {
        if (!m_history.empty())
        {
            if (str == m_history.back())
                return;
        }
    }
    m_history.push_back(std::string(str));
}

void
CommandHistory::Clear ()
{
    std::lock_guard<std::recursive_mutex> guard(m_mutex);
    m_history.clear();
}

void
CommandHistory::Dump (Stream& stream,
                      size_t start_idx,
                      size_t stop_idx) const
{
    std::lock_guard<std::recursive_mutex> guard(m_mutex);
    stop_idx = std::min(stop_idx + 1, m_history.size());
    for (size_t counter = start_idx;
         counter < stop_idx;
         counter++)
    {
        const std::string hist_item = m_history[counter];
        if (!hist_item.empty())
        {
            stream.Indent();
            stream.Printf("%4" PRIu64 ": %s\n", (uint64_t)counter, hist_item.c_str());
        }
    }
}
