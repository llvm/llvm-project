//===-- LogChannelDWARF.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LogChannelDWARF.h"

#include "SymbolFileDWARF.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Interpreter/Args.h"

using namespace lldb;
using namespace lldb_private;

// when the one and only logging channel is enabled, then this will be non NULL.
static LogChannelDWARF *g_log_channel = NULL;

LogChannelDWARF::LogChannelDWARF() : LogChannel() {}

LogChannelDWARF::~LogChannelDWARF() {}

void LogChannelDWARF::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(),
                                LogChannelDWARF::CreateInstance);
}

void LogChannelDWARF::Terminate() {
  PluginManager::UnregisterPlugin(LogChannelDWARF::CreateInstance);
}

LogChannel *LogChannelDWARF::CreateInstance() { return new LogChannelDWARF(); }

lldb_private::ConstString LogChannelDWARF::GetPluginNameStatic() {
  return SymbolFileDWARF::GetPluginNameStatic();
}

const char *LogChannelDWARF::GetPluginDescriptionStatic() {
  return "DWARF log channel for debugging plug-in issues.";
}

lldb_private::ConstString LogChannelDWARF::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t LogChannelDWARF::GetPluginVersion() { return 1; }

void LogChannelDWARF::Delete() { g_log_channel = NULL; }

void LogChannelDWARF::Disable(const char **categories, Stream *feedback_strm) {
  if (m_log_ap.get() == NULL)
    return;

  uint32_t flag_bits = m_log_ap->GetMask().Get();
  for (size_t i = 0; categories[i] != NULL; ++i) {
    const char *arg = categories[i];

    if (::strcasecmp(arg, "all") == 0)
      flag_bits &= ~DWARF_LOG_ALL;
    else if (::strcasecmp(arg, "info") == 0)
      flag_bits &= ~DWARF_LOG_DEBUG_INFO;
    else if (::strcasecmp(arg, "line") == 0)
      flag_bits &= ~DWARF_LOG_DEBUG_LINE;
    else if (::strcasecmp(arg, "pubnames") == 0)
      flag_bits &= ~DWARF_LOG_DEBUG_PUBNAMES;
    else if (::strcasecmp(arg, "pubtypes") == 0)
      flag_bits &= ~DWARF_LOG_DEBUG_PUBTYPES;
    else if (::strcasecmp(arg, "aranges") == 0)
      flag_bits &= ~DWARF_LOG_DEBUG_ARANGES;
    else if (::strcasecmp(arg, "lookups") == 0)
      flag_bits &= ~DWARF_LOG_LOOKUPS;
    else if (::strcasecmp(arg, "map") == 0)
      flag_bits &= ~DWARF_LOG_DEBUG_MAP;
    else if (::strcasecmp(arg, "default") == 0)
      flag_bits &= ~DWARF_LOG_DEFAULT;
    else if (::strcasecmp(arg, "verbose") == 0)
      flag_bits &= ~DWARF_LOG_VERBOSE;
    else if (::strncasecmp(arg, "comp", 4) == 0)
      flag_bits &= ~DWARF_LOG_TYPE_COMPLETION;
    else {
      feedback_strm->Printf("error: unrecognized log category '%s'\n", arg);
      ListCategories(feedback_strm);
    }
  }

  if (flag_bits == 0)
    Delete();
  else
    m_log_ap->GetMask().Reset(flag_bits);

  return;
}

bool LogChannelDWARF::Enable(
    StreamSP &log_stream_sp, uint32_t log_options,
    Stream *feedback_strm,  // Feedback stream for argument errors etc
    const char **categories // The categories to enable within this logging
                            // stream, if empty, enable default set
    ) {
  Delete();

  if (m_log_ap)
    m_log_ap->SetStream(log_stream_sp);
  else
    m_log_ap.reset(new Log(log_stream_sp));

  g_log_channel = this;
  uint32_t flag_bits = 0;
  bool got_unknown_category = false;
  for (size_t i = 0; categories[i] != NULL; ++i) {
    const char *arg = categories[i];

    if (::strcasecmp(arg, "all") == 0)
      flag_bits |= DWARF_LOG_ALL;
    else if (::strcasecmp(arg, "info") == 0)
      flag_bits |= DWARF_LOG_DEBUG_INFO;
    else if (::strcasecmp(arg, "line") == 0)
      flag_bits |= DWARF_LOG_DEBUG_LINE;
    else if (::strcasecmp(arg, "pubnames") == 0)
      flag_bits |= DWARF_LOG_DEBUG_PUBNAMES;
    else if (::strcasecmp(arg, "pubtypes") == 0)
      flag_bits |= DWARF_LOG_DEBUG_PUBTYPES;
    else if (::strcasecmp(arg, "aranges") == 0)
      flag_bits |= DWARF_LOG_DEBUG_ARANGES;
    else if (::strcasecmp(arg, "lookups") == 0)
      flag_bits |= DWARF_LOG_LOOKUPS;
    else if (::strcasecmp(arg, "map") == 0)
      flag_bits |= DWARF_LOG_DEBUG_MAP;
    else if (::strcasecmp(arg, "default") == 0)
      flag_bits |= DWARF_LOG_DEFAULT;
    else if (::strcasecmp(arg, "verbose") == 0)
      flag_bits |= DWARF_LOG_VERBOSE;
    else if (::strncasecmp(arg, "comp", 4) == 0)
      flag_bits |= DWARF_LOG_TYPE_COMPLETION;
    else {
      feedback_strm->Printf("error: unrecognized log category '%s'\n", arg);
      if (got_unknown_category == false) {
        got_unknown_category = true;
        ListCategories(feedback_strm);
      }
    }
  }
  if (flag_bits == 0)
    flag_bits = DWARF_LOG_DEFAULT;
  m_log_ap->GetMask().Reset(flag_bits);
  m_log_ap->GetOptions().Reset(log_options);
  return m_log_ap.get() != NULL;
}

void LogChannelDWARF::ListCategories(Stream *strm) {
  strm->Printf(
      "Logging categories for '%s':\n"
      "  all - turn on all available logging categories\n"
      "  info - log the parsing of .debug_info\n"
      "  line - log the parsing of .debug_line\n"
      "  pubnames - log the parsing of .debug_pubnames\n"
      "  pubtypes - log the parsing of .debug_pubtypes\n"
      "  aranges - log the parsing of .debug_aranges\n"
      "  lookups - log any lookups that happen by name, regex, or address\n"
      "  completion - log struct/unions/class type completions\n"
      "  map - log insertions of object files into DWARF debug maps\n",
      SymbolFileDWARF::GetPluginNameStatic().GetCString());
}

Log *LogChannelDWARF::GetLog() {
  if (g_log_channel)
    return g_log_channel->m_log_ap.get();

  return NULL;
}

Log *LogChannelDWARF::GetLogIfAll(uint32_t mask) {
  if (g_log_channel && g_log_channel->m_log_ap.get()) {
    if (g_log_channel->m_log_ap->GetMask().AllSet(mask))
      return g_log_channel->m_log_ap.get();
  }
  return NULL;
}

Log *LogChannelDWARF::GetLogIfAny(uint32_t mask) {
  if (g_log_channel && g_log_channel->m_log_ap.get()) {
    if (g_log_channel->m_log_ap->GetMask().AnySet(mask))
      return g_log_channel->m_log_ap.get();
  }
  return NULL;
}

void LogChannelDWARF::LogIf(uint32_t mask, const char *format, ...) {
  if (g_log_channel) {
    Log *log = g_log_channel->m_log_ap.get();
    if (log && log->GetMask().AnySet(mask)) {
      va_list args;
      va_start(args, format);
      log->VAPrintf(format, args);
      va_end(args);
    }
  }
}
