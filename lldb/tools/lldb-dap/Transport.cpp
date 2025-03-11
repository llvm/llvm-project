//===-- Transport.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Transport.h"
#include "DAPLog.h"
#include "Protocol.h"
#include "lldb/Utility/IOObject.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <utility>

using namespace llvm;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_dap;
using namespace lldb_dap::protocol;

static Expected<std::string> ReadFull(IOObject *descriptor, size_t length) {
  std::string data;
  data.resize(length);
  auto status = descriptor->Read(data.data(), length);
  if (status.Fail())
    return status.takeError();
  return data.substr(0, length);
}

static Expected<std::string> ReadUntil(IOObject *descriptor,
                                       StringRef delimiter) {
  std::string buffer;
  buffer.reserve(delimiter.size() + 1);
  while (!llvm::StringRef(buffer).ends_with(delimiter)) {
    Expected<std::string> next =
        ReadFull(descriptor, buffer.empty() ? delimiter.size() : 1);
    if (auto Err = next.takeError())
      return std::move(Err);
    // '' is returned on EOF.
    if (next->empty())
      return buffer;
    buffer += *next;
  }
  return buffer.substr(0, buffer.size() - delimiter.size());
}

/// DAP message format
/// ```
/// Content-Length: (?<length>\d+)\r\n\r\n(?<content>.{\k<length>})
/// ```
static constexpr StringLiteral kHeaderContentLength = "Content-Length: ";
static constexpr StringLiteral kHeaderSeparator = "\r\n\r\n";

namespace lldb_dap {

Transport::Transport(StringRef client_name, std::ofstream *log,
                     IOObjectSP input, IOObjectSP output)
    : m_client_name(client_name), m_log(log), m_input(std::move(input)),
      m_output(std::move(output)) {}

std::optional<Message> Transport::Read() {
  if (!m_input || !m_input->IsValid()) {
    DAP_LOG(m_log, "({0}) input is closed", m_client_name);
    return std::nullopt;
  }
  IOObject *input = m_input.get();
  Expected<std::string> message_header =
      ReadFull(input, kHeaderContentLength.size());
  if (!message_header) {
    DAP_LOG_ERROR(m_log, message_header.takeError(), "({1}) read failed: {0}",
                  m_client_name);
    return std::nullopt;
  }

  // '' returned on EOF.
  if (message_header->empty())
    return std::nullopt;
  if (*message_header != kHeaderContentLength) {
    DAP_LOG(m_log, "({0}) read failed: expected '{1}' and got '{2}'",
            m_client_name, kHeaderContentLength, *message_header);
    return std::nullopt;
  }

  Expected<std::string> raw_length = ReadUntil(input, kHeaderSeparator);
  if (!raw_length) {
    DAP_LOG_ERROR(m_log, raw_length.takeError(), "({1}) read failed: {0}",
                  m_client_name);
    return std::nullopt;
  }

  size_t length;
  if (!to_integer(*raw_length, length)) {
    DAP_LOG(m_log, "({0}) read failed: invalid content length {1}",
            m_client_name, *raw_length);
    return std::nullopt;
  }

  Expected<std::string> raw_json = ReadFull(input, length);
  if (!raw_json) {
    DAP_LOG_ERROR(m_log, raw_json.takeError(), "({1}) read failed: {0}",
                  m_client_name);
    return std::nullopt;
  }
  if (raw_json->length() != length) {
    DAP_LOG(m_log, "({0}) read failed: expected {1} bytes and got {2} bytes",
            m_client_name, length, raw_json->length());
    return std::nullopt;
  }

  DAP_LOG(m_log, "<-- ({0}) {1}", m_client_name, *raw_json);

  llvm::Expected<Message> message = json::parse<Message>(*raw_json);
  if (!message) {
    DAP_LOG_ERROR(m_log, message.takeError(), "({1}) read failed: {0}",
                  m_client_name);
    return std::nullopt;
  }

  return std::move(*message);
}

Error Transport::Write(const Message &message) {
  if (!m_output || !m_output->IsValid())
    return createStringError("transport output is closed");

  std::string json = formatv("{0}", toJSON(message)).str();

  DAP_LOG(m_log, "--> ({0}) {1}", m_client_name, json);

  std::string Output;
  raw_string_ostream OS(Output);
  OS << kHeaderContentLength << json.length() << kHeaderSeparator << json;
  size_t num_bytes = Output.size();
  return m_output->Write(Output.data(), num_bytes).takeError();
}

} // end namespace lldb_dap
