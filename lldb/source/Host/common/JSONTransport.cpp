//===-- JSONTransport.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/JSONTransport.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
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

JSONTransport::JSONTransport(IOObjectSP input, IOObjectSP output)
    : m_input(std::move(input)), m_output(std::move(output)) {}

void JSONTransport::Log(llvm::StringRef message) {
  LLDB_LOG(GetLog(LLDBLog::Host), "{0}", message);
}

// Parses messages based on
// https://microsoft.github.io/debug-adapter-protocol/overview#base-protocol
Expected<std::vector<std::string>> HTTPDelimitedJSONTransport::Parse() {
  if (m_buffer.empty())
    return std::vector<std::string>{};

  std::vector<std::string> messages;
  StringRef buffer = m_buffer;
  size_t content_length = 0, end_of_last_message = 0, cursor = 0;
  do {
    auto idx = buffer.find(kHeaderSeparator, cursor);
    // Separator not found, we need more data.
    if (idx == StringRef::npos)
      break;

    auto header = buffer.slice(cursor, idx);
    cursor = idx + kHeaderSeparator.size();

    // An empty line separates the headers from the message body.
    if (header.empty()) {
      // Check if we have enough data or wait for the next chunk to arrive.
      if (content_length + cursor > buffer.size())
        break;

      std::string body = buffer.substr(cursor, content_length).str();
      end_of_last_message = cursor + content_length;
      cursor += content_length;
      Logv("--> {0}", body);
      messages.emplace_back(std::move(body));
      content_length = 0;
      continue;
    }

    // HTTP Headers are formatted like `<field-name> ':' [<field-value>]`.
    if (!header.contains(kHeaderFieldSeparator))
      return createStringError("malformed content header");

    auto [name, value] = header.split(kHeaderFieldSeparator);

    // Handle known headers, at the moment only "Content-Length" is specified,
    // other headers are ignored.
    if (name.lower() == kHeaderContentLength.lower()) {
      value = value.trim();
      if (value.trim().consumeInteger(10, content_length))
        return createStringError(std::errc::invalid_argument,
                                 "invalid content length: %s",
                                 value.str().c_str());
    }
  } while (cursor < buffer.size());

  // Store the remainder of the buffer for the next read callback.
  m_buffer = buffer.substr(end_of_last_message);

  return messages;
}

Error HTTPDelimitedJSONTransport::WriteImpl(const std::string &message) {
  if (!m_output || !m_output->IsValid())
    return llvm::make_error<TransportInvalidError>();

  Logv("<-- {0}", message);

  std::string Output;
  raw_string_ostream OS(Output);
  OS << kHeaderContentLength << kHeaderFieldSeparator << ' ' << message.length()
     << kHeaderSeparator << kHeaderSeparator << message;
  size_t num_bytes = Output.size();
  return m_output->Write(Output.data(), num_bytes).takeError();
}

Expected<std::vector<std::string>> JSONRPCTransport::Parse() {
  std::vector<std::string> messages;
  StringRef buf = m_buffer;
  do {
    size_t idx = buf.find(kMessageSeparator);
    if (idx == StringRef::npos)
      break;
    std::string raw_json = buf.substr(0, idx).str();
    buf = buf.substr(idx + 1);
    Logv("--> {0}", raw_json);
    messages.push_back(raw_json);
  } while (!buf.empty());

  // Store the remainder of the buffer for the next read callback.
  m_buffer = buf.str();

  return messages;
}

Error JSONRPCTransport::WriteImpl(const std::string &message) {
  if (!m_output || !m_output->IsValid())
    return llvm::make_error<TransportInvalidError>();

  Logv("<-- {0}", message);

  std::string Output;
  llvm::raw_string_ostream OS(Output);
  OS << message << kMessageSeparator;
  size_t num_bytes = Output.size();
  return m_output->Write(Output.data(), num_bytes).takeError();
}

char TransportEOFError::ID;
char TransportUnhandledContentsError::ID;
char TransportInvalidError::ID;
