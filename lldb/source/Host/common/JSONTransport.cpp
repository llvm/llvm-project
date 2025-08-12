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

void TransportEOFError::log(llvm::raw_ostream &OS) const {
  OS << "transport EOF";
}

std::error_code TransportEOFError::convertToErrorCode() const {
  return std::make_error_code(std::errc::io_error);
}

TransportUnhandledContentsError::TransportUnhandledContentsError(
    std::string unhandled_contents)
    : m_unhandled_contents(unhandled_contents) {}

void TransportUnhandledContentsError::log(llvm::raw_ostream &OS) const {
  OS << "transport EOF with unhandled contents " << m_unhandled_contents;
}
std::error_code TransportUnhandledContentsError::convertToErrorCode() const {
  return std::make_error_code(std::errc::bad_message);
}

void TransportInvalidError::log(llvm::raw_ostream &OS) const {
  OS << "transport IO object invalid";
}
std::error_code TransportInvalidError::convertToErrorCode() const {
  return std::make_error_code(std::errc::not_connected);
}

JSONTransport::JSONTransport(IOObjectSP input, IOObjectSP output)
    : m_input(std::move(input)), m_output(std::move(output)) {}

void JSONTransport::Log(llvm::StringRef message) {
  LLDB_LOG(GetLog(LLDBLog::Host), "{0}", message);
}

// Parses messages based on
// https://microsoft.github.io/debug-adapter-protocol/overview#base-protocol
Expected<std::vector<std::string>> HTTPDelimitedJSONTransport::Parse() {
  std::vector<std::string> messages;
  StringRef buffer = m_buffer;
  while (buffer.contains(kEndOfHeader)) {
    auto [headers, rest] = buffer.split(kEndOfHeader);
    size_t content_length = 0;
    // HTTP Headers are formatted like `<field-name> ':' [<field-value>]`.
    for (const auto &header : llvm::split(headers, kHeaderSeparator)) {
      auto [key, value] = header.split(kHeaderFieldSeparator);
      // 'Content-Length' is the only meaningful key at the moment. Others are
      // ignored.
      if (!key.equals_insensitive(kHeaderContentLength))
        continue;

      value = value.trim();
      if (!llvm::to_integer(value, content_length, 10))
        return createStringError(std::errc::invalid_argument,
                                 "invalid content length: %s",
                                 value.str().c_str());
    }

    // Check if we have enough data.
    if (content_length > rest.size())
      break;

    StringRef body = rest.take_front(content_length);
    buffer = rest.drop_front(content_length);
    messages.emplace_back(body.str());
    Logv("--> {0}", body);
  }

  // Store the remainder of the buffer for the next read callback.
  m_buffer = buffer.str();

  return std::move(messages);
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
  while (buf.contains(kMessageSeparator)) {
    auto [raw_json, rest] = buf.split(kMessageSeparator);
    buf = rest;
    messages.emplace_back(raw_json.str());
    Logv("--> {0}", raw_json);
  }

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
