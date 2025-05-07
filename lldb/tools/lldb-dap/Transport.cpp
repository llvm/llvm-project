//===-- Transport.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Transport.h"
#include "DAPLog.h"
#include "Protocol/ProtocolBase.h"
#include "lldb/Utility/IOObject.h"
#include "lldb/Utility/SelectHelper.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>
#include <utility>

using namespace llvm;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_dap;
using namespace lldb_dap::protocol;

/// ReadFull attempts to read the specified number of bytes. If EOF is
/// encountered, an empty string is returned.
static Expected<std::string>
ReadFull(IOObject &descriptor, size_t length,
         std::optional<std::chrono::microseconds> timeout = std::nullopt) {
  if (!descriptor.IsValid())
    return createStringError("transport output is closed");

  bool timeout_supported = true;
  // FIXME: SelectHelper does not work with NativeFile on Win32.
#if _WIN32
  timeout_supported = descriptor.GetFdType() == IOObject::eFDTypeSocket;
#endif

  if (timeout && timeout_supported) {
    SelectHelper sh;
    sh.SetTimeout(*timeout);
    sh.FDSetRead(descriptor.GetWaitableHandle());
    Status status = sh.Select();
    if (status.Fail()) {
      // Convert timeouts into a specific error.
      if (status.GetType() == lldb::eErrorTypePOSIX &&
          status.GetError() == ETIMEDOUT)
        return make_error<TimeoutError>();
      return status.takeError();
    }
  }

  std::string data;
  data.resize(length);
  Status status = descriptor.Read(data.data(), length);
  if (status.Fail())
    return status.takeError();

  // Read returns '' on EOF.
  if (length == 0)
    return make_error<EndOfFileError>();

  // Return the actual number of bytes read.
  return data.substr(0, length);
}

static Expected<std::string>
ReadUntil(IOObject &descriptor, StringRef delimiter,
          std::optional<std::chrono::microseconds> timeout = std::nullopt) {
  std::string buffer;
  buffer.reserve(delimiter.size() + 1);
  while (!llvm::StringRef(buffer).ends_with(delimiter)) {
    Expected<std::string> next =
        ReadFull(descriptor, buffer.empty() ? delimiter.size() : 1, timeout);
    if (auto Err = next.takeError())
      return std::move(Err);
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

char EndOfFileError::ID;
char TimeoutError::ID;

Transport::Transport(StringRef client_name, Log *log, IOObjectSP input,
                     IOObjectSP output)
    : m_client_name(client_name), m_log(log), m_input(std::move(input)),
      m_output(std::move(output)) {}

Expected<Message> Transport::Read(const std::chrono::microseconds &timeout) {
  if (!m_input || !m_input->IsValid())
    return createStringError("transport output is closed");

  IOObject *input = m_input.get();
  Expected<std::string> message_header =
      ReadFull(*input, kHeaderContentLength.size(), timeout);
  if (!message_header)
    return message_header.takeError();
  if (*message_header != kHeaderContentLength)
    return createStringError(formatv("expected '{0}' and got '{1}'",
                                     kHeaderContentLength, *message_header)
                                 .str());

  Expected<std::string> raw_length = ReadUntil(*input, kHeaderSeparator);
  if (!raw_length)
    return handleErrors(raw_length.takeError(),
                        [&](const EndOfFileError &E) -> llvm::Error {
                          return createStringError(
                              "unexpected EOF while reading header separator");
                        });

  size_t length;
  if (!to_integer(*raw_length, length))
    return createStringError(
        formatv("invalid content length {0}", *raw_length).str());

  Expected<std::string> raw_json = ReadFull(*input, length);
  if (!raw_json)
    return handleErrors(
        raw_json.takeError(), [&](const EndOfFileError &E) -> llvm::Error {
          return createStringError("unexpected EOF while reading JSON");
        });

  DAP_LOG(m_log, "--> ({0}) {1}", m_client_name, *raw_json);

  return json::parse<Message>(/*JSON=*/*raw_json,
                              /*RootName=*/"protocol_message");
}

Error Transport::Write(const Message &message) {
  if (!m_output || !m_output->IsValid())
    return createStringError("transport output is closed");

  std::string json = formatv("{0}", toJSON(message)).str();

  DAP_LOG(m_log, "<-- ({0}) {1}", m_client_name, json);

  std::string Output;
  raw_string_ostream OS(Output);
  OS << kHeaderContentLength << json.length() << kHeaderSeparator << json;
  size_t num_bytes = Output.size();
  return m_output->Write(Output.data(), num_bytes).takeError();
}

} // end namespace lldb_dap
