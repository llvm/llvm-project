//===-- Transport.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#include "Transport.h"
#include "DAPLog.h"
#include "Protocol.h"
#include "lldb/Utility/IOObject.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

using namespace llvm;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_dap::protocol;

static Expected<std::string> ReadBytes(IOObjectSP &IO, size_t length) {
  std::string data;
  data.resize(length);

  auto status = IO->Read(data.data(), length);
  if (status.Fail())
    return status.takeError();
  // Return a slice of the amount read, which may be less than the requested
  // length.
  return data.substr(0, length);
}

static Expected<std::string> ReadUpTo(IOObjectSP &IO,
                                      const std::string &delimiter) {
  std::string buf;
  while (!StringRef(buf).ends_with(delimiter)) {
    auto byte = ReadBytes(IO, 1);
    if (!byte)
      return byte.takeError();
    buf += *byte;
  }
  return buf.substr(0, buf.size() - delimiter.size());
}

static Error ReadExpected(IOObjectSP &IO, const std::string &expected) {
  auto result = ReadBytes(IO, expected.size());
  if (!result)
    return result.takeError();
  if (*result != expected)
    return createStringError("expected %s, got %s", expected.c_str(),
                             result->c_str());
  return Error::success();
}

namespace lldb_dap {

Transport::Transport(StringRef client_name, IOObjectSP input, IOObjectSP output)
    : m_client_name(client_name), m_input(std::move(input)),
      m_output(std::move(output)) {}

Status Transport::Write(const ProtocolMessage &M) {
  if (!m_output || !m_output->IsValid())
    return Status("transport output is closed");

  std::string JSON = formatv("{0}", toJSON(M)).str();

  LLDB_LOG(GetLog(DAPLog::Transport), "--> ({0}) {1}", m_client_name, JSON);

  std::string Output;
  raw_string_ostream OS(Output);
  OS << "Content-Length: " << JSON.length() << "\r\n\r\n" << JSON;
  size_t num_bytes = Output.size();
  return m_output->Write(Output.data(), num_bytes);
}

Expected<ProtocolMessage> Transport::Read() {
  if (!m_input || !m_input->IsValid())
    return make_error<StringError>(inconvertibleErrorCode(),
                                   "transport input is closed");

  if (auto Err = ReadExpected(m_input, "Content-Length: "))
    return std::move(Err);

  auto rawLength = ReadUpTo(m_input, "\r\n\r\n");
  if (!rawLength)
    return rawLength.takeError();

  size_t length;
  if (!to_integer(*rawLength, length))
    return createStringError("invalid content length %s", rawLength->c_str());

  auto rawJSON = ReadBytes(m_input, length);
  if (!rawJSON)
    return rawJSON.takeError();
  if (rawJSON->length() != length)
    return createStringError(
        std::errc::message_size,
        "malformed request, expected %ld bytes, got %ld bytes", length,
        rawJSON->length());

  LLDB_LOG(GetLog(DAPLog::Transport), "<-- ({0}) {1}", m_client_name, *rawJSON);

  return json::parse<protocol::ProtocolMessage>(*rawJSON);
}

} // namespace lldb_dap
