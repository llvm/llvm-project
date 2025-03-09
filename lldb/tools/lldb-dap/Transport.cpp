//===-- Transport.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Transport.h"
#include "Protocol.h"
#include "c++/v1/__system_error/error_code.h"
#include "lldb/Utility/IOObject.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <system_error>
#include <utility>

using namespace llvm;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_dap;
using namespace lldb_dap::protocol;

static Expected<std::string> ReadFull(IOObjectSP &descriptor, size_t length) {
  if (!descriptor || !descriptor->IsValid())
    return createStringError("transport input is closed");

  std::string data;
  data.resize(length);

  auto status = descriptor->Read(data.data(), length);
  if (status.Fail())
    return status.takeError();

  // If we got back zero then we have reached EOF.
  if (length == 0)
    return createStringError(Transport::kEOF, "end-of-file");

  return data.substr(0, length);
}

static Expected<std::string> ReadUntil(IOObjectSP &descriptor,
                                       StringRef delimiter) {
  std::string buffer;
  buffer.reserve(delimiter.size() + 1);
  while (!llvm::StringRef(buffer).ends_with(delimiter)) {
    auto next = ReadFull(descriptor, 1);
    if (auto Err = next.takeError())
      return std::move(Err);
    buffer += *next;
  }
  return buffer.substr(0, buffer.size() - delimiter.size());
}

static Error ReadExpected(IOObjectSP &descriptor, StringRef want) {
  auto got = ReadFull(descriptor, want.size());
  if (auto Err = got.takeError())
    return Err;
  if (*got != want) {
    return createStringError("want %s, got %s", want.str().c_str(),
                             got->c_str());
  }
  return Error::success();
}

namespace lldb_dap {

const std::error_code Transport::kEOF =
    std::error_code(0x1001, std::generic_category());

Transport::Transport(StringRef client_name, IOObjectSP input, IOObjectSP output)
    : m_client_name(client_name), m_input(std::move(input)),
      m_output(std::move(output)) {}

Expected<protocol::Message> Transport::Read(std::ofstream *log) {
  // If we don't find the expected header we have reached EOF.
  if (auto Err = ReadExpected(m_input, "Content-Length: "))
    return std::move(Err);

  auto rawLength = ReadUntil(m_input, "\r\n\r\n");
  if (auto Err = rawLength.takeError())
    return std::move(Err);

  size_t length;
  if (!to_integer(*rawLength, length))
    return createStringError("invalid content length %s", rawLength->c_str());

  auto rawJSON = ReadFull(m_input, length);
  if (auto Err = rawJSON.takeError())
    return std::move(Err);
  if (rawJSON->length() != length)
    return createStringError(
        "malformed request, expected %ld bytes, got %ld bytes", length,
        rawJSON->length());

  if (log) {
    auto now = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch());
    *log << formatv("{0:f9} <-- ({1}) {2}\n", now.count(), m_client_name,
                    *rawJSON)
                .str();
  }

  auto JSON = json::parse(*rawJSON);
  if (auto Err = JSON.takeError()) {
    return createStringError("malformed JSON %s\n%s", rawJSON->c_str(),
                             llvm::toString(std::move(Err)).c_str());
  }

  protocol::Message M;
  llvm::json::Path::Root Root;
  if (!fromJSON(*JSON, M, Root)) {
    std::string error;
    raw_string_ostream OS(error);
    Root.printErrorContext(*JSON, OS);
    return createStringError("malformed request: %s", error.c_str());
  }
  return std::move(M);
}

lldb_private::Status Transport::Write(std::ofstream *log,
                                      const protocol::Message &M) {
  if (!m_output || !m_output->IsValid())
    return Status("transport output is closed");

  std::string JSON = formatv("{0}", toJSON(M)).str();

  if (log) {
    auto now = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch());
    *log << formatv("{0:f9} --> ({1}) {2}\n", now.count(), m_client_name, JSON)
                .str();
  }

  std::string Output;
  raw_string_ostream OS(Output);
  OS << "Content-Length: " << JSON.length() << "\r\n\r\n" << JSON;
  size_t num_bytes = Output.size();
  return m_output->Write(Output.data(), num_bytes);
}

} // end namespace lldb_dap
