//===-- Transport.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Transport.h"
#include "Protocol.h"
#include "lldb/Utility/IOObject.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

using namespace llvm;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_dap;
using namespace lldb_dap::protocol;

static Status ReadFull(IOObjectSP &descriptor, size_t length,
                       std::string &text) {
  if (!descriptor || !descriptor->IsValid())
    return Status("transport input is closed");

  std::string data;
  data.resize(length);

  auto status = descriptor->Read(data.data(), length);
  if (status.Fail())
    return status;

  // If we got back zero then we have reached EOF.
  if (length == 0)
    return Status(Transport::kEOF, lldb::eErrorTypeGeneric, "end-of-file");

  text += data.substr(0, length);
  return Status();
}

static Status ReadUpTo(IOObjectSP &descriptor, std::string &line,
                       const std::string &delimiter) {
  line.clear();
  while (true) {
    std::string next;
    auto status = ReadFull(descriptor, 1, next);
    if (status.Fail())
      return status;

    line += next;

    if (llvm::StringRef(line).ends_with(delimiter))
      break;
  }
  line.erase(line.size() - delimiter.size());
  return Status();
}

static Status ReadExpected(IOObjectSP &descriptor, llvm::StringRef expected) {
  std::string result;
  auto status = ReadFull(descriptor, expected.size(), result);
  if (status.Fail())
    return status;
  if (expected != result) {
    return Status::FromErrorStringWithFormatv("expected %s, got %s", expected,
                                              result);
  }
  return Status();
}

namespace lldb_dap {

Transport::Transport(StringRef client_name, IOObjectSP input, IOObjectSP output)
    : m_client_name(client_name), m_input(std::move(input)),
      m_output(std::move(output)) {}

Status Transport::Read(std::ofstream *log, Message &M) {
  // If we don't find the expected header we have reached EOF.
  auto status = ReadExpected(m_input, "Content-Length: ");
  if (status.Fail())
    return status;

  std::string rawLength;
  status = ReadUpTo(m_input, rawLength, "\r\n\r\n");
  if (status.Fail())
    return status;

  size_t length;
  if (!to_integer(rawLength, length))
    return Status::FromErrorStringWithFormatv("invalid content length {0}",
                                              rawLength);

  std::string rawJSON;
  status = ReadFull(m_input, length, rawJSON);
  if (status.Fail())
    return status;
  if (rawJSON.length() != length)
    return Status::FromErrorStringWithFormatv(
        "malformed request, expected {0} bytes, got {1} bytes", length,
        rawJSON.length());

  if (log) {
    auto now = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch());
    *log << formatv("{0:f9} <-- ({1}) {2}\n", now.count(), m_client_name,
                    rawJSON)
                .str();
  }

  auto JSON = json::parse(rawJSON);
  if (auto Err = JSON.takeError()) {
    return Status::FromErrorStringWithFormatv("malformed JSON {0}\n{1}",
                                              rawJSON, Err);
  }

  llvm::json::Path::Root Root;
  if (!fromJSON(*JSON, M, Root)) {
    std::string error;
    raw_string_ostream OS(error);
    Root.printErrorContext(*JSON, OS);
    return Status::FromErrorStringWithFormatv("malformed request: {0}", error);
  }
  return Status();
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
