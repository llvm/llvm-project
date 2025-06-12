//===-- TransportTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Transport.h"
#include "Protocol/ProtocolBase.h"
#include "TestBase.h"
#include "lldb/Host/File.h"
#include "lldb/Host/Pipe.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <chrono>
#include <memory>
#include <optional>

using namespace llvm;
using namespace lldb;
using namespace lldb_dap;
using namespace lldb_dap_tests;
using namespace lldb_dap::protocol;
using lldb_private::File;
using lldb_private::NativeFile;
using lldb_private::Pipe;
using lldb_private::TransportEOFError;
using lldb_private::TransportTimeoutError;

class TransportTest : public PipeBase {
protected:
  std::unique_ptr<Transport> transport;

  void SetUp() override {
    PipeBase::SetUp();
    transport = std::make_unique<Transport>(
        "stdio", nullptr,
        std::make_shared<NativeFile>(input.GetReadFileDescriptor(),
                                     File::eOpenOptionReadOnly,
                                     NativeFile::Unowned),
        std::make_shared<NativeFile>(output.GetWriteFileDescriptor(),
                                     File::eOpenOptionWriteOnly,
                                     NativeFile::Unowned));
  }
};

TEST_F(TransportTest, MalformedRequests) {
  std::string malformed_header = "COnTent-LenGth: -1{}\r\n\r\nnotjosn";
  ASSERT_THAT_EXPECTED(
      input.Write(malformed_header.data(), malformed_header.size()),
      Succeeded());
  ASSERT_THAT_EXPECTED(
      transport->Read<protocol::Message>(std::chrono::milliseconds(1)),
      FailedWithMessage(
          "expected 'Content-Length: ' and got 'COnTent-LenGth: '"));
}

TEST_F(TransportTest, Read) {
  std::string json =
      R"json({"seq": 1, "type": "request", "command": "abc"})json";
  std::string message =
      formatv("Content-Length: {0}\r\n\r\n{1}", json.size(), json).str();
  ASSERT_THAT_EXPECTED(input.Write(message.data(), message.size()),
                       Succeeded());
  ASSERT_THAT_EXPECTED(
      transport->Read<protocol::Message>(std::chrono::milliseconds(1)),
      HasValue(testing::VariantWith<Request>(testing::FieldsAre(
          /*seq=*/1, /*command=*/"abc", /*arguments=*/std::nullopt))));
}

TEST_F(TransportTest, ReadWithTimeout) {
  ASSERT_THAT_EXPECTED(
      transport->Read<protocol::Message>(std::chrono::milliseconds(1)),
      Failed<TransportTimeoutError>());
}

TEST_F(TransportTest, ReadWithEOF) {
  input.CloseWriteFileDescriptor();
  ASSERT_THAT_EXPECTED(
      transport->Read<protocol::Message>(std::chrono::milliseconds(1)),
      Failed<TransportEOFError>());
}

TEST_F(TransportTest, Write) {
  ASSERT_THAT_ERROR(transport->Write(Event{"my-event", std::nullopt}),
                    Succeeded());
  output.CloseWriteFileDescriptor();
  char buf[1024];
  Expected<size_t> bytes_read =
      output.Read(buf, sizeof(buf), std::chrono::milliseconds(1));
  ASSERT_THAT_EXPECTED(bytes_read, Succeeded());
  ASSERT_EQ(
      StringRef(buf, *bytes_read),
      StringRef("Content-Length: 43\r\n\r\n"
                R"json({"event":"my-event","seq":0,"type":"event"})json"));
}
