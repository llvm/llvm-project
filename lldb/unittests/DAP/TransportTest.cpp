//===-- TransportTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Transport.h"
#include "Protocol/ProtocolBase.h"
#include "lldb/Host/File.h"
#include "lldb/Host/Pipe.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <chrono>
#include <memory>
#include <optional>

using namespace llvm;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_dap;
using namespace lldb_dap::protocol;

class TransportTest : public testing::Test {
protected:
  Pipe input;
  Pipe output;
  std::unique_ptr<Transport> transport;

  void SetUp() override {
    ASSERT_THAT_ERROR(input.CreateNew(false).ToError(), llvm::Succeeded());
    ASSERT_THAT_ERROR(output.CreateNew(false).ToError(), llvm::Succeeded());
    transport = std::make_unique<Transport>(
        "stdio", nullptr,
        std::make_shared<NativeFile>(input.GetReadFileDescriptor(),
                                     File::eOpenOptionReadOnly,
                                     NativeFile::Unowned),
        std::make_shared<NativeFile>(output.GetWriteFileDescriptor(),
                                     File::eOpenOptionWriteOnly,
                                     NativeFile::Unowned));
  }

  void Write(StringRef json) {
    std::string message =
        formatv("Content-Length: {0}\r\n\r\n{1}", json.size(), json).str();
    ASSERT_THAT_EXPECTED(input.Write(message.data(), message.size()),
                         Succeeded());
  }
};

TEST_F(TransportTest, MalformedRequests) {
  std::string malformed_header = "COnTent-LenGth: -1{}\r\n\r\nnotjosn";
  ASSERT_THAT_EXPECTED(
      input.Write(malformed_header.data(), malformed_header.size()),
      Succeeded());
  ASSERT_THAT_EXPECTED(
      transport->Read(std::chrono::milliseconds(1)),
      FailedWithMessage(
          "expected 'Content-Length: ' and got 'COnTent-LenGth: '"));
}

TEST_F(TransportTest, Read) {
  Write(R"json({"seq": 1, "type": "request", "command": "abc"})json");
  ASSERT_THAT_EXPECTED(transport->Read(std::chrono::milliseconds(1)),
                       HasValue(testing::VariantWith<Request>(
                           testing::FieldsAre(1, "abc", std::nullopt))));
}

TEST_F(TransportTest, ReadWithTimeout) {
  ASSERT_THAT_EXPECTED(transport->Read(std::chrono::milliseconds(1)),
                       Failed<TimeoutError>());
}

TEST_F(TransportTest, ReadWithEOF) {
  input.CloseWriteFileDescriptor();
  ASSERT_THAT_EXPECTED(transport->Read(std::chrono::milliseconds(1)),
                       Failed<EndOfFileError>());
}
