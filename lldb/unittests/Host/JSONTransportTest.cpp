//===-- JSONTransportTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/JSONTransport.h"
#include "TestingSupport/Host/PipeTestUtilities.h"
#include "lldb/Host/File.h"

using namespace llvm;
using namespace lldb_private;

namespace {
template <typename T> class JSONTransportTest : public PipeTest {
protected:
  std::unique_ptr<JSONTransport> transport;

  void SetUp() override {
    PipeTest::SetUp();
    transport = std::make_unique<T>(
        std::make_shared<NativeFile>(input.GetReadFileDescriptor(),
                                     File::eOpenOptionReadOnly,
                                     NativeFile::Unowned),
        std::make_shared<NativeFile>(output.GetWriteFileDescriptor(),
                                     File::eOpenOptionWriteOnly,
                                     NativeFile::Unowned));
  }
};

class HTTPDelimitedJSONTransportTest
    : public JSONTransportTest<HTTPDelimitedJSONTransport> {
public:
  using JSONTransportTest::JSONTransportTest;
};

class JSONRPCTransportTest : public JSONTransportTest<JSONRPCTransport> {
public:
  using JSONTransportTest::JSONTransportTest;
};

struct JSONTestType {
  std::string str;
};

llvm::json::Value toJSON(const JSONTestType &T) {
  return llvm::json::Object{{"str", T.str}};
}

bool fromJSON(const llvm::json::Value &V, JSONTestType &T, llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  return O && O.map("str", T.str);
}
} // namespace

TEST_F(HTTPDelimitedJSONTransportTest, MalformedRequests) {
  std::string malformed_header = "COnTent-LenGth: -1{}\r\n\r\nnotjosn";
  ASSERT_THAT_EXPECTED(
      input.Write(malformed_header.data(), malformed_header.size()),
      Succeeded());
  ASSERT_THAT_EXPECTED(
      transport->Read<JSONTestType>(std::chrono::milliseconds(1)),
      FailedWithMessage(
          "expected 'Content-Length: ' and got 'COnTent-LenGth: '"));
}

TEST_F(HTTPDelimitedJSONTransportTest, Read) {
  std::string json = R"json({"str": "foo"})json";
  std::string message =
      formatv("Content-Length: {0}\r\n\r\n{1}", json.size(), json).str();
  ASSERT_THAT_EXPECTED(input.Write(message.data(), message.size()),
                       Succeeded());
  ASSERT_THAT_EXPECTED(
      transport->Read<JSONTestType>(std::chrono::milliseconds(1)),
      HasValue(testing::FieldsAre(/*str=*/"foo")));
}

TEST_F(HTTPDelimitedJSONTransportTest, ReadWithEOF) {
  input.CloseWriteFileDescriptor();
  ASSERT_THAT_EXPECTED(
      transport->Read<JSONTestType>(std::chrono::milliseconds(1)),
      Failed<TransportEOFError>());
}


TEST_F(HTTPDelimitedJSONTransportTest, InvalidTransport) {
  transport = std::make_unique<HTTPDelimitedJSONTransport>(nullptr, nullptr);
  ASSERT_THAT_EXPECTED(
      transport->Read<JSONTestType>(std::chrono::milliseconds(1)),
      Failed<TransportInvalidError>());
}

TEST_F(HTTPDelimitedJSONTransportTest, Write) {
  ASSERT_THAT_ERROR(transport->Write(JSONTestType{"foo"}), Succeeded());
  output.CloseWriteFileDescriptor();
  char buf[1024];
  Expected<size_t> bytes_read =
      output.Read(buf, sizeof(buf), std::chrono::milliseconds(1));
  ASSERT_THAT_EXPECTED(bytes_read, Succeeded());
  ASSERT_EQ(StringRef(buf, *bytes_read), StringRef("Content-Length: 13\r\n\r\n"
                                                   R"json({"str":"foo"})json"));
}

TEST_F(JSONRPCTransportTest, MalformedRequests) {
  std::string malformed_header = "notjson\n";
  ASSERT_THAT_EXPECTED(
      input.Write(malformed_header.data(), malformed_header.size()),
      Succeeded());
  ASSERT_THAT_EXPECTED(
      transport->Read<JSONTestType>(std::chrono::milliseconds(1)),
      llvm::Failed());
}

TEST_F(JSONRPCTransportTest, Read) {
  std::string json = R"json({"str": "foo"})json";
  std::string message = formatv("{0}\n", json).str();
  ASSERT_THAT_EXPECTED(input.Write(message.data(), message.size()),
                       Succeeded());
  ASSERT_THAT_EXPECTED(
      transport->Read<JSONTestType>(std::chrono::milliseconds(1)),
      HasValue(testing::FieldsAre(/*str=*/"foo")));
}

TEST_F(JSONRPCTransportTest, ReadWithEOF) {
  input.CloseWriteFileDescriptor();
  ASSERT_THAT_EXPECTED(
      transport->Read<JSONTestType>(std::chrono::milliseconds(1)),
      Failed<TransportEOFError>());
}

TEST_F(JSONRPCTransportTest, Write) {
  ASSERT_THAT_ERROR(transport->Write(JSONTestType{"foo"}), Succeeded());
  output.CloseWriteFileDescriptor();
  char buf[1024];
  Expected<size_t> bytes_read =
      output.Read(buf, sizeof(buf), std::chrono::milliseconds(1));
  ASSERT_THAT_EXPECTED(bytes_read, Succeeded());
  ASSERT_EQ(StringRef(buf, *bytes_read), StringRef(R"json({"str":"foo"})json"
                                                   "\n"));
}

TEST_F(JSONRPCTransportTest, InvalidTransport) {
  transport = std::make_unique<JSONRPCTransport>(nullptr, nullptr);
  ASSERT_THAT_EXPECTED(
      transport->Read<JSONTestType>(std::chrono::milliseconds(1)),
      Failed<TransportInvalidError>());
}

#ifndef _WIN32
TEST_F(HTTPDelimitedJSONTransportTest, ReadWithTimeout) {
  ASSERT_THAT_EXPECTED(
      transport->Read<JSONTestType>(std::chrono::milliseconds(1)),
      Failed<TransportTimeoutError>());
}

TEST_F(JSONRPCTransportTest, ReadWithTimeout) {
  ASSERT_THAT_EXPECTED(
      transport->Read<JSONTestType>(std::chrono::milliseconds(1)),
      Failed<TransportTimeoutError>());
}

// Windows CRT _read checks that the file descriptor is valid and calls a
// handler if not. This handler is normally a breakpoint, which looks like a
// crash when not handled by a debugger.
// https://learn.microsoft.com/en-us/%20cpp/c-runtime-library/reference/read?view=msvc-170
TEST_F(HTTPDelimitedJSONTransportTest, ReadAfterClosed) {
  input.CloseReadFileDescriptor();
  ASSERT_THAT_EXPECTED(
      transport->Read<JSONTestType>(std::chrono::milliseconds(1)),
      llvm::Failed());
}

TEST_F(JSONRPCTransportTest, ReadAfterClosed) {
  input.CloseReadFileDescriptor();
  ASSERT_THAT_EXPECTED(
      transport->Read<JSONTestType>(std::chrono::milliseconds(1)),
      llvm::Failed());
}
#endif
