//===-- GDBRemoteCommunicationClientTest.cpp --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(_MSC_VER) && (_HAS_EXCEPTIONS == 0)
// Workaround for MSVC standard library bug, which fails to include <thread> when
// exceptions are disabled.
#include <eh.h>
#endif
#include <future>

#include "GDBRemoteTestUtils.h"
#include "gtest/gtest.h"

#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationClient.h"

using namespace lldb_private::process_gdb_remote;
using namespace lldb_private;
using namespace lldb;

namespace
{

typedef GDBRemoteCommunication::PacketResult PacketResult;

struct TestClient : public GDBRemoteCommunicationClient
{
    TestClient() { m_send_acks = false; }
};

void
Handle_QThreadSuffixSupported(MockServer &server, bool supported)
{
    StringExtractorGDBRemote request;
    ASSERT_EQ(PacketResult::Success, server.GetPacket(request));
    ASSERT_EQ("QThreadSuffixSupported", request.GetStringRef());
    if (supported)
        ASSERT_EQ(PacketResult::Success, server.SendOKResponse());
    else
        ASSERT_EQ(PacketResult::Success, server.SendUnimplementedResponse(nullptr));
}

void
HandlePacket(MockServer &server, llvm::StringRef expected, llvm::StringRef response)
{
    StringExtractorGDBRemote request;
    ASSERT_EQ(PacketResult::Success, server.GetPacket(request));
    ASSERT_EQ(expected, request.GetStringRef());
    ASSERT_EQ(PacketResult::Success, server.SendPacket(response));
}

} // end anonymous namespace

class GDBRemoteCommunicationClientTest : public GDBRemoteTest
{
};

TEST_F(GDBRemoteCommunicationClientTest, WriteRegister)
{
    TestClient client;
    MockServer server;
    Connect(client, server);
    if (HasFailure())
        return;

    const lldb::tid_t tid = 0x47;
    const uint32_t reg_num = 4;
    std::future<bool> write_result =
        std::async(std::launch::async, [&client] { return client.WriteRegister(tid, reg_num, "ABCD"); });

    Handle_QThreadSuffixSupported(server, true);

    HandlePacket(server, "P4=41424344;thread:0047;", "OK");
    ASSERT_TRUE(write_result.get());

    write_result = std::async(std::launch::async,
                              [&client] { return client.WriteAllRegisters(tid, "404142434445464748494a4b4c4d4e4f"); });

    HandlePacket(server, "G404142434445464748494a4b4c4d4e4f;thread:0047;", "OK");
    ASSERT_TRUE(write_result.get());
}

TEST_F(GDBRemoteCommunicationClientTest, WriteRegisterNoSuffix)
{
    TestClient client;
    MockServer server;
    Connect(client, server);
    if (HasFailure())
        return;

    const lldb::tid_t tid = 0x47;
    const uint32_t reg_num = 4;
    std::future<bool> write_result =
        std::async(std::launch::async, [&client] { return client.WriteRegister(tid, reg_num, "ABCD"); });

    Handle_QThreadSuffixSupported(server, false);
    HandlePacket(server, "Hg47", "OK");
    HandlePacket(server, "P4=41424344", "OK");
    ASSERT_TRUE(write_result.get());

    write_result = std::async(std::launch::async,
                              [&client] { return client.WriteAllRegisters(tid, "404142434445464748494a4b4c4d4e4f"); });

    HandlePacket(server, "G404142434445464748494a4b4c4d4e4f", "OK");
    ASSERT_TRUE(write_result.get());
}
