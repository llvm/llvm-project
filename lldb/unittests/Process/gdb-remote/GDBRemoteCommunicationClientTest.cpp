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
#include "lldb/Core/DataBuffer.h"

#include "llvm/ADT/ArrayRef.h"

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

uint8_t all_registers[] = {'@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O'};
std::string all_registers_hex = "404142434445464748494a4b4c4d4e4f";
uint8_t one_register[] = {'A', 'B', 'C', 'D'};
std::string one_register_hex = "41424344";

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
        std::async(std::launch::async, [&] { return client.WriteRegister(tid, reg_num, one_register); });

    Handle_QThreadSuffixSupported(server, true);

    HandlePacket(server, "P4=" + one_register_hex + ";thread:0047;", "OK");
    ASSERT_TRUE(write_result.get());

    write_result = std::async(std::launch::async, [&] { return client.WriteAllRegisters(tid, all_registers); });

    HandlePacket(server, "G" + all_registers_hex + ";thread:0047;", "OK");
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
        std::async(std::launch::async, [&] { return client.WriteRegister(tid, reg_num, one_register); });

    Handle_QThreadSuffixSupported(server, false);
    HandlePacket(server, "Hg47", "OK");
    HandlePacket(server, "P4=" + one_register_hex, "OK");
    ASSERT_TRUE(write_result.get());

    write_result = std::async(std::launch::async, [&] { return client.WriteAllRegisters(tid, all_registers); });

    HandlePacket(server, "G" + all_registers_hex, "OK");
    ASSERT_TRUE(write_result.get());
}

TEST_F(GDBRemoteCommunicationClientTest, ReadRegister)
{
    TestClient client;
    MockServer server;
    Connect(client, server);
    if (HasFailure())
        return;

    const lldb::tid_t tid = 0x47;
    const uint32_t reg_num = 4;
    std::future<bool> async_result = std::async(std::launch::async, [&] { return client.GetpPacketSupported(tid); });
    Handle_QThreadSuffixSupported(server, true);
    HandlePacket(server, "p0;thread:0047;", one_register_hex);
    ASSERT_TRUE(async_result.get());

    std::future<DataBufferSP> read_result =
        std::async(std::launch::async, [&] { return client.ReadRegister(tid, reg_num); });
    HandlePacket(server, "p4;thread:0047;", "41424344");
    auto buffer_sp = read_result.get();
    ASSERT_TRUE(bool(buffer_sp));
    ASSERT_EQ(0, memcmp(buffer_sp->GetBytes(), one_register, sizeof one_register));

    read_result = std::async(std::launch::async, [&] { return client.ReadAllRegisters(tid); });
    HandlePacket(server, "g;thread:0047;", all_registers_hex);
    buffer_sp = read_result.get();
    ASSERT_TRUE(bool(buffer_sp));
    ASSERT_EQ(0, memcmp(buffer_sp->GetBytes(), all_registers, sizeof all_registers));
}

TEST_F(GDBRemoteCommunicationClientTest, SaveRestoreRegistersNoSuffix)
{
    TestClient client;
    MockServer server;
    Connect(client, server);
    if (HasFailure())
        return;

    const lldb::tid_t tid = 0x47;
    uint32_t save_id;
    std::future<bool> async_result =
        std::async(std::launch::async, [&] { return client.SaveRegisterState(tid, save_id); });
    Handle_QThreadSuffixSupported(server, false);
    HandlePacket(server, "Hg47", "OK");
    HandlePacket(server, "QSaveRegisterState", "1");
    ASSERT_TRUE(async_result.get());
    EXPECT_EQ(1u, save_id);

    async_result = std::async(std::launch::async, [&] { return client.RestoreRegisterState(tid, save_id); });
    HandlePacket(server, "QRestoreRegisterState:1", "OK");
    ASSERT_TRUE(async_result.get());
}

TEST_F(GDBRemoteCommunicationClientTest, SyncThreadState)
{
    TestClient client;
    MockServer server;
    Connect(client, server);
    if (HasFailure())
        return;

    const lldb::tid_t tid = 0x47;
    std::future<bool> async_result = std::async(std::launch::async, [&] { return client.SyncThreadState(tid); });
    HandlePacket(server, "qSyncThreadStateSupported", "OK");
    HandlePacket(server, "QSyncThreadState:0047;", "OK");
    ASSERT_TRUE(async_result.get());
}
