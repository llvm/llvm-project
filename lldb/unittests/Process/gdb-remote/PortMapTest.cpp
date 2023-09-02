//===-- PortMapTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServerPlatform.h"

using namespace lldb_private::process_gdb_remote;

TEST(PortMapTest, Constructors) {
  // Default construct to empty map
  GDBRemoteCommunicationServerPlatform::PortMap p1;
  ASSERT_TRUE(p1.empty());

  // Empty means no restrictions, return 0 and bind to get a port
  llvm::Expected<uint16_t> available_port = p1.GetNextAvailablePort();
  ASSERT_THAT_EXPECTED(available_port, llvm::HasValue(0));

  // Adding any port makes it not empty
  p1.AllowPort(1);
  ASSERT_FALSE(p1.empty());

  // So we will return the added port this time
  available_port = p1.GetNextAvailablePort();
  ASSERT_THAT_EXPECTED(available_port, llvm::HasValue(1));

  // Construct from a range of ports
  GDBRemoteCommunicationServerPlatform::PortMap p2(1, 4);
  ASSERT_FALSE(p2.empty());

  // Use up all the ports
  for (uint16_t expected = 1; expected < 4; ++expected) {
    available_port = p2.GetNextAvailablePort();
    ASSERT_THAT_EXPECTED(available_port, llvm::HasValue(expected));
    p2.AssociatePortWithProcess(*available_port, 1);
  }

  // Now we fail since we're not an empty port map but all ports are used
  available_port = p2.GetNextAvailablePort();
  ASSERT_THAT_EXPECTED(available_port, llvm::Failed());
}

TEST(PortMapTest, FreePort) {
  GDBRemoteCommunicationServerPlatform::PortMap p(1, 4);
  // Use up all the ports
  for (uint16_t port = 1; port < 4; ++port) {
    p.AssociatePortWithProcess(port, 1);
  }

  llvm::Expected<uint16_t> available_port = p.GetNextAvailablePort();
  ASSERT_THAT_EXPECTED(available_port, llvm::Failed());

  // Can't free a port that isn't in the map
  ASSERT_FALSE(p.FreePort(0));
  ASSERT_FALSE(p.FreePort(4));

  // After freeing a port it becomes available
  ASSERT_TRUE(p.FreePort(2));
  available_port = p.GetNextAvailablePort();
  ASSERT_THAT_EXPECTED(available_port, llvm::HasValue(2));
}

TEST(PortMapTest, FreePortForProcess) {
  GDBRemoteCommunicationServerPlatform::PortMap p;
  p.AllowPort(1);
  p.AllowPort(2);
  ASSERT_TRUE(p.AssociatePortWithProcess(1, 11));
  ASSERT_TRUE(p.AssociatePortWithProcess(2, 22));

  // All ports have been used
  llvm::Expected<uint16_t> available_port = p.GetNextAvailablePort();
  ASSERT_THAT_EXPECTED(available_port, llvm::Failed());

  // Can't free a port for a process that doesn't have any
  ASSERT_FALSE(p.FreePortForProcess(33));

  // You can move a used port to a new pid
  ASSERT_TRUE(p.AssociatePortWithProcess(1, 99));

  ASSERT_TRUE(p.FreePortForProcess(22));
  available_port = p.GetNextAvailablePort();
  ASSERT_THAT_EXPECTED(available_port, llvm::Succeeded());
  ASSERT_EQ(2, *available_port);

  // proces 22 no longer has a port
  ASSERT_FALSE(p.FreePortForProcess(22));
}

TEST(PortMapTest, AllowPort) {
  GDBRemoteCommunicationServerPlatform::PortMap p;

  // Allow port 1 and tie it to process 11
  p.AllowPort(1);
  ASSERT_TRUE(p.AssociatePortWithProcess(1, 11));

  // Allowing it a second time shouldn't change existing mapping
  p.AllowPort(1);
  llvm::Expected<uint16_t> available_port = p.GetNextAvailablePort();
  ASSERT_THAT_EXPECTED(available_port, llvm::Failed());

  // A new port is marked as free when allowed
  p.AllowPort(2);
  available_port = p.GetNextAvailablePort();
  ASSERT_THAT_EXPECTED(available_port, llvm::HasValue(2));

  // 11 should still be tied to port 1
  ASSERT_TRUE(p.FreePortForProcess(11));
}
