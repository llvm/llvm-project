//===-- NetworkSymbolManagerTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/NetworkSymbolManager.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Utility/Status.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

using namespace lldb_private;

class NetworkSymbolManagerTest : public ::testing::Test {
public:
  void SetUp() override {
    FileSystem::Initialize();
    Debugger::Initialize(nullptr);
    manager_ = std::make_unique<NetworkSymbolManager>();
  }

  void TearDown() override {
    manager_.reset();
    Debugger::Terminate();
    FileSystem::Terminate();
  }

protected:
  std::unique_ptr<NetworkSymbolManager> manager_;
};

TEST_F(NetworkSymbolManagerTest, DefaultConfiguration) {
  NetworkSymbolManager::Configuration config;

  // Test default values
  EXPECT_TRUE(config.enable_server_caching);
  EXPECT_EQ(config.debuginfod_timeout_ms, 2000u);
  EXPECT_EQ(config.symbol_server_timeout_ms, 2000u);
  EXPECT_FALSE(config.disable_network_symbols);
  EXPECT_TRUE(config.enable_adaptive_timeouts);
  EXPECT_EQ(config.cache_ttl_minutes, 5u);
}

TEST_F(NetworkSymbolManagerTest, ConfigurationValidation) {
  NetworkSymbolManager::Configuration config;

  // Valid configuration should pass
  EXPECT_TRUE(NetworkSymbolManager::ValidateConfiguration(config).Success());

  // Invalid timeout (too large)
  config.debuginfod_timeout_ms = 70000; // > 60 seconds
  EXPECT_FALSE(NetworkSymbolManager::ValidateConfiguration(config).Success());

  // Reset and test symbol server timeout
  config.debuginfod_timeout_ms = 2000;
  config.symbol_server_timeout_ms = 70000; // > 60 seconds
  EXPECT_FALSE(NetworkSymbolManager::ValidateConfiguration(config).Success());

  // Reset and test cache TTL
  config.symbol_server_timeout_ms = 2000;
  config.cache_ttl_minutes = 0; // Invalid
  EXPECT_FALSE(NetworkSymbolManager::ValidateConfiguration(config).Success());

  config.cache_ttl_minutes = 70; // Too large
  EXPECT_FALSE(NetworkSymbolManager::ValidateConfiguration(config).Success());
}

TEST_F(NetworkSymbolManagerTest, ServerAvailabilityTracking) {
  // Test server availability tracking
  std::string test_server = "http://test.server.com";

  // Initially unknown server should be considered for attempts
  EXPECT_TRUE(manager_->ShouldAttemptNetworkSymbolResolution(test_server));

  // Record a failure
  manager_->RecordServerResponse(test_server, std::chrono::milliseconds(5000),
                                false);

  // Should still attempt after one failure
  EXPECT_TRUE(manager_->ShouldAttemptNetworkSymbolResolution(test_server));

  // Record multiple consecutive failures
  for (int i = 0; i < 3; ++i) {
    manager_->RecordServerResponse(test_server, std::chrono::milliseconds(5000),
                                  false);
  }

  // After multiple failures, server should be temporarily blacklisted
  EXPECT_FALSE(manager_->ShouldAttemptNetworkSymbolResolution(test_server));
}

TEST_F(NetworkSymbolManagerTest, ResponsiveServerFiltering) {
  std::vector<llvm::StringRef> test_servers = {
    "http://good.server.com",
    "http://bad.server.com",
    "http://unknown.server.com"
  };

  // Record good server responses
  manager_->RecordServerResponse("http://good.server.com",
                                std::chrono::milliseconds(100), true);
  manager_->RecordServerResponse("http://good.server.com",
                                std::chrono::milliseconds(150), true);

  // Record bad server responses (blacklist it)
  for (int i = 0; i < 4; ++i) {
    manager_->RecordServerResponse("http://bad.server.com",
                                  std::chrono::milliseconds(5000), false);
  }

  // Get responsive servers
  auto responsive_servers = manager_->GetResponsiveServers(test_servers);

  // Verify expected server filtering behavior
  EXPECT_EQ(responsive_servers.size(), 2u);  // Exactly 2: good + unknown

  // Check that good server is included
  EXPECT_TRUE(llvm::is_contained(responsive_servers, "http://good.server.com"));

  // Check that unknown server is included (no history = assumed responsive)
  EXPECT_TRUE(llvm::is_contained(responsive_servers, "http://unknown.server.com"));

  // Check that bad server is excluded (blacklisted due to failures)
  EXPECT_FALSE(llvm::is_contained(responsive_servers, "http://bad.server.com"));
}

TEST_F(NetworkSymbolManagerTest, DisableNetworkSymbols) {
  NetworkSymbolManager::Configuration config;
  config.disable_network_symbols = true;

  EXPECT_TRUE(manager_->Configure(config).Success());

  // When disabled, should not attempt any network resolution
  EXPECT_FALSE(manager_->ShouldAttemptNetworkSymbolResolution("http://any.server.com"));

  // Should return empty list of responsive servers
  std::vector<llvm::StringRef> servers = {"http://server1.com", "http://server2.com"};
  auto responsive_servers = manager_->GetResponsiveServers(servers);
  EXPECT_TRUE(responsive_servers.empty());
}

TEST_F(NetworkSymbolManagerTest, AdaptiveTimeouts) {
  std::string fast_server = "http://fast.server.com";
  std::string slow_server = "http://slow.server.com";

  // Configure with adaptive timeouts enabled
  NetworkSymbolManager::Configuration config;
  config.enable_adaptive_timeouts = true;
  EXPECT_TRUE(manager_->Configure(config).Success());

  // Record fast server responses
  for (int i = 0; i < 5; ++i) {
    manager_->RecordServerResponse(fast_server, std::chrono::milliseconds(100), true);
  }

  // Record slow but successful server responses
  for (int i = 0; i < 5; ++i) {
    manager_->RecordServerResponse(slow_server, std::chrono::milliseconds(1500), true);
  }

  // Fast server should get shorter adaptive timeout
  auto fast_timeout = manager_->GetAdaptiveTimeout(fast_server);
  auto slow_timeout = manager_->GetAdaptiveTimeout(slow_server);

  // Fast server should have shorter timeout than slow server
  EXPECT_LT(fast_timeout, slow_timeout);

  // Both should be reasonable values
  EXPECT_GE(fast_timeout.count(), 100);
  EXPECT_LE(slow_timeout.count(), 2000);
}
