//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/SymbolLocator/SymStore/SymbolLocatorSymStore.h"

using namespace lldb_private;
using LookupEntry = SymbolLocatorSymStore::LookupEntry;

TEST(SymStoreTest, ParseEnvSymbolPaths_Srv) {
  auto check = [](const char *str) {
    std::vector<std::string> sources;
    for (LookupEntry entry : SymbolLocatorSymStore::ParseEnvSymbolPaths(str))
      sources.push_back(std::move(entry.source));
    return sources;
  };
  auto returns = [](auto... strs) { return std::vector<std::string>{strs...}; };

  // Local paths.
  EXPECT_EQ(check(""), returns());
  EXPECT_EQ(check("C:\\ProgramData\\Symbols"),
            returns("C:\\ProgramData\\Symbols"));
  EXPECT_EQ(check("C:\\symbols;\\\\buildserver\\syms;file://D:/pdb"),
            returns("C:\\symbols", "\\\\buildserver\\syms", "file://D:/pdb"));

  // Symbol servers.
  EXPECT_EQ(check("srv*https://msdl.microsoft.com/download/symbols"),
            returns("https://msdl.microsoft.com/download/symbols"));
  EXPECT_EQ(check("Srv*https://msdl.microsoft.com/download/symbols"),
            returns("https://msdl.microsoft.com/download/symbols"));
  EXPECT_EQ(check("SRV*http://localhost"), returns("http://localhost"));

  // Symbol servers and local paths with caches.
  EXPECT_EQ(check("SRV*C:\\symcache*\\\\corp\\symbols"),
            returns("\\\\corp\\symbols"));
  EXPECT_EQ(check("D:\\sym;srv*C:\\symcache*D:\\sym"),
            returns("D:\\sym", "D:\\sym"));
  EXPECT_EQ(check("srv**https://symbols.mozilla.org"),
            returns("https://symbols.mozilla.org"));

  // Symbol server with custom implementation (unsupported).
  EXPECT_EQ(check("symsrv*symsrv.dll*https://symbols.mozilla.org"), returns());
  EXPECT_EQ(check("symsrv*symsrv.dll*C:\\symbols*https://symbols.mozilla.org"),
            returns());
  EXPECT_EQ(check("symsrv*https://symbols.mozilla.org;D:\\sym"),
            returns("D:\\sym"));

  // Partially invalid specs.
  EXPECT_EQ(check("srv*;;D:\\sym;SRV*"), returns("", "D:\\sym", ""));
  EXPECT_EQ(check("srv*D:\\1*D:\\2*D:\\3;D:\\sym"), returns("D:\\sym"));
  EXPECT_EQ(check("symsrv*D:\\1;D:\\sym"), returns("D:\\sym"));
}

TEST(SymStoreTest, ParseEnvSymbolPaths_Cache) {
  auto check = [](const char *str) {
    std::vector<std::string> caches;
    for (LookupEntry entry : SymbolLocatorSymStore::ParseEnvSymbolPaths(str))
      if (entry.cache)
        caches.push_back(std::move(*entry.cache));
    return caches;
  };
  auto returns = [](auto... strs) { return std::vector<std::string>{strs...}; };

  // No caches.
  EXPECT_EQ(check(""), returns());
  EXPECT_EQ(check("C:\\ProgramData\\Symbols"), returns());
  EXPECT_EQ(check("C:\\symbols;\\\\buildserver\\syms;file://D:/pdb"),
            returns());
  EXPECT_EQ(check("SRV*http://localhost"), returns());

  // No cache without a server.
  EXPECT_EQ(check("cache*"), returns());
  EXPECT_EQ(check("cache*C:\\symcache"), returns());
  EXPECT_EQ(check("cache*C:\\symcache;D:\\sym"), returns());

  // Explicit caches for symbol servers.
  EXPECT_EQ(check("SRV*C:\\symcache*\\\\corp\\symbols"),
            returns("C:\\symcache"));
  EXPECT_EQ(check("D:\\sym;srv*C:\\symcache*D:\\sym"), returns("C:\\symcache"));

  // Implicit caches for following symbol servers.
  EXPECT_EQ(check("cache*D:\\s;srv*\\\\corp"), returns("D:\\s"));
  EXPECT_EQ(check("CACHE*D:\\s;srv*\\\\corp;SRV*http://localhost"),
            returns("D:\\s", "D:\\s"));
  EXPECT_EQ(check("Cache*D:\\s;srv*\\\\corp;SRV*C:\\X*http://localhost"),
            returns("D:\\s", "C:\\X"));
  EXPECT_EQ(check("srv*\\\\corp;cache*D:\\s;SRV*C:\\X*http://localhost"),
            returns("C:\\X"));
  EXPECT_EQ(check("srv*\\\\corp;SRV*C:\\X*http://localhost;cache*D:\\s"),
            returns("C:\\X"));

  // Fall back to default cache.
  auto default_cache = SymbolLocatorSymStore::GetSystemDefaultCachePath();
  EXPECT_EQ(check("cache*;srv*\\\\corp"), returns(default_cache));
  EXPECT_EQ(check("srv**https://symbols.mozilla.org"), returns(default_cache));

  // Symbol server with custom implementation (unsupported).
  EXPECT_EQ(check("symsrv*symsrv.dll*https://symbols.mozilla.org"), returns());
  EXPECT_EQ(check("symsrv*symsrv.dll*C:\\symbols*https://symbols.mozilla.org"),
            returns());

  // Partially invalid specs.
  EXPECT_EQ(check("cache*C:\\1;;D:\\sym;SRV*"), returns("C:\\1"));
  EXPECT_EQ(check("cache*C:\\1;srv*D:\\1*D:\\2*D:\\3;srv*D:\\sym"),
            returns("C:\\1"));
  EXPECT_EQ(check("cache*C:\\1;symsrv*D:\\1;srv*D:\\sym"), returns("C:\\1"));
}
