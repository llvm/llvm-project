//===- DlfcnHostTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's bedrock/host/posix/dlfcn-host.h APIs.
//
// The APIs forward to the host's dlfcn.h APIs, whose behavior we take as given.
// Each test checks only that a forwarder reaches the corresponding host API:
// that it gives the same result as the native call, or acts on the same host
// state.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-c/bedrock/host/posix/dlfcn-host.h"
#include "gtest/gtest.h"

#ifndef ORC_RT_TEST_DYLIB_PATH
#error "ORC_RT_TEST_DYLIB_PATH must be defined to the path of the test dylib"
#endif

static constexpr int TestDylibMode = RTLD_NOW | RTLD_LOCAL;

TEST(DlfcnHostTest, Dlopen) {
  // Opening an already-open library returns its existing handle, so a handle
  // from the forwarder should match one from the native call.
  void *Native = dlopen(ORC_RT_TEST_DYLIB_PATH, TestDylibMode);
  ASSERT_NE(Native, nullptr) << dlerror();

  void *Forwarded = orc_rt_dlopen_host(ORC_RT_TEST_DYLIB_PATH, TestDylibMode);
  EXPECT_EQ(Forwarded, Native);

  if (Forwarded)
    dlclose(Forwarded);
  dlclose(Native);
}

TEST(DlfcnHostTest, Dlclose) {
  void *H = dlopen(ORC_RT_TEST_DYLIB_PATH, TestDylibMode);
  ASSERT_NE(H, nullptr) << dlerror();
  EXPECT_EQ(orc_rt_dlclose_host(H), 0) << dlerror();
}

TEST(DlfcnHostTest, Dlsym) {
  void *H = dlopen(ORC_RT_TEST_DYLIB_PATH, TestDylibMode);
  ASSERT_NE(H, nullptr) << dlerror();

  void *Native = dlsym(H, "orc_rt_test_dylib_return42");
  ASSERT_NE(Native, nullptr) << dlerror();
  EXPECT_EQ(orc_rt_dlsym_host(H, "orc_rt_test_dylib_return42"), Native);

  dlclose(H);
}

TEST(DlfcnHostTest, Dladdr) {
  void *H = dlopen(ORC_RT_TEST_DYLIB_PATH, TestDylibMode);
  ASSERT_NE(H, nullptr) << dlerror();
  void *Addr = dlsym(H, "orc_rt_test_dylib_return42");
  ASSERT_NE(Addr, nullptr) << dlerror();

  Dl_info Native = {};
  Dl_info Forwarded = {};
  ASSERT_NE(dladdr(Addr, &Native), 0);
  ASSERT_NE(orc_rt_dladdr_host(Addr, &Forwarded), 0);
  EXPECT_EQ(Forwarded.dli_fname, Native.dli_fname);
  EXPECT_EQ(Forwarded.dli_fbase, Native.dli_fbase);
  EXPECT_EQ(Forwarded.dli_sname, Native.dli_sname);
  EXPECT_EQ(Forwarded.dli_saddr, Native.dli_saddr);

  dlclose(H);
}

TEST(DlfcnHostTest, Dlerror) {
  // Leave an error in the host's dlerror state with a native call. The
  // forwarder should report it, and in doing so consume it from that state.
  dlerror();
  ASSERT_EQ(dlopen("/nonexistent/liborc-rt-dlfcn-host-test.so", RTLD_LAZY),
            nullptr);
  EXPECT_NE(orc_rt_dlerror_host(), nullptr);
  EXPECT_EQ(dlerror(), nullptr);
}
