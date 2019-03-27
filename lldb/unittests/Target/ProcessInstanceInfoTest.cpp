//===-- ProcessInstanceInfoTest.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Process.h"
#include "gtest/gtest.h"

using namespace lldb_private;

namespace {
/// A very simple resolver which fails for even ids and returns a simple string
/// for odd ones.
class DummyUserIDResolver : public UserIDResolver {
protected:
  llvm::Optional<std::string> DoGetUserName(id_t uid) {
    if (uid % 2)
      return ("user" + llvm::Twine(uid)).str();
    return llvm::None;
  }

  llvm::Optional<std::string> DoGetGroupName(id_t gid) {
    if (gid % 2)
      return ("group" + llvm::Twine(gid)).str();
    return llvm::None;
  }
};
} // namespace

TEST(ProcessInstanceInfo, Dump) {
  ProcessInstanceInfo info("a.out", ArchSpec("x86_64-pc-linux"), 47);
  info.SetUserID(1);
  info.SetEffectiveUserID(2);
  info.SetGroupID(3);
  info.SetEffectiveGroupID(4);

  DummyUserIDResolver resolver;
  StreamString s;
  info.Dump(s, resolver);
  EXPECT_STREQ(R"(    pid = 47
   name = a.out
   file = a.out
   arch = x86_64-pc-linux
    uid = 1     (user1)
    gid = 3     (group3)
   euid = 2     ()
   egid = 4     ()
)",
               s.GetData());
}

TEST(ProcessInstanceInfo, DumpTable) {
  ProcessInstanceInfo info("a.out", ArchSpec("x86_64-pc-linux"), 47);
  info.SetUserID(1);
  info.SetEffectiveUserID(2);
  info.SetGroupID(3);
  info.SetEffectiveGroupID(4);

  DummyUserIDResolver resolver;
  StreamString s;

  const bool show_args = false;
  const bool verbose = true;
  ProcessInstanceInfo::DumpTableHeader(s, show_args, verbose);
  info.DumpAsTableRow(s, resolver, show_args, verbose);
  EXPECT_STREQ(
      R"(PID    PARENT USER       GROUP      EFF USER   EFF GROUP  TRIPLE                   ARGUMENTS
====== ====== ========== ========== ========== ========== ======================== ============================
47     0      user1      group3     2          4          x86_64-pc-linux          
)",
      s.GetData());
}
