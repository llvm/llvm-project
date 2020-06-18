//===-- ProcessInstanceInfoTest.cpp ---------------------------------------===//
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
  llvm::Optional<std::string> DoGetUserName(id_t uid) override {
    if (uid % 2)
      return ("user" + llvm::Twine(uid)).str();
    return llvm::None;
  }

  llvm::Optional<std::string> DoGetGroupName(id_t gid) override {
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
      R"(PID    PARENT USER       GROUP      EFF USER   EFF GROUP  TRIPLE                         ARGUMENTS
====== ====== ========== ========== ========== ========== ============================== ============================
47     0      user1      group3     2          4          x86_64-pc-linux                
)",
      s.GetData());
}

TEST(ProcessInstanceInfo, DumpTable_invalidUID) {
  ProcessInstanceInfo info("a.out", ArchSpec("aarch64-unknown-linux-android"), 47);

  DummyUserIDResolver resolver;
  StreamString s;

  const bool show_args = false;
  const bool verbose = false;
  ProcessInstanceInfo::DumpTableHeader(s, show_args, verbose);
  info.DumpAsTableRow(s, resolver, show_args, verbose);
  EXPECT_STREQ(
      R"(PID    PARENT USER       TRIPLE                         NAME
====== ====== ========== ============================== ============================
47     0                 aarch64-unknown-linux-android  a.out
)",
      s.GetData());
}

TEST(ProcessInstanceInfoMatch, Name) {
  ProcessInstanceInfo info_bar, info_empty;
  info_bar.GetExecutableFile().SetFile("/foo/bar", FileSpec::Style::posix);

  ProcessInstanceInfoMatch match;
  match.SetNameMatchType(NameMatch::Equals);
  match.GetProcessInfo().GetExecutableFile().SetFile("bar",
                                                     FileSpec::Style::posix);

  EXPECT_TRUE(match.Matches(info_bar));
  EXPECT_FALSE(match.Matches(info_empty));

  match.GetProcessInfo().GetExecutableFile() = FileSpec();
  EXPECT_TRUE(match.Matches(info_bar));
  EXPECT_TRUE(match.Matches(info_empty));
}

TEST(ProcessInstanceInfo, Yaml) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);

  // Serialize.
  ProcessInstanceInfo info("a.out", ArchSpec("x86_64-pc-linux"), 47);
  info.SetUserID(1);
  info.SetEffectiveUserID(2);
  info.SetGroupID(3);
  info.SetEffectiveGroupID(4);
  llvm::yaml::Output yout(os);
  yout << info;
  os.flush();

  // Deserialize.
  ProcessInstanceInfo deserialized;
  llvm::yaml::Input yin(buffer);
  yin >> deserialized;

  EXPECT_EQ(deserialized.GetNameAsStringRef(), info.GetNameAsStringRef());
  EXPECT_EQ(deserialized.GetArchitecture(), info.GetArchitecture());
  EXPECT_EQ(deserialized.GetUserID(), info.GetUserID());
  EXPECT_EQ(deserialized.GetGroupID(), info.GetGroupID());
  EXPECT_EQ(deserialized.GetEffectiveUserID(), info.GetEffectiveUserID());
  EXPECT_EQ(deserialized.GetEffectiveGroupID(), info.GetEffectiveGroupID());
}

TEST(ProcessInstanceInfoList, Yaml) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);

  // Serialize.
  ProcessInstanceInfo info("a.out", ArchSpec("x86_64-pc-linux"), 47);
  info.SetUserID(1);
  info.SetEffectiveUserID(2);
  info.SetGroupID(3);
  info.SetEffectiveGroupID(4);
  ProcessInstanceInfoList list;
  list.push_back(info);
  llvm::yaml::Output yout(os);
  yout << list;
  os.flush();

  // Deserialize.
  ProcessInstanceInfoList deserialized;
  llvm::yaml::Input yin(buffer);
  yin >> deserialized;

  ASSERT_EQ(deserialized.size(), static_cast<size_t>(1));
  EXPECT_EQ(deserialized[0].GetNameAsStringRef(), info.GetNameAsStringRef());
  EXPECT_EQ(deserialized[0].GetArchitecture(), info.GetArchitecture());
  EXPECT_EQ(deserialized[0].GetUserID(), info.GetUserID());
  EXPECT_EQ(deserialized[0].GetGroupID(), info.GetGroupID());
  EXPECT_EQ(deserialized[0].GetEffectiveUserID(), info.GetEffectiveUserID());
  EXPECT_EQ(deserialized[0].GetEffectiveGroupID(), info.GetEffectiveGroupID());
}
