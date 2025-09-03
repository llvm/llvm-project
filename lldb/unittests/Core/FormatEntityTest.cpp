//===-- FormatEntityTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/FormatEntity.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace llvm;

using Definition = FormatEntity::Entry::Definition;
using Entry = FormatEntity::Entry;

static Expected<std::string> Format(StringRef format_str) {
  StreamString stream;
  FormatEntity::Entry format;
  Status status = FormatEntity::Parse(format_str, format);
  if (status.Fail())
    return status.ToError();

  FormatEntity::Format(format, stream, nullptr, nullptr, nullptr, nullptr,
                       false, false);
  return stream.GetString().str();
}

TEST(FormatEntityTest, DefinitionConstructionNameAndType) {
  Definition d("foo", FormatEntity::Entry::Type::Invalid);

  EXPECT_STREQ(d.name, "foo");
  EXPECT_EQ(d.string, nullptr);
  EXPECT_EQ(d.type, FormatEntity::Entry::Type::Invalid);
  EXPECT_EQ(d.data, 0UL);
  EXPECT_EQ(d.num_children, 0UL);
  EXPECT_EQ(d.children, nullptr);
  EXPECT_FALSE(d.keep_separator);
}

TEST(FormatEntityTest, DefinitionConstructionNameAndString) {
  Definition d("foo", "string");

  EXPECT_STREQ(d.name, "foo");
  EXPECT_STREQ(d.string, "string");
  EXPECT_EQ(d.type, FormatEntity::Entry::Type::EscapeCode);
  EXPECT_EQ(d.data, 0UL);
  EXPECT_EQ(d.num_children, 0UL);
  EXPECT_EQ(d.children, nullptr);
  EXPECT_FALSE(d.keep_separator);
}

TEST(FormatEntityTest, DefinitionConstructionNameTypeData) {
  Definition d("foo", FormatEntity::Entry::Type::Invalid, 33);

  EXPECT_STREQ(d.name, "foo");
  EXPECT_EQ(d.string, nullptr);
  EXPECT_EQ(d.type, FormatEntity::Entry::Type::Invalid);
  EXPECT_EQ(d.data, 33UL);
  EXPECT_EQ(d.num_children, 0UL);
  EXPECT_EQ(d.children, nullptr);
  EXPECT_FALSE(d.keep_separator);
}

TEST(FormatEntityTest, DefinitionConstructionNameTypeChildren) {
  Definition d("foo", FormatEntity::Entry::Type::Invalid, 33);
  Definition parent("parent", FormatEntity::Entry::Type::Invalid, 1, &d);
  EXPECT_STREQ(parent.name, "parent");
  EXPECT_STREQ(parent.string, nullptr);
  EXPECT_EQ(parent.type, FormatEntity::Entry::Type::Invalid);
  EXPECT_EQ(parent.num_children, 1UL);
  EXPECT_EQ(parent.children, &d);
  EXPECT_FALSE(parent.keep_separator);

  EXPECT_STREQ(parent.children[0].name, "foo");
  EXPECT_EQ(parent.children[0].string, nullptr);
  EXPECT_EQ(parent.children[0].type, FormatEntity::Entry::Type::Invalid);
  EXPECT_EQ(parent.children[0].data, 33UL);
  EXPECT_EQ(parent.children[0].num_children, 0UL);
  EXPECT_EQ(parent.children[0].children, nullptr);
  EXPECT_FALSE(d.keep_separator);
}

constexpr llvm::StringRef lookupStrings[] = {
    "${addr.load}",
    "${addr.file}",
    "${ansi.fg.black}",
    "${ansi.fg.red}",
    "${ansi.fg.green}",
    "${ansi.fg.yellow}",
    "${ansi.fg.blue}",
    "${ansi.fg.purple}",
    "${ansi.fg.cyan}",
    "${ansi.fg.white}",
    "${ansi.bg.black}",
    "${ansi.bg.red}",
    "${ansi.bg.green}",
    "${ansi.bg.yellow}",
    "${ansi.bg.blue}",
    "${ansi.bg.purple}",
    "${ansi.bg.cyan}",
    "${ansi.bg.white}",
    "${file.basename}",
    "${file.dirname}",
    "${file.fullpath}",
    "${frame.index}",
    "${frame.pc}",
    "${frame.fp}",
    "${frame.sp}",
    "${frame.flags}",
    "${frame.no-debug}",
    "${frame.reg.*}",
    "${frame.is-artificial}",
    "${frame.kind}",
    "${function.id}",
    "${function.name}",
    "${function.name-without-args}",
    "${function.name-with-args}",
    "${function.mangled-name}",
    "${function.addr-offset}",
    "${function.concrete-only-addr-offset-no-padding}",
    "${function.line-offset}",
    "${function.pc-offset}",
    "${function.initial-function}",
    "${function.changed}",
    "${function.is-optimized}",
    "${function.is-inlined}",
    "${line.file.basename}",
    "${line.file.dirname}",
    "${line.file.fullpath}",
    "${line.number}",
    "${line.column}",
    "${line.start-addr}",
    "${line.end-addr}",
    "${module.file.basename}",
    "${module.file.dirname}",
    "${module.file.fullpath}",
    "${process.id}",
    "${process.name}",
    "${process.file.basename}",
    "${process.file.dirname}",
    "${process.file.fullpath}",
    "${script.frame}",
    "${script.process}",
    "${script.target}",
    "${script.thread}",
    "${script.var}",
    "${script.svar}",
    "${script.thread}",
    "${svar.dummy-svar-to-test-wildcard}",
    "${thread.id}",
    "${thread.protocol_id}",
    "${thread.index}",
    "${thread.info.*}",
    "${thread.queue}",
    "${thread.name}",
    "${thread.stop-reason}",
    "${thread.stop-reason-raw}",
    "${thread.return-value}",
    "${thread.completed-expression}",
    "${target.arch}",
    "${target.file.basename}",
    "${target.file.dirname}",
    "${target.file.fullpath}",
    "${var.dummy-var-to-test-wildcard}"};

TEST(FormatEntityTest, LookupAllEntriesInTree) {
  for (const llvm::StringRef testString : lookupStrings) {
    Entry e;
    EXPECT_TRUE(FormatEntity::Parse(testString, e).Success())
        << "Formatting " << testString << " did not succeed";
  }
}

TEST(FormatEntityTest, Scope) {
  // Scope with  one alternative.
  EXPECT_THAT_EXPECTED(Format("{${frame.pc}|foo}"), HasValue("foo"));

  // Scope with multiple alternatives.
  EXPECT_THAT_EXPECTED(Format("{${frame.pc}|${function.name}|foo}"),
                       HasValue("foo"));

  // Escaped pipe inside a scope.
  EXPECT_THAT_EXPECTED(Format("{foo\\|bar}"), HasValue("foo|bar"));

  // Unescaped pipe outside a scope.
  EXPECT_THAT_EXPECTED(Format("foo|bar"), HasValue("foo|bar"));

  // Nested scopes. Note that scopes always resolve.
  EXPECT_THAT_EXPECTED(Format("{{${frame.pc}|foo}|{bar}}"), HasValue("foo"));
  EXPECT_THAT_EXPECTED(Format("{{${frame.pc}}|{bar}}"), HasValue(""));

  // Pipe between scopes.
  EXPECT_THAT_EXPECTED(Format("{foo}|{bar}"), HasValue("foo|bar"));
  EXPECT_THAT_EXPECTED(Format("{foo}||{bar}"), HasValue("foo||bar"));

  // Empty space between pipes.
  EXPECT_THAT_EXPECTED(Format("{{foo}||{bar}}"), HasValue("foo"));
  EXPECT_THAT_EXPECTED(Format("{${frame.pc}||{bar}}"), HasValue(""));
}
