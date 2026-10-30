//===-- LanguageOptsTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/TypeSystem/Clike/LanguageOpts.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Triple.h"

#include "gtest/gtest.h"

using namespace lldb_private::clike;

static LanguageOpts OptsFor(const char *triple) {
  return llvm::cantFail(LanguageOpts::Create(llvm::Triple(triple)));
}

TEST(LanguageOptsTest, X86_64Sizes) {
  LanguageOpts opts = OptsFor("x86_64-pc-linux-gnu");
  const LanguageOpts::BuiltinSizes &sizes = opts.GetBuiltinSizes();
  EXPECT_EQ(sizes.bool_size, 1u);
  EXPECT_EQ(sizes.short_size, 2u);
  EXPECT_EQ(sizes.int_size, 4u);
  EXPECT_EQ(sizes.long_size, 8u);
  EXPECT_EQ(sizes.long_long_size, 8u);
  EXPECT_EQ(sizes.float_size, 4u);
  EXPECT_EQ(sizes.double_size, 8u);
  EXPECT_EQ(sizes.pointer_size, 8u);
}

TEST(LanguageOptsTest, I386Sizes) {
  LanguageOpts opts = OptsFor("i386-pc-linux-gnu");
  const LanguageOpts::BuiltinSizes &sizes = opts.GetBuiltinSizes();
  EXPECT_EQ(sizes.long_size, 4u);
  EXPECT_EQ(sizes.pointer_size, 4u);
}

TEST(LanguageOptsTest, CreateFailsForUnknownTriple) {
  llvm::Expected<LanguageOpts> opts =
      LanguageOpts::Create(llvm::Triple("totally-bogus-triple-value"));
  EXPECT_FALSE(static_cast<bool>(opts));
  llvm::consumeError(opts.takeError());
}

TEST(LanguageOptsTest, FloatTypeSemanticsBySize) {
  LanguageOpts opts = OptsFor("x86_64-pc-linux-gnu");
  EXPECT_EQ(&opts.GetFloatTypeSemantics(4, lldb::eFormatFloat),
            &llvm::APFloat::IEEEsingle());
  EXPECT_EQ(&opts.GetFloatTypeSemantics(8, lldb::eFormatFloat),
            &llvm::APFloat::IEEEdouble());
}

TEST(LanguageOptsTest, FloatTypeSemanticsUnknownSizeIsBogus) {
  LanguageOpts opts = OptsFor("x86_64-pc-linux-gnu");
  EXPECT_EQ(&opts.GetFloatTypeSemantics(3, lldb::eFormatFloat),
            &llvm::APFloat::Bogus());
}

TEST(LanguageOptsTest, BitIntByteSizeRoundsUpToAlignment) {
  LanguageOpts opts = OptsFor("x86_64-pc-linux-gnu");
  std::optional<uint64_t> size = opts.GetBitIntByteSize(1);
  ASSERT_TRUE(size.has_value());
  // A 1-bit _BitInt still occupies at least one byte.
  EXPECT_GE(*size, 1u);

  std::optional<uint64_t> size9 = opts.GetBitIntByteSize(9);
  ASSERT_TRUE(size9.has_value());
  // 9 bits need more than 1 byte of storage.
  EXPECT_GT(*size9, 1u);
}

TEST(LanguageOptsTest, BitIntByteSizeZeroIsInvalid) {
  LanguageOpts opts = OptsFor("x86_64-pc-linux-gnu");
  EXPECT_FALSE(opts.GetBitIntByteSize(0).has_value());
}
