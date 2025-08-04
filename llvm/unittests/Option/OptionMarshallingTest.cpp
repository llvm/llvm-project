//===- OptionMarshallingTest.cpp - OptionParserEmitter tests -================//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringTable.h"
#include "gtest/gtest.h"

#define OPTTABLE_STR_TABLE_CODE
#include "Opts.inc"
#undef OPTTABLE_STR_TABLE_CODE

struct OptionWithMarshallingInfo {
  int PrefixedNameOffset;
  const char *KeyPath;
  const char *ImpliedCheck;
  const char *ImpliedValue;

  llvm::StringRef getPrefixedName() const {
    return OptionStrTable[PrefixedNameOffset];
  }
};

static const OptionWithMarshallingInfo MarshallingTable[] = {
#define OPTION_WITH_MARSHALLING(                                               \
    PREFIX_TYPE, PREFIXED_NAME_OFFSET, ID, KIND, GROUP, ALIAS, ALIASARGS,      \
    FLAGS, VISIBILITY, PARAM, HELPTEXT, HELPTEXTSFORVARIANTS, METAVAR, VALUES, \
    SHOULD_PARSE, ALWAYS_EMIT, KEYPATH, DEFAULT_VALUE, IMPLIED_CHECK,          \
    IMPLIED_VALUE, NORMALIZER, DENORMALIZER, MERGER, EXTRACTOR, TABLE_INDEX)   \
  {PREFIXED_NAME_OFFSET, #KEYPATH, #IMPLIED_CHECK, #IMPLIED_VALUE},
#include "Opts.inc"
#undef OPTION_WITH_MARSHALLING
};

TEST(OptionMarshalling, EmittedOrderSameAsDefinitionOrder) {
  ASSERT_EQ(MarshallingTable[0].getPrefixedName(), "-marshalled-flag-d");
  ASSERT_EQ(MarshallingTable[1].getPrefixedName(), "-marshalled-flag-c");
  ASSERT_EQ(MarshallingTable[2].getPrefixedName(), "-marshalled-flag-b");
  ASSERT_EQ(MarshallingTable[3].getPrefixedName(), "-marshalled-flag-a");
}

TEST(OptionMarshalling, EmittedSpecifiedKeyPath) {
  ASSERT_STREQ(MarshallingTable[0].KeyPath, "X->MarshalledFlagD");
  ASSERT_STREQ(MarshallingTable[1].KeyPath, "X->MarshalledFlagC");
  ASSERT_STREQ(MarshallingTable[2].KeyPath, "X->MarshalledFlagB");
  ASSERT_STREQ(MarshallingTable[3].KeyPath, "X->MarshalledFlagA");
}

TEST(OptionMarshalling, ImpliedCheckContainsDisjunctionOfKeypaths) {
  ASSERT_STREQ(MarshallingTable[0].ImpliedCheck, "false");

  ASSERT_STREQ(MarshallingTable[1].ImpliedCheck, "false || X->MarshalledFlagD");
  ASSERT_STREQ(MarshallingTable[1].ImpliedValue, "true");

  ASSERT_STREQ(MarshallingTable[2].ImpliedCheck, "false || X->MarshalledFlagD");
  ASSERT_STREQ(MarshallingTable[2].ImpliedValue, "true");

  ASSERT_STREQ(MarshallingTable[3].ImpliedCheck,
               "false || X->MarshalledFlagC || X->MarshalledFlagB");
  ASSERT_STREQ(MarshallingTable[3].ImpliedValue, "true");
}
