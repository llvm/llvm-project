//===- unittest/Support/OptionMarshallingTest.cpp - OptParserEmitter tests ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

struct OptionWithMarshallingInfo {
  const char *Name;
  const char *KeyPath;
  const char *ImpliedCheck;
  const char *ImpliedValue;
};

static const OptionWithMarshallingInfo MarshallingTable[] = {
#define OPTION_WITH_MARSHALLING(                                               \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  {NAME, #KEYPATH, #IMPLIED_CHECK, #IMPLIED_VALUE},
#include "Opts.inc"
#undef OPTION_WITH_MARSHALLING
};

TEST(OptionMarshalling, EmittedOrderSameAsDefinitionOrder) {
  ASSERT_STREQ(MarshallingTable[0].Name, "marshalled-flag-d");
  ASSERT_STREQ(MarshallingTable[1].Name, "marshalled-flag-c");
  ASSERT_STREQ(MarshallingTable[2].Name, "marshalled-flag-b");
  ASSERT_STREQ(MarshallingTable[3].Name, "marshalled-flag-a");
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
