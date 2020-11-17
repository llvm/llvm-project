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
  const char *DefaultValue;
};

static const OptionWithMarshallingInfo MarshallingTable[] = {
#define OPTION_WITH_MARSHALLING(                                               \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, ALWAYS_EMIT, KEYPATH, DEFAULT_VALUE,  \
    TYPE, NORMALIZER, DENORMALIZER, MERGER, EXTRACTOR, TABLE_INDEX)            \
  {NAME, #KEYPATH, #DEFAULT_VALUE},
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
  ASSERT_STREQ(MarshallingTable[0].KeyPath, "MarshalledFlagD");
  ASSERT_STREQ(MarshallingTable[1].KeyPath, "MarshalledFlagC");
  ASSERT_STREQ(MarshallingTable[2].KeyPath, "MarshalledFlagB");
  ASSERT_STREQ(MarshallingTable[3].KeyPath, "MarshalledFlagA");
}

TEST(OptionMarshalling, DefaultAnyOfConstructedDisjunctionOfKeypaths) {
  ASSERT_STREQ(MarshallingTable[0].DefaultValue, "false");
  ASSERT_STREQ(MarshallingTable[1].DefaultValue, "false || MarshalledFlagD");
  ASSERT_STREQ(MarshallingTable[2].DefaultValue, "false || MarshalledFlagD");
  ASSERT_STREQ(MarshallingTable[3].DefaultValue,
            "false || MarshalledFlagC || MarshalledFlagB");
}
