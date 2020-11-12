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
#define OPTION_WITH_MARSHALLING_FLAG(PREFIX_TYPE, NAME, ID, KIND, GROUP,       \
                                     ALIAS, ALIASARGS, FLAGS, PARAM, HELPTEXT, \
                                     METAVAR, VALUES, SPELLING, ALWAYS_EMIT,   \
                                     KEYPATH, DEFAULT_VALUE, IS_POSITIVE)      \
  { NAME, #KEYPATH, #DEFAULT_VALUE },
#include "Opts.inc"
#undef OPTION_WITH_MARSHALLING_FLAG
};

TEST(OptionMarshalling, EmittedOrderSameAsDefinitionOrder) {
  ASSERT_STREQ(MarshallingTable[0].Name, "marshalled-flag-0");
  ASSERT_STREQ(MarshallingTable[1].Name, "marshalled-flag-1");
  ASSERT_STREQ(MarshallingTable[2].Name, "marshalled-flag-2");
  ASSERT_STREQ(MarshallingTable[3].Name, "marshalled-flag-3");
}

TEST(OptionMarshalling, EmittedSpecifiedKeyPath) {
  ASSERT_STREQ(MarshallingTable[0].KeyPath, "MarshalledFlag0");
  ASSERT_STREQ(MarshallingTable[1].KeyPath, "MarshalledFlag1");
  ASSERT_STREQ(MarshallingTable[2].KeyPath, "MarshalledFlag2");
  ASSERT_STREQ(MarshallingTable[3].KeyPath, "MarshalledFlag3");
}

TEST(OptionMarshalling, DefaultAnyOfConstructedDisjunctionOfKeypaths) {
  ASSERT_STREQ(MarshallingTable[0].DefaultValue, "false");
  ASSERT_STREQ(MarshallingTable[1].DefaultValue, "false || MarshalledFlag0");
  ASSERT_STREQ(MarshallingTable[2].DefaultValue, "false || MarshalledFlag0");
  ASSERT_STREQ(MarshallingTable[3].DefaultValue,
            "false || MarshalledFlag1 || MarshalledFlag2");
}
