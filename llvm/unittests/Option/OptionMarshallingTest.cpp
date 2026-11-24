//===- OptionMarshallingTest.cpp - OptionParserEmitter tests -================//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Option/OptTable.h"
#include "gtest/gtest.h"

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Opts.inc"
  LastOption
#undef OPTION
};

struct OptionWithMarshallingInfo {
  ID Opt;
  const char *KeyPath;
  const char *ImpliedCheck;
  const char *ImpliedValue;
};

static const OptionWithMarshallingInfo MarshallingTable[] = {
#define OPTION_WITH_MARSHALLING(                                               \
    PREFIX_TYPE, PREFIXED_NAME_OFFSET, ID, KIND, GROUP, ALIAS, ALIASARGS,      \
    FLAGS, VISIBILITY, PARAM, HELPTEXT, HELPTEXTSFORVARIANTS, METAVAR, VALUES, \
    SUBCOMMANDIDS_OFFSET, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH, DEFAULT_VALUE,   \
    IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER, TABLE_INDEX)       \
  {OPT_##ID, #KEYPATH, #IMPLIED_CHECK, #IMPLIED_VALUE},
#include "Opts.inc"
#undef OPTION_WITH_MARSHALLING
};

TEST(OptionMarshalling, EmittedOrderSameAsDefinitionOrder) {
  ASSERT_EQ(MarshallingTable[0].Opt, OPT_marshalled_flag_d);
  ASSERT_EQ(MarshallingTable[1].Opt, OPT_marshalled_flag_c);
  ASSERT_EQ(MarshallingTable[2].Opt, OPT_marshalled_flag_b);
  ASSERT_EQ(MarshallingTable[3].Opt, OPT_marshalled_flag_a);
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
