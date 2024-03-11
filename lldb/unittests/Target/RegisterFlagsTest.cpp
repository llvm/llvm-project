//===-- RegisterFlagsTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/RegisterFlags.h"
#include "lldb/Utility/StreamString.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;

TEST(RegisterFlagsTest, Field) {
  // We assume that start <= end is always true, so that is not tested here.

  RegisterFlags::Field f1("abc", 0);
  ASSERT_EQ(f1.GetName(), "abc");
  // start == end means a 1 bit field.
  ASSERT_EQ(f1.GetSizeInBits(), (unsigned)1);
  ASSERT_EQ(f1.GetMask(), (uint64_t)1);
  ASSERT_EQ(f1.GetValue(0), (uint64_t)0);
  ASSERT_EQ(f1.GetValue(3), (uint64_t)1);

  // End is inclusive meaning that start 0 to end 1 includes bit 1
  // to make a 2 bit field.
  RegisterFlags::Field f2("", 0, 1);
  ASSERT_EQ(f2.GetSizeInBits(), (unsigned)2);
  ASSERT_EQ(f2.GetMask(), (uint64_t)3);
  ASSERT_EQ(f2.GetValue(UINT64_MAX), (uint64_t)3);
  ASSERT_EQ(f2.GetValue(UINT64_MAX & ~(uint64_t)3), (uint64_t)0);

  // If the field doesn't start at 0 we need to shift up/down
  // to account for it.
  RegisterFlags::Field f3("", 2, 5);
  ASSERT_EQ(f3.GetSizeInBits(), (unsigned)4);
  ASSERT_EQ(f3.GetMask(), (uint64_t)0x3c);
  ASSERT_EQ(f3.GetValue(UINT64_MAX), (uint64_t)0xf);
  ASSERT_EQ(f3.GetValue(UINT64_MAX & ~(uint64_t)0x3c), (uint64_t)0);

  // Fields are sorted lowest starting bit first.
  ASSERT_TRUE(f2 < f3);
  ASSERT_FALSE(f3 < f1);
  ASSERT_FALSE(f1 < f2);
  ASSERT_FALSE(f1 < f1);
}

static RegisterFlags::Field make_field(unsigned start, unsigned end) {
  return RegisterFlags::Field("", start, end);
}

static RegisterFlags::Field make_field(unsigned bit) {
  return RegisterFlags::Field("", bit);
}

TEST(RegisterFlagsTest, FieldOverlaps) {
  // Single bit fields
  ASSERT_FALSE(make_field(0, 0).Overlaps(make_field(1)));
  ASSERT_TRUE(make_field(1, 1).Overlaps(make_field(1)));
  ASSERT_FALSE(make_field(1, 1).Overlaps(make_field(3)));

  ASSERT_TRUE(make_field(0, 1).Overlaps(make_field(1, 2)));
  ASSERT_TRUE(make_field(1, 2).Overlaps(make_field(0, 1)));
  ASSERT_FALSE(make_field(0, 1).Overlaps(make_field(2, 3)));
  ASSERT_FALSE(make_field(2, 3).Overlaps(make_field(0, 1)));

  ASSERT_FALSE(make_field(1, 5).Overlaps(make_field(10, 20)));
  ASSERT_FALSE(make_field(15, 30).Overlaps(make_field(7, 12)));
}

TEST(RegisterFlagsTest, PaddingDistance) {
  // We assume that this method is always called with a more significant
  // (start bit is higher) field first and that they do not overlap.

  // [field 1][field 2]
  ASSERT_EQ(make_field(1, 1).PaddingDistance(make_field(0)), 0ULL);
  // [field 1][..][field 2]
  ASSERT_EQ(make_field(2, 2).PaddingDistance(make_field(0)), 1ULL);
  // [field 1][field 1][field 2]
  ASSERT_EQ(make_field(1, 2).PaddingDistance(make_field(0)), 0ULL);
  // [field 1][30 bits free][field 2]
  ASSERT_EQ(make_field(31, 31).PaddingDistance(make_field(0)), 30ULL);
}

static void test_padding(const std::vector<RegisterFlags::Field> &fields,
                         const std::vector<RegisterFlags::Field> &expected) {
  RegisterFlags rf("", 4, fields);
  EXPECT_THAT(expected, ::testing::ContainerEq(rf.GetFields()));
}

TEST(RegisterFlagsTest, RegisterFlagsPadding) {
  // When creating a set of flags we assume that:
  // * There are >= 1 fields.
  // * They are sorted in descending order.
  // * There may be gaps between each field.

  // Needs no padding
  auto fields =
      std::vector<RegisterFlags::Field>{make_field(16, 31), make_field(0, 15)};
  test_padding(fields, fields);

  // Needs padding in between the fields, single bit.
  test_padding({make_field(17, 31), make_field(0, 15)},
               {make_field(17, 31), make_field(16), make_field(0, 15)});
  // Multiple bits of padding.
  test_padding({make_field(17, 31), make_field(0, 14)},
               {make_field(17, 31), make_field(15, 16), make_field(0, 14)});

  // Padding before first field, single bit.
  test_padding({make_field(0, 30)}, {make_field(31), make_field(0, 30)});
  // Multiple bits.
  test_padding({make_field(0, 15)}, {make_field(16, 31), make_field(0, 15)});

  // Padding after last field, single bit.
  test_padding({make_field(1, 31)}, {make_field(1, 31), make_field(0)});
  // Multiple bits.
  test_padding({make_field(2, 31)}, {make_field(2, 31), make_field(0, 1)});

  // Fields need padding before, in between and after.
  // [31-28][field 27-24][23-22][field 21-20][19-12][field 11-8][7-0]
  test_padding({make_field(24, 27), make_field(20, 21), make_field(8, 11)},
               {make_field(28, 31), make_field(24, 27), make_field(22, 23),
                make_field(20, 21), make_field(12, 19), make_field(8, 11),
                make_field(0, 7)});
}

TEST(RegisterFieldsTest, ReverseFieldOrder) {
  // Unchanged
  RegisterFlags rf("", 4, {make_field(0, 31)});
  ASSERT_EQ(0x12345678ULL, (unsigned long long)rf.ReverseFieldOrder(0x12345678));

  // Swap the two halves around.
  RegisterFlags rf2("", 4, {make_field(16, 31), make_field(0, 15)});
  ASSERT_EQ(0x56781234ULL, (unsigned long long)rf2.ReverseFieldOrder(0x12345678));

  // Many small fields.
  RegisterFlags rf3(
      "", 4, {make_field(31), make_field(30), make_field(29), make_field(28)});
  ASSERT_EQ(0x00000005ULL, rf3.ReverseFieldOrder(0xA0000000));
}

TEST(RegisterFlagsTest, AsTable) {
  // Anonymous fields are shown with an empty name cell,
  // whether they are known up front or added during construction.
  RegisterFlags anon_field("", 4, {make_field(0, 31)});
  ASSERT_EQ("| 31-0 |\n"
            "|------|\n"
            "|      |",
            anon_field.AsTable(100));

  RegisterFlags anon_with_pad("", 4, {make_field(16, 31)});
  ASSERT_EQ("| 31-16 | 15-0 |\n"
            "|-------|------|\n"
            "|       |      |",
            anon_with_pad.AsTable(100));

  // Use the wider of position and name to set the column width.
  RegisterFlags name_wider("", 4, {RegisterFlags::Field("aardvark", 0, 31)});
  ASSERT_EQ("|   31-0   |\n"
            "|----------|\n"
            "| aardvark |",
            name_wider.AsTable(100));
  // When the padding is an odd number, put the remaining 1 on the right.
  RegisterFlags pos_wider("", 4, {RegisterFlags::Field("?", 0, 31)});
  ASSERT_EQ("| 31-0 |\n"
            "|------|\n"
            "|  ?   |",
            pos_wider.AsTable(100));

  // Single bit fields don't need to show start and end, just one of them.
  RegisterFlags single_bit("", 4, {make_field(31)});
  ASSERT_EQ("| 31 | 30-0 |\n"
            "|----|------|\n"
            "|    |      |",
            single_bit.AsTable(100));

  // Columns are printed horizontally if max width allows.
  RegisterFlags many_fields("", 4,
                            {RegisterFlags::Field("cat", 28, 31),
                             RegisterFlags::Field("pigeon", 20, 23),
                             RegisterFlags::Field("wolf", 12),
                             RegisterFlags::Field("x", 0, 4)});
  ASSERT_EQ("| 31-28 | 27-24 | 23-20  | 19-13 |  12  | 11-5 | 4-0 |\n"
            "|-------|-------|--------|-------|------|------|-----|\n"
            "|  cat  |       | pigeon |       | wolf |      |  x  |",
            many_fields.AsTable(100));

  // max_width tells us when we need to split into further tables.
  // Here no split is needed.
  RegisterFlags exact_max_single_col("", 4, {RegisterFlags::Field("?", 0, 31)});
  ASSERT_EQ("| 31-0 |\n"
            "|------|\n"
            "|  ?   |",
            exact_max_single_col.AsTable(9));
  RegisterFlags exact_max_two_col(
      "", 4,
      {RegisterFlags::Field("?", 16, 31), RegisterFlags::Field("#", 0, 15)});
  ASSERT_EQ("| 31-16 | 15-0 |\n"
            "|-------|------|\n"
            "|   ?   |  #   |",
            exact_max_two_col.AsTable(16));

  // If max is less than a single column, just print the single column. The user
  // will have to put up with some wrapping in this niche case.
  RegisterFlags zero_max_single_col("", 4, {RegisterFlags::Field("?", 0, 31)});
  ASSERT_EQ("| 31-0 |\n"
            "|------|\n"
            "|  ?   |",
            zero_max_single_col.AsTable(0));
  // Same logic for any following columns. Effectively making a "vertical"
  // table, just with more grid lines.
  RegisterFlags zero_max_two_col(
      "", 4,
      {RegisterFlags::Field("?", 16, 31), RegisterFlags::Field("#", 0, 15)});
  ASSERT_EQ("| 31-16 |\n"
            "|-------|\n"
            "|   ?   |\n"
            "\n"
            "| 15-0 |\n"
            "|------|\n"
            "|  #   |",
            zero_max_two_col.AsTable(0));

  RegisterFlags max_less_than_single_col("", 4,
                                         {RegisterFlags::Field("?", 0, 31)});
  ASSERT_EQ("| 31-0 |\n"
            "|------|\n"
            "|  ?   |",
            max_less_than_single_col.AsTable(3));
  RegisterFlags max_less_than_two_col(
      "", 4,
      {RegisterFlags::Field("?", 16, 31), RegisterFlags::Field("#", 0, 15)});
  ASSERT_EQ("| 31-16 |\n"
            "|-------|\n"
            "|   ?   |\n"
            "\n"
            "| 15-0 |\n"
            "|------|\n"
            "|  #   |",
            max_less_than_two_col.AsTable(9));
  RegisterFlags max_many_columns(
      "", 4,
      {RegisterFlags::Field("A", 24, 31), RegisterFlags::Field("B", 16, 23),
       RegisterFlags::Field("C", 8, 15),
       RegisterFlags::Field("really long name", 0, 7)});
  ASSERT_EQ("| 31-24 | 23-16 |\n"
            "|-------|-------|\n"
            "|   A   |   B   |\n"
            "\n"
            "| 15-8 |\n"
            "|------|\n"
            "|  C   |\n"
            "\n"
            "|       7-0        |\n"
            "|------------------|\n"
            "| really long name |",
            max_many_columns.AsTable(23));
}

TEST(RegisterFlagsTest, DumpEnums) {
  ASSERT_EQ(RegisterFlags("", 8, {RegisterFlags::Field{"A", 0}}).DumpEnums(80),
            "");

  FieldEnum basic_enum("test", {{0, "an_enumerator"}});
  ASSERT_EQ(RegisterFlags("", 8, {RegisterFlags::Field{"A", 0, 0, &basic_enum}})
                .DumpEnums(80),
            "A: 0 = an_enumerator");

  // If width is smaller than the enumerator name, print it anyway.
  ASSERT_EQ(RegisterFlags("", 8, {RegisterFlags::Field{"A", 0, 0, &basic_enum}})
                .DumpEnums(5),
            "A: 0 = an_enumerator");

  // Mutliple values can go on the same line, up to the width.
  FieldEnum more_enum("long_enum",
                      {{0, "an_enumerator"},
                       {1, "another_enumerator"},
                       {2, "a_very_very_long_enumerator_has_its_own_line"},
                       {3, "small"},
                       {4, "small2"}});
  ASSERT_EQ(RegisterFlags("", 8, {RegisterFlags::Field{"A", 0, 2, &more_enum}})
                // Width is chosen to be exactly enough to allow 0 and 1
                // enumerators on the first line.
                .DumpEnums(45),
            "A: 0 = an_enumerator, 1 = another_enumerator,\n"
            "   2 = a_very_very_long_enumerator_has_its_own_line,\n"
            "   3 = small, 4 = small2");

  // If they all exceed width, one per line.
  FieldEnum another_enum("another_enum", {{0, "an_enumerator"},
                                          {1, "another_enumerator"},
                                          {2, "a_longer_enumerator"}});
  ASSERT_EQ(
      RegisterFlags("", 8, {RegisterFlags::Field{"A", 0, 1, &another_enum}})
          .DumpEnums(5),
      "A: 0 = an_enumerator,\n"
      "   1 = another_enumerator,\n"
      "   2 = a_longer_enumerator");

  // If the name is already > the width, put one value per line.
  FieldEnum short_enum("short_enum", {{0, "a"}, {1, "b"}, {2, "c"}});
  ASSERT_EQ(RegisterFlags("", 8,
                          {RegisterFlags::Field{"AReallyLongFieldName", 0, 1,
                                                &short_enum}})
                .DumpEnums(10),
            "AReallyLongFieldName: 0 = a,\n"
            "                      1 = b,\n"
            "                      2 = c");

  // Fields are separated by a blank line. Indentation of lines split by width
  // is set by the size of the fields name (as opposed to some max of all field
  // names).
  FieldEnum enum_1("enum_1", {{0, "an_enumerator"}, {1, "another_enumerator"}});
  FieldEnum enum_2("enum_2",
                   {{0, "Cdef_enumerator_1"}, {1, "Cdef_enumerator_2"}});
  ASSERT_EQ(RegisterFlags("", 8,
                          {RegisterFlags::Field{"Ab", 1, 1, &enum_1},
                           RegisterFlags::Field{"Cdef", 0, 0, &enum_2}})
                .DumpEnums(10),
            "Ab: 0 = an_enumerator,\n"
            "    1 = another_enumerator\n"
            "\n"
            "Cdef: 0 = Cdef_enumerator_1,\n"
            "      1 = Cdef_enumerator_2");

  // Having fields without enumerators shouldn't produce any extra newlines.
  ASSERT_EQ(RegisterFlags("", 8,
                          {
                              RegisterFlags::Field{"A", 4, 4},
                              RegisterFlags::Field{"B", 3, 3, &enum_1},
                              RegisterFlags::Field{"C", 2, 2},
                              RegisterFlags::Field{"D", 1, 1, &enum_1},
                              RegisterFlags::Field{"E", 0, 0},
                          })
                .DumpEnums(80),
            "B: 0 = an_enumerator, 1 = another_enumerator\n"
            "\n"
            "D: 0 = an_enumerator, 1 = another_enumerator");
}

TEST(RegisterFieldsTest, FlagsToXML) {
  StreamString strm;

  // RegisterFlags requires that some fields be given, so no testing of empty
  // input.

  // Unnamed fields are padding that are ignored. This applies to fields passed
  // in, and those generated to fill the other bits (31-1 here).
  RegisterFlags("Foo", 4, {RegisterFlags::Field("", 0, 0)}).ToXML(strm);
  ASSERT_EQ(strm.GetString(), "<flags id=\"Foo\" size=\"4\">\n"
                              "</flags>\n");

  strm.Clear();
  RegisterFlags("Foo", 4, {RegisterFlags::Field("abc", 0, 0)}).ToXML(strm);
  ASSERT_EQ(strm.GetString(), "<flags id=\"Foo\" size=\"4\">\n"
                              "  <field name=\"abc\" start=\"0\" end=\"0\"/>\n"
                              "</flags>\n");

  strm.Clear();
  // Should use the current indentation level as a starting point.
  strm.IndentMore();
  RegisterFlags(
      "Bar", 5,
      {RegisterFlags::Field("f1", 25, 32), RegisterFlags::Field("f2", 10, 24)})
      .ToXML(strm);
  ASSERT_EQ(strm.GetString(),
            "  <flags id=\"Bar\" size=\"5\">\n"
            "    <field name=\"f1\" start=\"25\" end=\"32\"/>\n"
            "    <field name=\"f2\" start=\"10\" end=\"24\"/>\n"
            "  </flags>\n");

  strm.Clear();
  strm.IndentLess();
  // Should replace any XML unsafe characters in field names.
  RegisterFlags("Safe", 8,
                {RegisterFlags::Field("A<", 4), RegisterFlags::Field("B>", 3),
                 RegisterFlags::Field("C'", 2), RegisterFlags::Field("D\"", 1),
                 RegisterFlags::Field("E&", 0)})
      .ToXML(strm);
  ASSERT_EQ(strm.GetString(),
            "<flags id=\"Safe\" size=\"8\">\n"
            "  <field name=\"A&lt;\" start=\"4\" end=\"4\"/>\n"
            "  <field name=\"B&gt;\" start=\"3\" end=\"3\"/>\n"
            "  <field name=\"C&apos;\" start=\"2\" end=\"2\"/>\n"
            "  <field name=\"D&quot;\" start=\"1\" end=\"1\"/>\n"
            "  <field name=\"E&amp;\" start=\"0\" end=\"0\"/>\n"
            "</flags>\n");

  // Should include enumerators as the "type".
  strm.Clear();
  FieldEnum enum_single("enum_single", {{0, "a"}});
  RegisterFlags("Enumerators", 8,
                {RegisterFlags::Field("NoEnumerators", 4),
                 RegisterFlags::Field("OneEnumerator", 3, 3, &enum_single)})
      .ToXML(strm);
  ASSERT_EQ(strm.GetString(),
            "<flags id=\"Enumerators\" size=\"8\">\n"
            "  <field name=\"NoEnumerators\" start=\"4\" end=\"4\"/>\n"
            "  <field name=\"OneEnumerator\" start=\"3\" end=\"3\" "
            "type=\"enum_single\"/>\n"
            "</flags>\n");
}

TEST(RegisterFlagsTest, EnumeratorToXML) {
  StreamString strm;

  FieldEnum::Enumerator(1234, "test").ToXML(strm);
  ASSERT_EQ(strm.GetString(), "<evalue name=\"test\" value=\"1234\"/>");

  // Special XML chars in names must be escaped.
  std::array special_names = {
      std::make_pair(FieldEnum::Enumerator(0, "A<"),
                     "<evalue name=\"A&lt;\" value=\"0\"/>"),
      std::make_pair(FieldEnum::Enumerator(1, "B>"),
                     "<evalue name=\"B&gt;\" value=\"1\"/>"),
      std::make_pair(FieldEnum::Enumerator(2, "C'"),
                     "<evalue name=\"C&apos;\" value=\"2\"/>"),
      std::make_pair(FieldEnum::Enumerator(3, "D\""),
                     "<evalue name=\"D&quot;\" value=\"3\"/>"),
      std::make_pair(FieldEnum::Enumerator(4, "E&"),
                     "<evalue name=\"E&amp;\" value=\"4\"/>"),
  };

  for (const auto &[enumerator, expected] : special_names) {
    strm.Clear();
    enumerator.ToXML(strm);
    ASSERT_EQ(strm.GetString(), expected);
  }
}

TEST(RegisterFlagsTest, EnumToXML) {
  StreamString strm;

  FieldEnum("empty_enum", {}).ToXML(strm, 4);
  ASSERT_EQ(strm.GetString(), "<enum id=\"empty_enum\" size=\"4\"/>\n");

  strm.Clear();
  FieldEnum("single_enumerator", {FieldEnum::Enumerator(0, "zero")})
      .ToXML(strm, 5);
  ASSERT_EQ(strm.GetString(), "<enum id=\"single_enumerator\" size=\"5\">\n"
                              "  <evalue name=\"zero\" value=\"0\"/>\n"
                              "</enum>\n");

  strm.Clear();
  FieldEnum("multiple_enumerator",
            {FieldEnum::Enumerator(0, "zero"), FieldEnum::Enumerator(1, "one")})
      .ToXML(strm, 8);
  ASSERT_EQ(strm.GetString(), "<enum id=\"multiple_enumerator\" size=\"8\">\n"
                              "  <evalue name=\"zero\" value=\"0\"/>\n"
                              "  <evalue name=\"one\" value=\"1\"/>\n"
                              "</enum>\n");
}

TEST(RegisterFlagsTest, EnumsToXML) {
  // This method should output all the enums used by the register flag set,
  // only once.

  StreamString strm;
  FieldEnum enum_a("enum_a", {FieldEnum::Enumerator(0, "zero")});
  FieldEnum enum_b("enum_b", {FieldEnum::Enumerator(1, "one")});
  FieldEnum enum_c("enum_c", {FieldEnum::Enumerator(2, "two")});
  llvm::StringSet<> seen;
  // Pretend that enum_c was already emitted for a different flag set.
  seen.insert("enum_c");

  RegisterFlags("Test", 4,
                {
                    RegisterFlags::Field("f1", 31, 31, &enum_a),
                    RegisterFlags::Field("f2", 30, 30, &enum_a),
                    RegisterFlags::Field("f3", 29, 29, &enum_b),
                    RegisterFlags::Field("f4", 27, 28, &enum_c),
                })
      .EnumsToXML(strm, seen);
  ASSERT_EQ(strm.GetString(), "<enum id=\"enum_a\" size=\"4\">\n"
                              "  <evalue name=\"zero\" value=\"0\"/>\n"
                              "</enum>\n"
                              "<enum id=\"enum_b\" size=\"4\">\n"
                              "  <evalue name=\"one\" value=\"1\"/>\n"
                              "</enum>\n");
}