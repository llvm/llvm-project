//===-- XMLTest.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/XML.h"
#include "gtest/gtest.h"

using namespace lldb_private;

#if LLDB_ENABLE_LIBXML2

static void assertGetElement(XMLNode &root, const char *element_name,
                             bool expected_uint_success, uint64_t expected_uint,
                             bool expected_double_success,
                             double expected_double) {
  XMLNode node = root.FindFirstChildElementWithName(element_name);
  ASSERT_TRUE(node.IsValid());

  uint64_t uint_val;
  EXPECT_EQ(node.GetElementTextAsUnsigned(uint_val, 66, 0),
            expected_uint_success);
  EXPECT_EQ(uint_val, expected_uint);

  double double_val;
  EXPECT_EQ(node.GetElementTextAsFloat(double_val, 66.0),
            expected_double_success);
  EXPECT_EQ(double_val, expected_double);

  XMLNode attr_node = root.FindFirstChildElementWithName("attr");
  ASSERT_TRUE(node.IsValid());

  EXPECT_EQ(
      attr_node.GetAttributeValueAsUnsigned(element_name, uint_val, 66, 0),
      expected_uint_success);
  EXPECT_EQ(uint_val, expected_uint);
}

#define ASSERT_GET(element_name, ...)                                          \
  {                                                                            \
    SCOPED_TRACE("at element/attribute " element_name);                        \
    assertGetElement(root, element_name, __VA_ARGS__);                         \
  }

TEST(XML, GetAs) {
  std::string test_xml =
      "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
      "<test>\n"
      "  <empty/>\n"
      "  <text>123foo</text>\n"
      "  <positive-int>11</positive-int>\n"
      "  <negative-int>-11</negative-int>\n"
      "  <positive-overflow>18446744073709551616</positive-overflow>\n"
      "  <negative-overflow>-9223372036854775809</negative-overflow>\n"
      "  <hex>0x1234</hex>\n"
      "  <positive-float>12.5</positive-float>\n"
      "  <negative-float>-12.5</negative-float>\n"
      "  <attr empty=\"\"\n"
      "        text=\"123foo\"\n"
      "        positive-int=\"11\"\n"
      "        negative-int=\"-11\"\n"
      "        positive-overflow=\"18446744073709551616\"\n"
      "        negative-overflow=\"-9223372036854775809\"\n"
      "        hex=\"0x1234\"\n"
      "        positive-float=\"12.5\"\n"
      "        negative-float=\"-12.5\"\n"
      "       />\n"
      "</test>\n";

  XMLDocument doc;
  ASSERT_TRUE(doc.ParseMemory(test_xml.data(), test_xml.size()));

  XMLNode root = doc.GetRootElement();
  ASSERT_TRUE(root.IsValid());

  ASSERT_GET("empty", false, 66, false, 66.0);
  ASSERT_GET("text", false, 66, false, 66.0);
  ASSERT_GET("positive-int", true, 11, true, 11.0);
  ASSERT_GET("negative-int", false, 66, true, -11.0);
  ASSERT_GET("positive-overflow", false, 66, true, 18446744073709551616.0);
  ASSERT_GET("negative-overflow", false, 66, true, -9223372036854775809.0);
  ASSERT_GET("hex", true, 0x1234, true, 4660.0);
  ASSERT_GET("positive-float", false, 66, true, 12.5);
  ASSERT_GET("negative-float", false, 66, true, -12.5);
}

#else // !LLDB_ENABLE_LIBXML2

TEST(XML, GracefulNoXML) {
  std::string test_xml =
      "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
      "<test>\n"
      "  <text attribute=\"123\">123</text>\n"
      "</test>\n";

  XMLDocument doc;
  ASSERT_FALSE(doc.ParseMemory(test_xml.data(), test_xml.size()));

  XMLNode root = doc.GetRootElement();
  EXPECT_FALSE(root.IsValid());

  XMLNode node = root.FindFirstChildElementWithName("text");
  EXPECT_FALSE(node.IsValid());

  uint64_t uint_val;
  EXPECT_FALSE(node.GetElementTextAsUnsigned(uint_val, 66, 0));
  EXPECT_EQ(uint_val, 66U);
  EXPECT_FALSE(node.GetAttributeValueAsUnsigned("attribute", uint_val, 66, 0));
  EXPECT_EQ(uint_val, 66U);

  double double_val;
  EXPECT_FALSE(node.GetElementTextAsFloat(double_val, 66.0));
  EXPECT_EQ(double_val, 66.0);
}

#endif // LLDB_ENABLE_LIBXML2
