#include <limits.h>
#include "gtest/gtest.h"

#include "lldb/Utility/StringExtractor.h"

namespace
{
    class StringExtractorTest: public ::testing::Test
    {
    };
}

TEST_F (StringExtractorTest, InitEmpty)
{
    const char kEmptyString[] = "";
    StringExtractor ex (kEmptyString);

    ASSERT_EQ (true, ex.IsGood());
    ASSERT_EQ (0u, ex.GetFilePos());
    ASSERT_STREQ (kEmptyString, ex.GetStringRef().c_str());
    ASSERT_EQ (true, ex.Empty());
    ASSERT_EQ (0u, ex.GetBytesLeft());
    ASSERT_EQ (nullptr, ex.Peek());
}

TEST_F (StringExtractorTest, InitMisc)
{
    const char kInitMiscString[] = "Hello, StringExtractor!";
    StringExtractor ex (kInitMiscString);

    ASSERT_EQ (true, ex.IsGood());
    ASSERT_EQ (0u, ex.GetFilePos());
    ASSERT_STREQ (kInitMiscString, ex.GetStringRef().c_str());
    ASSERT_EQ (false, ex.Empty());
    ASSERT_EQ (sizeof(kInitMiscString)-1, ex.GetBytesLeft());
    ASSERT_EQ (kInitMiscString[0], *ex.Peek());
}

TEST_F (StringExtractorTest, DecodeHexU8_Underflow)
{
    const char kEmptyString[] = "";
    StringExtractor ex (kEmptyString);

    ASSERT_EQ (-1, ex.DecodeHexU8());
    ASSERT_EQ (true, ex.IsGood());
    ASSERT_EQ (0u, ex.GetFilePos());
    ASSERT_EQ (true, ex.Empty());
    ASSERT_EQ (0u, ex.GetBytesLeft());
    ASSERT_EQ (nullptr, ex.Peek());
}

TEST_F (StringExtractorTest, DecodeHexU8_Underflow2)
{
    const char kEmptyString[] = "1";
    StringExtractor ex (kEmptyString);

    ASSERT_EQ (-1, ex.DecodeHexU8());
    ASSERT_EQ (true, ex.IsGood());
    ASSERT_EQ (0u, ex.GetFilePos());
    ASSERT_EQ (1u, ex.GetBytesLeft());
    ASSERT_EQ ('1', *ex.Peek());
}

TEST_F (StringExtractorTest, DecodeHexU8_InvalidHex)
{
    const char kInvalidHex[] = "xa";
    StringExtractor ex (kInvalidHex);

    ASSERT_EQ (-1, ex.DecodeHexU8());
    ASSERT_EQ (true, ex.IsGood());
    ASSERT_EQ (0u, ex.GetFilePos());
    ASSERT_EQ (2u, ex.GetBytesLeft());
    ASSERT_EQ ('x', *ex.Peek());
}

TEST_F (StringExtractorTest, DecodeHexU8_InvalidHex2)
{
    const char kInvalidHex[] = "ax";
    StringExtractor ex (kInvalidHex);

    ASSERT_EQ (-1, ex.DecodeHexU8());
    ASSERT_EQ (true, ex.IsGood());
    ASSERT_EQ (0u, ex.GetFilePos());
    ASSERT_EQ (2u, ex.GetBytesLeft());
    ASSERT_EQ ('a', *ex.Peek());
}

TEST_F (StringExtractorTest, DecodeHexU8_Exact)
{
    const char kValidHexPair[] = "12";
    StringExtractor ex (kValidHexPair);

    ASSERT_EQ (0x12, ex.DecodeHexU8());
    ASSERT_EQ (true, ex.IsGood());
    ASSERT_EQ (2u, ex.GetFilePos());
    ASSERT_EQ (0u, ex.GetBytesLeft());
    ASSERT_EQ (nullptr, ex.Peek());
}

TEST_F (StringExtractorTest, DecodeHexU8_Extra)
{
    const char kValidHexPair[] = "1234";
    StringExtractor ex (kValidHexPair);

    ASSERT_EQ (0x12, ex.DecodeHexU8());
    ASSERT_EQ (true, ex.IsGood());
    ASSERT_EQ (2u, ex.GetFilePos());
    ASSERT_EQ (2u, ex.GetBytesLeft());
    ASSERT_EQ ('3', *ex.Peek());
}

TEST_F (StringExtractorTest, GetHexU8_Underflow)
{
    const char kEmptyString[] = "";
    StringExtractor ex (kEmptyString);

    ASSERT_EQ (0xab, ex.GetHexU8(0xab));
    ASSERT_EQ (false, ex.IsGood());
    ASSERT_EQ (UINT64_MAX, ex.GetFilePos());
    ASSERT_EQ (true, ex.Empty());
    ASSERT_EQ (0u, ex.GetBytesLeft());
    ASSERT_EQ (nullptr, ex.Peek());
}

TEST_F (StringExtractorTest, GetHexU8_Underflow2)
{
    const char kOneNibble[] = "1";
    StringExtractor ex (kOneNibble);

    ASSERT_EQ (0xbc, ex.GetHexU8(0xbc));
    ASSERT_EQ (false, ex.IsGood());
    ASSERT_EQ (UINT64_MAX, ex.GetFilePos());
    ASSERT_EQ (0u, ex.GetBytesLeft());
    ASSERT_EQ (nullptr, ex.Peek());
}

TEST_F (StringExtractorTest, GetHexU8_InvalidHex)
{
    const char kInvalidHex[] = "xx";
    StringExtractor ex (kInvalidHex);

    ASSERT_EQ (0xcd, ex.GetHexU8(0xcd));
    ASSERT_EQ (false, ex.IsGood());
    ASSERT_EQ (UINT64_MAX, ex.GetFilePos());
    ASSERT_EQ (0u, ex.GetBytesLeft());
    ASSERT_EQ (nullptr, ex.Peek());
}

TEST_F (StringExtractorTest, GetHexU8_Exact)
{
    const char kValidHexPair[] = "12";
    StringExtractor ex (kValidHexPair);

    ASSERT_EQ (0x12, ex.GetHexU8(0x12));
    ASSERT_EQ (true, ex.IsGood());
    ASSERT_EQ (2u, ex.GetFilePos());
    ASSERT_EQ (0u, ex.GetBytesLeft());
    ASSERT_EQ (nullptr, ex.Peek());
}

TEST_F (StringExtractorTest, GetHexU8_Extra)
{
    const char kValidHexPair[] = "1234";
    StringExtractor ex (kValidHexPair);

    ASSERT_EQ (0x12, ex.GetHexU8(0x12));
    ASSERT_EQ (true, ex.IsGood());
    ASSERT_EQ (2u, ex.GetFilePos());
    ASSERT_EQ (2u, ex.GetBytesLeft());
    ASSERT_EQ ('3', *ex.Peek());
}

TEST_F (StringExtractorTest, GetHexU8_Underflow_NoEof)
{
    const char kEmptyString[] = "";
    StringExtractor ex (kEmptyString);
    const bool kSetEofOnFail = false;

    ASSERT_EQ (0xab, ex.GetHexU8(0xab, kSetEofOnFail));
    ASSERT_EQ (false, ex.IsGood()); // this result seems inconsistent with kSetEofOnFail == false
    ASSERT_EQ (UINT64_MAX, ex.GetFilePos());
    ASSERT_EQ (true, ex.Empty());
    ASSERT_EQ (0u, ex.GetBytesLeft());
    ASSERT_EQ (nullptr, ex.Peek());
}

TEST_F (StringExtractorTest, GetHexU8_Underflow2_NoEof)
{
    const char kOneNibble[] = "1";
    StringExtractor ex (kOneNibble);
    const bool kSetEofOnFail = false;

    ASSERT_EQ (0xbc, ex.GetHexU8(0xbc, kSetEofOnFail));
    ASSERT_EQ (true, ex.IsGood());
    ASSERT_EQ (0u, ex.GetFilePos());
    ASSERT_EQ (1u, ex.GetBytesLeft());
    ASSERT_EQ ('1', *ex.Peek());
}

TEST_F (StringExtractorTest, GetHexU8_InvalidHex_NoEof)
{
    const char kInvalidHex[] = "xx";
    StringExtractor ex (kInvalidHex);
    const bool kSetEofOnFail = false;

    ASSERT_EQ (0xcd, ex.GetHexU8(0xcd, kSetEofOnFail));
    ASSERT_EQ (true, ex.IsGood());
    ASSERT_EQ (0u, ex.GetFilePos());
    ASSERT_EQ (2u, ex.GetBytesLeft());
    ASSERT_EQ ('x', *ex.Peek());
}

TEST_F (StringExtractorTest, GetHexU8_Exact_NoEof)
{
    const char kValidHexPair[] = "12";
    StringExtractor ex (kValidHexPair);
    const bool kSetEofOnFail = false;

    ASSERT_EQ (0x12, ex.GetHexU8(0x12, kSetEofOnFail));
    ASSERT_EQ (true, ex.IsGood());
    ASSERT_EQ (2u, ex.GetFilePos());
    ASSERT_EQ (0u, ex.GetBytesLeft());
    ASSERT_EQ (nullptr, ex.Peek());
}

TEST_F (StringExtractorTest, GetHexU8_Extra_NoEof)
{
    const char kValidHexPair[] = "1234";
    StringExtractor ex (kValidHexPair);
    const bool kSetEofOnFail = false;

    ASSERT_EQ (0x12, ex.GetHexU8(0x12, kSetEofOnFail));
    ASSERT_EQ (true, ex.IsGood());
    ASSERT_EQ (2u, ex.GetFilePos());
    ASSERT_EQ (2u, ex.GetBytesLeft());
    ASSERT_EQ ('3', *ex.Peek());
}

TEST_F (StringExtractorTest, GetHexBytes)
{
    const char kHexEncodedBytes[] = "abcdef0123456789xyzw";
    const size_t kValidHexPairs = 8;
    StringExtractor ex(kHexEncodedBytes);

    uint8_t dst[kValidHexPairs];
    ASSERT_EQ(kValidHexPairs, ex.GetHexBytes (dst, kValidHexPairs, 0xde));
    EXPECT_EQ(0xab,dst[0]);
    EXPECT_EQ(0xcd,dst[1]);
    EXPECT_EQ(0xef,dst[2]);
    EXPECT_EQ(0x01,dst[3]);
    EXPECT_EQ(0x23,dst[4]);
    EXPECT_EQ(0x45,dst[5]);
    EXPECT_EQ(0x67,dst[6]);
    EXPECT_EQ(0x89,dst[7]);

    ASSERT_EQ(true, ex.IsGood());
    ASSERT_EQ(2*kValidHexPairs, ex.GetFilePos());
    ASSERT_EQ(false, ex.Empty());
    ASSERT_EQ(4u, ex.GetBytesLeft());
    ASSERT_EQ('x', *ex.Peek());
}

TEST_F (StringExtractorTest, GetHexBytes_Underflow)
{
    const char kHexEncodedBytes[] = "abcdef0123456789xyzw";
    const size_t kValidHexPairs = 8;
    StringExtractor ex(kHexEncodedBytes);

    uint8_t dst[12];
    ASSERT_EQ(kValidHexPairs, ex.GetHexBytes (dst, sizeof(dst), 0xde));
    EXPECT_EQ(0xab,dst[0]);
    EXPECT_EQ(0xcd,dst[1]);
    EXPECT_EQ(0xef,dst[2]);
    EXPECT_EQ(0x01,dst[3]);
    EXPECT_EQ(0x23,dst[4]);
    EXPECT_EQ(0x45,dst[5]);
    EXPECT_EQ(0x67,dst[6]);
    EXPECT_EQ(0x89,dst[7]);
    // these bytes should be filled with fail_fill_value 0xde
    EXPECT_EQ(0xde,dst[8]);
    EXPECT_EQ(0xde,dst[9]);
    EXPECT_EQ(0xde,dst[10]);
    EXPECT_EQ(0xde,dst[11]);

    ASSERT_EQ(false, ex.IsGood());
    ASSERT_EQ(UINT64_MAX, ex.GetFilePos());
    ASSERT_EQ(false, ex.Empty());
    ASSERT_EQ(0u, ex.GetBytesLeft());
    ASSERT_EQ(0, ex.Peek());
}

TEST_F (StringExtractorTest, GetHexBytes_Partial)
{
    const char kHexEncodedBytes[] = "abcdef0123456789xyzw";
    const size_t kReadBytes = 4;
    StringExtractor ex(kHexEncodedBytes);

    uint8_t dst[12];
    memset(dst, 0xab, sizeof(dst));
    ASSERT_EQ(kReadBytes, ex.GetHexBytes (dst, kReadBytes, 0xde));
    EXPECT_EQ(0xab,dst[0]);
    EXPECT_EQ(0xcd,dst[1]);
    EXPECT_EQ(0xef,dst[2]);
    EXPECT_EQ(0x01,dst[3]);
    // these bytes should be unchanged
    EXPECT_EQ(0xab,dst[4]);
    EXPECT_EQ(0xab,dst[5]);
    EXPECT_EQ(0xab,dst[6]);
    EXPECT_EQ(0xab,dst[7]);
    EXPECT_EQ(0xab,dst[8]);
    EXPECT_EQ(0xab,dst[9]);
    EXPECT_EQ(0xab,dst[10]);
    EXPECT_EQ(0xab,dst[11]);

    ASSERT_EQ(true, ex.IsGood());
    ASSERT_EQ(kReadBytes*2, ex.GetFilePos());
    ASSERT_EQ(false, ex.Empty());
    ASSERT_EQ(12u, ex.GetBytesLeft());
    ASSERT_EQ('2', *ex.Peek());
}

TEST_F (StringExtractorTest, GetHexBytesAvail)
{
    const char kHexEncodedBytes[] = "abcdef0123456789xyzw";
    const size_t kValidHexPairs = 8;
    StringExtractor ex(kHexEncodedBytes);

    uint8_t dst[kValidHexPairs];
    ASSERT_EQ(kValidHexPairs, ex.GetHexBytesAvail (dst, kValidHexPairs));
    EXPECT_EQ(0xab,dst[0]);
    EXPECT_EQ(0xcd,dst[1]);
    EXPECT_EQ(0xef,dst[2]);
    EXPECT_EQ(0x01,dst[3]);
    EXPECT_EQ(0x23,dst[4]);
    EXPECT_EQ(0x45,dst[5]);
    EXPECT_EQ(0x67,dst[6]);
    EXPECT_EQ(0x89,dst[7]);

    ASSERT_EQ(true, ex.IsGood());
    ASSERT_EQ(2*kValidHexPairs, ex.GetFilePos());
    ASSERT_EQ(false, ex.Empty());
    ASSERT_EQ(4u, ex.GetBytesLeft());
    ASSERT_EQ('x', *ex.Peek());
}

TEST_F (StringExtractorTest, GetHexBytesAvail_Underflow)
{
    const char kHexEncodedBytes[] = "abcdef0123456789xyzw";
    const size_t kValidHexPairs = 8;
    StringExtractor ex(kHexEncodedBytes);

    uint8_t dst[12];
    memset(dst, 0xef, sizeof(dst));
    ASSERT_EQ(kValidHexPairs, ex.GetHexBytesAvail (dst, sizeof(dst)));
    EXPECT_EQ(0xab,dst[0]);
    EXPECT_EQ(0xcd,dst[1]);
    EXPECT_EQ(0xef,dst[2]);
    EXPECT_EQ(0x01,dst[3]);
    EXPECT_EQ(0x23,dst[4]);
    EXPECT_EQ(0x45,dst[5]);
    EXPECT_EQ(0x67,dst[6]);
    EXPECT_EQ(0x89,dst[7]);
    // these bytes should be unchanged
    EXPECT_EQ(0xef,dst[8]);
    EXPECT_EQ(0xef,dst[9]);
    EXPECT_EQ(0xef,dst[10]);
    EXPECT_EQ(0xef,dst[11]);

    ASSERT_EQ(true, ex.IsGood());
    ASSERT_EQ(kValidHexPairs*2, ex.GetFilePos());
    ASSERT_EQ(false, ex.Empty());
    ASSERT_EQ(4u, ex.GetBytesLeft());
    ASSERT_EQ('x', *ex.Peek());
}

TEST_F (StringExtractorTest, GetHexBytesAvail_Partial)
{
    const char kHexEncodedBytes[] = "abcdef0123456789xyzw";
    const size_t kReadBytes = 4;
    StringExtractor ex(kHexEncodedBytes);

    uint8_t dst[12];
    memset(dst, 0xab, sizeof(dst));
    ASSERT_EQ(kReadBytes, ex.GetHexBytesAvail (dst, kReadBytes));
    EXPECT_EQ(0xab,dst[0]);
    EXPECT_EQ(0xcd,dst[1]);
    EXPECT_EQ(0xef,dst[2]);
    EXPECT_EQ(0x01,dst[3]);
    // these bytes should be unchanged
    EXPECT_EQ(0xab,dst[4]);
    EXPECT_EQ(0xab,dst[5]);
    EXPECT_EQ(0xab,dst[6]);
    EXPECT_EQ(0xab,dst[7]);
    EXPECT_EQ(0xab,dst[8]);
    EXPECT_EQ(0xab,dst[9]);
    EXPECT_EQ(0xab,dst[10]);
    EXPECT_EQ(0xab,dst[11]);

    ASSERT_EQ(true, ex.IsGood());
    ASSERT_EQ(kReadBytes*2, ex.GetFilePos());
    ASSERT_EQ(false, ex.Empty());
    ASSERT_EQ(12u, ex.GetBytesLeft());
    ASSERT_EQ('2', *ex.Peek());
}

TEST_F(StringExtractorTest, GetNameColonValueSuccess)
{
    const char kNameColonPairs[] = "key1:value1;key2:value2;";
    StringExtractor ex(kNameColonPairs);

    std::string name;
    std::string value;
    EXPECT_TRUE(ex.GetNameColonValue(name, value));
    EXPECT_EQ("key1", name);
    EXPECT_EQ("value1", value);
    EXPECT_TRUE(ex.GetNameColonValue(name, value));
    EXPECT_EQ("key2", name);
    EXPECT_EQ("value2", value);
    EXPECT_EQ(0, ex.GetBytesLeft());
}


TEST_F(StringExtractorTest, GetNameColonValueContainsColon)
{
    const char kNameColonPairs[] = "key1:value1:value2;key2:value3;";
    StringExtractor ex(kNameColonPairs);

    std::string name;
    std::string value;
    EXPECT_TRUE(ex.GetNameColonValue(name, value));
    EXPECT_EQ("key1", name);
    EXPECT_EQ("value1:value2", value);
    EXPECT_TRUE(ex.GetNameColonValue(name, value));
    EXPECT_EQ("key2", name);
    EXPECT_EQ("value3", value);
    EXPECT_EQ(0, ex.GetBytesLeft());
}

TEST_F(StringExtractorTest, GetNameColonValueNoSemicolon)
{
    const char kNameColonPairs[] = "key1:value1";
    StringExtractor ex(kNameColonPairs);

    std::string name;
    std::string value;
    EXPECT_FALSE(ex.GetNameColonValue(name, value));
    EXPECT_EQ(0, ex.GetBytesLeft());
}

TEST_F(StringExtractorTest, GetNameColonValueNoColon)
{
    const char kNameColonPairs[] = "key1value1;";
    StringExtractor ex(kNameColonPairs);

    std::string name;
    std::string value;
    EXPECT_FALSE(ex.GetNameColonValue(name, value));
    EXPECT_EQ(0, ex.GetBytesLeft());
}
