//===- llvm/unittest/Support/ScopedPrinterTest.cpp - ScopedPrinter tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ScopedPrinter.h"
#include "llvm/ADT/APSInt.h"
#include "gtest/gtest.h"
#include <vector>

using namespace llvm;

class ScopedPrinterTest : public ::testing::Test {
protected:
  std::string StreamBuffer;
  raw_string_ostream OS;
  ScopedPrinter Writer;

  ScopedPrinterTest() : OS(StreamBuffer), Writer(OS) {}

  using PrintFunc = function_ref<void(ScopedPrinter &)>;

  void verifyScopedPrinter(StringRef Expected, PrintFunc Func) {
    Func(Writer);
    Writer.flush();
    EXPECT_EQ(Expected.str(), OS.str());
  }
};

TEST_F(ScopedPrinterTest, Indent) {
  auto PrintFunc = [](ScopedPrinter &W) {
    W.printString("|");
    W.indent();
    W.printString("|");
    W.indent(2);
    W.printString("|");
  };

  const char *ExpectedOut = R"(|
  |
      |
)";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}

TEST_F(ScopedPrinterTest, Unindent) {
  auto PrintFunc = [](ScopedPrinter &W) {
    W.indent(3);
    W.printString("|");
    W.unindent(2);
    W.printString("|");
    W.unindent();
    W.printString("|");
    W.unindent();
    W.printString("|");
  };

  const char *ExpectedOut = R"(      |
  |
|
|
)";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}

TEST_F(ScopedPrinterTest, ResetIndent) {
  auto PrintFunc = [](ScopedPrinter &W) {
    W.indent(4);
    W.printString("|");
    W.resetIndent();
    W.printString("|");
  };

  const char *ExpectedOut = R"(        |
|
)";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}

TEST_F(ScopedPrinterTest, PrintIndent) {
  auto PrintFunc = [](ScopedPrinter &W) {
    W.printIndent();
    W.printString("|");
    W.indent();
    W.printIndent();
    W.printString("|");
  };

  const char *ExpectedOut = R"(|
    |
)";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}

TEST_F(ScopedPrinterTest, GetIndentLevel) {
  EXPECT_EQ(Writer.getIndentLevel(), 0);
  Writer.indent();
  EXPECT_EQ(Writer.getIndentLevel(), 1);
  Writer.indent();
  EXPECT_EQ(Writer.getIndentLevel(), 2);
  Writer.unindent();
  EXPECT_EQ(Writer.getIndentLevel(), 1);
  Writer.indent();
  Writer.resetIndent();
  EXPECT_EQ(Writer.getIndentLevel(), 0);
  Writer.unindent();
  EXPECT_EQ(Writer.getIndentLevel(), 0);
  Writer.indent();
  EXPECT_EQ(Writer.getIndentLevel(), 1);
}

TEST_F(ScopedPrinterTest, SetPrefix) {
  auto PrintFunc = [](ScopedPrinter &W) {
    W.setPrefix("Prefix1");
    W.indent();
    W.printIndent();
    W.printString("|");
    W.unindent();
    W.printIndent();
    W.printString("|");
    W.setPrefix("Prefix2");
    W.printIndent();
    W.printString("|");
  };

  const char *ExpectedOut = R"(Prefix1  Prefix1  |
Prefix1Prefix1|
Prefix2Prefix2|
)";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}

TEST_F(ScopedPrinterTest, PrintEnum) {
  auto PrintFunc = [](ScopedPrinter &W) {
    const EnumEntry<int> EnumList[] = {{"Name1", "AltName1", 1},
                                       {"Name2", "AltName2", 2},
                                       {"Name3", "AltName3", 3},
                                       {"Name4", "AltName4", 2}};
    EnumEntry<int> OtherEnum{"Name5", "AltName5", 5};
    W.printEnum("Exists", EnumList[1].Value, makeArrayRef(EnumList));
    W.printEnum("DoesNotExist", OtherEnum.Value, makeArrayRef(EnumList));
  };

  const char *ExpectedOut = R"(Exists: Name2 (0x2)
DoesNotExist: 0x5
)";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}

TEST_F(ScopedPrinterTest, PrintFlag) {
  auto PrintFunc = [](ScopedPrinter &W) {
    const EnumEntry<uint16_t> SingleBitFlags[] = {
        {"Name0", "AltName0", 0},
        {"Name1", "AltName1", 1},
        {"Name2", "AltName2", 1 << 1},
        {"Name3", "AltName3", 1 << 2}};
    const EnumEntry<uint16_t> UnsortedFlags[] = {
        {"C", "c", 1}, {"B", "b", 1 << 1}, {"A", "a", 1 << 2}};
    const EnumEntry<uint16_t> EnumFlags[] = {
        {"FirstByte1", "First1", 0x1u},    {"FirstByte2", "First2", 0x2u},
        {"FirstByte3", "First3", 0x3u},    {"SecondByte1", "Second1", 0x10u},
        {"SecondByte2", "Second2", 0x20u}, {"SecondByte3", "Second3", 0x30u},
        {"ThirdByte1", "Third1", 0x100u},  {"ThirdByte2", "Third2", 0x200u},
        {"ThirdByte3", "Third3", 0x300u}};
    W.printFlags("ZeroFlag", 0, makeArrayRef(SingleBitFlags));
    W.printFlags("NoFlag", 1 << 3, makeArrayRef(SingleBitFlags));
    W.printFlags("Flag1", SingleBitFlags[1].Value,
                 makeArrayRef(SingleBitFlags));
    W.printFlags("Flag1&3", (1 << 2) + 1, makeArrayRef(SingleBitFlags));

    W.printFlags("ZeroFlagRaw", 0);
    W.printFlags("NoFlagRaw", 1 << 3);
    W.printFlags("Flag1Raw", SingleBitFlags[1].Value);
    W.printFlags("Flag1&3Raw", (1 << 2) + 1);

    W.printFlags("FlagSorted", (1 << 2) + (1 << 1) + 1,
                 makeArrayRef(UnsortedFlags));

    uint16_t NoBitMask = 0;
    uint16_t FirstByteMask = 0xFu;
    uint16_t SecondByteMask = 0xF0u;
    uint16_t ThirdByteMask = 0xF00u;
    W.printFlags("NoBitMask", 0xFFFu, makeArrayRef(EnumFlags), NoBitMask);
    W.printFlags("FirstByteMask", 0x3u, makeArrayRef(EnumFlags), FirstByteMask);
    W.printFlags("SecondByteMask", 0x30u, makeArrayRef(EnumFlags),
                 SecondByteMask);
    W.printFlags("ValueOutsideMask", 0x1u, makeArrayRef(EnumFlags),
                 SecondByteMask);
    W.printFlags("FirstSecondByteMask", 0xFFu, makeArrayRef(EnumFlags),
                 FirstByteMask, SecondByteMask);
    W.printFlags("FirstSecondThirdByteMask", 0x333u, makeArrayRef(EnumFlags),
                 FirstByteMask, SecondByteMask, ThirdByteMask);
  };

  const char *ExpectedOut = R"(ZeroFlag [ (0x0)
]
NoFlag [ (0x8)
]
Flag1 [ (0x1)
  Name1 (0x1)
]
Flag1&3 [ (0x5)
  Name1 (0x1)
  Name3 (0x4)
]
ZeroFlagRaw [ (0x0)
]
NoFlagRaw [ (0x8)
  0x8
]
Flag1Raw [ (0x1)
  0x1
]
Flag1&3Raw [ (0x5)
  0x1
  0x4
]
FlagSorted [ (0x7)
  A (0x4)
  B (0x2)
  C (0x1)
]
NoBitMask [ (0xFFF)
  FirstByte1 (0x1)
  FirstByte2 (0x2)
  FirstByte3 (0x3)
  SecondByte1 (0x10)
  SecondByte2 (0x20)
  SecondByte3 (0x30)
  ThirdByte1 (0x100)
  ThirdByte2 (0x200)
  ThirdByte3 (0x300)
]
FirstByteMask [ (0x3)
  FirstByte3 (0x3)
]
SecondByteMask [ (0x30)
  SecondByte3 (0x30)
]
ValueOutsideMask [ (0x1)
  FirstByte1 (0x1)
]
FirstSecondByteMask [ (0xFF)
]
FirstSecondThirdByteMask [ (0x333)
  FirstByte3 (0x3)
  SecondByte3 (0x30)
  ThirdByte3 (0x300)
]
)";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}

TEST_F(ScopedPrinterTest, PrintNumber) {
  auto PrintFunc = [](ScopedPrinter &W) {
    uint64_t Unsigned64Max = std::numeric_limits<uint64_t>::max();
    uint64_t Unsigned64Min = std::numeric_limits<uint64_t>::min();
    W.printNumber("uint64_t-max", Unsigned64Max);
    W.printNumber("uint64_t-min", Unsigned64Min);

    uint32_t Unsigned32Max = std::numeric_limits<uint32_t>::max();
    uint32_t Unsigned32Min = std::numeric_limits<uint32_t>::min();
    W.printNumber("uint32_t-max", Unsigned32Max);
    W.printNumber("uint32_t-min", Unsigned32Min);

    uint16_t Unsigned16Max = std::numeric_limits<uint16_t>::max();
    uint16_t Unsigned16Min = std::numeric_limits<uint16_t>::min();
    W.printNumber("uint16_t-max", Unsigned16Max);
    W.printNumber("uint16_t-min", Unsigned16Min);

    uint8_t Unsigned8Max = std::numeric_limits<uint8_t>::max();
    uint8_t Unsigned8Min = std::numeric_limits<uint8_t>::min();
    W.printNumber("uint8_t-max", Unsigned8Max);
    W.printNumber("uint8_t-min", Unsigned8Min);

    int64_t Signed64Max = std::numeric_limits<int64_t>::max();
    int64_t Signed64Min = std::numeric_limits<int64_t>::min();
    W.printNumber("int64_t-max", Signed64Max);
    W.printNumber("int64_t-min", Signed64Min);

    int32_t Signed32Max = std::numeric_limits<int32_t>::max();
    int32_t Signed32Min = std::numeric_limits<int32_t>::min();
    W.printNumber("int32_t-max", Signed32Max);
    W.printNumber("int32_t-min", Signed32Min);

    int16_t Signed16Max = std::numeric_limits<int16_t>::max();
    int16_t Signed16Min = std::numeric_limits<int16_t>::min();
    W.printNumber("int16_t-max", Signed16Max);
    W.printNumber("int16_t-min", Signed16Min);

    int8_t Signed8Max = std::numeric_limits<int8_t>::max();
    int8_t Signed8Min = std::numeric_limits<int8_t>::min();
    W.printNumber("int8_t-max", Signed8Max);
    W.printNumber("int8_t-min", Signed8Min);

    APSInt LargeNum("9999999999999999999999");
    W.printNumber("apsint", LargeNum);

    W.printNumber("label", "value", 0);
  };

  const char *ExpectedOut = R"(uint64_t-max: 18446744073709551615
uint64_t-min: 0
uint32_t-max: 4294967295
uint32_t-min: 0
uint16_t-max: 65535
uint16_t-min: 0
uint8_t-max: 255
uint8_t-min: 0
int64_t-max: 9223372036854775807
int64_t-min: -9223372036854775808
int32_t-max: 2147483647
int32_t-min: -2147483648
int16_t-max: 32767
int16_t-min: -32768
int8_t-max: 127
int8_t-min: -128
apsint: 9999999999999999999999
label: value (0)
)";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}

TEST_F(ScopedPrinterTest, PrintBoolean) {
  auto PrintFunc = [](ScopedPrinter &W) {
    W.printBoolean("True", true);
    W.printBoolean("False", false);
  };

  const char *ExpectedOut = R"(True: Yes
False: No
)";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}

TEST_F(ScopedPrinterTest, PrintVersion) {
  auto PrintFunc = [](ScopedPrinter &W) {
    W.printVersion("Version", "123", "456", "789");
  };
  const char *ExpectedOut = R"(Version: 123.456.789
)";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}

TEST_F(ScopedPrinterTest, PrintList) {
  auto PrintFunc = [](ScopedPrinter &W) {
    const std::vector<uint64_t> EmptyList;
    const std::vector<std::string> StringList = {"foo", "bar", "baz"};
    const bool BoolList[] = {true, false};
    const std::vector<uint64_t> Unsigned64List = {
        std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::min()};
    const std::vector<uint32_t> Unsigned32List = {
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::min()};
    const std::vector<uint16_t> Unsigned16List = {
        std::numeric_limits<uint16_t>::max(),
        std::numeric_limits<uint16_t>::min()};
    const std::vector<uint8_t> Unsigned8List = {
        std::numeric_limits<uint8_t>::max(),
        std::numeric_limits<uint8_t>::min()};
    const std::vector<int64_t> Signed64List = {
        std::numeric_limits<int64_t>::max(),
        std::numeric_limits<int64_t>::min()};
    const std::vector<int32_t> Signed32List = {
        std::numeric_limits<int32_t>::max(),
        std::numeric_limits<int32_t>::min()};
    const std::vector<int16_t> Signed16List = {
        std::numeric_limits<int16_t>::max(),
        std::numeric_limits<int16_t>::min()};
    const std::vector<int8_t> Signed8List = {
        std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::min()};
    const std::vector<APSInt> APSIntList = {APSInt("9999999999999999999999"),
                                            APSInt("-9999999999999999999999")};
    W.printList("EmptyList", EmptyList);
    W.printList("StringList", StringList);
    W.printList("BoolList", makeArrayRef(BoolList));
    W.printList("uint64List", Unsigned64List);
    W.printList("uint32List", Unsigned32List);
    W.printList("uint16List", Unsigned16List);
    W.printList("uint8List", Unsigned8List);
    W.printList("int64List", Signed64List);
    W.printList("int32List", Signed32List);
    W.printList("int16List", Signed16List);
    W.printList("int8List", Signed8List);
    W.printList("APSIntList", APSIntList);
  };

  const char *ExpectedOut = R"(EmptyList: []
StringList: [foo, bar, baz]
BoolList: [1, 0]
uint64List: [18446744073709551615, 0]
uint32List: [4294967295, 0]
uint16List: [65535, 0]
uint8List: [255, 0]
int64List: [9223372036854775807, -9223372036854775808]
int32List: [2147483647, -2147483648]
int16List: [32767, -32768]
int8List: [127, -128]
APSIntList: [9999999999999999999999, -9999999999999999999999]
)";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}

TEST_F(ScopedPrinterTest, PrintListPrinter) {
  auto PrintFunc = [](ScopedPrinter &W) {
    const std::string StringList[] = {"a", "ab", "abc"};
    W.printList("StringSizeList", StringList,
                [](raw_ostream &OS, StringRef Item) { OS << Item.size(); });
  };

  const char *ExpectedOut = R"(StringSizeList: [1, 2, 3]
)";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}

TEST_F(ScopedPrinterTest, PrintHex) {
  auto PrintFunc = [](ScopedPrinter &W) {
    W.printHex("HexNumber", 0x10);
    W.printHex("HexLabel", "Name", 0x10);
  };

  const char *ExpectedOut = R"(HexNumber: 0x10
HexLabel: Name (0x10)
)";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}

TEST_F(ScopedPrinterTest, PrintHexList) {
  auto PrintFunc = [](ScopedPrinter &W) {
    const uint64_t HexList[] = {0x1, 0x10, 0x100};
    W.printHexList("HexList", HexList);
  };
  const char *ExpectedOut = R"(HexList: [0x1, 0x10, 0x100]
)";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}

TEST_F(ScopedPrinterTest, PrintSymbolOffset) {
  auto PrintFunc = [](ScopedPrinter &W) {
    W.printSymbolOffset("SymbolOffset", "SymbolName", 0x10);
    W.printSymbolOffset("NoSymbolOffset", "SymbolName", 0);
  };
  const char *ExpectedOut = R"(SymbolOffset: SymbolName+0x10
NoSymbolOffset: SymbolName+0x0
)";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}

TEST_F(ScopedPrinterTest, PrintString) {
  auto PrintFunc = [](ScopedPrinter &W) {
    const StringRef StringRefValue("Value");
    const std::string StringValue = "Value";
    const char *CharArrayValue = "Value";
    W.printString("StringRef", StringRefValue);
    W.printString("String", StringValue);
    W.printString("CharArray", CharArrayValue);
    ListScope L(W, "StringList");
    W.printString(StringRefValue);
  };

  const char *ExpectedOut = R"(StringRef: Value
String: Value
CharArray: Value
StringList [
  Value
]
)";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}

TEST_F(ScopedPrinterTest, PrintBinary) {
  auto PrintFunc = [](ScopedPrinter &W) {
    std::vector<uint8_t> IntArray = {70, 111, 111, 66, 97, 114};
    std::vector<char> CharArray = {'F', 'o', 'o', 'B', 'a', 'r'};
    std::vector<uint8_t> InvalidChars = {255, 255};
    W.printBinary("Binary1", "FooBar", IntArray);
    W.printBinary("Binary2", "FooBar", CharArray);
    W.printBinary("Binary3", IntArray);
    W.printBinary("Binary4", CharArray);
    W.printBinary("Binary5", StringRef("FooBar"));
    W.printBinary("Binary6", StringRef("Multiple Line FooBar"));
    W.printBinaryBlock("Binary7", IntArray, 20);
    W.printBinaryBlock("Binary8", IntArray);
    W.printBinaryBlock("Binary9", "FooBar");
    W.printBinaryBlock("Binary10", "Multiple Line FooBar");
    W.printBinaryBlock("Binary11", InvalidChars);
  };

  const char *ExpectedOut = R"(Binary1: FooBar (46 6F 6F 42 61 72)
Binary2: FooBar (46 6F 6F 42 61 72)
Binary3: (46 6F 6F 42 61 72)
Binary4: (46 6F 6F 42 61 72)
Binary5: (46 6F 6F 42 61 72)
Binary6 (
  0000: 4D756C74 69706C65 204C696E 6520466F  |Multiple Line Fo|
  0010: 6F426172                             |oBar|
)
Binary7 (
  0014: 466F6F42 6172                        |FooBar|
)
Binary8 (
  0000: 466F6F42 6172                        |FooBar|
)
Binary9 (
  0000: 466F6F42 6172                        |FooBar|
)
Binary10 (
  0000: 4D756C74 69706C65 204C696E 6520466F  |Multiple Line Fo|
  0010: 6F426172                             |oBar|
)
Binary11 (
  0000: FFFF                                 |..|
)
)";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}

TEST_F(ScopedPrinterTest, PrintObject) {
  auto PrintFunc = [](ScopedPrinter &W) { W.printObject("Object", "Value"); };

  const char *ExpectedOut = R"(Object: Value
)";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}

TEST_F(ScopedPrinterTest, StartLine) {
  auto PrintFunc = [](ScopedPrinter &W) {
    W.startLine() << "|";
    W.indent(2);
    W.startLine() << "|";
    W.unindent();
    W.startLine() << "|";
  };

  const char *ExpectedOut = "|    |  |";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}

TEST_F(ScopedPrinterTest, GetOStream) {
  auto PrintFunc = [](ScopedPrinter &W) { W.getOStream() << "Test"; };

  const char *ExpectedOut = "Test";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}

TEST_F(ScopedPrinterTest, PrintScope) {
  auto PrintFunc = [](ScopedPrinter &W) {
    {
      DictScope O(W, "Object");
      { DictScope OO(W, "ObjectInObject"); }
      { ListScope LO(W, "ListInObject"); }
    }
    {
      ListScope L(W, "List");
      { DictScope OL(W, "ObjectInList"); }
      { ListScope LL(W, "ListInList"); }
    }
  };

  const char *ExpectedOut = R"(Object {
  ObjectInObject {
  }
  ListInObject [
  ]
}
List [
  ObjectInList {
  }
  ListInList [
  ]
]
)";
  verifyScopedPrinter(ExpectedOut, PrintFunc);
}
