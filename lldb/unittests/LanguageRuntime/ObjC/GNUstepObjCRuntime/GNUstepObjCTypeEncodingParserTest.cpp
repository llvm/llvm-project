//===-- GNUstepObjCTypeEncodingParserTest.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/LanguageRuntime/ObjC/GNUstepObjCRuntime/GNUstepObjCTypeEncodingParser.h"

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/CompilerType.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

using namespace lldb_private;

namespace {

/// The parser needs no target state, so each data model is just a triple.
struct DataModel {
  const char *triple;
  /// Width of `long` in bits, which is what differs between LP64 and LLP64.
  unsigned long_bits;
  /// Width of a pointer in bits, which is what drives aggregate layout.
  unsigned pointer_bits;
};

class GNUstepTypeEncodingParserTest
    : public ::testing::TestWithParam<DataModel> {
public:
  void SetUp() override {
    m_triple = llvm::Triple(GetParam().triple);
    m_ast_sp = std::make_shared<TypeSystemClang>("test ASTContext", m_triple);
    // No runtime: `@"Name"` has no DeclVendor to resolve against and must
    // degrade to `id` rather than crash.
    m_parser = std::make_unique<GNUstepObjCTypeEncodingParser>(m_triple);
  }

  void TearDown() override {
    m_parser.reset();
    m_ast_sp.reset();
  }

  /// Realizes \p encoding and returns its type name, or "" if it failed.
  std::string Realize(llvm::StringRef encoding, bool for_expression = false) {
    CompilerType type = m_parser->RealizeType(*m_ast_sp, encoding.str().c_str(),
                                              for_expression);
    if (!type)
      return "";
    return type.GetTypeName().GetString();
  }

  std::optional<uint64_t> SizeOf(llvm::StringRef encoding) {
    CompilerType type =
        m_parser->RealizeType(*m_ast_sp, encoding.str().c_str(), false);
    if (!type)
      return std::nullopt;
    llvm::Expected<uint64_t> size = type.GetByteSize(nullptr);
    if (!size) {
      llvm::consumeError(size.takeError());
      return std::nullopt;
    }
    return *size;
  }

  SubsystemRAII<FileSystem, HostInfo> subsystems;
  llvm::Triple m_triple;
  std::shared_ptr<TypeSystemClang> m_ast_sp;
  std::unique_ptr<GNUstepObjCTypeEncodingParser> m_parser;
};

TEST_P(GNUstepTypeEncodingParserTest, Scalars) {
  EXPECT_EQ(Realize("c"), "char");
  EXPECT_EQ(Realize("i"), "int");
  EXPECT_EQ(Realize("s"), "short");
  EXPECT_EQ(Realize("q"), "long long");
  EXPECT_EQ(Realize("C"), "unsigned char");
  EXPECT_EQ(Realize("I"), "unsigned int");
  EXPECT_EQ(Realize("S"), "unsigned short");
  EXPECT_EQ(Realize("Q"), "unsigned long long");
  EXPECT_EQ(Realize("f"), "float");
  EXPECT_EQ(Realize("d"), "double");
  EXPECT_EQ(Realize("B"), "bool");
  EXPECT_EQ(Realize("v"), "void");
  EXPECT_EQ(Realize("*"), "char *");
}

// clang emits 'l' only where `long` is 32 bits and 'q' otherwise
// (ASTContext::getObjCEncodingForPrimitiveType), so 'l' always means 32 bits
// no matter which data model the target uses.
TEST_P(GNUstepTypeEncodingParserTest, LongIsAlways32Bits) {
  EXPECT_EQ(SizeOf("l"), std::optional<uint64_t>(4u));
  EXPECT_EQ(SizeOf("L"), std::optional<uint64_t>(4u));
  EXPECT_EQ(SizeOf("q"), std::optional<uint64_t>(8u));
  EXPECT_EQ(SizeOf("Q"), std::optional<uint64_t>(8u));
}

TEST_P(GNUstepTypeEncodingParserTest, LongDouble) {
  // Unhandled by Apple's parser; an encoding mentioning one would otherwise
  // fail and take the whole method with it.
  EXPECT_EQ(Realize("D"), "long double");
}

TEST_P(GNUstepTypeEncodingParserTest, ObjCBuiltins) {
  EXPECT_EQ(Realize("@"), "id");
  EXPECT_EQ(Realize("#"), "Class");
  EXPECT_EQ(Realize(":"), "SEL");
  // A block. The extended form carries a signature nothing here needs.
  EXPECT_EQ(Realize("@?"), "id");
}

// Without a DeclVendor the class name cannot be resolved. That must degrade
// to `id`, not assert - libobjc2 encodings routinely name classes that are
// not realized.
TEST_P(GNUstepTypeEncodingParserTest, UnknownClassNameDegradesToId) {
  EXPECT_EQ(Realize("@\"NSString\""), "id");
  EXPECT_EQ(Realize("@\"NSString\"", /*for_expression=*/true), "id");
  EXPECT_EQ(Realize("@\"<NSCopying>\"", /*for_expression=*/true), "id");
  EXPECT_EQ(Realize("@\"NSFoo<NSCopying>\"", /*for_expression=*/true), "id");
}

TEST_P(GNUstepTypeEncodingParserTest, Pointers) {
  EXPECT_EQ(Realize("^i"), "int *");
  EXPECT_EQ(Realize("^^v"), "void **");
  EXPECT_EQ(Realize("^?"), "void *");
  EXPECT_EQ(Realize("ri"), "const int");
  EXPECT_EQ(Realize("^ri"), "const int *");
}

TEST_P(GNUstepTypeEncodingParserTest, Arrays) {
  EXPECT_EQ(Realize("[10i]"), "int[10]");
  EXPECT_EQ(SizeOf("[10i]"), std::optional<uint64_t>(40u));
  EXPECT_EQ(SizeOf("[3[4f]]"), std::optional<uint64_t>(48u));
}

TEST_P(GNUstepTypeEncodingParserTest, ArrayExtentDoesNotWrap) {
  // The encoding is inferior data, so an extent that does not fit must fail
  // rather than wrap: 10000000000 truncated to 32 bits is 1410065408, and an
  // array of that many ints would be accepted as if it were the real size.
  EXPECT_EQ(Realize("[10000000000i]"), "");
  EXPECT_EQ(Realize("[4294967296i]"), "");
  // One below the limit still parses, so the bound is not off by one.
  EXPECT_EQ(Realize("[4294967295i]"), "int[4294967295]");
}

TEST_P(GNUstepTypeEncodingParserTest, Structs) {
  EXPECT_EQ(Realize("{CGPoint=dd}"), "CGPoint");
  EXPECT_EQ(SizeOf("{CGPoint=dd}"), std::optional<uint64_t>(16u));
  // An opaque struct, which is how libobjc2 spells a forward declaration.
  EXPECT_EQ(Realize("{CGImage=}"), "CGImage");
  EXPECT_EQ(Realize("^{CGImage=}"), "CGImage *");
  EXPECT_EQ(Realize("(U=ic)"), "U");
}

// This is the encoding shape libobjc2 actually emits for a struct ivar,
// verified against the shipped library: members are expanded positionally
// with no field names.
TEST_P(GNUstepTypeEncodingParserTest, StructWithoutFieldNames) {
  EXPECT_EQ(Realize("{_NSRange=QQ}"), "_NSRange");
  EXPECT_EQ(SizeOf("{_NSRange=QQ}"), std::optional<uint64_t>(16u));
}

// A quoted string after '@' is a class name, never a field name: unlike
// Apple's runtime, clang's GNUstep output does not emit quoted field names,
// so peeking to disambiguate would mis-parse the `i` here as evidence that
// "NSString" was a field name.
TEST_P(GNUstepTypeEncodingParserTest, ClassNameInStructIsNotAFieldName) {
  // An object pointer followed by an int: the size follows the pointer
  // width, so LLP64 and LP64 agree here and only ILP32 differs.
  EXPECT_EQ(SizeOf("{Foo=@\"NSString\"i}"),
            std::optional<uint64_t>(GetParam().pointer_bits == 32 ? 8u : 16u));
}

TEST_P(GNUstepTypeEncodingParserTest, Bitfields) {
  // A bitfield outside a struct has nowhere to report its width.
  EXPECT_EQ(Realize("b3"), "");
  // Bitfields are given an unsigned int base type, so eight bits of them
  // still occupy one int rather than one byte.
  EXPECT_EQ(SizeOf("{Flags=b1b1b6}"), std::optional<uint64_t>(4u));
}

// clang emits these for declarations such as `- (oneway void)release`.
// Leaving them unhandled would drop the whole method.
TEST_P(GNUstepTypeEncodingParserTest, SkipsArgumentQualifiers) {
  EXPECT_EQ(Realize("Vv"), "void");
  EXPECT_EQ(Realize("ni"), "int");
  EXPECT_EQ(Realize("Ni"), "int");
  EXPECT_EQ(Realize("oi"), "int");
  EXPECT_EQ(Realize("Oi"), "int");
  EXPECT_EQ(Realize("Ri"), "int");
  // libobjc2's addition, for _Atomic.
  EXPECT_EQ(Realize("Ai"), "int");
  // 'r' is const, not an argument qualifier, and must keep its meaning.
  EXPECT_NE(Realize("ri"), "int");
}

// The same struct appearing twice must yield the same type rather than two
// separately laid out decls sharing a name.
TEST_P(GNUstepTypeEncodingParserTest, RecordsAreCached) {
  CompilerType first = m_parser->RealizeType(*m_ast_sp, "{CGPoint=dd}", false);
  CompilerType second = m_parser->RealizeType(*m_ast_sp, "{CGPoint=dd}", false);
  ASSERT_TRUE(first);
  EXPECT_EQ(first.GetOpaqueQualType(), second.GetOpaqueQualType());
}

// Two different types can legitimately share a tag - an opaque declaration
// and its definition - so the cache must not collapse them.
TEST_P(GNUstepTypeEncodingParserTest, SameTagDifferentBodyIsNotCached) {
  CompilerType opaque = m_parser->RealizeType(*m_ast_sp, "{Foo=}", false);
  CompilerType defined = m_parser->RealizeType(*m_ast_sp, "{Foo=ii}", false);
  ASSERT_TRUE(opaque);
  ASSERT_TRUE(defined);
  EXPECT_NE(opaque.GetOpaqueQualType(), defined.GetOpaqueQualType());
}

// Nothing here may crash, hang, or recurse without bound. The encoding comes
// from inferior memory, so it cannot be assumed well-formed.
TEST_P(GNUstepTypeEncodingParserTest, MalformedInputIsRejected) {
  for (llvm::StringRef bad :
       {"", "{", "{Foo=", "{Foo=ii", "[10", "[10i", "^", "b", "@\"", "(U=ic",
        "{Foo=@\"NSString", "]", ")", "}", "="}) {
    // No assertion on the result beyond "it returned"; the point is that it
    // terminates and does not trip an assertion inside clang.
    Realize(bad);
  }
}

TEST_P(GNUstepTypeEncodingParserTest, DeeplyNestedInputTerminates) {
  EXPECT_EQ(Realize(std::string(4096, '^') + "v"), "");
  EXPECT_EQ(Realize(std::string(1000, '{')), "");
}

// Truncating a valid encoding at every length is the cheapest way to reach
// the partial states a hand-written parser gets wrong.
TEST_P(GNUstepTypeEncodingParserTest, EveryTruncationTerminates) {
  const std::string full =
      "{Outer=@\"NSString\"^{Inner=ii}[4f]b3q{_NSRange=QQ}}";
  for (size_t n = 1; n <= full.size(); ++n)
    Realize(llvm::StringRef(full).take_front(n));
}

INSTANTIATE_TEST_SUITE_P(
    DataModels, GNUstepTypeEncodingParserTest,
    ::testing::Values(DataModel{"x86_64-pc-linux", 64, 64},
                      // Windows is LLP64: pointers are 64 bits, long is 32.
                      DataModel{"x86_64-pc-windows-msvc", 32, 64},
                      DataModel{"i386-pc-linux", 32, 32}),
    [](const ::testing::TestParamInfo<DataModel> &info) {
      std::string name = info.param.triple;
      for (char &c : name)
        if (!std::isalnum(static_cast<unsigned char>(c)))
          c = '_';
      return name;
    });

} // namespace
