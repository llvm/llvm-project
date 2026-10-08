//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/Mach-O/MachOTrie.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/MachO.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <set>
#include <string>
#include <vector>

using namespace lldb;
using namespace lldb_private;
using namespace llvm::MachO;

namespace {

/// Number of bytes a value occupies as a minimal ULEB128.
unsigned ULEB128Len(uint64_t value) {
  unsigned len = 1;
  while (value >= 0x80) {
    value >>= 7;
    ++len;
  }
  return len;
}

void AppendULEB128(std::vector<uint8_t> &out, uint64_t value) {
  do {
    uint8_t byte = value & 0x7f;
    value >>= 7;
    if (value != 0)
      byte |= 0x80;
    out.push_back(byte);
  } while (value != 0);
}

/// Child node offsets are emitted as a fixed-width (padded) ULEB128 so that a
/// node's serialized size is independent of the offset values it stores. That
/// keeps the byte layout — and therefore every node offset — predictable when
/// hand-crafting tries below.
constexpr unsigned kOffsetWidth = 5;

void AppendOffset(std::vector<uint8_t> &out, uint64_t value) {
  for (unsigned i = 0; i < kOffsetWidth; ++i) {
    uint8_t byte = value & 0x7f;
    value >>= 7;
    if (i + 1 < kOffsetWidth)
      byte |= 0x80;
    out.push_back(byte);
  }
}

void AppendCStr(std::vector<uint8_t> &out, llvm::StringRef str) {
  out.insert(out.end(), str.begin(), str.end());
  out.push_back('\0');
}

/// A small builder that lays out a well-formed export trie.
class TrieBuilder {
public:
  struct Node {
    bool terminal = false;
    uint64_t flags = 0;
    uint64_t address = 0;
    uint64_t other = 0;      // dylib ordinal (re-export) or resolver address
    std::string import_name; // re-export target, including the leading '_'
    std::vector<std::pair<std::string, Node *>> children;

    uint64_t offset = 0; // assigned during layout
  };

  TrieBuilder() : m_root(Make()) {}

  Node *Root() { return m_root; }

  Node *Make() {
    m_nodes.push_back(std::make_unique<Node>());
    return m_nodes.back().get();
  }

  /// Add a terminal export child reached by edge \a edge.
  Node *AddExport(Node *parent, llvm::StringRef edge, uint64_t address,
                  uint64_t flags = 0, uint64_t other = 0) {
    Node *child = Make();
    child->terminal = true;
    child->flags = flags;
    child->address = address;
    child->other = other;
    parent->children.push_back({edge.str(), child});
    return child;
  }

  /// Add a terminal re-export child reached by edge \a edge.
  Node *AddReexport(Node *parent, llvm::StringRef edge, uint64_t dylib_ordinal,
                    llvm::StringRef import_name) {
    Node *child = Make();
    child->terminal = true;
    child->flags = EXPORT_SYMBOL_FLAGS_REEXPORT;
    child->other = dylib_ordinal;
    child->import_name = import_name.str();
    parent->children.push_back({edge.str(), child});
    return child;
  }

  /// Add an interior (non-terminal) child reached by edge \a edge.
  Node *AddEdge(Node *parent, llvm::StringRef edge) {
    Node *child = Make();
    parent->children.push_back({edge.str(), child});
    return child;
  }

  std::vector<uint8_t> Build() {
    AssignOffsets(m_root, 0);
    std::vector<uint8_t> out;
    Emit(m_root, out);
    return out;
  }

private:
  static std::vector<uint8_t> ExportInfo(const Node &node) {
    std::vector<uint8_t> info;
    if (!node.terminal)
      return info;
    AppendULEB128(info, node.flags);
    if (node.flags & EXPORT_SYMBOL_FLAGS_REEXPORT) {
      AppendULEB128(info, node.other); // dylib ordinal
      info.insert(info.end(), node.import_name.begin(), node.import_name.end());
      info.push_back('\0');
    } else {
      AppendULEB128(info, node.address);
      if (node.flags & EXPORT_SYMBOL_FLAGS_STUB_AND_RESOLVER)
        AppendULEB128(info, node.other); // resolver address
    }
    return info;
  }

  static uint64_t SelfSize(const Node &node) {
    std::vector<uint8_t> info = ExportInfo(node);
    uint64_t size = ULEB128Len(info.size()) + info.size() + /*childrenCount=*/1;
    for (const auto &child : node.children)
      size += child.first.size() + 1 + kOffsetWidth;
    return size;
  }

  uint64_t AssignOffsets(Node *node, uint64_t offset) {
    node->offset = offset;
    offset += SelfSize(*node);
    for (auto &child : node->children)
      offset = AssignOffsets(child.second, offset);
    return offset;
  }

  static void Emit(Node *node, std::vector<uint8_t> &out) {
    std::vector<uint8_t> info = ExportInfo(*node);
    AppendULEB128(out, info.size()); // terminalSize
    out.insert(out.end(), info.begin(), info.end());
    out.push_back(static_cast<uint8_t>(node->children.size()));
    for (auto &child : node->children) {
      AppendCStr(out, child.first);
      AppendOffset(out, child.second->offset);
    }
    for (auto &child : node->children)
      Emit(child.second, out);
  }

  std::vector<std::unique_ptr<Node>> m_nodes;
  Node *m_root;
};

struct ParseResult {
  bool ok = false;
  std::vector<TrieEntryWithOffset> reexports;
  std::vector<TrieEntryWithOffset> ext_symbols;
  std::set<lldb::addr_t> resolver_addresses;
};

ParseResult Parse(llvm::ArrayRef<uint8_t> bytes, bool is_arm = false,
                  lldb::addr_t text_seg_base_addr = LLDB_INVALID_ADDRESS) {
  DataExtractor data(bytes.data(), bytes.size(), lldb::eByteOrderLittle,
                     /*addr_size=*/8);
  ParseResult result;
  result.ok = ParseTrieEntries(data, is_arm, text_seg_base_addr,
                               result.resolver_addresses, result.reexports,
                               result.ext_symbols);
  return result;
}

} // namespace

TEST(MachOTrieTest, Empty) {
  ParseResult result = Parse({});
  EXPECT_TRUE(result.ok);
  EXPECT_TRUE(result.ext_symbols.empty());
  EXPECT_TRUE(result.reexports.empty());
}

TEST(MachOTrieTest, SingleExport) {
  TrieBuilder b;
  b.AddExport(b.Root(), "_foo", 0x1000);

  ParseResult result = Parse(b.Build());
  ASSERT_TRUE(result.ok);
  ASSERT_EQ(result.ext_symbols.size(), 1u);
  EXPECT_EQ(result.ext_symbols[0].entry.name.GetStringRef(), "foo");
  EXPECT_EQ(result.ext_symbols[0].entry.address, 0x1000u);
  EXPECT_TRUE(result.reexports.empty());
}

TEST(MachOTrieTest, ExportAddressBiasedByTextSegment) {
  TrieBuilder b;
  b.AddExport(b.Root(), "_foo", 0x1000);

  ParseResult result = Parse(b.Build(), /*is_arm=*/false,
                             /*text_seg_base_addr=*/0x4000);
  ASSERT_TRUE(result.ok);
  ASSERT_EQ(result.ext_symbols.size(), 1u);
  EXPECT_EQ(result.ext_symbols[0].entry.address, 0x5000u);
}

TEST(MachOTrieTest, Reexport) {
  TrieBuilder b;
  b.AddReexport(b.Root(), "_bar", /*dylib_ordinal=*/2, "_baz");

  ParseResult result = Parse(b.Build());
  ASSERT_TRUE(result.ok);
  EXPECT_TRUE(result.ext_symbols.empty());
  ASSERT_EQ(result.reexports.size(), 1u);
  EXPECT_EQ(result.reexports[0].entry.name.GetStringRef(), "bar");
  EXPECT_EQ(result.reexports[0].entry.import_name.GetStringRef(), "baz");
  EXPECT_EQ(result.reexports[0].entry.other, 2u);
}

TEST(MachOTrieTest, StubAndResolverCollectsResolverAddress) {
  TrieBuilder b;
  b.AddExport(b.Root(), "_foo", /*address=*/0x2000,
              EXPORT_SYMBOL_FLAGS_STUB_AND_RESOLVER, /*resolver=*/0x3000);

  ParseResult result = Parse(b.Build());
  ASSERT_TRUE(result.ok);
  // A stub-and-resolver entry is not externally visible, but its resolver
  // address is collected.
  EXPECT_TRUE(result.ext_symbols.empty());
  EXPECT_EQ(result.resolver_addresses, std::set<lldb::addr_t>{0x3000});
}

TEST(MachOTrieTest, ArmThumbBit) {
  TrieBuilder b;
  b.AddExport(b.Root(), "_foo", /*address=*/0x1001);

  ParseResult result = Parse(b.Build(), /*is_arm=*/true);
  ASSERT_TRUE(result.ok);
  ASSERT_EQ(result.ext_symbols.size(), 1u);
  EXPECT_EQ(result.ext_symbols[0].entry.address, 0x1000u);
  EXPECT_TRUE(result.ext_symbols[0].entry.flags & TRIE_SYMBOL_IS_THUMB);
}

TEST(MachOTrieTest, MultiLevel) {
  TrieBuilder b;
  TrieBuilder::Node *mid = b.AddEdge(b.Root(), "_");
  b.AddExport(mid, "foo", 0x1000);
  b.AddExport(mid, "bar", 0x2000);

  ParseResult result = Parse(b.Build());
  ASSERT_TRUE(result.ok);
  ASSERT_EQ(result.ext_symbols.size(), 2u);
  std::set<std::string> names;
  for (const auto &e : result.ext_symbols)
    names.insert(e.entry.name.GetStringRef().str());
  EXPECT_EQ(names, (std::set<std::string>{"foo", "bar"}));
}

TEST(MachOTrieTest, TerminalNodeWithChildren) {
  // "_foo" is itself a symbol and also a prefix of "_foobar".
  TrieBuilder b;
  TrieBuilder::Node *foo = b.AddExport(b.Root(), "_foo", 0x1000);
  b.AddExport(foo, "bar", 0x2000);

  ParseResult result = Parse(b.Build());
  ASSERT_TRUE(result.ok);
  ASSERT_EQ(result.ext_symbols.size(), 2u);
  std::set<std::string> names;
  for (const auto &e : result.ext_symbols)
    names.insert(e.entry.name.GetStringRef().str());
  EXPECT_EQ(names, (std::set<std::string>{"foo", "foobar"}));
}

TEST(MachOTrieTest, ManySiblings) {
  TrieBuilder b;
  TrieBuilder::Node *root = b.Root();
  b.AddExport(root, "_a", 0x1000);
  b.AddExport(root, "_b", 0x2000);
  b.AddExport(root, "_c", 0x3000);
  b.AddExport(root, "_d", 0x4000);
  b.AddExport(root, "_e", 0x5000);

  ParseResult result = Parse(b.Build());
  ASSERT_TRUE(result.ok);
  ASSERT_EQ(result.ext_symbols.size(), 5u);
  std::set<std::string> names;
  for (const auto &e : result.ext_symbols)
    names.insert(e.entry.name.GetStringRef().str());
  EXPECT_EQ(names, (std::set<std::string>{"a", "b", "c", "d", "e"}));
}

TEST(MachOTrieTest, EmptyEdgeLabel) {
  // An empty edge label contributes nothing to the symbol name.
  TrieBuilder b;
  TrieBuilder::Node *mid = b.AddEdge(b.Root(), "_foo");
  b.AddExport(mid, "", 0x1000);

  ParseResult result = Parse(b.Build());
  ASSERT_TRUE(result.ok);
  ASSERT_EQ(result.ext_symbols.size(), 1u);
  EXPECT_EQ(result.ext_symbols[0].entry.name.GetStringRef(), "foo");
}

TEST(MachOTrieTest, UnnamedRootChild) {
  // A symbol reached by a single-character "_" edge has a one-byte prefix, so
  // dropping the leading underscore leaves an empty name.
  TrieBuilder b;
  b.AddExport(b.Root(), "_", 0x1000);

  ParseResult result = Parse(b.Build());
  ASSERT_TRUE(result.ok);
  ASSERT_EQ(result.ext_symbols.size(), 1u);
  EXPECT_TRUE(result.ext_symbols[0].entry.name.GetStringRef().empty());
}

TEST(MachOTrieTest, LargeAddress) {
  // A multi-byte ULEB128 address round-trips intact.
  TrieBuilder b;
  b.AddExport(b.Root(), "_foo", 0x123456789ABULL);

  ParseResult result = Parse(b.Build());
  ASSERT_TRUE(result.ok);
  ASSERT_EQ(result.ext_symbols.size(), 1u);
  EXPECT_EQ(result.ext_symbols[0].entry.address, 0x123456789ABULL);
}

TEST(MachOTrieTest, ThumbBitNotSetWhenNotArm) {
  TrieBuilder b;
  b.AddExport(b.Root(), "_foo", /*address=*/0x1001);

  ParseResult result = Parse(b.Build(), /*is_arm=*/false);
  ASSERT_TRUE(result.ok);
  ASSERT_EQ(result.ext_symbols.size(), 1u);
  // Without is_arm the low bit is left untouched and the Thumb flag is unset.
  EXPECT_EQ(result.ext_symbols[0].entry.address, 0x1001u);
  EXPECT_FALSE(result.ext_symbols[0].entry.flags & TRIE_SYMBOL_IS_THUMB);
}

TEST(MachOTrieTest, ArmStubAndResolverMasksThumb) {
  TrieBuilder b;
  b.AddExport(b.Root(), "_foo", /*address=*/0x2000,
              EXPORT_SYMBOL_FLAGS_STUB_AND_RESOLVER, /*resolver=*/0x3001);

  ParseResult result = Parse(b.Build(), /*is_arm=*/true);
  ASSERT_TRUE(result.ok);
  // The resolver address has its Thumb bit stripped on ARM.
  EXPECT_EQ(result.resolver_addresses, std::set<lldb::addr_t>{0x3000});
}

TEST(MachOTrieTest, ResolverAddressBiasedByTextSegment) {
  TrieBuilder b;
  b.AddExport(b.Root(), "_foo", /*address=*/0x2000,
              EXPORT_SYMBOL_FLAGS_STUB_AND_RESOLVER, /*resolver=*/0x3000);

  ParseResult result = Parse(b.Build(), /*is_arm=*/false,
                             /*text_seg_base_addr=*/0x4000);
  ASSERT_TRUE(result.ok);
  EXPECT_EQ(result.resolver_addresses, std::set<lldb::addr_t>{0x7000});
}

TEST(MachOTrieTest, ReexportWithEmptyImportName) {
  TrieBuilder b;
  b.AddReexport(b.Root(), "_bar", /*dylib_ordinal=*/2, /*import_name=*/"");

  ParseResult result = Parse(b.Build());
  ASSERT_TRUE(result.ok);
  // A re-export with no import name is dropped from both tables.
  EXPECT_TRUE(result.reexports.empty());
  EXPECT_TRUE(result.ext_symbols.empty());
}

TEST(MachOTrieTest, MixedExportsAndReexports) {
  TrieBuilder b;
  TrieBuilder::Node *root = b.Root();
  b.AddExport(root, "_foo", 0x1000);
  b.AddReexport(root, "_bar", /*dylib_ordinal=*/3, "_baz");

  ParseResult result = Parse(b.Build());
  ASSERT_TRUE(result.ok);
  ASSERT_EQ(result.ext_symbols.size(), 1u);
  EXPECT_EQ(result.ext_symbols[0].entry.name.GetStringRef(), "foo");
  ASSERT_EQ(result.reexports.size(), 1u);
  EXPECT_EQ(result.reexports[0].entry.name.GetStringRef(), "bar");
  EXPECT_EQ(result.reexports[0].entry.import_name.GetStringRef(), "baz");
}

TEST(MachOTrieTest, MalformedChildOffsetZero) {
  // A child offset of 0 points back at the root, which is a cycle.
  std::vector<uint8_t> t;
  AppendULEB128(t, 0); // terminalSize
  t.push_back(1);      // childrenCount
  AppendCStr(t, "_foo");
  AppendOffset(t, 0); // childNodeOffset 0 -> points back at the root

  ParseResult result = Parse(t);
  EXPECT_FALSE(result.ok);
}

TEST(MachOTrieTest, MalformedSelfCycle) {
  // Root --"a"--> node1 (@9) --"b"--> node1 (points to itself).
  std::vector<uint8_t> t;
  AppendULEB128(t, 0); // root terminalSize
  t.push_back(1);      // root childrenCount
  AppendCStr(t, "a");
  AppendOffset(t, 9); // root SelfSize == 1 + 1 + 2 + 5 == 9
  ASSERT_EQ(t.size(), 9u);
  AppendULEB128(t, 0); // node1 terminalSize
  t.push_back(1);      // node1 childrenCount
  AppendCStr(t, "b");
  AppendOffset(t, 9); // points back at node1 -> cycle

  ParseResult result = Parse(t);
  EXPECT_FALSE(result.ok);
}

TEST(MachOTrieTest, MalformedBackEdgeCycle) {
  // Root(@0) --"a"--> n1(@9) --"b"--> n2(@18) --"c"--> n1 (ancestor).
  std::vector<uint8_t> t;
  AppendULEB128(t, 0);
  t.push_back(1);
  AppendCStr(t, "a");
  AppendOffset(t, 9);
  ASSERT_EQ(t.size(), 9u);
  AppendULEB128(t, 0);
  t.push_back(1);
  AppendCStr(t, "b");
  AppendOffset(t, 18);
  ASSERT_EQ(t.size(), 18u);
  AppendULEB128(t, 0);
  t.push_back(1);
  AppendCStr(t, "c");
  AppendOffset(t, 9); // back edge to n1

  ParseResult result = Parse(t);
  EXPECT_FALSE(result.ok);
}

TEST(MachOTrieTest, MalformedSharedSubtree) {
  // Root(@0) has two children that both point at the same node (@34). A real
  // trie reaches every node by exactly one path, so the visited-set rejects the
  // reuse just like a cycle.
  std::vector<uint8_t> t;
  AppendULEB128(t, 0); // root terminalSize
  t.push_back(2);      // root childrenCount
  AppendCStr(t, "a");
  AppendOffset(t, 16); // -> n1
  AppendCStr(t, "b");
  AppendOffset(t, 25); // -> n2
  ASSERT_EQ(t.size(), 16u);
  AppendULEB128(t, 0); // n1 terminalSize
  t.push_back(1);      // n1 childrenCount
  AppendCStr(t, "c");
  AppendOffset(t, 34); // -> shared
  ASSERT_EQ(t.size(), 25u);
  AppendULEB128(t, 0); // n2 terminalSize
  t.push_back(1);      // n2 childrenCount
  AppendCStr(t, "d");
  AppendOffset(t, 34); // -> shared (same node as n1's child)
  ASSERT_EQ(t.size(), 34u);
  std::vector<uint8_t> info;   // shared: a terminal export leaf
  AppendULEB128(info, 0);      // flags
  AppendULEB128(info, 0x1000); // address
  AppendULEB128(t, info.size());
  t.insert(t.end(), info.begin(), info.end());
  t.push_back(0); // shared childrenCount

  ParseResult result = Parse(t);
  EXPECT_FALSE(result.ok);
}

TEST(MachOTrieTest, MalformedUnterminatedEdgeString) {
  std::vector<uint8_t> t;
  AppendULEB128(t, 0); // terminalSize
  t.push_back(1);      // childrenCount
  t.push_back('a');    // edge bytes with no null terminator before EOF
  t.push_back('b');
  t.push_back('c');

  ParseResult result = Parse(t);
  EXPECT_FALSE(result.ok);
}

TEST(MachOTrieTest, MalformedExcessChildrenCount) {
  std::vector<uint8_t> t;
  AppendULEB128(t, 0); // terminalSize
  t.push_back(5);      // claims five children
  AppendCStr(t, "a");
  AppendOffset(t, 1000); // first child offset, out of range and tolerated
  // ...but no data for the remaining four claimed children.

  ParseResult result = Parse(t);
  EXPECT_FALSE(result.ok);
}

TEST(MachOTrieTest, MalformedTruncatedTerminalSize) {
  // A lone ULEB128 continuation byte with no following byte.
  std::vector<uint8_t> t = {0x80};

  ParseResult result = Parse(t);
  EXPECT_TRUE(result.ok); // bounds-safe: decodes to 0, no children
  EXPECT_TRUE(result.ext_symbols.empty());
}

TEST(MachOTrieTest, ChildOffsetOutOfRange) {
  // A child node offset past the end of the data is ignored, not followed.
  std::vector<uint8_t> t;
  AppendULEB128(t, 0); // terminalSize
  t.push_back(1);      // childrenCount
  AppendCStr(t, "_foo");
  AppendOffset(t, 1000); // child node offset well past the end

  ParseResult result = Parse(t);
  EXPECT_TRUE(result.ok);
  EXPECT_TRUE(result.ext_symbols.empty());
}

TEST(MachOTrieTest, OversizedEdgeLabelIsRejected) {
  // A corrupt export trie can encode an edge label far longer than any real
  // symbol name.  ParseTrieEntries appends every edge label onto the running
  // symbol name without a length bound, so such a label drives an unbounded
  // allocation.  The parser must treat an implausibly long name as corrupt data
  // and bail instead of accepting it.
  constexpr size_t kOversizedLabelLen = 8 * 1024 * 1024;
  TrieBuilder b;
  b.AddExport(b.Root(), std::string(kOversizedLabelLen, 'A'), 0x1000);

  ParseResult result = Parse(b.Build());
  EXPECT_FALSE(result.ok);
  EXPECT_TRUE(result.ext_symbols.empty());
}
