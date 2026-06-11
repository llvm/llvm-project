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
