//===- llvm/unittest/CodeGen/DIEHashTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/CodeGen/AsmPrinter/DIEHash.h"
#include "TestAsmPrinter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/CodeGen/DwarfStringPoolEntry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Host.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

// Test fixture
class DIEHashTest : public testing::Test {
public:
  BumpPtrAllocator Alloc;

private:
  StringMap<DwarfStringPoolEntry> Pool;
  std::unique_ptr<TestAsmPrinter> TestPrinter;

  void setupTestPrinter() {
    auto ExpectedTestPrinter = TestAsmPrinter::create(
        sys::getDefaultTargetTriple(), /*DwarfVersion=*/4, dwarf::DWARF32);
    ASSERT_THAT_EXPECTED(ExpectedTestPrinter, Succeeded());
    TestPrinter = std::move(ExpectedTestPrinter.get());
  }

public:
  DIEString getString(StringRef S) {
    DwarfStringPoolEntry Entry = {nullptr, 1, 1};
    return DIEString(DwarfStringPoolEntryRef(
        *Pool.insert(std::make_pair(S, Entry)).first, Entry.isIndexed()));
  }

  AsmPrinter *getAsmPrinter() {
    if (!TestPrinter)
      setupTestPrinter();
    return TestPrinter ? TestPrinter->getAP() : nullptr;
  }
};

TEST_F(DIEHashTest, Data1) {
  DIEHash Hash;
  DIE &Die = *DIE::get(Alloc, dwarf::DW_TAG_base_type);
  DIEInteger Size(4);
  Die.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, Size);
  uint64_t MD5Res = Hash.computeTypeSignature(Die);
  ASSERT_EQ(0x1AFE116E83701108ULL, MD5Res);
}

// struct {};
TEST_F(DIEHashTest, TrivialType) {
  DIE &Unnamed = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
  DIEInteger One(1);
  Unnamed.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, One);

  // Line and file number are ignored.
  Unnamed.addValue(Alloc, dwarf::DW_AT_decl_file, dwarf::DW_FORM_data1, One);
  Unnamed.addValue(Alloc, dwarf::DW_AT_decl_line, dwarf::DW_FORM_data1, One);
  uint64_t MD5Res = DIEHash().computeTypeSignature(Unnamed);

  // The exact same hash GCC produces for this DIE.
  ASSERT_EQ(0x715305ce6cfd9ad1ULL, MD5Res);
}

// struct foo { };
TEST_F(DIEHashTest, NamedType) {
  DIE &Foo = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
  DIEInteger One(1);
  DIEString FooStr = getString("foo");
  Foo.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, FooStr);
  Foo.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, One);

  uint64_t MD5Res = DIEHash().computeTypeSignature(Foo);

  // The exact same hash GCC produces for this DIE.
  ASSERT_EQ(0xd566dbd2ca5265ffULL, MD5Res);
}

// namespace space { struct foo { }; }
TEST_F(DIEHashTest, NamespacedType) {
  DIE &CU = *DIE::get(Alloc, dwarf::DW_TAG_compile_unit);

  auto Space = DIE::get(Alloc, dwarf::DW_TAG_namespace);
  DIEInteger One(1);
  DIEString SpaceStr = getString("space");
  Space->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, SpaceStr);
  // DW_AT_declaration is ignored.
  Space->addValue(Alloc, dwarf::DW_AT_declaration, dwarf::DW_FORM_flag_present,
                  One);
  // sibling?

  auto Foo = DIE::get(Alloc, dwarf::DW_TAG_structure_type);
  DIEString FooStr = getString("foo");
  Foo->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, FooStr);
  Foo->addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, One);

  DIE &N = *Foo;
  Space->addChild(std::move(Foo));
  CU.addChild(std::move(Space));

  uint64_t MD5Res = DIEHash().computeTypeSignature(N);

  // The exact same hash GCC produces for this DIE.
  ASSERT_EQ(0x7b80381fd17f1e33ULL, MD5Res);
}

// struct { int member; };
TEST_F(DIEHashTest, TypeWithMember) {
  DIE &Unnamed = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
  DIEInteger Four(4);
  Unnamed.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, Four);

  DIE &Int = *DIE::get(Alloc, dwarf::DW_TAG_base_type);
  DIEString IntStr = getString("int");
  Int.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, IntStr);
  Int.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, Four);
  DIEInteger Five(5);
  Int.addValue(Alloc, dwarf::DW_AT_encoding, dwarf::DW_FORM_data1, Five);

  DIEEntry IntRef(Int);

  auto Member = DIE::get(Alloc, dwarf::DW_TAG_member);
  DIEString MemberStr = getString("member");
  Member->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, MemberStr);
  DIEInteger Zero(0);
  Member->addValue(Alloc, dwarf::DW_AT_data_member_location,
                   dwarf::DW_FORM_data1, Zero);
  Member->addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4, IntRef);

  Unnamed.addChild(std::move(Member));

  uint64_t MD5Res = DIEHash().computeTypeSignature(Unnamed);

  ASSERT_EQ(0x5646aa436b7e07c6ULL, MD5Res);
}

// struct foo { int mem1, mem2; };
TEST_F(DIEHashTest, ReusedType) {
  DIE &Unnamed = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
  DIEInteger Eight(8);
  Unnamed.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, Eight);

  DIEInteger Four(4);
  DIE &Int = *DIE::get(Alloc, dwarf::DW_TAG_base_type);
  DIEString IntStr = getString("int");
  Int.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, IntStr);
  Int.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, Four);
  DIEInteger Five(5);
  Int.addValue(Alloc, dwarf::DW_AT_encoding, dwarf::DW_FORM_data1, Five);

  DIEEntry IntRef(Int);

  auto Mem1 = DIE::get(Alloc, dwarf::DW_TAG_member);
  DIEString Mem1Str = getString("mem1");
  Mem1->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, Mem1Str);
  DIEInteger Zero(0);
  Mem1->addValue(Alloc, dwarf::DW_AT_data_member_location, dwarf::DW_FORM_data1,
                 Zero);
  Mem1->addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4, IntRef);

  Unnamed.addChild(std::move(Mem1));

  auto Mem2 = DIE::get(Alloc, dwarf::DW_TAG_member);
  DIEString Mem2Str = getString("mem2");
  Mem2->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, Mem2Str);
  Mem2->addValue(Alloc, dwarf::DW_AT_data_member_location, dwarf::DW_FORM_data1,
                 Four);
  Mem2->addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4, IntRef);

  Unnamed.addChild(std::move(Mem2));

  uint64_t MD5Res = DIEHash().computeTypeSignature(Unnamed);

  ASSERT_EQ(0x3a7dc3ed7b76b2f8ULL, MD5Res);
}

// struct foo { static foo f; };
TEST_F(DIEHashTest, RecursiveType) {
  DIE &Foo = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
  DIEInteger One(1);
  Foo.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, One);
  DIEString FooStr = getString("foo");
  Foo.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, FooStr);

  auto Mem = DIE::get(Alloc, dwarf::DW_TAG_member);
  DIEString MemStr = getString("mem");
  Mem->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, MemStr);
  DIEEntry FooRef(Foo);
  Mem->addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4, FooRef);
  // DW_AT_external and DW_AT_declaration are ignored anyway, so skip them.

  Foo.addChild(std::move(Mem));

  uint64_t MD5Res = DIEHash().computeTypeSignature(Foo);

  ASSERT_EQ(0x73d8b25aef227b06ULL, MD5Res);
}

// struct foo { foo *mem; };
TEST_F(DIEHashTest, Pointer) {
  DIE &Foo = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
  DIEInteger Eight(8);
  Foo.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, Eight);
  DIEString FooStr = getString("foo");
  Foo.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, FooStr);

  auto Mem = DIE::get(Alloc, dwarf::DW_TAG_member);
  DIEString MemStr = getString("mem");
  Mem->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, MemStr);
  DIEInteger Zero(0);
  Mem->addValue(Alloc, dwarf::DW_AT_data_member_location, dwarf::DW_FORM_data1,
                Zero);

  DIE &FooPtr = *DIE::get(Alloc, dwarf::DW_TAG_pointer_type);
  FooPtr.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, Eight);
  DIEEntry FooRef(Foo);
  FooPtr.addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4, FooRef);

  DIEEntry FooPtrRef(FooPtr);
  Mem->addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4, FooPtrRef);

  Foo.addChild(std::move(Mem));

  uint64_t MD5Res = DIEHash().computeTypeSignature(Foo);

  ASSERT_EQ(0x74ea73862e8708d2ULL, MD5Res);
}

// struct foo { foo &mem; };
TEST_F(DIEHashTest, Reference) {
  DIE &Foo = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
  DIEInteger Eight(8);
  Foo.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, Eight);
  DIEString FooStr = getString("foo");
  Foo.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, FooStr);

  auto Mem = DIE::get(Alloc, dwarf::DW_TAG_member);
  DIEString MemStr = getString("mem");
  Mem->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, MemStr);
  DIEInteger Zero(0);
  Mem->addValue(Alloc, dwarf::DW_AT_data_member_location, dwarf::DW_FORM_data1,
                Zero);

  DIE &FooRef = *DIE::get(Alloc, dwarf::DW_TAG_reference_type);
  FooRef.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, Eight);
  DIEEntry FooEntry(Foo);
  FooRef.addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4, FooEntry);

  DIE &FooRefConst = *DIE::get(Alloc, dwarf::DW_TAG_const_type);
  DIEEntry FooRefRef(FooRef);
  FooRefConst.addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4,
                       FooRefRef);

  DIEEntry FooRefConstRef(FooRefConst);
  Mem->addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4, FooRefConstRef);

  Foo.addChild(std::move(Mem));

  uint64_t MD5Res = DIEHash().computeTypeSignature(Foo);

  ASSERT_EQ(0xa0b15f467ad4525bULL, MD5Res);
}

// struct foo { foo &&mem; };
TEST_F(DIEHashTest, RValueReference) {
  DIE &Foo = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
  DIEInteger Eight(8);
  Foo.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, Eight);
  DIEString FooStr = getString("foo");
  Foo.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, FooStr);

  auto Mem = DIE::get(Alloc, dwarf::DW_TAG_member);
  DIEString MemStr = getString("mem");
  Mem->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, MemStr);
  DIEInteger Zero(0);
  Mem->addValue(Alloc, dwarf::DW_AT_data_member_location, dwarf::DW_FORM_data1,
                Zero);

  DIE &FooRef = *DIE::get(Alloc, dwarf::DW_TAG_rvalue_reference_type);
  FooRef.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, Eight);
  DIEEntry FooEntry(Foo);
  FooRef.addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4, FooEntry);

  DIE &FooRefConst = *DIE::get(Alloc, dwarf::DW_TAG_const_type);
  DIEEntry FooRefRef(FooRef);
  FooRefConst.addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4,
                       FooRefRef);

  DIEEntry FooRefConstRef(FooRefConst);
  Mem->addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4, FooRefConstRef);

  Foo.addChild(std::move(Mem));

  uint64_t MD5Res = DIEHash().computeTypeSignature(Foo);

  ASSERT_EQ(0xad211c8c3b31e57ULL, MD5Res);
}

// struct foo { foo foo::*mem; };
TEST_F(DIEHashTest, PtrToMember) {
  DIE &Foo = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
  DIEInteger Eight(8);
  Foo.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, Eight);
  DIEString FooStr = getString("foo");
  Foo.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, FooStr);

  auto Mem = DIE::get(Alloc, dwarf::DW_TAG_member);
  DIEString MemStr = getString("mem");
  Mem->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, MemStr);
  DIEInteger Zero(0);
  Mem->addValue(Alloc, dwarf::DW_AT_data_member_location, dwarf::DW_FORM_data1,
                Zero);

  DIE &PtrToFooMem = *DIE::get(Alloc, dwarf::DW_TAG_ptr_to_member_type);
  DIEEntry FooEntry(Foo);
  PtrToFooMem.addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4, FooEntry);
  PtrToFooMem.addValue(Alloc, dwarf::DW_AT_containing_type, dwarf::DW_FORM_ref4,
                       FooEntry);

  DIEEntry PtrToFooMemRef(PtrToFooMem);
  Mem->addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4, PtrToFooMemRef);

  Foo.addChild(std::move(Mem));

  uint64_t MD5Res = DIEHash().computeTypeSignature(Foo);

  ASSERT_EQ(0x852e0c9ff7c04ebULL, MD5Res);
}

// Check that the hash for a pointer-to-member matches regardless of whether the
// pointed-to type is a declaration or a definition.
//
//   struct bar; // { };
//   struct foo { bar foo::*mem; };
TEST_F(DIEHashTest, PtrToMemberDeclDefMatch) {
  DIEInteger Zero(0);
  DIEInteger One(1);
  DIEInteger Eight(8);
  DIEString FooStr = getString("foo");
  DIEString BarStr = getString("bar");
  DIEString MemStr = getString("mem");
  uint64_t MD5ResDecl;
  {
    DIE &Bar = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
    Bar.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, BarStr);
    Bar.addValue(Alloc, dwarf::DW_AT_declaration, dwarf::DW_FORM_flag_present,
                 One);

    DIE &Foo = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
    Foo.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, Eight);
    Foo.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, FooStr);

    auto Mem = DIE::get(Alloc, dwarf::DW_TAG_member);
    Mem->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, MemStr);
    Mem->addValue(Alloc, dwarf::DW_AT_data_member_location,
                  dwarf::DW_FORM_data1, Zero);

    DIE &PtrToFooMem = *DIE::get(Alloc, dwarf::DW_TAG_ptr_to_member_type);
    DIEEntry BarEntry(Bar);
    PtrToFooMem.addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4,
                         BarEntry);
    DIEEntry FooEntry(Foo);
    PtrToFooMem.addValue(Alloc, dwarf::DW_AT_containing_type,
                         dwarf::DW_FORM_ref4, FooEntry);

    DIEEntry PtrToFooMemRef(PtrToFooMem);
    Mem->addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4,
                  PtrToFooMemRef);

    Foo.addChild(std::move(Mem));

    MD5ResDecl = DIEHash().computeTypeSignature(Foo);
  }
  uint64_t MD5ResDef;
  {
    DIE &Bar = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
    Bar.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, BarStr);
    Bar.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, One);

    DIE &Foo = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
    Foo.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, Eight);
    Foo.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, FooStr);

    auto Mem = DIE::get(Alloc, dwarf::DW_TAG_member);
    Mem->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, MemStr);
    Mem->addValue(Alloc, dwarf::DW_AT_data_member_location,
                  dwarf::DW_FORM_data1, Zero);

    DIE &PtrToFooMem = *DIE::get(Alloc, dwarf::DW_TAG_ptr_to_member_type);
    DIEEntry BarEntry(Bar);
    PtrToFooMem.addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4,
                         BarEntry);
    DIEEntry FooEntry(Foo);
    PtrToFooMem.addValue(Alloc, dwarf::DW_AT_containing_type,
                         dwarf::DW_FORM_ref4, FooEntry);

    DIEEntry PtrToFooMemRef(PtrToFooMem);
    Mem->addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4,
                  PtrToFooMemRef);

    Foo.addChild(std::move(Mem));

    MD5ResDef = DIEHash().computeTypeSignature(Foo);
  }
  ASSERT_EQ(MD5ResDef, MD5ResDecl);
}

// Check that the hash for a pointer-to-member matches regardless of whether the
// pointed-to type is a declaration or a definition.
//
//   struct bar; // { };
//   struct foo { bar bar::*mem; };
TEST_F(DIEHashTest, PtrToMemberDeclDefMisMatch) {
  DIEInteger Zero(0);
  DIEInteger One(1);
  DIEInteger Eight(8);
  DIEString FooStr = getString("foo");
  DIEString BarStr = getString("bar");
  DIEString MemStr = getString("mem");
  uint64_t MD5ResDecl;
  {
    DIE &Bar = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
    Bar.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, BarStr);
    Bar.addValue(Alloc, dwarf::DW_AT_declaration, dwarf::DW_FORM_flag_present,
                 One);

    DIE &Foo = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
    Foo.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, Eight);
    Foo.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, FooStr);

    auto Mem = DIE::get(Alloc, dwarf::DW_TAG_member);
    Mem->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, MemStr);
    Mem->addValue(Alloc, dwarf::DW_AT_data_member_location,
                  dwarf::DW_FORM_data1, Zero);

    DIE &PtrToFooMem = *DIE::get(Alloc, dwarf::DW_TAG_ptr_to_member_type);
    DIEEntry BarEntry(Bar);
    PtrToFooMem.addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4,
                         BarEntry);
    PtrToFooMem.addValue(Alloc, dwarf::DW_AT_containing_type,
                         dwarf::DW_FORM_ref4, BarEntry);

    DIEEntry PtrToFooMemRef(PtrToFooMem);
    Mem->addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4,
                  PtrToFooMemRef);

    Foo.addChild(std::move(Mem));

    MD5ResDecl = DIEHash().computeTypeSignature(Foo);
  }
  uint64_t MD5ResDef;
  {
    DIE &Bar = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
    Bar.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, BarStr);
    Bar.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, One);

    DIE &Foo = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
    Foo.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, Eight);
    Foo.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, FooStr);

    auto Mem = DIE::get(Alloc, dwarf::DW_TAG_member);
    Mem->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, MemStr);
    Mem->addValue(Alloc, dwarf::DW_AT_data_member_location,
                  dwarf::DW_FORM_data1, Zero);

    DIE &PtrToFooMem = *DIE::get(Alloc, dwarf::DW_TAG_ptr_to_member_type);
    DIEEntry BarEntry(Bar);
    PtrToFooMem.addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4,
                         BarEntry);
    PtrToFooMem.addValue(Alloc, dwarf::DW_AT_containing_type,
                         dwarf::DW_FORM_ref4, BarEntry);

    DIEEntry PtrToFooMemRef(PtrToFooMem);
    Mem->addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4,
                  PtrToFooMemRef);

    Foo.addChild(std::move(Mem));

    MD5ResDef = DIEHash().computeTypeSignature(Foo);
  }
  // FIXME: This seems to be a bug in the DWARF type hashing specification that
  // only uses the brief name hashing for types referenced via DW_AT_type. In
  // this case the type is referenced via DW_AT_containing_type and full hashing
  // causes a hash to differ when the containing type is a declaration in one TU
  // and a definition in another.
  ASSERT_NE(MD5ResDef, MD5ResDecl);
}

// struct { } a;
// struct foo { decltype(a) mem; };
TEST_F(DIEHashTest, RefUnnamedType) {
  DIEInteger Zero(0);
  DIEInteger One(1);
  DIEInteger Eight(8);
  DIEString FooStr = getString("foo");
  DIEString MemStr = getString("mem");

  DIE &Unnamed = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
  Unnamed.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, One);

  DIE &Foo = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
  Foo.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, Eight);
  Foo.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, FooStr);

  auto Mem = DIE::get(Alloc, dwarf::DW_TAG_member);
  Mem->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, MemStr);
  Mem->addValue(Alloc, dwarf::DW_AT_data_member_location, dwarf::DW_FORM_data1,
                Zero);

  DIE &UnnamedPtr = *DIE::get(Alloc, dwarf::DW_TAG_pointer_type);
  UnnamedPtr.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1,
                      Eight);
  DIEEntry UnnamedRef(Unnamed);
  UnnamedPtr.addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4,
                      UnnamedRef);

  DIEEntry UnnamedPtrRef(UnnamedPtr);
  Mem->addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4, UnnamedPtrRef);

  Foo.addChild(std::move(Mem));

  uint64_t MD5Res = DIEHash().computeTypeSignature(Foo);

  ASSERT_EQ(0x954e026f01c02529ULL, MD5Res);
}

// struct { struct foo { }; };
TEST_F(DIEHashTest, NestedType) {
  DIE &Unnamed = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
  DIEInteger One(1);
  Unnamed.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, One);

  auto Foo = DIE::get(Alloc, dwarf::DW_TAG_structure_type);
  DIEString FooStr = getString("foo");
  Foo->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, FooStr);
  Foo->addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, One);

  Unnamed.addChild(std::move(Foo));

  uint64_t MD5Res = DIEHash().computeTypeSignature(Unnamed);

  // The exact same hash GCC produces for this DIE.
  ASSERT_EQ(0xde8a3b7b43807f4aULL, MD5Res);
}

// struct { static void func(); };
TEST_F(DIEHashTest, MemberFunc) {
  DIE &Unnamed = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
  DIEInteger One(1);
  Unnamed.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, One);

  auto Func = DIE::get(Alloc, dwarf::DW_TAG_subprogram);
  DIEString FuncStr = getString("func");
  Func->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, FuncStr);

  Unnamed.addChild(std::move(Func));

  uint64_t MD5Res = DIEHash().computeTypeSignature(Unnamed);

  // The exact same hash GCC produces for this DIE.
  ASSERT_EQ(0xd36a1b6dfb604ba0ULL, MD5Res);
}

// struct A {
//   static void func();
// };
TEST_F(DIEHashTest, MemberFuncFlag) {
  DIE &A = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
  DIEInteger One(1);
  DIEString AStr = getString("A");
  A.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, AStr);
  A.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, One);
  A.addValue(Alloc, dwarf::DW_AT_decl_file, dwarf::DW_FORM_data1, One);
  A.addValue(Alloc, dwarf::DW_AT_decl_line, dwarf::DW_FORM_data1, One);

  auto Func = DIE::get(Alloc, dwarf::DW_TAG_subprogram);
  DIEString FuncStr = getString("func");
  DIEString FuncLinkage = getString("_ZN1A4funcEv");
  DIEInteger Two(2);
  Func->addValue(Alloc, dwarf::DW_AT_external, dwarf::DW_FORM_flag_present,
                 One);
  Func->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, FuncStr);
  Func->addValue(Alloc, dwarf::DW_AT_decl_file, dwarf::DW_FORM_data1, One);
  Func->addValue(Alloc, dwarf::DW_AT_decl_line, dwarf::DW_FORM_data1, Two);
  Func->addValue(Alloc, dwarf::DW_AT_linkage_name, dwarf::DW_FORM_strp,
                 FuncLinkage);
  Func->addValue(Alloc, dwarf::DW_AT_declaration, dwarf::DW_FORM_flag_present,
                 One);

  A.addChild(std::move(Func));

  uint64_t MD5Res = DIEHash().computeTypeSignature(A);

  // The exact same hash GCC produces for this DIE.
  ASSERT_EQ(0x8f78211ddce3df10ULL, MD5Res);
}

// Derived from:
// struct A {
//   const static int PI = -3;
// };
// A a;
TEST_F(DIEHashTest, MemberSdata) {
  DIE &A = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
  DIEInteger One(1);
  DIEString AStr = getString("A");
  A.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, AStr);
  A.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, One);
  A.addValue(Alloc, dwarf::DW_AT_decl_file, dwarf::DW_FORM_data1, One);
  A.addValue(Alloc, dwarf::DW_AT_decl_line, dwarf::DW_FORM_data1, One);

  DIEInteger Four(4);
  DIEInteger Five(5);
  DIEString FStr = getString("int");
  DIE &IntTyDIE = *DIE::get(Alloc, dwarf::DW_TAG_base_type);
  IntTyDIE.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, Four);
  IntTyDIE.addValue(Alloc, dwarf::DW_AT_encoding, dwarf::DW_FORM_data1, Five);
  IntTyDIE.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, FStr);

  DIEEntry IntTy(IntTyDIE);
  auto PITyDIE = DIE::get(Alloc, dwarf::DW_TAG_const_type);
  PITyDIE->addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4, IntTy);

  DIEEntry PITy(*PITyDIE);
  auto PI = DIE::get(Alloc, dwarf::DW_TAG_member);
  DIEString PIStr = getString("PI");
  DIEInteger Two(2);
  DIEInteger NegThree(-3);
  PI->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, PIStr);
  PI->addValue(Alloc, dwarf::DW_AT_decl_file, dwarf::DW_FORM_data1, One);
  PI->addValue(Alloc, dwarf::DW_AT_decl_line, dwarf::DW_FORM_data1, Two);
  PI->addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4, PITy);
  PI->addValue(Alloc, dwarf::DW_AT_external, dwarf::DW_FORM_flag_present, One);
  PI->addValue(Alloc, dwarf::DW_AT_declaration, dwarf::DW_FORM_flag_present,
               One);
  PI->addValue(Alloc, dwarf::DW_AT_const_value, dwarf::DW_FORM_sdata, NegThree);

  A.addChild(std::move(PI));

  uint64_t MD5Res = DIEHash().computeTypeSignature(A);
  ASSERT_EQ(0x9a216000dd3788a7ULL, MD5Res);
}

// Derived from:
// struct A {
//   const static float PI = 3.14;
// };
// A a;
TEST_F(DIEHashTest, MemberBlock) {
  if (!this->getAsmPrinter())
    GTEST_SKIP();

  DIE &A = *DIE::get(Alloc, dwarf::DW_TAG_structure_type);
  DIEInteger One(1);
  DIEString AStr = getString("A");
  A.addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, AStr);
  A.addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1, One);
  A.addValue(Alloc, dwarf::DW_AT_decl_file, dwarf::DW_FORM_data1, One);
  A.addValue(Alloc, dwarf::DW_AT_decl_line, dwarf::DW_FORM_data1, One);

  DIEInteger Four(4);
  DIEString FStr = getString("float");
  auto FloatTyDIE = DIE::get(Alloc, dwarf::DW_TAG_base_type);
  FloatTyDIE->addValue(Alloc, dwarf::DW_AT_byte_size, dwarf::DW_FORM_data1,
                       Four);
  FloatTyDIE->addValue(Alloc, dwarf::DW_AT_encoding, dwarf::DW_FORM_data1,
                       Four);
  FloatTyDIE->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, FStr);
  DIEEntry FloatTy(*FloatTyDIE);
  auto PITyDIE = DIE::get(Alloc, dwarf::DW_TAG_const_type);
  PITyDIE->addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4, FloatTy);

  DIEEntry PITy(*PITyDIE);
  auto PI = DIE::get(Alloc, dwarf::DW_TAG_member);
  DIEString PIStr = getString("PI");
  DIEInteger Two(2);
  PI->addValue(Alloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp, PIStr);
  PI->addValue(Alloc, dwarf::DW_AT_decl_file, dwarf::DW_FORM_data1, One);
  PI->addValue(Alloc, dwarf::DW_AT_decl_line, dwarf::DW_FORM_data1, Two);
  PI->addValue(Alloc, dwarf::DW_AT_type, dwarf::DW_FORM_ref4, PITy);
  PI->addValue(Alloc, dwarf::DW_AT_external, dwarf::DW_FORM_flag_present, One);
  PI->addValue(Alloc, dwarf::DW_AT_declaration, dwarf::DW_FORM_flag_present,
               One);

  DIEBlock PIBlock;
  DIEInteger Blk1(0xc3);
  DIEInteger Blk2(0xf5);
  DIEInteger Blk3(0x48);
  DIEInteger Blk4(0x40);

  PIBlock.addValue(Alloc, (dwarf::Attribute)0, dwarf::DW_FORM_data1, Blk1);
  PIBlock.addValue(Alloc, (dwarf::Attribute)0, dwarf::DW_FORM_data1, Blk2);
  PIBlock.addValue(Alloc, (dwarf::Attribute)0, dwarf::DW_FORM_data1, Blk3);
  PIBlock.addValue(Alloc, (dwarf::Attribute)0, dwarf::DW_FORM_data1, Blk4);

  PI->addValue(Alloc, dwarf::DW_AT_const_value, dwarf::DW_FORM_block1,
               &PIBlock);

  A.addChild(std::move(PI));

  uint64_t MD5Res = DIEHash(this->getAsmPrinter()).computeTypeSignature(A);
  ASSERT_EQ(0x493af53ad3d3f651ULL, MD5Res);
}
}
