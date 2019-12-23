//===-- MangledTest.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"

#include "lldb/Core/Mangled.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/SymbolContext.h"

#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

TEST(MangledTest, ResultForValidName) {
  ConstString MangledName("_ZN1a1b1cIiiiEEvm");
  Mangled TheMangled(MangledName);
  ConstString TheDemangled =
      TheMangled.GetDemangledName(eLanguageTypeC_plus_plus);

  ConstString ExpectedResult("void a::b::c<int, int, int>(unsigned long)");
  EXPECT_STREQ(ExpectedResult.GetCString(), TheDemangled.GetCString());
}

TEST(MangledTest, ResultForBlockInvocation) {
  ConstString MangledName("___Z1fU13block_pointerFviE_block_invoke");
  Mangled TheMangled(MangledName);
  ConstString TheDemangled =
      TheMangled.GetDemangledName(eLanguageTypeC_plus_plus);

  ConstString ExpectedResult(
      "invocation function for block in f(void (int) block_pointer)");
  EXPECT_STREQ(ExpectedResult.GetCString(), TheDemangled.GetCString());
}

TEST(MangledTest, EmptyForInvalidName) {
  ConstString MangledName("_ZN1a1b1cmxktpEEvm");
  Mangled TheMangled(MangledName);
  ConstString TheDemangled =
      TheMangled.GetDemangledName(eLanguageTypeC_plus_plus);

  EXPECT_STREQ("", TheDemangled.GetCString());
}

TEST(MangledTest, NameIndexes_FindFunctionSymbols) {
  SubsystemRAII<FileSystem, HostInfo, ObjectFileELF, SymbolFileSymtab>
      subsystems;

  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:      
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_EXEC
  Machine:         EM_X86_64
Sections:        
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    AddressAlign:    0x0000000000000010
    Size:            0x20
  - Name:            .anothertext
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x0000000000000010
    AddressAlign:    0x0000000000000010
    Size:            0x40
  - Name:            .data
    Type:            SHT_PROGBITS
    Flags:           [ SHF_WRITE, SHF_ALLOC ]
    Address:         0x00000000000000A8
    AddressAlign:    0x0000000000000004
    Content:         '01000000'
Symbols:
  - Name:            somedata
    Type:            STT_OBJECT
    Section:         .anothertext
    Value:           0x0000000000000045
    Binding:         STB_GLOBAL
  - Name:            main
    Type:            STT_FUNC
    Section:         .anothertext
    Value:           0x0000000000000010
    Size:            0x000000000000003F
    Binding:         STB_GLOBAL
  - Name:            _Z3foov
    Type:            STT_FUNC
    Section:         .text
    Size:            0x000000000000000D
    Binding:         STB_GLOBAL
  - Name:            puts@GLIBC_2.5
    Type:            STT_FUNC
    Section:         .text
    Size:            0x000000000000000D
    Binding:         STB_GLOBAL
  - Name:            puts@GLIBC_2.6
    Type:            STT_FUNC
    Section:         .text
    Size:            0x000000000000000D
    Binding:         STB_GLOBAL
  - Name:            _Z5annotv@VERSION3
    Type:            STT_FUNC
    Section:         .text
    Size:            0x000000000000000D
    Binding:         STB_GLOBAL
  - Name:            _ZN1AC2Ev
    Type:            STT_FUNC
    Section:         .text
    Size:            0x000000000000000D
    Binding:         STB_GLOBAL
  - Name:            _ZN1AD2Ev
    Type:            STT_FUNC
    Section:         .text
    Size:            0x000000000000000D
    Binding:         STB_GLOBAL
  - Name:            _ZN1A3barEv
    Type:            STT_FUNC
    Section:         .text
    Size:            0x000000000000000D
    Binding:         STB_GLOBAL
  - Name:            _ZGVZN4llvm4dbgsEvE7thestrm
    Type:            STT_FUNC
    Section:         .text
    Size:            0x000000000000000D
    Binding:         STB_GLOBAL
  - Name:            _ZZN4llvm4dbgsEvE7thestrm
    Type:            STT_FUNC
    Section:         .text
    Size:            0x000000000000000D
    Binding:         STB_GLOBAL
  - Name:            _ZTVN5clang4DeclE
    Type:            STT_FUNC
    Section:         .text
    Size:            0x000000000000000D
    Binding:         STB_GLOBAL
  - Name:            -[ObjCfoo]
    Type:            STT_FUNC
    Section:         .text
    Size:            0x000000000000000D
    Binding:         STB_GLOBAL
  - Name:            +[B ObjCbar(WithCategory)]
    Type:            STT_FUNC
    Section:         .text
    Size:            0x000000000000000D
    Binding:         STB_GLOBAL
  - Name:            _Z12undemangableEvx42
    Type:            STT_FUNC
    Section:         .text
    Size:            0x000000000000000D
    Binding:         STB_GLOBAL
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  ModuleSpec Spec{FileSpec(ExpectedFile->name())};
  auto M = std::make_shared<Module>(Spec);

  auto Count = [M](const char *Name, FunctionNameType Type) -> int {
    SymbolContextList SymList;
    M->FindFunctionSymbols(ConstString(Name), Type, SymList);
    return SymList.GetSize();
  };

  // Unmangled
  EXPECT_EQ(1, Count("main", eFunctionNameTypeFull));
  EXPECT_EQ(1, Count("main", eFunctionNameTypeBase));
  EXPECT_EQ(0, Count("main", eFunctionNameTypeMethod));

  // Itanium mangled
  EXPECT_EQ(1, Count("_Z3foov", eFunctionNameTypeFull));
  EXPECT_EQ(1, Count("_Z3foov", eFunctionNameTypeBase));
  EXPECT_EQ(1, Count("foo", eFunctionNameTypeBase));
  EXPECT_EQ(0, Count("foo", eFunctionNameTypeMethod));

  // Unmangled with linker annotation
  EXPECT_EQ(1, Count("puts@GLIBC_2.5", eFunctionNameTypeFull));
  EXPECT_EQ(1, Count("puts@GLIBC_2.6", eFunctionNameTypeFull));
  EXPECT_EQ(2, Count("puts", eFunctionNameTypeFull));
  EXPECT_EQ(2, Count("puts", eFunctionNameTypeBase));
  EXPECT_EQ(0, Count("puts", eFunctionNameTypeMethod));

  // Itanium mangled with linker annotation
  EXPECT_EQ(1, Count("_Z5annotv@VERSION3", eFunctionNameTypeFull));
  EXPECT_EQ(1, Count("_Z5annotv", eFunctionNameTypeFull));
  EXPECT_EQ(1, Count("_Z5annotv", eFunctionNameTypeBase));
  EXPECT_EQ(0, Count("annot", eFunctionNameTypeBase));
  EXPECT_EQ(0, Count("annot", eFunctionNameTypeMethod));

  // Itanium mangled ctor A::A()
  EXPECT_EQ(1, Count("_ZN1AC2Ev", eFunctionNameTypeFull));
  EXPECT_EQ(1, Count("_ZN1AC2Ev", eFunctionNameTypeBase));
  EXPECT_EQ(1, Count("A", eFunctionNameTypeMethod));
  EXPECT_EQ(0, Count("A", eFunctionNameTypeBase));

  // Itanium mangled dtor A::~A()
  EXPECT_EQ(1, Count("_ZN1AD2Ev", eFunctionNameTypeFull));
  EXPECT_EQ(1, Count("_ZN1AD2Ev", eFunctionNameTypeBase));
  EXPECT_EQ(1, Count("~A", eFunctionNameTypeMethod));
  EXPECT_EQ(0, Count("~A", eFunctionNameTypeBase));

  // Itanium mangled method A::bar()
  EXPECT_EQ(1, Count("_ZN1A3barEv", eFunctionNameTypeFull));
  EXPECT_EQ(1, Count("_ZN1A3barEv", eFunctionNameTypeBase));
  EXPECT_EQ(1, Count("bar", eFunctionNameTypeMethod));
  EXPECT_EQ(0, Count("bar", eFunctionNameTypeBase));

  // Itanium mangled names that are explicitly excluded from parsing
  EXPECT_EQ(1, Count("_ZGVZN4llvm4dbgsEvE7thestrm", eFunctionNameTypeFull));
  EXPECT_EQ(1, Count("_ZGVZN4llvm4dbgsEvE7thestrm", eFunctionNameTypeBase));
  EXPECT_EQ(0, Count("dbgs", eFunctionNameTypeMethod));
  EXPECT_EQ(0, Count("dbgs", eFunctionNameTypeBase));
  EXPECT_EQ(1, Count("_ZZN4llvm4dbgsEvE7thestrm", eFunctionNameTypeFull));
  EXPECT_EQ(1, Count("_ZZN4llvm4dbgsEvE7thestrm", eFunctionNameTypeBase));
  EXPECT_EQ(0, Count("dbgs", eFunctionNameTypeMethod));
  EXPECT_EQ(0, Count("dbgs", eFunctionNameTypeBase));
  EXPECT_EQ(1, Count("_ZTVN5clang4DeclE", eFunctionNameTypeFull));
  EXPECT_EQ(1, Count("_ZTVN5clang4DeclE", eFunctionNameTypeBase));
  EXPECT_EQ(0, Count("Decl", eFunctionNameTypeMethod));
  EXPECT_EQ(0, Count("Decl", eFunctionNameTypeBase));

  // ObjC mangled static
  EXPECT_EQ(1, Count("-[ObjCfoo]", eFunctionNameTypeFull));
  EXPECT_EQ(1, Count("-[ObjCfoo]", eFunctionNameTypeBase));
  EXPECT_EQ(0, Count("ObjCfoo", eFunctionNameTypeMethod));

  // ObjC mangled method with category
  EXPECT_EQ(1, Count("+[B ObjCbar(WithCategory)]", eFunctionNameTypeFull));
  EXPECT_EQ(1, Count("+[B ObjCbar(WithCategory)]", eFunctionNameTypeBase));
  EXPECT_EQ(0, Count("ObjCbar", eFunctionNameTypeMethod));

  // Invalid things: unable to decode but still possible to find by full name
  EXPECT_EQ(1, Count("_Z12undemangableEvx42", eFunctionNameTypeFull));
  EXPECT_EQ(1, Count("_Z12undemangableEvx42", eFunctionNameTypeBase));
  EXPECT_EQ(0, Count("_Z12undemangableEvx42", eFunctionNameTypeMethod));
  EXPECT_EQ(0, Count("undemangable", eFunctionNameTypeBase));
  EXPECT_EQ(0, Count("undemangable", eFunctionNameTypeMethod));
}
