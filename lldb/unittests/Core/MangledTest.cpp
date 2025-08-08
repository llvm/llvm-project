//===-- MangledTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Language/CPlusPlus/CPlusPlusLanguage.h"
#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"

#include "lldb/Core/DemangledNameInfo.h"
#include "lldb/Core/Mangled.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/SymbolContext.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

#include <cstdlib>
#include <memory>

using namespace lldb;
using namespace lldb_private;

/// Custom deleter to use with unique_ptr.
///
/// Usage:
/// \code{.cpp}
///
/// auto OB =
///     std::unique_ptr<TrackingOutputBuffer, TrackingOutputBufferDeleter>(
///         new TrackingOutputBuffer());
///
/// \endcode
struct TrackingOutputBufferDeleter {
  void operator()(TrackingOutputBuffer *TOB) {
    if (!TOB)
      return;
    std::free(TOB->getBuffer());
    delete TOB;
  }
};

TEST(MangledTest, ResultForValidName) {
  ConstString MangledName("_ZN1a1b1cIiiiEEvm");
  Mangled TheMangled(MangledName);
  ConstString TheDemangled = TheMangled.GetDemangledName();

  ConstString ExpectedResult("void a::b::c<int, int, int>(unsigned long)");
  EXPECT_STREQ(ExpectedResult.GetCString(), TheDemangled.GetCString());
}

TEST(MangledTest, ResultForBlockInvocation) {
  ConstString MangledName("___Z1fU13block_pointerFviE_block_invoke");
  Mangled TheMangled(MangledName);
  ConstString TheDemangled = TheMangled.GetDemangledName();

  ConstString ExpectedResult(
      "invocation function for block in f(void (int) block_pointer)");
  EXPECT_STREQ(ExpectedResult.GetCString(), TheDemangled.GetCString());
}

TEST(MangledTest, EmptyForInvalidName) {
  ConstString MangledName("_ZN1a1b1cmxktpEEvm");
  Mangled TheMangled(MangledName);
  ConstString TheDemangled = TheMangled.GetDemangledName();

  EXPECT_STREQ("", TheDemangled.GetCString());
}

TEST(MangledTest, ResultForValidRustV0Name) {
  ConstString mangled_name("_RNvC1a4main");
  Mangled the_mangled(mangled_name);
  ConstString the_demangled = the_mangled.GetDemangledName();

  ConstString expected_result("a::main");
  EXPECT_STREQ(expected_result.GetCString(), the_demangled.GetCString());
}

TEST(MangledTest, EmptyForInvalidRustV0Name) {
  ConstString mangled_name("_RRR");
  Mangled the_mangled(mangled_name);
  ConstString the_demangled = the_mangled.GetDemangledName();

  EXPECT_STREQ("", the_demangled.GetCString());
}

TEST(MangledTest, ResultForValidDLangName) {
  ConstString mangled_name("_Dmain");
  Mangled the_mangled(mangled_name);
  ConstString the_demangled = the_mangled.GetDemangledName();

  ConstString expected_result("D main");
  EXPECT_STREQ(expected_result.GetCString(), the_demangled.GetCString());
}

TEST(MangledTest, SameForInvalidDLangPrefixedName) {
  ConstString mangled_name("_DDD");
  Mangled the_mangled(mangled_name);
  ConstString the_demangled = the_mangled.GetDemangledName();

  EXPECT_STREQ("_DDD", the_demangled.GetCString());
}

TEST(MangledTest, RecognizeSwiftMangledNames) {
  llvm::StringRef valid_swift_mangled_names[] = {
      "_TtC4main7MyClass",   // Mangled objc class name
      "_TtP4main3Foo_",      // Mangld objc protocol name
      "$s4main3BarCACycfC",  // Mangled name
      "_$s4main3BarCACycfC", // Mangled name with leading underscore
      "$S4main3BarCACycfC",  // Older swift mangled name
      "_$S4main3BarCACycfC", // Older swift mangled name
                             // with leading underscore
      // Mangled swift filename
      "@__swiftmacro_4main16FunVariableNames9OptionSetfMm_.swift",
  };

  for (llvm::StringRef mangled : valid_swift_mangled_names)
    EXPECT_EQ(Mangled::GetManglingScheme(mangled),
              Mangled::eManglingSchemeSwift);
}

TEST(MangledTest, BoolConversionOperator) {
  {
    ConstString MangledName("_ZN1a1b1cIiiiEEvm");
    Mangled TheMangled(MangledName);
    EXPECT_EQ(true, bool(TheMangled));
    EXPECT_EQ(false, !TheMangled);
  }
  {
    ConstString UnmangledName("puts");
    Mangled TheMangled(UnmangledName);
    EXPECT_EQ(true, bool(TheMangled));
    EXPECT_EQ(false, !TheMangled);
  }
  {
    Mangled TheMangled{};
    EXPECT_EQ(false, bool(TheMangled));
    EXPECT_EQ(true, !TheMangled);
  }
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

  auto M = std::make_shared<Module>(ExpectedFile->moduleSpec());

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

static bool NameInfoEquals(const DemangledNameInfo &lhs,
                           const DemangledNameInfo &rhs) {
  return std::tie(lhs.BasenameRange, lhs.ArgumentsRange, lhs.ScopeRange,
                  lhs.QualifiersRange) ==
         std::tie(rhs.BasenameRange, rhs.ArgumentsRange, rhs.ScopeRange,
                  rhs.QualifiersRange);
}

TEST(MangledTest, DemangledNameInfo_SetMangledResets) {
  Mangled mangled;
  EXPECT_EQ(mangled.GetDemangledInfo(), std::nullopt);

  mangled.SetMangledName(ConstString("_Z3foov"));
  ASSERT_TRUE(mangled);

  auto info1 = mangled.GetDemangledInfo();
  EXPECT_NE(info1, std::nullopt);
  EXPECT_TRUE(info1->hasBasename());

  mangled.SetMangledName(ConstString("_Z4funcv"));

  // Should have re-calculated demangled-info since mangled name changed.
  auto info2 = mangled.GetDemangledInfo();
  ASSERT_NE(info2, std::nullopt);
  EXPECT_TRUE(info2->hasBasename());

  EXPECT_FALSE(NameInfoEquals(info1.value(), info2.value()));
  EXPECT_EQ(mangled.GetDemangledName(), "func()");
}

TEST(MangledTest, DemangledNameInfo_SetDemangledResets) {
  Mangled mangled("_Z3foov");
  ASSERT_TRUE(mangled);

  mangled.SetDemangledName(ConstString(""));

  // Mangled name hasn't changed, so GetDemangledInfo causes re-demangling
  // of previously set mangled name.
  EXPECT_NE(mangled.GetDemangledInfo(), std::nullopt);
  EXPECT_EQ(mangled.GetDemangledName(), "foo()");
}

TEST(MangledTest, DemangledNameInfo_Clear) {
  Mangled mangled("_Z3foov");
  ASSERT_TRUE(mangled);
  EXPECT_NE(mangled.GetDemangledInfo(), std::nullopt);

  mangled.Clear();

  EXPECT_EQ(mangled.GetDemangledInfo(), std::nullopt);
}

TEST(MangledTest, DemangledNameInfo_SetValue) {
  Mangled mangled("_Z4funcv");
  ASSERT_TRUE(mangled);

  auto demangled_func = mangled.GetDemangledInfo();

  // SetValue(mangled) resets demangled-info
  mangled.SetValue(ConstString("_Z3foov"));
  auto demangled_foo = mangled.GetDemangledInfo();
  EXPECT_NE(demangled_foo, std::nullopt);
  EXPECT_FALSE(NameInfoEquals(demangled_foo.value(), demangled_func.value()));

  // SetValue(demangled) resets demangled-info
  mangled.SetValue(ConstString("_Z4funcv"));
  EXPECT_TRUE(NameInfoEquals(mangled.GetDemangledInfo().value(),
                             demangled_func.value()));

  // SetValue(empty) resets demangled-info
  mangled.SetValue(ConstString());
  EXPECT_EQ(mangled.GetDemangledInfo(), std::nullopt);

  // Demangling invalid mangled name will set demangled-info
  // (without a valid basename).
  mangled.SetValue(ConstString("_Zinvalid"));
  ASSERT_NE(mangled.GetDemangledInfo(), std::nullopt);
  EXPECT_FALSE(mangled.GetDemangledInfo()->hasBasename());
}

struct DemanglingPartsTestCase {
  const char *mangled;
  DemangledNameInfo expected_info;
  std::string_view basename;
  std::string_view scope;
  std::string_view qualifiers;
  bool valid_basename = true;
};

DemanglingPartsTestCase g_demangling_parts_test_cases[] = {
    // clang-format off
   { "_ZNVKO3BarIN2ns3QuxIiEEE1CIPFi3FooIS_IiES6_EEE6methodIS6_EENS5_IT_SC_E5InnerIiEESD_SD_",
     {
       /*.BasenameRange=*/{92, 98}, /*.TemplateArgumentsRange=*/{98, 108}, /*.ScopeRange=*/{36, 92},
       /*.ArgumentsRange=*/{108, 158}, /*.QualifiersRange=*/{158, 176}, /*.NameQualifiersRange=*/{0, 0},
       /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{0, 0}
     },
     /*.basename=*/"method",
     /*.scope=*/"Bar<ns::Qux<int>>::C<int (*)(Foo<Bar<int>, Bar<int>>)>::",
     /*.qualifiers=*/" const volatile &&"
   },
   { "_Z7getFuncIfEPFiiiET_",
     {
       /*.BasenameRange=*/{6, 13}, /*.TemplateArgumentsRange=*/{13, 20}, /*.ScopeRange=*/{6, 6},
       /*.ArgumentsRange=*/{20, 27}, /*.QualifiersRange=*/{38, 38}, /*.NameQualifiersRange=*/{0, 0},
       /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{0, 0}
     },
     /*.basename=*/"getFunc",
     /*.scope=*/"",
     /*.qualifiers=*/""
   },
   { "_ZN1f1b1c1gEv",
     {
       /*.BasenameRange=*/{9, 10}, /*.TemplateArgumentsRange=*/{0, 0}, /*.ScopeRange=*/{0, 9},
       /*.ArgumentsRange=*/{10, 12}, /*.QualifiersRange=*/{12, 12}, /*.NameQualifiersRange=*/{0, 0},
       /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{0, 0}
     },
     /*.basename=*/"g",
     /*.scope=*/"f::b::c::",
     /*.qualifiers=*/""
   },
   { "_ZN5test73fD1IiEEDTcmtlNS_1DEL_ZNS_1bEEEcvT__EES2_",
     {
       /*.BasenameRange=*/{45, 48}, /*.TemplateArgumentsRange=*/{48, 53}, /*.ScopeRange=*/{38, 45},
       /*.ArgumentsRange=*/{53, 58}, /*.QualifiersRange=*/{58, 58}, /*.NameQualifiersRange=*/{0, 0},
       /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{0, 0}
     },
     /*.basename=*/"fD1",
     /*.scope=*/"test7::",
     /*.qualifiers=*/""
   },
   { "_ZN5test73fD1IiEEDTcmtlNS_1DEL_ZNS_1bINDT1cE1dEEEEEcvT__EES2_",
     {
       /*.BasenameRange=*/{61, 64}, /*.TemplateArgumentsRange=*/{64, 69},/*.ScopeRange=*/{54, 61},
       /*.ArgumentsRange=*/{69, 79}, /*.QualifiersRange=*/{79, 79}, /*.NameQualifiersRange=*/{0, 0},
       /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{0, 0}
     },
     /*.basename=*/"fD1",
     /*.scope=*/"test7::",
     /*.qualifiers=*/""
   },
   { "_ZN5test7INDT1cE1dINDT1cE1dEEEE3fD1INDT1cE1dINDT1cE1dEEEEEDTcmtlNS_1DEL_ZNS_1bINDT1cE1dEEEEEcvT__EES2_",
     {
       /*.BasenameRange=*/{120, 123}, /*.TemplateArgumentsRange=*/{123, 155}, /*.ScopeRange=*/{81, 120},
       /*.ArgumentsRange=*/{155, 168}, /*.QualifiersRange=*/{168, 168}, /*.NameQualifiersRange=*/{0, 0},
       /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{0, 0}
     },
     /*.basename=*/"fD1",
     /*.scope=*/"test7<decltype(c)::d<decltype(c)::d>>::",
     /*.qualifiers=*/""
   },
   { "_ZN8nlohmann16json_abi_v3_11_310basic_jsonINSt3__13mapENS2_6vectorENS2_12basic_stringIcNS2_11char_traitsIcEENS2_9allocatorIcEEEEbxydS8_NS0_14adl_serializerENS4_IhNS8_IhEEEEvE5parseIRA29_KcEESE_OT_NS2_8functionIFbiNS0_6detail13parse_event_tERSE_EEEbb",
     {
       /*.BasenameRange=*/{687, 692}, /*.TemplateArgumentsRange=*/{692, 713}, /*.ScopeRange=*/{343, 687},
       /*.ArgumentsRange=*/{713, 1174}, /*.QualifiersRange=*/{1174, 1174}, /*.NameQualifiersRange=*/{0, 0},
       /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{0, 0}
     },
     /*.basename=*/"parse",
     /*.scope=*/"nlohmann::json_abi_v3_11_3::basic_json<std::__1::map, std::__1::vector, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, bool, long long, unsigned long long, double, std::__1::allocator, nlohmann::json_abi_v3_11_3::adl_serializer, std::__1::vector<unsigned char, std::__1::allocator<unsigned char>>, void>::",
     /*.qualifiers=*/""
   },
   { "_ZN8nlohmann16json_abi_v3_11_310basic_jsonINSt3__13mapENS2_6vectorENS2_12basic_stringIcNS2_11char_traitsIcEENS2_9allocatorIcEEEEbxydS8_NS0_14adl_serializerENS4_IhNS8_IhEEEEvEC1EDn",
     {
       /*.BasenameRange=*/{344, 354}, /*.TemplateArgumentsRange=*/{0, 0}, /*.ScopeRange=*/{0, 344},
       /*.ArgumentsRange=*/{354, 370}, /*.QualifiersRange=*/{370, 370}, /*.NameQualifiersRange=*/{0, 0},
       /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{0, 0}
     },
     /*.basename=*/"basic_json",
     /*.scope=*/"nlohmann::json_abi_v3_11_3::basic_json<std::__1::map, std::__1::vector, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, bool, long long, unsigned long long, double, std::__1::allocator, nlohmann::json_abi_v3_11_3::adl_serializer, std::__1::vector<unsigned char, std::__1::allocator<unsigned char>>, void>::",
     /*.qualifiers=*/""
   },
   { "_Z3fppIiEPFPFvvEiEf",
     {
       /*.BasenameRange=*/{10, 13}, /*.TemplateArgumentsRange=*/{13, 18}, /*.ScopeRange=*/{10, 10},
      /*.ArgumentsRange=*/{18, 25}, /*.QualifiersRange=*/{34,34}, /*.NameQualifiersRange=*/{0, 0},
      /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{0, 0}
     },
     /*.basename=*/"fpp",
     /*.scope=*/"",
     /*.qualifiers=*/""
   },
   { "_Z3fppIiEPFPFvvEN2ns3FooIiEEEf",
     {
       /*.BasenameRange=*/{10, 13}, /*.TemplateArgumentsRange=*/{13, 18}, /*.ScopeRange=*/{10, 10},
       /*.ArgumentsRange=*/{18, 25}, /*.QualifiersRange=*/{43, 43}, /*.NameQualifiersRange=*/{0, 0},
       /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{0, 0}
     },
     /*.basename=*/"fpp",
     /*.scope=*/"",
     /*.qualifiers=*/""
   },
   { "_Z3fppIiEPFPFvPFN2ns3FooIiEENS2_3BarIfE3QuxEEEPFS2_S2_EEf",
     {
       /*.BasenameRange=*/{10, 13}, /*.TemplateArgumentsRange=*/{13, 18}, /*.ScopeRange=*/{10, 10},
       /*.ArgumentsRange=*/{18, 25}, /*.QualifiersRange=*/{108, 108}, /*.NameQualifiersRange=*/{0, 0},
       /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{0, 0}
     },
     /*.basename=*/"fpp",
     /*.scope=*/"",
     /*.qualifiers=*/""
   },
   { "_ZN2ns8HasFuncsINS_3FooINS1_IiE3BarIfE3QuxEEEE3fppIiEEPFPFvvEiEf",
     {
       /*.BasenameRange=*/{64, 67}, /*.TemplateArgumentsRange=*/{67, 72}, /*.ScopeRange=*/{10, 64},
       /*.ArgumentsRange=*/{72, 79}, /*.QualifiersRange=*/{88, 88}, /*.NameQualifiersRange=*/{0, 0},
       /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{0, 0}
     },
     /*.basename=*/"fpp",
     /*.scope=*/"ns::HasFuncs<ns::Foo<ns::Foo<int>::Bar<float>::Qux>>::",
     /*.qualifiers=*/""
   },
   { "_ZN2ns8HasFuncsINS_3FooINS1_IiE3BarIfE3QuxEEEE3fppIiEEPFPFvvES2_Ef",
     {
       /*.BasenameRange=*/{64, 67}, /*.TemplateArgumentsRange=*/{67, 72}, /*.ScopeRange=*/{10, 64},
       /*.ArgumentsRange=*/{72, 79}, /*.QualifiersRange=*/{97, 97}, /*.NameQualifiersRange=*/{0, 0},
       /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{0, 0}
     },
     /*.basename=*/"fpp",
     /*.scope=*/"ns::HasFuncs<ns::Foo<ns::Foo<int>::Bar<float>::Qux>>::",
     /*.qualifiers=*/"",
   },
   { "_ZN2ns8HasFuncsINS_3FooINS1_IiE3BarIfE3QuxEEEE3fppIiEEPFPFvPFS2_S5_EEPFS2_S2_EEf",
     {
       /*.BasenameRange=*/{64, 67}, /*.TemplateArgumentsRange=*/{67, 72}, /*.ScopeRange=*/{10, 64},
       /*.ArgumentsRange=*/{72, 79}, /*.QualifiersRange=*/{162, 162}, /*.NameQualifiersRange=*/{0, 0},
       /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{0, 0}
     },
     /*.basename=*/"fpp",
     /*.scope=*/"ns::HasFuncs<ns::Foo<ns::Foo<int>::Bar<float>::Qux>>::",
     /*.qualifiers=*/"",
   },
   { "_ZNKO2ns3ns23Bar3fooIiEEPFPFNS0_3FooIiEEiENS3_IfEEEi",
     {
       /*.BasenameRange=*/{37, 40}, /*.TemplateArgumentsRange=*/{40, 45}, /*.ScopeRange=*/{23, 37},
       /*.ArgumentsRange=*/{45, 50}, /*.QualifiersRange=*/{78, 87}, /*.NameQualifiersRange=*/{0, 0},
       /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{0, 0}
     },
     /*.basename=*/"foo",
     /*.scope=*/"ns::ns2::Bar::",
     /*.qualifiers=*/" const &&",
   },
   { "_ZTV11ImageLoader",
     {
       /*.BasenameRange=*/{0, 0}, /*.TemplateArgumentsRange=*/{0, 0}, /*.ScopeRange=*/{0, 0},
       /*.ArgumentsRange=*/{0, 0}, /*.QualifiersRange=*/{0, 0}, /*.NameQualifiersRange=*/{0, 0},
       /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{0, 0}
     },
     /*.basename=*/"",
     /*.scope=*/"",
     /*.qualifiers=*/"",
     /*.valid_basename=*/false
   },
   { "___ZNK5dyld313MachOAnalyzer18forEachInitializerER11DiagnosticsRKNS0_15VMAddrConverterEU13block_pointerFvjEPKv_block_invoke.204",
     {
       /*.BasenameRange=*/{55, 73}, /*.TemplateArgumentsRange=*/{0, 0}, /*.ScopeRange=*/{33, 55},
       /*.ArgumentsRange=*/{73, 181}, /*.QualifiersRange=*/{181, 187}, /*.NameQualifiersRange=*/{0, 0},
       /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{0, 0}
     },
     /*.basename=*/"forEachInitializer",
     /*.scope=*/"dyld3::MachOAnalyzer::",
     /*.qualifiers=*/" const",
   },
   { "_ZZN5dyld45startEPNS_10KernelArgsEPvS2_ENK3$_1clEv",
     {
       /*.BasenameRange=*/{53, 63}, /*.TemplateArgumentsRange=*/{0, 0}, /*.ScopeRange=*/{0, 53},
       /*.ArgumentsRange=*/{63, 65}, /*.QualifiersRange=*/{65, 71}, /*.NameQualifiersRange=*/{0, 0},
       /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{0, 0}
     },
     /*.basename=*/"operator()",
     /*.scope=*/"dyld4::start(dyld4::KernelArgs*, void*, void*)::$_1::",
     /*.qualifiers=*/" const",
   },
   { "_ZZNK5dyld46Loader38runInitializersBottomUpPlusUpwardLinksERNS_12RuntimeStateEENK3$_0clEv",
     {
       /*.BasenameRange=*/{88, 98}, /*.TemplateArgumentsRange=*/{0, 0}, /*.ScopeRange=*/{0, 88},
       /*.ArgumentsRange=*/{98, 100}, /*.QualifiersRange=*/{100, 106}, /*.NameQualifiersRange=*/{0, 0},
       /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{0, 0}
     },
     /*.basename=*/"operator()",
     /*.scope=*/"dyld4::Loader::runInitializersBottomUpPlusUpwardLinks(dyld4::RuntimeState&) const::$_0::",
     /*.qualifiers=*/" const",
   },
   { "_ZZNK5dyld46Loader38runInitializersBottomUpPlusUpwardLinksERNS_12RuntimeStateEENK3$_0clEv.cold",
     {
       /*.BasenameRange=*/{88, 98}, /*.TemplateArgumentsRange=*/{0, 0}, /*.ScopeRange=*/{0, 88},
       /*.ArgumentsRange=*/{98, 100}, /*.QualifiersRange=*/{100, 106}, /*.NameQualifiersRange=*/{0, 0},
       /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{0, 0}
     },
     /*.basename=*/"operator()",
     /*.scope=*/"dyld4::Loader::runInitializersBottomUpPlusUpwardLinks(dyld4::RuntimeState&) const::$_0::",
     /*.qualifiers=*/" const",
   }
    // clang-format on
};

struct DemanglingPartsTestFixture
    : public ::testing::TestWithParam<DemanglingPartsTestCase> {};

namespace {
class TestAllocator {
  llvm::BumpPtrAllocator Alloc;

public:
  void reset() { Alloc.Reset(); }

  template <typename T, typename... Args> T *makeNode(Args &&...args) {
    return new (Alloc.Allocate(sizeof(T), alignof(T)))
        T(std::forward<Args>(args)...);
  }

  void *allocateNodeArray(size_t sz) {
    return Alloc.Allocate(sizeof(llvm::itanium_demangle::Node *) * sz,
                          alignof(llvm::itanium_demangle::Node *));
  }
};
} // namespace

TEST_P(DemanglingPartsTestFixture, DemanglingParts) {
  const auto &[mangled, info, basename, scope, qualifiers, valid_basename] =
      GetParam();

  llvm::itanium_demangle::ManglingParser<TestAllocator> Parser(
      mangled, mangled + ::strlen(mangled));

  const auto *Root = Parser.parse();

  ASSERT_NE(nullptr, Root);

  auto OB = std::unique_ptr<TrackingOutputBuffer, TrackingOutputBufferDeleter>(
      new TrackingOutputBuffer());
  Root->print(*OB);
  auto demangled = std::string_view(*OB);

  ASSERT_EQ(OB->NameInfo.hasBasename(), valid_basename);
  EXPECT_EQ(OB->NameInfo.BasenameRange, info.BasenameRange);
  EXPECT_EQ(OB->NameInfo.TemplateArgumentsRange, info.TemplateArgumentsRange);
  EXPECT_EQ(OB->NameInfo.ScopeRange, info.ScopeRange);
  EXPECT_EQ(OB->NameInfo.ArgumentsRange, info.ArgumentsRange);
  EXPECT_EQ(OB->NameInfo.QualifiersRange, info.QualifiersRange);
  EXPECT_EQ(OB->NameInfo.NameQualifiersRange, info.NameQualifiersRange);

  auto get_part = [&](const std::pair<size_t, size_t> &loc) {
    return demangled.substr(loc.first, loc.second - loc.first);
  };

  EXPECT_EQ(get_part(OB->NameInfo.BasenameRange), basename);
  EXPECT_EQ(get_part(OB->NameInfo.ScopeRange), scope);
  EXPECT_EQ(get_part(OB->NameInfo.QualifiersRange), qualifiers);
}

INSTANTIATE_TEST_SUITE_P(DemanglingPartsTests, DemanglingPartsTestFixture,
                         ::testing::ValuesIn(g_demangling_parts_test_cases));

struct DemangledNameInfoTestCase {
  DemangledNameInfo expected_info;
  bool valid_basename;
  bool valid_template;
  bool valid_scope;
  bool valid_arguments;
  bool valid_qualifiers;
  bool valid_name_qualifiers;
  bool valid_prefix;
  bool valid_suffix;
};

DemangledNameInfoTestCase g_demangled_name_info_test_cases[] = {
    // clang-format off
  {
    {
      /*.BasenameRange=*/{0, 10}, /*.TemplateArgumentsRange=*/{1, 0}, /*.ScopeRange=*/{1, 0},
      /*.ArgumentsRange=*/{1, 0}, /*.QualifiersRange=*/{1, 0}, /*.NameQualifiersRange=*/{1, 0},
      /*.PrefixRange=*/{1, 0}, /*.SuffixRange=*/{1, 0}
    },
    /*valid_basename=*/true, /*valid_template=*/false, /*valid_scope=*/false,
    /*valid_arguments=*/false, /*valid_qualifiers=*/false, /*valid_name_qualifiers=*/false,
    /*valid_prefix=*/false, /*valid_suffix=*/false,
  },
  {
    {
      /*.BasenameRange=*/{1, 0}, /*.TemplateArgumentsRange=*/{0, 10}, /*.ScopeRange=*/{1, 0},
      /*.ArgumentsRange=*/{1, 0}, /*.QualifiersRange=*/{1, 0}, /*.NameQualifiersRange=*/{1, 0},
      /*.PrefixRange=*/{1, 0}, /*.SuffixRange=*/{1, 0}
    },
    /*valid_basename=*/false, /*valid_name_qualifiers=*/true, /*valid_scope=*/false,
    /*valid_arguments=*/false, /*valid_qualifiers=*/false, /*valid_name_qualifiers=*/false,
    /*valid_prefix=*/false, /*valid_suffix=*/false,
  },
  {
    {
      /*.BasenameRange=*/{1, 0}, /*.TemplateArgumentsRange=*/{1, 0}, /*.ScopeRange=*/{0, 10},
      /*.ArgumentsRange=*/{1, 0}, /*.QualifiersRange=*/{1, 0}, /*.NameQualifiersRange=*/{1, 0},
      /*.PrefixRange=*/{1, 0}, /*.SuffixRange=*/{1, 0}
    },
    /*valid_basename=*/false, /*valid_template=*/false, /*valid_scope=*/true,
    /*valid_arguments=*/false, /*valid_qualifiers=*/false, /*valid_name_qualifiers=*/false,
    /*valid_prefix=*/false, /*valid_suffix=*/false,
  },
  {
    {
      /*.BasenameRange=*/{1, 0}, /*.TemplateArgumentsRange=*/{1, 0}, /*.ScopeRange=*/{1, 0},
      /*.ArgumentsRange=*/{0, 10}, /*.QualifiersRange=*/{1, 0}, /*.NameQualifiersRange=*/{1, 0},
      /*.PrefixRange=*/{1, 0}, /*.SuffixRange=*/{1, 0}
    },
    /*valid_basename=*/false, /*valid_template=*/false, /*valid_scope=*/false,
    /*valid_arguments=*/true, /*valid_qualifiers=*/false, /*valid_name_qualifiers=*/false,
    /*valid_prefix=*/false, /*valid_suffix=*/false,
  },
  {
    {
      /*.BasenameRange=*/{1, 0}, /*.TemplateArgumentsRange=*/{1, 0}, /*.ScopeRange=*/{1, 0},
      /*.ArgumentsRange=*/{1, 0}, /*.QualifiersRange=*/{0, 10}, /*.NameQualifiersRange=*/{1, 0},
      /*.PrefixRange=*/{1, 0}, /*.SuffixRange=*/{1, 0}
    },
    /*valid_basename=*/false, /*valid_template=*/false, /*valid_scope=*/false,
    /*valid_arguments=*/false, /*valid_qualifiers=*/true, /*valid_name_qualifiers=*/false,
    /*valid_prefix=*/false, /*valid_suffix=*/false,
  },
  {
    {
      /*.BasenameRange=*/{1, 0}, /*.TemplateArgumentsRange=*/{1, 0}, /*.ScopeRange=*/{1, 0},
      /*.ArgumentsRange=*/{1, 0}, /*.QualifiersRange=*/{1, 0}, /*.NameQualifiersRange=*/{0, 10},
      /*.PrefixRange=*/{1, 0}, /*.SuffixRange=*/{1, 0}
    },
    /*valid_basename=*/false, /*valid_template=*/false, /*valid_scope=*/false,
    /*valid_arguments=*/false, /*valid_qualifiers=*/false, /*valid_name_qualifiers=*/true,
    /*valid_prefix=*/false, /*valid_suffix=*/false,
  },
  {
    {
      /*.BasenameRange=*/{1, 0}, /*.TemplateArgumentsRange=*/{1, 0}, /*.ScopeRange=*/{1, 0},
      /*.ArgumentsRange=*/{1, 0}, /*.QualifiersRange=*/{1, 0}, /*.NameQualifiersRange=*/{1, 0},
      /*.PrefixRange=*/{0, 10}, /*.SuffixRange=*/{1, 0}
    },
    /*valid_basename=*/false, /*valid_template=*/false, /*valid_scope=*/false,
    /*valid_arguments=*/false, /*valid_qualifiers=*/false, /*valid_name_qualifiers=*/false,
    /*valid_prefix=*/true, /*valid_suffix=*/false,
  },
  {
    {
      /*.BasenameRange=*/{1, 0}, /*.TemplateArgumentsRange=*/{1, 0}, /*.ScopeRange=*/{1, 0},
      /*.ArgumentsRange=*/{1, 0}, /*.QualifiersRange=*/{1, 0}, /*.NameQualifiersRange=*/{1, 0},
      /*.PrefixRange=*/{1, 0}, /*.SuffixRange=*/{0, 10}
    },
    /*valid_basename=*/false, /*valid_template=*/false, /*valid_scope=*/false,
    /*valid_arguments=*/false, /*valid_qualifiers=*/false, /*valid_name_qualifiers=*/false,
    /*valid_prefix=*/false, /*valid_suffix=*/true,
  },
    // clang-format on
};

struct DemangledNameInfoTestFixture
    : public ::testing::TestWithParam<DemangledNameInfoTestCase> {};

TEST_P(DemangledNameInfoTestFixture, DemangledNameInfoRanges) {
  const auto &[info, valid_basename, valid_template_arguments, valid_scope,
               valid_arguments, valid_qualifiers, valid_name_qualifiers,
               valid_prefix, valid_suffix] = GetParam();

  ASSERT_EQ(info.hasBasename(), valid_basename);
  ASSERT_EQ(info.hasTemplateArguments(), valid_template_arguments);
  ASSERT_EQ(info.hasScope(), valid_scope);
  ASSERT_EQ(info.hasArguments(), valid_arguments);
  ASSERT_EQ(info.hasQualifiers(), valid_qualifiers);
  ASSERT_EQ(info.hasNameQualifiers(), valid_name_qualifiers);
  ASSERT_EQ(info.hasPrefix(), valid_prefix);
  ASSERT_EQ(info.hasSuffix(), valid_suffix);
}

INSTANTIATE_TEST_SUITE_P(DemangledNameInfoRangesTests,
                         DemangledNameInfoTestFixture,
                         ::testing::ValuesIn(g_demangled_name_info_test_cases));

struct DemanglingInfoCorrectnessTestCase {
  const char *mangled;
  const char *demangled;
};

DemanglingInfoCorrectnessTestCase g_demangling_correctness_test_cases[] = {
#include "llvm/Testing/Demangle/DemangleTestCases.inc"
};

struct DemanglingInfoCorrectnessTestFixutre
    : public ::testing::TestWithParam<DemanglingInfoCorrectnessTestCase> {};

TEST_P(DemanglingInfoCorrectnessTestFixutre, Correctness) {
  auto [mangled, demangled] = GetParam();

  llvm::itanium_demangle::ManglingParser<TestAllocator> Parser(
      mangled, mangled + ::strlen(mangled));

  const auto *Root = Parser.parse();

  ASSERT_NE(nullptr, Root);

  auto OB = std::unique_ptr<TrackingOutputBuffer, TrackingOutputBufferDeleter>(
      new TrackingOutputBuffer());
  Root->print(*OB);

  // Filter out cases which would never show up in frames. We only care about
  // function names.
  if (Root->getKind() !=
          llvm::itanium_demangle::Node::Kind::KFunctionEncoding &&
      Root->getKind() != llvm::itanium_demangle::Node::Kind::KDotSuffix)
    return;

  ASSERT_TRUE(OB->NameInfo.hasBasename());

  auto tracked_name = llvm::StringRef(*OB);

  std::string reconstructed_name;

  auto return_left =
      CPlusPlusLanguage::GetDemangledReturnTypeLHS(tracked_name, OB->NameInfo);
  EXPECT_THAT_EXPECTED(return_left, llvm::Succeeded());
  reconstructed_name += *return_left;

  auto scope = CPlusPlusLanguage::GetDemangledScope(tracked_name, OB->NameInfo);
  EXPECT_THAT_EXPECTED(scope, llvm::Succeeded());
  reconstructed_name += *scope;

  auto basename =
      CPlusPlusLanguage::GetDemangledBasename(tracked_name, OB->NameInfo);
  reconstructed_name += basename;

  auto template_args = CPlusPlusLanguage::GetDemangledTemplateArguments(
      tracked_name, OB->NameInfo);
  EXPECT_THAT_EXPECTED(template_args, llvm::Succeeded());
  reconstructed_name += *template_args;

  auto args = CPlusPlusLanguage::GetDemangledFunctionArguments(tracked_name,
                                                               OB->NameInfo);
  EXPECT_THAT_EXPECTED(args, llvm::Succeeded());
  reconstructed_name += *args;

  auto return_right =
      CPlusPlusLanguage::GetDemangledReturnTypeRHS(tracked_name, OB->NameInfo);
  EXPECT_THAT_EXPECTED(return_right, llvm::Succeeded());
  reconstructed_name += *return_right;

  auto qualifiers = CPlusPlusLanguage::GetDemangledFunctionQualifiers(
      tracked_name, OB->NameInfo);
  EXPECT_THAT_EXPECTED(qualifiers, llvm::Succeeded());
  reconstructed_name += *qualifiers;

  auto suffix =
      CPlusPlusLanguage::GetDemangledFunctionSuffix(tracked_name, OB->NameInfo);
  EXPECT_THAT_EXPECTED(suffix, llvm::Succeeded());
  reconstructed_name += *suffix;

  EXPECT_EQ(reconstructed_name, demangled);
}

INSTANTIATE_TEST_SUITE_P(
    DemanglingInfoCorrectnessTests, DemanglingInfoCorrectnessTestFixutre,
    ::testing::ValuesIn(g_demangling_correctness_test_cases));
