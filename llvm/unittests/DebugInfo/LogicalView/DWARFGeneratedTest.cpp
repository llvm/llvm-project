//===- llvm/unittest/DebugInfo/LogicalView/DWARFGeneratedTest.cpp ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../DWARF/DwarfGenerator.h"
#include "../DWARF/DwarfUtils.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/LogicalView/Core/LVReader.h"
#include "llvm/DebugInfo/LogicalView/Core/LVScope.h"
#include "llvm/DebugInfo/LogicalView/Core/LVSymbol.h"
#include "llvm/DebugInfo/LogicalView/Core/LVType.h"
#include "llvm/DebugInfo/LogicalView/LVReaderHandler.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/COM.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::logicalview;
using namespace llvm::dwarf;
using namespace llvm::dwarf::utils;

namespace {

// Helper function to get the first compile unit.
LVScopeCompileUnit *getFirstCompileUnit(LVScopeRoot *Root) {
  if (!Root)
    return nullptr;

  const LVScopes *CompileUnits = Root->getScopes();
  if (!CompileUnits)
    return nullptr;

  LVScopes::const_iterator Iter = CompileUnits->begin();
  return (Iter != CompileUnits->end())
             ? static_cast<LVScopeCompileUnit *>(*Iter)
             : nullptr;
}

// Helper function to create a reader.
std::unique_ptr<LVReader> createReader(LVReaderHandler &ReaderHandler,
                                       SmallString<128> &InputsDir,
                                       StringRef Filename) {
  SmallString<128> ObjectName(InputsDir);
  llvm::sys::path::append(ObjectName, Filename);

  Expected<std::unique_ptr<LVReader>> ReaderOrErr =
      ReaderHandler.createReader(std::string(ObjectName));
  EXPECT_THAT_EXPECTED(ReaderOrErr, Succeeded());
  std::unique_ptr<LVReader> Reader = std::move(*ReaderOrErr);
  return Reader;
}

// Create a file with generated DWARF.
void generateDebugInfo(StringRef Path, Triple &Triple) {
  uint16_t Version = 5;
  auto ExpectedDG = dwarfgen::Generator::create(Triple, Version);
  ASSERT_THAT_EXPECTED(ExpectedDG, Succeeded());
  dwarfgen::Generator *DG = ExpectedDG.get().get();
  dwarfgen::CompileUnit &CU = DG->addCompileUnit();

  dwarfgen::DIE CUDie = CU.getUnitDIE();
  CUDie.addAttribute(DW_AT_name, DW_FORM_strp, "test.cpp");
  CUDie.addAttribute(DW_AT_language, DW_FORM_data2, DW_LANG_C_plus_plus);
  CUDie.addAttribute(DW_AT_producer, DW_FORM_strp, "dwarfgen::Generator");

  dwarfgen::DIE ScopeValueDie = CUDie.addChild(DW_TAG_inlined_subroutine);
  ScopeValueDie.addAttribute(DW_AT_accessibility, DW_FORM_data1, 1);
  ScopeValueDie.addAttribute(DW_AT_inline, DW_FORM_data4, 2);
  ScopeValueDie.addAttribute(DW_AT_virtuality, DW_FORM_data4, 3);
  ScopeValueDie.addAttribute(DW_AT_call_file, DW_FORM_data4, 4);
  ScopeValueDie.addAttribute(DW_AT_call_line, DW_FORM_data4, 5);
  ScopeValueDie.addAttribute(DW_AT_decl_file, DW_FORM_data4, 6);
  ScopeValueDie.addAttribute(DW_AT_decl_line, DW_FORM_data4, 7);
  ScopeValueDie.addAttribute(DW_AT_GNU_discriminator, DW_FORM_data4, 8);

  dwarfgen::DIE ScopeNoValueDie = CUDie.addChild(DW_TAG_inlined_subroutine);
  ScopeNoValueDie.addAttribute(DW_AT_accessibility, DW_FORM_sdata, 1);
  ScopeNoValueDie.addAttribute(DW_AT_inline, DW_FORM_sdata, 2);
  ScopeNoValueDie.addAttribute(DW_AT_virtuality, DW_FORM_sdata, 3);
  ScopeNoValueDie.addAttribute(DW_AT_call_file, DW_FORM_sdata, 4);
  ScopeNoValueDie.addAttribute(DW_AT_call_line, DW_FORM_sdata, 5);
  ScopeNoValueDie.addAttribute(DW_AT_decl_file, DW_FORM_sdata, 6);
  ScopeNoValueDie.addAttribute(DW_AT_decl_line, DW_FORM_sdata, 7);
  ScopeNoValueDie.addAttribute(DW_AT_GNU_discriminator, DW_FORM_sdata, 8);

  dwarfgen::DIE ScopeImplicitDie = CUDie.addChild(DW_TAG_inlined_subroutine);
  ScopeImplicitDie.addAttribute(DW_AT_accessibility, DW_FORM_implicit_const, 1);
  ScopeImplicitDie.addAttribute(DW_AT_inline, DW_FORM_implicit_const, 2);
  ScopeImplicitDie.addAttribute(DW_AT_virtuality, DW_FORM_implicit_const, 3);
  ScopeImplicitDie.addAttribute(DW_AT_call_file, DW_FORM_implicit_const, 4);
  ScopeImplicitDie.addAttribute(DW_AT_call_line, DW_FORM_implicit_const, 5);
  ScopeImplicitDie.addAttribute(DW_AT_decl_file, DW_FORM_implicit_const, 6);
  ScopeImplicitDie.addAttribute(DW_AT_decl_line, DW_FORM_implicit_const, 7);
  ScopeImplicitDie.addAttribute(DW_AT_GNU_discriminator, DW_FORM_implicit_const,
                                8);

  dwarfgen::DIE SymbolValueDie = CUDie.addChild(DW_TAG_variable);
  SymbolValueDie.addAttribute(DW_AT_bit_size, DW_FORM_data1, 1);

  dwarfgen::DIE SymbolNoValueDie = CUDie.addChild(DW_TAG_variable);
  SymbolNoValueDie.addAttribute(DW_AT_bit_size, DW_FORM_sdata, 1);

  dwarfgen::DIE SymbolImplicitDie = CUDie.addChild(DW_TAG_variable);
  SymbolImplicitDie.addAttribute(DW_AT_bit_size, DW_FORM_implicit_const, 1);

  dwarfgen::DIE TypeValueCountDie = CUDie.addChild(DW_TAG_subrange_type);
  TypeValueCountDie.addAttribute(DW_AT_count, DW_FORM_data4, 1);

  dwarfgen::DIE TypeNoValueCountDie = CUDie.addChild(DW_TAG_subrange_type);
  TypeNoValueCountDie.addAttribute(DW_AT_count, DW_FORM_sdata, 1);

  dwarfgen::DIE TypeImplicitCountDie = CUDie.addChild(DW_TAG_subrange_type);
  TypeImplicitCountDie.addAttribute(DW_AT_count, DW_FORM_implicit_const, 1);

  dwarfgen::DIE TypeValueRangeDie = CUDie.addChild(DW_TAG_subrange_type);
  TypeValueRangeDie.addAttribute(DW_AT_lower_bound, DW_FORM_data4, 1);
  TypeValueRangeDie.addAttribute(DW_AT_upper_bound, DW_FORM_data4, 2);

  dwarfgen::DIE TypeNoValueRangeDie = CUDie.addChild(DW_TAG_subrange_type);
  TypeNoValueRangeDie.addAttribute(DW_AT_lower_bound, DW_FORM_addr, 3);
  TypeNoValueRangeDie.addAttribute(DW_AT_upper_bound, DW_FORM_addr, 4);

  dwarfgen::DIE TypeImplicitRangeDie = CUDie.addChild(DW_TAG_subrange_type);
  TypeImplicitRangeDie.addAttribute(DW_AT_lower_bound, DW_FORM_implicit_const,
                                    5);
  TypeImplicitRangeDie.addAttribute(DW_AT_upper_bound, DW_FORM_implicit_const,
                                    6);

  // Generate the DWARF.
  StringRef FileBytes = DG->generate();
  MemoryBufferRef FileBuffer(FileBytes, "dwarf");
  auto Obj = object::ObjectFile::createObjectFile(FileBuffer);
  EXPECT_TRUE((bool)Obj);
  std::unique_ptr<DWARFContext> DwarfContext = DWARFContext::create(**Obj);

  // Verify the number of compile units is correct.
  uint32_t NumCUs = DwarfContext->getNumCompileUnits();
  EXPECT_EQ(NumCUs, 1u);
  DWARFCompileUnit *U = cast<DWARFCompileUnit>(DwarfContext->getUnitAtIndex(0));
  auto DieDG = U->getUnitDIE(false);
  EXPECT_TRUE(DieDG.isValid());

  // Verify the siblings correct order.
  //   ScopeValue
  //   ScopeNoValue
  //   ScopeImplicit
  auto ScopeValueDieDG = DieDG.getFirstChild();
  EXPECT_TRUE(ScopeValueDieDG.isValid());
  EXPECT_EQ(ScopeValueDieDG.getTag(), DW_TAG_inlined_subroutine);
  auto ScopeNoValueDieDG = ScopeValueDieDG.getSibling();
  EXPECT_TRUE(ScopeNoValueDieDG.isValid());
  EXPECT_EQ(ScopeNoValueDieDG.getTag(), DW_TAG_inlined_subroutine);
  auto ScopeImplicitDieDG = ScopeNoValueDieDG.getSibling();
  EXPECT_TRUE(ScopeImplicitDieDG.isValid());
  EXPECT_EQ(ScopeImplicitDieDG.getTag(), DW_TAG_inlined_subroutine);

  // Verify the siblings correct order.
  //   SymbolValue
  //   SymbolNoValue
  //   SymbolImplicitValue
  auto SymbolValueDieDG = ScopeImplicitDieDG.getSibling();
  EXPECT_TRUE(SymbolValueDieDG.isValid());
  EXPECT_EQ(SymbolValueDieDG.getTag(), DW_TAG_variable);
  auto SymbolNoValueDieDG = SymbolValueDieDG.getSibling();
  EXPECT_TRUE(SymbolNoValueDieDG.isValid());
  EXPECT_EQ(SymbolNoValueDieDG.getTag(), DW_TAG_variable);
  auto SymbolImplicitDieDG = SymbolNoValueDieDG.getSibling();
  EXPECT_TRUE(SymbolImplicitDieDG.isValid());
  EXPECT_EQ(SymbolImplicitDieDG.getTag(), DW_TAG_variable);

  // Verify the siblings correct order.
  //   TypeValueCount
  //   TypeNoValueCount
  //   TypeImplicitValueCount
  auto TypeValueCountDieDG = SymbolImplicitDieDG.getSibling();
  EXPECT_TRUE(TypeValueCountDieDG.isValid());
  EXPECT_EQ(TypeValueCountDieDG.getTag(), DW_TAG_subrange_type);
  auto TypeNoValueCountDieDG = TypeValueCountDieDG.getSibling();
  EXPECT_TRUE(TypeNoValueCountDieDG.isValid());
  EXPECT_EQ(TypeNoValueCountDieDG.getTag(), DW_TAG_subrange_type);
  auto TypeImplicitCountDieDG = TypeNoValueCountDieDG.getSibling();
  EXPECT_TRUE(TypeImplicitCountDieDG.isValid());
  EXPECT_EQ(TypeImplicitCountDieDG.getTag(), DW_TAG_subrange_type);

  // Verify the siblings correct order.
  //   TypeValueRange
  //   TypeNoValueRange
  //   TypeImplicitValueRange
  auto TypeValueRangeDieDG = TypeImplicitCountDieDG.getSibling();
  EXPECT_TRUE(TypeValueRangeDieDG.isValid());
  EXPECT_EQ(TypeValueRangeDieDG.getTag(), DW_TAG_subrange_type);
  auto TypeNoValueRangeDieDG = TypeValueRangeDieDG.getSibling();
  EXPECT_TRUE(TypeNoValueRangeDieDG.isValid());
  EXPECT_EQ(TypeNoValueRangeDieDG.getTag(), DW_TAG_subrange_type);
  auto TypeImplicitRangeDieDG = TypeNoValueRangeDieDG.getSibling();
  EXPECT_TRUE(TypeImplicitRangeDieDG.isValid());
  EXPECT_EQ(TypeImplicitRangeDieDG.getTag(), DW_TAG_subrange_type);

  // Save the generated DWARF file to disk.
  EXPECT_TRUE(DG->saveFile(Path));
}

// Check the logical elements basic properties.
void checkElementAttributes(LVReader *Reader) {
  LVScopeRoot *Root = Reader->getScopesRoot();
  ASSERT_NE(Root, nullptr);
  LVScopeCompileUnit *CompileUnit = getFirstCompileUnit(Root);
  ASSERT_NE(CompileUnit, nullptr);

  const LVScopes *Scopes = CompileUnit->getScopes();
  ASSERT_NE(Scopes, nullptr);
  ASSERT_EQ(Scopes->size(), 3u);

  // Check values.
  LVScopes::const_iterator ScopeIter = Scopes->begin();
  ASSERT_NE(ScopeIter, Scopes->end());
  LVScope *Scope = static_cast<LVScope *>(*ScopeIter);
  ASSERT_NE(Scope, nullptr);
  EXPECT_EQ(Scope->getAccessibilityCode(), 1u); // Element
  EXPECT_EQ(Scope->getInlineCode(), 2u);        // Element
  EXPECT_EQ(Scope->getVirtualityCode(), 3u);    // Element
  EXPECT_EQ(Scope->getCallFilenameIndex(), 5u); // ScopeFunctionInlined
  EXPECT_EQ(Scope->getCallLineNumber(), 5u);    // ScopeFunctionInlined
  EXPECT_EQ(Scope->getFilenameIndex(), 7u);     // Element
  EXPECT_EQ(Scope->getLineNumber(), 7u);        // Element
  EXPECT_EQ(Scope->getDiscriminator(), 8u);     // ScopeFunctionInlined

  // Check no-values.
  ASSERT_NE(++ScopeIter, Scopes->end());
  Scope = static_cast<LVScope *>(*ScopeIter);
  ASSERT_NE(Scope, nullptr);
  EXPECT_EQ(Scope->getAccessibilityCode(), 0u); // Element
  EXPECT_EQ(Scope->getInlineCode(), 0u);        // Element
  EXPECT_EQ(Scope->getVirtualityCode(), 0u);    // Element
  EXPECT_EQ(Scope->getCallFilenameIndex(), 1u); // ScopeFunctionInlined
  EXPECT_EQ(Scope->getCallLineNumber(), 0u);    // ScopeFunctionInlined
  EXPECT_EQ(Scope->getFilenameIndex(), 1u);     // Element
  EXPECT_EQ(Scope->getLineNumber(), 0u);        // Element
  EXPECT_EQ(Scope->getDiscriminator(), 0u);     // ScopeFunctionInlined

  // Check implicit values.
  ASSERT_NE(++ScopeIter, Scopes->end());
  Scope = static_cast<LVScope *>(*ScopeIter);
  ASSERT_NE(Scope, nullptr);
  EXPECT_EQ(Scope->getAccessibilityCode(), 1u); // Element
  EXPECT_EQ(Scope->getInlineCode(), 2u);        // Element
  EXPECT_EQ(Scope->getVirtualityCode(), 3u);    // Element
  EXPECT_EQ(Scope->getCallFilenameIndex(), 5u); // ScopeFunctionInlined
  EXPECT_EQ(Scope->getCallLineNumber(), 5u);    // ScopeFunctionInlined
  EXPECT_EQ(Scope->getFilenameIndex(), 7u);     // Element
  EXPECT_EQ(Scope->getLineNumber(), 7u);        // Element
  EXPECT_EQ(Scope->getDiscriminator(), 8u);     // ScopeFunctionInlined

  const LVSymbols *Symbols = CompileUnit->getSymbols();
  ASSERT_NE(Symbols, nullptr);
  ASSERT_EQ(Symbols->size(), 3u);

  LVSymbols::const_iterator SymbolIter = Symbols->begin();
  ASSERT_NE(SymbolIter, Symbols->end());
  LVSymbol *Symbol = static_cast<LVSymbol *>(*SymbolIter);
  ASSERT_NE(Symbol, nullptr);
  EXPECT_EQ(Symbol->getBitSize(), 1u); // Symbol

  ASSERT_NE(++SymbolIter, Symbols->end());
  Symbol = static_cast<LVSymbol *>(*SymbolIter);
  ASSERT_NE(Symbol, nullptr);
  EXPECT_EQ(Symbol->getBitSize(), 0u); // Symbol

  ASSERT_NE(++SymbolIter, Symbols->end());
  Symbol = static_cast<LVSymbol *>(*SymbolIter);
  ASSERT_NE(Symbol, nullptr);
  EXPECT_EQ(Symbol->getBitSize(), 1u); // Symbol

  const LVTypes *Types = CompileUnit->getTypes();
  ASSERT_NE(Types, nullptr);
  ASSERT_EQ(Types->size(), 6u);

  LVTypes::const_iterator TypeIter = Types->begin();
  ASSERT_NE(TypeIter, Types->end());
  LVType *Type = static_cast<LVType *>(*TypeIter);
  ASSERT_NE(Type, nullptr);
  EXPECT_EQ(Type->getCount(), 1u); // Type

  ASSERT_NE(++TypeIter, Types->end());
  Type = static_cast<LVType *>(*TypeIter);
  ASSERT_NE(Type, nullptr);
  EXPECT_EQ(Type->getCount(), 0u); // Type

  ASSERT_NE(++TypeIter, Types->end());
  Type = static_cast<LVType *>(*TypeIter);
  ASSERT_NE(Type, nullptr);
  EXPECT_EQ(Type->getCount(), 1u); // Type

  ASSERT_NE(++TypeIter, Types->end());
  Type = static_cast<LVType *>(*TypeIter);
  ASSERT_NE(Type, nullptr);
  EXPECT_EQ(Type->getLowerBound(), 1u); // Type
  EXPECT_EQ(Type->getUpperBound(), 2u); // Type

  ASSERT_NE(++TypeIter, Types->end());
  Type = static_cast<LVType *>(*TypeIter);
  ASSERT_NE(Type, nullptr);
  EXPECT_EQ(Type->getLowerBound(), 0u); // Type
  EXPECT_EQ(Type->getUpperBound(), 0u); // Type

  ASSERT_NE(++TypeIter, Types->end());
  Type = static_cast<LVType *>(*TypeIter);
  ASSERT_NE(Type, nullptr);
  EXPECT_EQ(Type->getLowerBound(), 5u); // Type
  EXPECT_EQ(Type->getUpperBound(), 6u); // Type
}

TEST(LogicalViewTest, ElementAttributes) {
  // Initialize targets and assembly printers/parsers.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  InitializeAllDisassemblers();

  llvm::sys::InitializeCOMRAII COM(llvm::sys::COMThreadingMode::MultiThreaded);

  Triple Triple(Triple::normalize("x86_64-pc-linux-gnu"));
  if (!isConfigurationSupported(Triple))
    GTEST_SKIP();

  unittest::TempDir TestDirectory("dwarf-test", /*Unique=*/true);
  llvm::SmallString<128> DirName(TestDirectory.path());
  StringRef Filename("test.o");
  llvm::SmallString<128> Path(TestDirectory.path(Filename));
  generateDebugInfo(Path, Triple);

  // Reader options.
  LVOptions ReaderOptions;
  ReaderOptions.setPrintScopes();
  ReaderOptions.setPrintSymbols();
  ReaderOptions.setPrintTypes();
  ReaderOptions.resolveDependencies();

  std::vector<std::string> Objects;
  ScopedPrinter W(outs());
  LVReaderHandler ReaderHandler(Objects, W, ReaderOptions);

  // Check logical elements properties.
  std::unique_ptr<LVReader> Reader =
      createReader(ReaderHandler, DirName, Filename);
  ASSERT_NE(Reader, nullptr);

  checkElementAttributes(Reader.get());
}

} // namespace
