//===- llvm/unittest/DebugInfo/LogicalView/DWARFReaderTest.cpp ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/LogicalView/Core/LVCompare.h"
#include "llvm/DebugInfo/LogicalView/Core/LVLine.h"
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

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::logicalview;

extern const char *TestMainArgv0;

namespace {

const char *DwarfClang = "test-dwarf-clang.o";
// Two compile units: one declares `extern int foo_printf(const char *, ...);`
// and another one that defines the function.
const char *DwarfClangUnspecParams = "test-dwarf-clang-unspec-params.elf";
const char *DwarfClangModule = "test-dwarf-clang-module.o";
const char *DwarfGcc = "test-dwarf-gcc.o";

// Helper function to get the first compile unit.
LVScopeCompileUnit *getFirstCompileUnit(LVScopeRoot *Root) {
  EXPECT_NE(Root, nullptr);
  const LVScopes *CompileUnits = Root->getScopes();
  EXPECT_NE(CompileUnits, nullptr);
  EXPECT_GT(CompileUnits->size(), 0u);

  LVScopes::const_iterator Iter = CompileUnits->begin();
  EXPECT_NE(Iter, nullptr);
  LVScopeCompileUnit *CompileUnit = static_cast<LVScopeCompileUnit *>(*Iter);
  EXPECT_NE(CompileUnit, nullptr);
  return CompileUnit;
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
  EXPECT_NE(Reader, nullptr);
  return Reader;
}

// Check the logical elements basic properties.
void checkElementProperties(LVReader *Reader) {
  LVScopeRoot *Root = Reader->getScopesRoot();
  LVScopeCompileUnit *CompileUnit = getFirstCompileUnit(Root);

  EXPECT_EQ(Root->getFileFormatName(), "elf64-x86-64");
  EXPECT_EQ(Root->getName(), DwarfClang);

  EXPECT_EQ(CompileUnit->getBaseAddress(), 0u);
  EXPECT_TRUE(CompileUnit->getProducer().starts_with("clang"));
  EXPECT_EQ(CompileUnit->getName(), "test.cpp");

  EXPECT_EQ(CompileUnit->lineCount(), 0u);
  EXPECT_EQ(CompileUnit->scopeCount(), 1u);
  EXPECT_EQ(CompileUnit->symbolCount(), 0u);
  EXPECT_EQ(CompileUnit->typeCount(), 7u);
  EXPECT_EQ(CompileUnit->rangeCount(), 1u);

  const LVLocations *Ranges = CompileUnit->getRanges();
  ASSERT_NE(Ranges, nullptr);
  ASSERT_EQ(Ranges->size(), 1u);
  LVLocations::const_iterator IterLocation = Ranges->begin();
  LVLocation *Location = (*IterLocation);
  EXPECT_STREQ(Location->getIntervalInfo().c_str(),
               "{Range} Lines 2:9 [0x0000000000:0x000000003a]");

  LVRange RangeList;
  CompileUnit->getRanges(RangeList);

  const LVRangeEntries &RangeEntries = RangeList.getEntries();
  ASSERT_EQ(RangeEntries.size(), 2u);
  LVRangeEntries::const_iterator IterRanges = RangeEntries.cbegin();
  LVRangeEntry RangeEntry = *IterRanges;
  EXPECT_EQ(RangeEntry.lower(), 0u);
  EXPECT_EQ(RangeEntry.upper(), 0x3au);
  EXPECT_EQ(RangeEntry.scope()->getLineNumber(), 0u);
  EXPECT_EQ(RangeEntry.scope()->getName(), "test.cpp");
  EXPECT_EQ(RangeEntry.scope()->getOffset(), 0x0bu);

  ++IterRanges;
  RangeEntry = *IterRanges;
  EXPECT_EQ(RangeEntry.lower(), 0x1cu);
  EXPECT_EQ(RangeEntry.upper(), 0x2fu);
  EXPECT_EQ(RangeEntry.scope()->getLineNumber(), 0u);
  EXPECT_EQ(RangeEntry.scope()->getName(), "foo::?");
  EXPECT_EQ(RangeEntry.scope()->getOffset(), 0x71u);

  const LVPublicNames &PublicNames = CompileUnit->getPublicNames();
  ASSERT_EQ(PublicNames.size(), 1u);
  LVPublicNames::const_iterator IterNames = PublicNames.cbegin();
  LVScope *Function = (*IterNames).first;
  EXPECT_EQ(Function->getName(), "foo");
  EXPECT_EQ(Function->getLineNumber(), 2u);
  LVNameInfo NameInfo = (*IterNames).second;
  EXPECT_EQ(NameInfo.first, 0u);
  EXPECT_EQ(NameInfo.second, 0x3au);

  // Lines (debug and assembler) for 'foo'.
  const LVLines *Lines = Function->getLines();
  ASSERT_NE(Lines, nullptr);
  ASSERT_EQ(Lines->size(), 0x12u);

  // Check size of types in CompileUnit.
  const LVTypes *Types = CompileUnit->getTypes();
  ASSERT_NE(Types, nullptr);
  EXPECT_EQ(Types->size(), 7u);

  const auto BoolType =
      std::find_if(Types->begin(), Types->end(), [](const LVElement *elt) {
        return elt->getName() == "bool";
      });
  ASSERT_NE(BoolType, Types->end());
  const auto IntType =
      std::find_if(Types->begin(), Types->end(), [](const LVElement *elt) {
        return elt->getName() == "int";
      });
  ASSERT_NE(IntType, Types->end());
  EXPECT_EQ(static_cast<LVType *>(*BoolType)->getBitSize(), 8u);
  EXPECT_EQ(static_cast<LVType *>(*BoolType)->getStorageSizeInBytes(), 1u);
  EXPECT_EQ(static_cast<LVType *>(*IntType)->getBitSize(), 32u);
  EXPECT_EQ(static_cast<LVType *>(*IntType)->getStorageSizeInBytes(), 4u);
}

// Check proper handling of DW_AT_unspecified_parameters in
// LVScope::addMissingElements().
void checkUnspecifiedParameters(LVReader *Reader) {
  LVScopeRoot *Root = Reader->getScopesRoot();
  LVScopeCompileUnit *CompileUnit = getFirstCompileUnit(Root);

  EXPECT_EQ(Root->getFileFormatName(), "elf64-x86-64");
  EXPECT_EQ(Root->getName(), DwarfClangUnspecParams);

  const LVPublicNames &PublicNames = CompileUnit->getPublicNames();
  ASSERT_EQ(PublicNames.size(), 1u);

  LVPublicNames::const_iterator IterNames = PublicNames.cbegin();
  LVScope *Function = (*IterNames).first;
  EXPECT_EQ(Function->getName(), "foo_printf");
  const LVElements *Elements = Function->getChildren();
  ASSERT_NE(Elements, nullptr);
  // foo_printf is a variadic function whose prototype is
  // `int foo_printf(const char *, ...)`, where the '...' is represented by a
  // DW_TAG_unspecified_parameters, i.e. we expect to find at least one child
  // for which getIsUnspecified() returns true.
  EXPECT_EQ(std::any_of(
                Elements->begin(), Elements->end(),
                [](const LVElement *elt) {
                  return elt->getIsSymbol() &&
                         static_cast<const LVSymbol *>(elt)->getIsUnspecified();
                }),
            true);
}

// Check the basic properties on parsed DW_TAG_module.
void checkScopeModule(LVReader *Reader) {
  LVScopeRoot *Root = Reader->getScopesRoot();
  LVScopeCompileUnit *CompileUnit = getFirstCompileUnit(Root);

  EXPECT_EQ(Root->getFileFormatName(), "Mach-O 64-bit x86-64");
  EXPECT_EQ(Root->getName(), DwarfClangModule);

  ASSERT_NE(CompileUnit->getChildren(), nullptr);
  LVElement *FirstChild = *(CompileUnit->getChildren()->begin());
  EXPECT_EQ(FirstChild->getIsScope(), 1);
  LVScopeModule *Module = static_cast<LVScopeModule *>(FirstChild);
  EXPECT_EQ(Module->getIsModule(), 1);
  EXPECT_EQ(Module->getName(), "DebugModule");
}

// Check the logical elements selection.
void checkElementSelection(LVReader *Reader) {
  LVScopeRoot *Root = Reader->getScopesRoot();
  LVScopeCompileUnit *CompileUnit = getFirstCompileUnit(Root);

  // Get the matched elements.
  LVElements MatchedElements = CompileUnit->getMatchedElements();
  std::map<LVOffset, LVElement *> MapElements;
  for (LVElement *Element : MatchedElements)
    MapElements[Element->getOffset()] = Element;
  ASSERT_EQ(MapElements.size(), 0xeu);

  LVElement *Element = MapElements[0x000000004b]; // 'foo'
  ASSERT_NE(Element, nullptr);
  EXPECT_NE(Element->getName().find("foo"), StringRef::npos);
  EXPECT_EQ(Element->getIsScope(), 1);

  Element = MapElements[0x00000000c0]; // 'CONSTANT'
  ASSERT_NE(Element, nullptr);
  EXPECT_NE(Element->getName().find("CONSTANT"), StringRef::npos);
  EXPECT_EQ(Element->getIsSymbol(), 1);

  Element = MapElements[0x000000002d]; // 'INTPTR'
  ASSERT_NE(Element, nullptr);
  EXPECT_NE(Element->getName().find("INTPTR"), StringRef::npos);
  EXPECT_EQ(Element->getIsType(), 1);

  Element = MapElements[0x00000000af]; // 'INTEGER'
  ASSERT_NE(Element, nullptr);
  EXPECT_NE(Element->getName().find("INTEGER"), StringRef::npos);
  EXPECT_EQ(Element->getIsType(), 1);

  Element = MapElements[0x000000000f]; // 'movl	%edx, %eax'
  ASSERT_NE(Element, nullptr);
  EXPECT_NE(Element->getName().find("movl"), StringRef::npos);
  EXPECT_EQ(Element->getIsLine(), 1);

  // Get the parents for the matched elements.
  LVScopes MatchedScopes = CompileUnit->getMatchedScopes();
  std::set<LVOffset> SetScopes;
  for (LVScope *Scope : MatchedScopes)
    SetScopes.insert(Scope->getOffset());
  std::set<LVOffset>::iterator Iter;
  ASSERT_EQ(SetScopes.size(), 3u);

  Iter = SetScopes.find(0x000000000b); // CompileUnit <- 'foo'
  EXPECT_NE(Iter, SetScopes.end());
  Iter = SetScopes.find(0x000000009e); // Function <- 'movl	%edx, %eax'
  EXPECT_NE(Iter, SetScopes.end());
  Iter = SetScopes.find(0x000000009e); // LexicalScope <- 'INTEGER'
  EXPECT_NE(Iter, SetScopes.end());
}

// Check the logical elements comparison.
void checkElementComparison(LVReader *Reference, LVReader *Target) {
  LVCompare Compare(nulls());
  Error Err = Compare.execute(Reference, Target);
  ASSERT_THAT_ERROR(std::move(Err), Succeeded());

  // Get comparison table.
  LVPassTable PassTable = Compare.getPassTable();
  ASSERT_EQ(PassTable.size(), 5u);

  LVReader *Reader;
  LVElement *Element;
  LVComparePass Pass;

  // Reference: Missing Variable 'CONSTANT'
  std::tie(Reader, Element, Pass) = PassTable[0];
  ASSERT_NE(Reader, nullptr);
  ASSERT_NE(Element, nullptr);
  EXPECT_EQ(Reader, Reference);
  EXPECT_EQ(Element->getLevel(), 4u);
  EXPECT_EQ(Element->getLineNumber(), 5u);
  EXPECT_EQ(Element->getName(), "CONSTANT");
  EXPECT_EQ(Pass, LVComparePass::Missing);

  // Reference: Missing TypeDefinition 'INTEGER'
  std::tie(Reader, Element, Pass) = PassTable[1];
  ASSERT_NE(Reader, nullptr);
  ASSERT_NE(Element, nullptr);
  EXPECT_EQ(Reader, Reference);
  EXPECT_EQ(Element->getLevel(), 3u);
  EXPECT_EQ(Element->getLineNumber(), 4u);
  EXPECT_EQ(Element->getName(), "INTEGER");
  EXPECT_EQ(Pass, LVComparePass::Missing);

  // Reference: Missing DebugLine
  std::tie(Reader, Element, Pass) = PassTable[2];
  ASSERT_NE(Reader, nullptr);
  ASSERT_NE(Element, nullptr);
  EXPECT_EQ(Reader, Reference);
  EXPECT_EQ(Element->getLevel(), 3u);
  EXPECT_EQ(Element->getLineNumber(), 8u);
  EXPECT_EQ(Element->getName(), "");
  EXPECT_EQ(Pass, LVComparePass::Missing);

  // Target: Added Variable 'CONSTANT'
  std::tie(Reader, Element, Pass) = PassTable[3];
  ASSERT_NE(Reader, nullptr);
  ASSERT_NE(Element, nullptr);
  EXPECT_EQ(Reader, Target);
  EXPECT_EQ(Element->getLevel(), 4u);
  EXPECT_EQ(Element->getLineNumber(), 5u);
  EXPECT_EQ(Element->getName(), "CONSTANT");
  EXPECT_EQ(Pass, LVComparePass::Added);

  // Target: Added TypeDefinition 'INTEGER'
  std::tie(Reader, Element, Pass) = PassTable[4];
  ASSERT_NE(Reader, nullptr);
  ASSERT_NE(Element, nullptr);
  EXPECT_EQ(Reader, Target);
  EXPECT_EQ(Element->getLevel(), 4u);
  EXPECT_EQ(Element->getLineNumber(), 4u);
  EXPECT_EQ(Element->getName(), "INTEGER");
  EXPECT_EQ(Pass, LVComparePass::Added);
}

// Logical elements properties.
void elementProperties(SmallString<128> &InputsDir) {
  // Reader options.
  LVOptions ReaderOptions;
  ReaderOptions.setAttributeOffset();
  ReaderOptions.setAttributeFormat();
  ReaderOptions.setAttributeFilename();
  ReaderOptions.setAttributeProducer();
  ReaderOptions.setAttributePublics();
  ReaderOptions.setAttributeRange();
  ReaderOptions.setAttributeLocation();
  ReaderOptions.setAttributeInserted();
  ReaderOptions.setAttributeSize();
  ReaderOptions.setPrintAll();
  ReaderOptions.resolveDependencies();

  std::vector<std::string> Objects;
  ScopedPrinter W(outs());
  LVReaderHandler ReaderHandler(Objects, W, ReaderOptions);

  // Check logical elements properties.
  std::unique_ptr<LVReader> Reader =
      createReader(ReaderHandler, InputsDir, DwarfClang);
  checkElementProperties(Reader.get());

  Reader = createReader(ReaderHandler, InputsDir, DwarfClangUnspecParams);
  checkUnspecifiedParameters(Reader.get());

  Reader = createReader(ReaderHandler, InputsDir, DwarfClangModule);
  checkScopeModule(Reader.get());
}

// Logical elements selection.
void elementSelection(SmallString<128> &InputsDir) {
  // Reader options.
  LVOptions ReaderOptions;
  ReaderOptions.setAttributeOffset();
  ReaderOptions.setPrintAll();

  ReaderOptions.setSelectIgnoreCase();
  ReaderOptions.setSelectUseRegex();

  ReaderOptions.setReportList(); // Matched elements.
  ReaderOptions.setReportView(); // Parents for matched elements.

  // Add patterns.
  ReaderOptions.Select.Generic.insert("foo");
  ReaderOptions.Select.Generic.insert("movl[ \t]?%");
  ReaderOptions.Select.Generic.insert("INT[a-z]*");
  ReaderOptions.Select.Generic.insert("CONSTANT");

  ReaderOptions.resolveDependencies();

  std::vector<std::string> Objects;
  ScopedPrinter W(outs());
  LVReaderHandler ReaderHandler(Objects, W, ReaderOptions);

  // Check logical elements selection.
  std::unique_ptr<LVReader> Reader =
      createReader(ReaderHandler, InputsDir, DwarfGcc);
  checkElementSelection(Reader.get());
}

// Compare logical elements.
void compareElements(SmallString<128> &InputsDir) {
  // Reader options.
  LVOptions ReaderOptions;
  ReaderOptions.setAttributeOffset();
  ReaderOptions.setPrintLines();
  ReaderOptions.setPrintSymbols();
  ReaderOptions.setPrintTypes();
  ReaderOptions.setCompareLines();
  ReaderOptions.setCompareSymbols();
  ReaderOptions.setCompareTypes();

  ReaderOptions.resolveDependencies();

  std::vector<std::string> Objects;
  ScopedPrinter W(outs());
  LVReaderHandler ReaderHandler(Objects, W, ReaderOptions);

  // Check logical comparison.
  std::unique_ptr<LVReader> Reference =
      createReader(ReaderHandler, InputsDir, DwarfClang);
  std::unique_ptr<LVReader> Target =
      createReader(ReaderHandler, InputsDir, DwarfGcc);
  checkElementComparison(Reference.get(), Target.get());
}

TEST(LogicalViewTest, DWARFReader) {
  // Initialize targets and assembly printers/parsers.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  InitializeAllDisassemblers();

  llvm::sys::InitializeCOMRAII COM(llvm::sys::COMThreadingMode::MultiThreaded);

  // This test requires a x86-registered-target.
  Triple TT;
  TT.setArch(Triple::x86_64);
  TT.setVendor(Triple::UnknownVendor);
  TT.setOS(Triple::UnknownOS);

  std::string TargetLookupError;
  if (!TargetRegistry::lookupTarget(TT, TargetLookupError))
    GTEST_SKIP();

  SmallString<128> InputsDir = unittest::getInputFileDirectory(TestMainArgv0);

  // Logical elements general properties and selection.
  elementProperties(InputsDir);
  elementSelection(InputsDir);

  // Compare logical elements.
  compareElements(InputsDir);
}

} // namespace
