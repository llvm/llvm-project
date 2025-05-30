//===- llvm/unittest/DebugInfo/LogicalView/CodeViewReaderTest.cpp ---------===//
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
#include <algorithm>

using namespace llvm;
using namespace llvm::logicalview;

extern const char *TestMainArgv0;

namespace {

const char *CodeViewClang = "test-codeview-clang.o";
const char *CodeViewMsvc = "test-codeview-msvc.o";
const char *CodeViewMsvcLib = "test-codeview-msvc.lib";
const char *CodeViewMsvcLibContentName =
    "test-codeview-msvc.lib(test-codeview-msvc.o)";
const char *CodeViewPdbMsvc = "test-codeview-pdb-msvc.o";

// Helper function to get the first scope child from the given parent.
LVScope *getFirstScopeChild(LVScope *Parent) {
  EXPECT_NE(Parent, nullptr);
  const LVScopes *Scopes = Parent->getScopes();
  EXPECT_NE(Scopes, nullptr);
  EXPECT_EQ(Scopes->size(), 1u);

  LVScopes::const_iterator Iter = Scopes->begin();
  LVScope *Child = *Iter;
  EXPECT_NE(Child, nullptr);
  return Child;
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

// Check the logical elements basic properties (Clang - Codeview).
void checkElementPropertiesClangCodeview(LVReader *Reader) {
  LVScopeRoot *Root = Reader->getScopesRoot();
  LVScopeCompileUnit *CompileUnit =
      static_cast<LVScopeCompileUnit *>(getFirstScopeChild(Root));
  LVScopeFunction *Function =
      static_cast<LVScopeFunction *>(getFirstScopeChild(CompileUnit));

  EXPECT_EQ(Root->getFileFormatName(), "COFF-x86-64");
  EXPECT_EQ(Root->getName(), CodeViewClang);

  EXPECT_EQ(CompileUnit->getBaseAddress(), 0u);
  EXPECT_TRUE(CompileUnit->getProducer().starts_with("clang"));
  EXPECT_EQ(CompileUnit->getName(), "test.cpp");

  EXPECT_EQ(Function->lineCount(), 16u);
  EXPECT_EQ(Function->scopeCount(), 1u);
  EXPECT_EQ(Function->symbolCount(), 3u);
  EXPECT_EQ(Function->typeCount(), 1u);
  EXPECT_EQ(Function->rangeCount(), 1u);

  const LVLocations *Ranges = Function->getRanges();
  ASSERT_NE(Ranges, nullptr);
  ASSERT_EQ(Ranges->size(), 1u);
  LVLocations::const_iterator IterLocation = Ranges->begin();
  LVLocation *Location = (*IterLocation);
  EXPECT_STREQ(Location->getIntervalInfo().c_str(),
               "{Range} Lines 2:9 [0x0000000000:0x0000000046]");

  LVRange RangeList;
  Function->getRanges(RangeList);

  const LVRangeEntries &RangeEntries = RangeList.getEntries();
  ASSERT_EQ(RangeEntries.size(), 2u);
  LVRangeEntries::const_iterator IterRanges = RangeEntries.cbegin();
  LVRangeEntry RangeEntry = *IterRanges;
  EXPECT_EQ(RangeEntry.lower(), 0u);
  EXPECT_EQ(RangeEntry.upper(), 0x46u);
  EXPECT_EQ(RangeEntry.scope()->getLineNumber(), 0u);
  EXPECT_EQ(RangeEntry.scope()->getName(), "foo");
  EXPECT_EQ(RangeEntry.scope()->getOffset(), 0u);

  ++IterRanges;
  RangeEntry = *IterRanges;
  EXPECT_EQ(RangeEntry.lower(), 0x21u);
  EXPECT_EQ(RangeEntry.upper(), 0x35u);
  EXPECT_EQ(RangeEntry.scope()->getLineNumber(), 0u);
  EXPECT_EQ(RangeEntry.scope()->getName(), "foo::?");
  EXPECT_EQ(RangeEntry.scope()->getOffset(), 0u);

  const LVPublicNames &PublicNames = CompileUnit->getPublicNames();
  ASSERT_EQ(PublicNames.size(), 1u);
  LVPublicNames::const_iterator IterNames = PublicNames.cbegin();
  LVScope *Foo = (*IterNames).first;
  EXPECT_EQ(Foo->getName(), "foo");
  EXPECT_EQ(Foo->getLineNumber(), 0u);
  LVNameInfo NameInfo = (*IterNames).second;
  EXPECT_EQ(NameInfo.first, 0u);
  EXPECT_EQ(NameInfo.second, 0x46u);

  // Lines (debug and assembler) for 'foo'.
  const LVLines *Lines = Foo->getLines();
  ASSERT_NE(Lines, nullptr);
  EXPECT_EQ(Lines->size(), 0x10u);

  // Check size of types in CompileUnit.
  const LVTypes *Types = CompileUnit->getTypes();
  ASSERT_NE(Types, nullptr);
  EXPECT_EQ(Types->size(), 6u);

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

// Check the logical elements basic properties (MSVC - Codeview).
void checkElementPropertiesMsvcCodeview(LVReader *Reader) {
  LVScopeRoot *Root = Reader->getScopesRoot();
  LVScopeCompileUnit *CompileUnit =
      static_cast<LVScopeCompileUnit *>(getFirstScopeChild(Root));
  LVScopeFunction *Function =
      static_cast<LVScopeFunction *>(getFirstScopeChild(CompileUnit));

  EXPECT_EQ(Root->getFileFormatName(), "COFF-x86-64");
  EXPECT_EQ(Root->getName(), CodeViewMsvc);

  EXPECT_EQ(CompileUnit->getBaseAddress(), 0u);
  EXPECT_TRUE(CompileUnit->getProducer().starts_with("Microsoft"));
  EXPECT_EQ(CompileUnit->getName(), "test.cpp");

  EXPECT_EQ(Function->lineCount(), 14u);
  EXPECT_EQ(Function->scopeCount(), 1u);
  EXPECT_EQ(Function->symbolCount(), 3u);
  EXPECT_EQ(Function->typeCount(), 0u);
  EXPECT_EQ(Function->rangeCount(), 1u);

  const LVLocations *Ranges = Function->getRanges();
  ASSERT_NE(Ranges, nullptr);
  ASSERT_EQ(Ranges->size(), 1u);
  LVLocations::const_iterator IterLocation = Ranges->begin();
  LVLocation *Location = (*IterLocation);
  EXPECT_STREQ(Location->getIntervalInfo().c_str(),
               "{Range} Lines 2:9 [0x0000000000:0x0000000031]");

  LVRange RangeList;
  Function->getRanges(RangeList);

  const LVRangeEntries &RangeEntries = RangeList.getEntries();
  ASSERT_EQ(RangeEntries.size(), 2u);
  LVRangeEntries::const_iterator IterRanges = RangeEntries.cbegin();
  LVRangeEntry RangeEntry = *IterRanges;
  EXPECT_EQ(RangeEntry.lower(), 0u);
  EXPECT_EQ(RangeEntry.upper(), 0x31u);
  EXPECT_EQ(RangeEntry.scope()->getLineNumber(), 0u);
  EXPECT_EQ(RangeEntry.scope()->getName(), "foo");
  EXPECT_EQ(RangeEntry.scope()->getOffset(), 0u);

  ++IterRanges;
  RangeEntry = *IterRanges;
  EXPECT_EQ(RangeEntry.lower(), 0x1bu);
  EXPECT_EQ(RangeEntry.upper(), 0x28u);
  EXPECT_EQ(RangeEntry.scope()->getLineNumber(), 0u);
  EXPECT_EQ(RangeEntry.scope()->getName(), "foo::?");
  EXPECT_EQ(RangeEntry.scope()->getOffset(), 0u);

  const LVPublicNames &PublicNames = CompileUnit->getPublicNames();
  ASSERT_EQ(PublicNames.size(), 1u);
  LVPublicNames::const_iterator IterNames = PublicNames.cbegin();
  LVScope *Foo = (*IterNames).first;
  EXPECT_EQ(Foo->getName(), "foo");
  EXPECT_EQ(Foo->getLineNumber(), 0u);
  LVNameInfo NameInfo = (*IterNames).second;
  EXPECT_EQ(NameInfo.first, 0u);
  EXPECT_EQ(NameInfo.second, 0x31u);

  // Lines (debug and assembler) for 'foo'.
  const LVLines *Lines = Foo->getLines();
  ASSERT_NE(Lines, nullptr);
  EXPECT_EQ(Lines->size(), 0x0eu);

  // Check size of types in CompileUnit.
  const LVTypes *Types = CompileUnit->getTypes();
  ASSERT_NE(Types, nullptr);
  EXPECT_EQ(Types->size(), 8u);

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

// Check the logical elements basic properties (MSVC library - Codeview).
void checkElementPropertiesMsvcLibraryCodeview(LVReader *Reader) {
  LVScopeRoot *Root = Reader->getScopesRoot();
  LVScopeCompileUnit *CompileUnit =
      static_cast<LVScopeCompileUnit *>(getFirstScopeChild(Root));
  LVScopeFunction *Function =
      static_cast<LVScopeFunction *>(getFirstScopeChild(CompileUnit));

  EXPECT_EQ(Root->getFileFormatName(), "COFF-x86-64");
  EXPECT_EQ(Root->getName(), CodeViewMsvcLibContentName);

  EXPECT_EQ(CompileUnit->getBaseAddress(), 0u);
  EXPECT_TRUE(CompileUnit->getProducer().starts_with("Microsoft"));
  EXPECT_EQ(CompileUnit->getName(), "test.cpp");

  EXPECT_EQ(Function->lineCount(), 14u);
  EXPECT_EQ(Function->scopeCount(), 1u);
  EXPECT_EQ(Function->symbolCount(), 3u);
  EXPECT_EQ(Function->typeCount(), 0u);
  EXPECT_EQ(Function->rangeCount(), 1u);

  const LVLocations *Ranges = Function->getRanges();
  ASSERT_NE(Ranges, nullptr);
  ASSERT_EQ(Ranges->size(), 1u);
  LVLocations::const_iterator IterLocation = Ranges->begin();
  LVLocation *Location = (*IterLocation);
  EXPECT_STREQ(Location->getIntervalInfo().c_str(),
               "{Range} Lines 2:9 [0x0000000000:0x0000000031]");

  LVRange RangeList;
  Function->getRanges(RangeList);

  const LVRangeEntries &RangeEntries = RangeList.getEntries();
  ASSERT_EQ(RangeEntries.size(), 2u);
  LVRangeEntries::const_iterator IterRanges = RangeEntries.cbegin();
  LVRangeEntry RangeEntry = *IterRanges;
  EXPECT_EQ(RangeEntry.lower(), 0u);
  EXPECT_EQ(RangeEntry.upper(), 0x31u);
  EXPECT_EQ(RangeEntry.scope()->getLineNumber(), 0u);
  EXPECT_EQ(RangeEntry.scope()->getName(), "foo");
  EXPECT_EQ(RangeEntry.scope()->getOffset(), 0u);

  ++IterRanges;
  RangeEntry = *IterRanges;
  EXPECT_EQ(RangeEntry.lower(), 0x1bu);
  EXPECT_EQ(RangeEntry.upper(), 0x28u);
  EXPECT_EQ(RangeEntry.scope()->getLineNumber(), 0u);
  EXPECT_EQ(RangeEntry.scope()->getName(), "foo::?");
  EXPECT_EQ(RangeEntry.scope()->getOffset(), 0u);

  const LVPublicNames &PublicNames = CompileUnit->getPublicNames();
  ASSERT_EQ(PublicNames.size(), 1u);
  LVPublicNames::const_iterator IterNames = PublicNames.cbegin();
  LVScope *Foo = (*IterNames).first;
  EXPECT_EQ(Foo->getName(), "foo");
  EXPECT_EQ(Foo->getLineNumber(), 0u);
  LVNameInfo NameInfo = (*IterNames).second;
  EXPECT_EQ(NameInfo.first, 0u);
  EXPECT_EQ(NameInfo.second, 0x31u);

  // Lines (debug and assembler) for 'foo'.
  const LVLines *Lines = Foo->getLines();
  ASSERT_NE(Lines, nullptr);
  EXPECT_EQ(Lines->size(), 0x0eu);
}

// Check the logical elements basic properties (MSVC - PDB).
void checkElementPropertiesMsvcCodeviewPdb(LVReader *Reader) {
  LVScopeRoot *Root = Reader->getScopesRoot();
  LVScopeCompileUnit *CompileUnit =
      static_cast<LVScopeCompileUnit *>(getFirstScopeChild(Root));
  LVScopeFunction *Function =
      static_cast<LVScopeFunction *>(getFirstScopeChild(CompileUnit));

  EXPECT_EQ(Root->getFileFormatName(), "COFF-x86-64");
  EXPECT_EQ(Root->getName(), CodeViewPdbMsvc);

  EXPECT_EQ(CompileUnit->getBaseAddress(), 0u);
  EXPECT_TRUE(CompileUnit->getProducer().starts_with("Microsoft"));
  EXPECT_EQ(CompileUnit->getName(), "test.cpp");

  EXPECT_EQ(Function->lineCount(), 14u);
  EXPECT_EQ(Function->scopeCount(), 1u);
  EXPECT_EQ(Function->symbolCount(), 3u);
  EXPECT_EQ(Function->typeCount(), 0u);
  EXPECT_EQ(Function->rangeCount(), 1u);

  const LVLocations *Ranges = Function->getRanges();
  ASSERT_NE(Ranges, nullptr);
  ASSERT_EQ(Ranges->size(), 1u);
  LVLocations::const_iterator IterLocation = Ranges->begin();
  LVLocation *Location = (*IterLocation);
  EXPECT_STREQ(Location->getIntervalInfo().c_str(),
               "{Range} Lines 2:9 [0x0000000000:0x0000000031]");

  LVRange RangeList;
  Function->getRanges(RangeList);

  const LVRangeEntries &RangeEntries = RangeList.getEntries();
  ASSERT_EQ(RangeEntries.size(), 2u);
  LVRangeEntries::const_iterator IterRanges = RangeEntries.cbegin();
  LVRangeEntry RangeEntry = *IterRanges;
  EXPECT_EQ(RangeEntry.lower(), 0u);
  EXPECT_EQ(RangeEntry.upper(), 0x31u);
  EXPECT_EQ(RangeEntry.scope()->getLineNumber(), 0u);
  EXPECT_EQ(RangeEntry.scope()->getName(), "foo");
  EXPECT_EQ(RangeEntry.scope()->getOffset(), 0u);

  ++IterRanges;
  RangeEntry = *IterRanges;
  EXPECT_EQ(RangeEntry.lower(), 0x1bu);
  EXPECT_EQ(RangeEntry.upper(), 0x28u);
  EXPECT_EQ(RangeEntry.scope()->getLineNumber(), 0u);
  EXPECT_EQ(RangeEntry.scope()->getName(), "foo::?");
  EXPECT_EQ(RangeEntry.scope()->getOffset(), 0u);

  const LVPublicNames &PublicNames = CompileUnit->getPublicNames();
  ASSERT_EQ(PublicNames.size(), 1u);
  LVPublicNames::const_iterator IterNames = PublicNames.cbegin();
  LVScope *Foo = (*IterNames).first;
  EXPECT_EQ(Foo->getName(), "foo");
  EXPECT_EQ(Foo->getLineNumber(), 0u);
  LVNameInfo NameInfo = (*IterNames).second;
  EXPECT_EQ(NameInfo.first, 0u);
  EXPECT_EQ(NameInfo.second, 0x31u);

  // Lines (debug and assembler) for 'foo'.
  const LVLines *Lines = Foo->getLines();
  ASSERT_NE(Lines, nullptr);
  EXPECT_EQ(Lines->size(), 0x0eu);
}

struct SelectionInfo {
  const char *Name;
  LVElementGetFunction Function;
};

// Check the logical elements selection.
void checkElementSelection(LVReader *Reader, std::vector<SelectionInfo> &Data,
                           size_t Size) {
  LVScopeRoot *Root = Reader->getScopesRoot();
  LVScopeCompileUnit *CompileUnit =
      static_cast<LVScopeCompileUnit *>(getFirstScopeChild(Root));

  // Get the matched elements.
  LVElements MatchedElements = CompileUnit->getMatchedElements();
  std::map<StringRef, LVElement *> MapElements;
  for (LVElement *Element : MatchedElements)
    MapElements[Element->getName()] = Element;
  ASSERT_EQ(MapElements.size(), Size);

  std::map<StringRef, LVElement *>::iterator Iter = MapElements.begin();
  for (const SelectionInfo &Entry : Data) {
    // Get matched element.
    EXPECT_NE(Iter, MapElements.end());
    LVElement *Element = Iter->second;
    ASSERT_NE(Element, nullptr);
    EXPECT_NE(Element->getName().find(Entry.Name), StringRef::npos);
    EXPECT_EQ((Element->*Entry.Function)(), 1u);
    ++Iter;
  }

  // Get the parents for the matched elements.
  LVScopes MatchedScopes = CompileUnit->getMatchedScopes();
  std::set<StringRef> SetScopes;
  for (LVScope *Scope : MatchedScopes)
    SetScopes.insert(Scope->getName());
  ASSERT_EQ(SetScopes.size(), 3u);

  // Parents of selected elements.
  std::set<StringRef>::iterator IterScope;
  IterScope = SetScopes.find("foo");
  EXPECT_NE(IterScope, SetScopes.end());
  IterScope = SetScopes.find("foo::?");
  EXPECT_NE(IterScope, SetScopes.end());
  IterScope = SetScopes.find("test.cpp");
  EXPECT_NE(IterScope, SetScopes.end());
}

// Check the logical elements comparison.
void checkElementComparison(LVReader *Reference, LVReader *Target) {
  LVCompare Compare(nulls());
  Error Err = Compare.execute(Reference, Target);
  ASSERT_THAT_ERROR(std::move(Err), Succeeded());

  // Get comparison table.
  LVPassTable PassTable = Compare.getPassTable();
  ASSERT_EQ(PassTable.size(), 2u);

  LVReader *Reader;
  LVElement *Element;
  LVComparePass Pass;

  // Reference: Missing TypeDefinition 'INTEGER'
  std::tie(Reader, Element, Pass) = PassTable[0];
  ASSERT_NE(Reader, nullptr);
  ASSERT_NE(Element, nullptr);
  EXPECT_EQ(Reader, Reference);
  EXPECT_EQ(Element->getLevel(), 3u);
  EXPECT_EQ(Element->getLineNumber(), 0u);
  EXPECT_EQ(Element->getName(), "INTEGER");
  EXPECT_EQ(Pass, LVComparePass::Missing);

  // Target: Added TypeDefinition 'INTEGER'
  std::tie(Reader, Element, Pass) = PassTable[1];
  ASSERT_NE(Reader, nullptr);
  ASSERT_NE(Element, nullptr);
  EXPECT_EQ(Reader, Target);
  EXPECT_EQ(Element->getLevel(), 4u);
  EXPECT_EQ(Element->getLineNumber(), 0u);
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
  ReaderOptions.setAttributeSize();
  ReaderOptions.setPrintAll();
  ReaderOptions.resolveDependencies();

  std::vector<std::string> Objects;
  ScopedPrinter W(outs());
  LVReaderHandler ReaderHandler(Objects, W, ReaderOptions);

  // Check logical elements properties.
  {
    std::unique_ptr<LVReader> Reader =
        createReader(ReaderHandler, InputsDir, CodeViewClang);
    checkElementPropertiesClangCodeview(Reader.get());
  }
  {
    std::unique_ptr<LVReader> Reader =
        createReader(ReaderHandler, InputsDir, CodeViewMsvc);
    checkElementPropertiesMsvcCodeview(Reader.get());
  }
  {
    std::unique_ptr<LVReader> Reader =
        createReader(ReaderHandler, InputsDir, CodeViewMsvcLib);
    checkElementPropertiesMsvcLibraryCodeview(Reader.get());
  }
  {
    std::unique_ptr<LVReader> Reader =
        createReader(ReaderHandler, InputsDir, CodeViewPdbMsvc);
    checkElementPropertiesMsvcCodeviewPdb(Reader.get());
  }
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
  {
    std::vector<SelectionInfo> DataClang = {
        {"* const int", &LVElement::getIsType},
        {"CONSTANT", &LVElement::getIsSymbol},
        {"INTEGER", &LVElement::getIsType},
        {"INTPTR", &LVElement::getIsType},
        {"ParamPtr", &LVElement::getIsSymbol},
        {"const int", &LVElement::getIsType},
        {"foo", &LVElement::getIsScope},
        {"foo::?", &LVElement::getIsScope},
        {"int", &LVElement::getIsType},
        {"movl", &LVElement::getIsLine},
        {"movl", &LVElement::getIsLine}};
    std::unique_ptr<LVReader> Reader =
        createReader(ReaderHandler, InputsDir, CodeViewClang);
    checkElementSelection(Reader.get(), DataClang, DataClang.size());
  }
  {
    std::vector<SelectionInfo> DataMsvc = {
        {"* const int", &LVElement::getIsType},
        {"CONSTANT", &LVElement::getIsSymbol},
        {"INTEGER", &LVElement::getIsType},
        {"INTPTR", &LVElement::getIsType},
        {"ParamPtr", &LVElement::getIsSymbol},
        {"const int", &LVElement::getIsType},
        {"foo", &LVElement::getIsScope},
        {"foo::?", &LVElement::getIsScope},
        {"int", &LVElement::getIsType},
        {"movl", &LVElement::getIsLine}};
    std::unique_ptr<LVReader> Reader =
        createReader(ReaderHandler, InputsDir, CodeViewMsvc);
    checkElementSelection(Reader.get(), DataMsvc, DataMsvc.size());
  }
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
      createReader(ReaderHandler, InputsDir, CodeViewClang);
  std::unique_ptr<LVReader> Target =
      createReader(ReaderHandler, InputsDir, CodeViewMsvc);
  checkElementComparison(Reference.get(), Target.get());
}

TEST(LogicalViewTest, CodeViewReader) {
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
    return;

  SmallString<128> InputsDir = unittest::getInputFileDirectory(TestMainArgv0);

  // Logical elements general properties and selection.
  elementProperties(InputsDir);
  elementSelection(InputsDir);

  // Compare logical elements.
  compareElements(InputsDir);
}

} // namespace
