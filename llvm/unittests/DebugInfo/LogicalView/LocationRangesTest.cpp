//===- llvm/unittest/DebugInfo/LogicalView/LocationRangesTest.cpp ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/LogicalView/Core/LVLine.h"
#include "llvm/DebugInfo/LogicalView/Core/LVReader.h"
#include "llvm/DebugInfo/LogicalView/Core/LVScope.h"
#include "llvm/DebugInfo/LogicalView/Core/LVSymbol.h"
#include "llvm/DebugInfo/LogicalView/Core/LVType.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::logicalview;

namespace {

class ReaderTest : public LVReader {
protected:
  void add(LVSymbol *Symbol, LVLine *LowerLine, LVLine *UpperLine);
  void add(LVScope *Parent, LVElement *Element);
  void set(LVElement *Element, StringRef Name, LVOffset Offset,
           uint32_t LineNumber = 0, LVElement *Type = nullptr);
  void set(LVLocation *Location, LVLine *LowerLine, LVLine *UpperLine,
           LVAddress LowerAddress, LVAddress UpperAddress);

public:
  ReaderTest(ScopedPrinter &W) : LVReader("", "", W) { setInstance(this); }

  Error createScopes() { return LVReader::createScopes(); }
};

// Helper function to add a logical element to a given scope.
void ReaderTest::add(LVScope *Parent, LVElement *Child) {
  Parent->addElement(Child);
  EXPECT_EQ(Child->getParent(), Parent);
  EXPECT_EQ(Child->getLevel(), Parent->getLevel() + 1);
}

// Helper function to set the initial values for a given logical element.
void ReaderTest::set(LVElement *Element, StringRef Name, LVOffset Offset,
                     uint32_t LineNumber, LVElement *Type) {
  Element->setName(Name);
  Element->setOffset(Offset);
  Element->setLineNumber(LineNumber);
  Element->setType(Type);
  EXPECT_EQ(Element->getName(), Name);
  EXPECT_EQ(Element->getOffset(), Offset);
  EXPECT_EQ(Element->getLineNumber(), LineNumber);
  EXPECT_EQ(Element->getType(), Type);
}

// Helper function to set the initial values for a given logical location.
void ReaderTest::set(LVLocation *Location, LVLine *LowerLine, LVLine *UpperLine,
                     LVAddress LowerAddress, LVAddress UpperAddress) {
  Location->setLowerLine(LowerLine);
  Location->setUpperLine(UpperLine);
  Location->setLowerAddress(LowerAddress);
  Location->setUpperAddress(UpperAddress);
  EXPECT_EQ(Location->getLowerLine(), LowerLine);
  EXPECT_EQ(Location->getUpperLine(), UpperLine);
  EXPECT_EQ(Location->getLowerAddress(), LowerAddress);
  EXPECT_EQ(Location->getUpperAddress(), UpperAddress);
}

// Helper function to add a logical location to a logical symbol.
void ReaderTest::add(LVSymbol *Symbol, LVLine *LowerLine, LVLine *UpperLine) {
  dwarf::Attribute Attr = dwarf::DW_AT_location;

  Symbol->addLocation(Attr, LowerLine->getAddress(), UpperLine->getAddress(),
                      /*SectionOffset=*/0, /*LocDesOffset=*/0);
}

class ReaderTestLocations : public ReaderTest {
#define CREATE(VARIABLE, CREATE_FUNCTION)                                      \
  VARIABLE = CREATE_FUNCTION();                                                \
  EXPECT_NE(VARIABLE, nullptr);

  // Types.
  LVType *IntegerType = nullptr;

  // Scopes.
  LVScope *NestedScope = nullptr;
  LVScopeFunction *Function = nullptr;

  // Symbols.
  LVSymbol *LocalVariable = nullptr;
  LVSymbol *NestedVariable = nullptr;
  LVSymbol *Parameter = nullptr;

  // Lines.
  LVLine *LineOne = nullptr;
  LVLine *LineTwo = nullptr;
  LVLine *LineThree = nullptr;
  LVLine *LineFour = nullptr;
  LVLine *LineFive = nullptr;
  LVLine *LineSix = nullptr;

  // Locations.
  LVLocation *LocationOne = nullptr;
  LVLocation *LocationTwo = nullptr;
  LVLocation *LocationThree = nullptr;
  LVLocation *LocationFour = nullptr;
  LVLocation *LocationFive = nullptr;
  LVLocation *LocationSix = nullptr;

public:
  ReaderTestLocations(ScopedPrinter &W) : ReaderTest(W) {}

  void createElements();
  void addElements();
  void initElements();
};

// Create the logical elements.
void ReaderTestLocations::createElements() {
  // Create scope root.
  Error Err = createScopes();
  ASSERT_THAT_ERROR(std::move(Err), Succeeded());
  Root = getScopesRoot();
  EXPECT_NE(Root, nullptr);

  // Create the logical types.
  CREATE(IntegerType, createType);

  // Create the logical scopes.
  CREATE(NestedScope, createScope);
  CREATE(CompileUnit, createScopeCompileUnit);
  CREATE(Function, createScopeFunction);

  // Create the logical symbols.
  CREATE(LocalVariable, createSymbol);
  CREATE(NestedVariable, createSymbol);
  CREATE(Parameter, createSymbol);

  // Create the logical lines.
  CREATE(LineOne, createLine);
  CREATE(LineTwo, createLine);
  CREATE(LineThree, createLine);
  CREATE(LineFour, createLine);
  CREATE(LineFive, createLine);
  CREATE(LineSix, createLine);

  // Create the logical locations.
  CREATE(LocationOne, createLocation);
  CREATE(LocationTwo, createLocation);
  CREATE(LocationThree, createLocation);
  CREATE(LocationFour, createLocation);
  CREATE(LocationFive, createLocation);
  CREATE(LocationSix, createLocation);
}

// Create the logical view adding the created logical elements.
void ReaderTestLocations::addElements() {
  setCompileUnit(CompileUnit);

  // Root
  //   CompileUnit
  //     IntegerType
  //     Function
  //       LocationOne
  //       LocationTwo
  //       LocationFive
  //       LocationSix
  //       Parameter
  //       LocalVariable
  //       LineOne
  //       LineTwo
  //       NestedScope
  //         LocationThree
  //         LocationFour
  //         NestedVariable
  //         LineThree
  //         LineFour
  //       LineFive
  //       LineSix

  // Add elements to Root.
  add(Root, CompileUnit);

  // Add elements to CompileUnit.
  add(CompileUnit, IntegerType);
  add(CompileUnit, Function);

  // Add elements to Function.
  add(Function, Parameter);
  add(Function, LocalVariable);
  add(Function, LineOne);
  add(Function, LineTwo);
  add(Function, LineFive);
  add(Function, LineSix);
  add(Function, NestedScope);

  // Add elements to NestedScope.
  add(NestedScope, NestedVariable);
  add(NestedScope, LineThree);
  add(NestedScope, LineFour);
}

// Set initial values to logical elements.
void ReaderTestLocations::initElements() {
  // Types.
  set(IntegerType, "int", 0x1000);

  // Scopes.
  set(CompileUnit, "foo.cpp", 0x2000);
  set(Function, "foo", 0x2010, 100, IntegerType);
  set(NestedScope, "", 0x2020, 300);

  // Symbols.
  set(Parameter, "Param", 0x3000, 110, IntegerType);
  set(LocalVariable, "LocalVariable", 0x3000, 120, IntegerType);
  set(NestedVariable, "NestedVariable", 0x3010, 310, IntegerType);

  // Lines.
  set(LineOne, "", 0x5000, 100);
  set(LineTwo, "", 0x5200, 200);
  set(LineThree, "", 0x5400, 300);
  set(LineFour, "", 0x5600, 400);
  set(LineFive, "", 0x5800, 500);
  set(LineSix, "", 0x6000, 600);

  // Locations.
  set(LocationOne, LineOne, LineOne, 0x5000, 0x5100);
  EXPECT_STREQ(LocationOne->getIntervalInfo().c_str(),
               " Lines 100:100 [0x0000005000:0x0000005100]");

  set(LocationTwo, LineTwo, LineTwo, 0x5200, 0x5300);
  EXPECT_STREQ(LocationTwo->getIntervalInfo().c_str(),
               " Lines 200:200 [0x0000005200:0x0000005300]");

  set(LocationThree, LineThree, LineThree, 0x5400, 0x5500);
  EXPECT_STREQ(LocationThree->getIntervalInfo().c_str(),
               " Lines 300:300 [0x0000005400:0x0000005500]");

  set(LocationFour, LineFour, LineFour, 0x5600, 0x5700);
  LocationFour->setIsAddressRange();
  EXPECT_STREQ(LocationFour->getIntervalInfo().c_str(),
               "{Range} Lines 400:400 [0x0000005600:0x0000005700]");

  set(LocationFive, LineFive, LineFive, 0x5800, 0x5900);
  LocationFive->setIsAddressRange();
  EXPECT_STREQ(LocationFive->getIntervalInfo().c_str(),
               "{Range} Lines 500:500 [0x0000005800:0x0000005900]");

  set(LocationSix, LineSix, LineSix, 0x6000, 0x6100);
  LocationSix->setIsAddressRange();
  EXPECT_STREQ(LocationSix->getIntervalInfo().c_str(),
               "{Range} Lines 600:600 [0x0000006000:0x0000006100]");

  // Add ranges to Function.
  // Function: LocationOne, LocationTwo, LocationFive, LocationSix
  Function->addObject(LocationOne);
  Function->addObject(LocationTwo);
  Function->addObject(LocationFive);
  Function->addObject(LocationSix);
  EXPECT_EQ(Function->rangeCount(), 4u);

  // Add ranges to NestedScope.
  // NestedScope: LocationThree, LocationFour
  NestedScope->addObject(LocationThree);
  NestedScope->addObject(LocationFour);
  EXPECT_EQ(NestedScope->rangeCount(), 2u);

  // Get all ranges.
  LVRange Ranges;
  CompileUnit->getRanges(Ranges);
  Ranges.startSearch();
  EXPECT_EQ(Ranges.getEntry(0x4000), nullptr);

  EXPECT_EQ(Ranges.getEntry(0x5060), Function);
  EXPECT_EQ(Ranges.getEntry(0x5850), Function);
  EXPECT_EQ(Ranges.getEntry(0x5010, 0x5090), Function);
  EXPECT_EQ(Ranges.getEntry(0x5210, 0x5290), Function);
  EXPECT_EQ(Ranges.getEntry(0x5810, 0x5890), Function);
  EXPECT_EQ(Ranges.getEntry(0x6010, 0x6090), Function);

  EXPECT_EQ(Ranges.getEntry(0x5400), NestedScope);
  EXPECT_EQ(Ranges.getEntry(0x5650), NestedScope);
  EXPECT_EQ(Ranges.getEntry(0x5410, 0x5490), NestedScope);
  EXPECT_EQ(Ranges.getEntry(0x5610, 0x5690), NestedScope);

  EXPECT_EQ(Ranges.getEntry(0x8000), nullptr);
  Ranges.endSearch();

  // Add locations to symbols.
  // Parameter:       [LineOne, LineSix]
  // LocalVariable:   [LineTwo, LineSix], [LineFour, LineFive]
  // NestedVariable:  [LineThree, LineFour]
  add(Parameter, LineOne, LineSix);
  add(LocalVariable, LineTwo, LineSix);
  add(LocalVariable, LineFour, LineFive);
  add(NestedVariable, LineThree, LineFour);

  LVLocation *Location;
  LVLocations Locations;
  Parameter->getLocations(Locations);
  ASSERT_EQ(Locations.size(), 1u);
  Location = Locations[0];
  EXPECT_EQ(Location->getLowerAddress(), LineOne->getAddress());
  EXPECT_EQ(Location->getUpperAddress(), LineSix->getAddress());

  Locations.clear();
  LocalVariable->getLocations(Locations);
  ASSERT_EQ(Locations.size(), 2u);
  Location = Locations[0];
  EXPECT_EQ(Location->getLowerAddress(), LineTwo->getAddress());
  EXPECT_EQ(Location->getUpperAddress(), LineSix->getAddress());
  Location = Locations[1];
  EXPECT_EQ(Location->getLowerAddress(), LineFour->getAddress());
  EXPECT_EQ(Location->getUpperAddress(), LineFive->getAddress());

  Locations.clear();
  NestedVariable->getLocations(Locations);
  ASSERT_EQ(Locations.size(), 1u);
  Location = Locations[0];
  EXPECT_EQ(Location->getLowerAddress(), LineThree->getAddress());
  EXPECT_EQ(Location->getUpperAddress(), LineFour->getAddress());
}

class ReaderTestCoverage : public ReaderTest {
  // Types.
  LVType *IntegerType = nullptr;

  // Scopes.
  LVScopeFunction *Function = nullptr;
  LVScopeFunctionInlined *InlinedFunction = nullptr;

  // Symbols.
  LVSymbol *Variable = nullptr;
  LVSymbol *Parameter = nullptr;

  // Lines.
  LVLine *LineOne = nullptr;
  LVLine *LineTwo = nullptr;
  LVLine *LineThree = nullptr;
  LVLine *LineFour = nullptr;
  LVLine *LineFive = nullptr;
  LVLine *LineSix = nullptr;

  // Locations.
  LVLocation *LocationOne = nullptr;
  LVLocation *LocationTwo = nullptr;
  LVLocation *LocationFive = nullptr;
  LVLocation *LocationSix = nullptr;

public:
  ReaderTestCoverage(ScopedPrinter &W) : ReaderTest(W) {}

  void createElements();
  void addElements();
  void initElements();
};

// Create the logical elements.
void ReaderTestCoverage::createElements() {
  // Create scope root.
  Error Err = createScopes();
  ASSERT_THAT_ERROR(std::move(Err), Succeeded());
  Root = getScopesRoot();
  EXPECT_NE(Root, nullptr);

  // Create the logical types.
  IntegerType = createType();
  EXPECT_NE(IntegerType, nullptr);

  // Create the logical scopes.
  CompileUnit = createScopeCompileUnit();
  EXPECT_NE(CompileUnit, nullptr);
  Function = createScopeFunction();
  EXPECT_NE(Function, nullptr);
  InlinedFunction = createScopeFunctionInlined();
  EXPECT_NE(InlinedFunction, nullptr);

  // Create the logical symbols.
  Variable = createSymbol();
  EXPECT_NE(Variable, nullptr);
  Parameter = createSymbol();
  EXPECT_NE(Parameter, nullptr);

  // Create the logical lines.
  LineOne = createLine();
  EXPECT_NE(LineOne, nullptr);
  LineTwo = createLine();
  EXPECT_NE(LineTwo, nullptr);
  LineThree = createLine();
  EXPECT_NE(LineThree, nullptr);
  LineFour = createLine();
  EXPECT_NE(LineFour, nullptr);
  LineFive = createLine();
  EXPECT_NE(LineFive, nullptr);
  LineSix = createLine();
  EXPECT_NE(LineSix, nullptr);

  // Create the logical locations.
  LocationOne = createLocation();
  EXPECT_NE(LocationOne, nullptr);
  LocationTwo = createLocation();
  EXPECT_NE(LocationTwo, nullptr);
  LocationFive = createLocation();
  EXPECT_NE(LocationFive, nullptr);
  LocationSix = createLocation();
  EXPECT_NE(LocationSix, nullptr);
}

// Create the logical view adding the created logical elements.
void ReaderTestCoverage::addElements() {
  setCompileUnit(CompileUnit);

  // Root
  //   CompileUnit
  //     IntegerType
  //     Function
  //       Ranges
  //         [LineOne, LineOne]
  //         [LineTwo, LineSix]
  //         [LineSix, LineSix]
  //       LineOne
  //       LineTwo
  //       InlinedFunction
  //         Ranges
  //           [LineFive, LineFive]
  //         Parameter
  //           Location
  //             [LineThree, LineThree]
  //         Variable
  //           Location
  //             [LineFour, LineFive]
  //             [LineFive, LineSix]
  //         LineThree
  //         LineFour
  //         LineFive
  //       LineSix

  // Add elements to Root.
  add(Root, CompileUnit);

  // Add elements to CompileUnit.
  add(CompileUnit, IntegerType);
  add(CompileUnit, Function);

  // Add elements to Function.
  add(Function, InlinedFunction);
  add(Function, LineOne);
  add(Function, LineTwo);
  add(Function, LineSix);

  // Add elements to function InlinedFunction.
  add(InlinedFunction, Parameter);
  add(InlinedFunction, Variable);
  add(InlinedFunction, LineThree);
  add(InlinedFunction, LineFour);
  add(InlinedFunction, LineFive);
}

// Set initial values to logical elements.
void ReaderTestCoverage::initElements() {
  // Types.
  set(IntegerType, "int", 0x1000);

  // Scopes.
  set(CompileUnit, "foo.cpp", 0x2000);
  set(Function, "foo", 0x2500, 100, IntegerType);
  set(InlinedFunction, "InlinedFunction", 0x3000, 300);

  // Symbols.
  set(Parameter, "Parameter", 0x3100, 310, IntegerType);
  set(Variable, "Variable", 0x3200, 320, IntegerType);

  // Lines.
  set(LineOne, "", 0x5000, 100);
  set(LineTwo, "", 0x5200, 200);
  set(LineThree, "", 0x5400, 300);
  set(LineFour, "", 0x5600, 400);
  set(LineFive, "", 0x5800, 500);
  set(LineSix, "", 0x6000, 600);

  // Locations.
  set(LocationOne, LineOne, LineOne, 0x5000, 0x5199);
  EXPECT_STREQ(LocationOne->getIntervalInfo().c_str(),
               " Lines 100:100 [0x0000005000:0x0000005199]");

  set(LocationTwo, LineTwo, LineSix, 0x5200, 0x6100);
  EXPECT_STREQ(LocationTwo->getIntervalInfo().c_str(),
               " Lines 200:600 [0x0000005200:0x0000006100]");

  set(LocationFive, LineFive, LineFive, 0x5800, 0x5900);
  EXPECT_STREQ(LocationFive->getIntervalInfo().c_str(),
               " Lines 500:500 [0x0000005800:0x0000005900]");

  set(LocationSix, LineSix, LineSix, 0x6000, 0x6100);
  EXPECT_STREQ(LocationSix->getIntervalInfo().c_str(),
               " Lines 600:600 [0x0000006000:0x0000006100]");

  // Add ranges to Function.
  // Function: LocationOne, LocationTwo, LocationSix
  Function->addObject(LocationOne);
  Function->addObject(LocationTwo);
  Function->addObject(LocationSix);
  EXPECT_EQ(Function->rangeCount(), 3u);

  // Add ranges to Inlined.
  // Inlined: LocationFive
  InlinedFunction->addObject(LocationFive);
  EXPECT_EQ(InlinedFunction->rangeCount(), 1u);

  // Add locations to symbols.
  // Parameter: [LineThree, LineThree]
  // Variable:  [LineFour, LineFive], [LineFive, LineSix]
  add(Parameter, LineThree, LineThree);
  add(Variable, LineFour, LineFive);
  add(Variable, LineFive, LineSix);

  CompileUnit->processRangeLocationCoverage();

  LVLocation *Location;
  LVLocations Locations;
  Parameter->getLocations(Locations);
  ASSERT_EQ(Locations.size(), 1u);
  Location = Locations[0];
  EXPECT_EQ(Location->getLowerAddress(), LineThree->getAddress());
  EXPECT_EQ(Location->getUpperAddress(), LineThree->getAddress());

  Locations.clear();
  Variable->getLocations(Locations);
  ASSERT_EQ(Locations.size(), 2u);
  Location = Locations[0];
  EXPECT_EQ(Location->getLowerAddress(), LineFour->getAddress());
  EXPECT_EQ(Location->getUpperAddress(), LineFive->getAddress());
  Location = Locations[1];
  EXPECT_EQ(Location->getLowerAddress(), LineFive->getAddress());
  EXPECT_EQ(Location->getUpperAddress(), LineSix->getAddress());

  // Test the changes done to 'LVScope::outermostParent' to use the
  // ranges allocated int the current scope during the scopes traversal.
  // These are the pre-conditions for the symbol:
  // - Its parent must be an inlined function.
  // - Have more than one location description.

  // Before the changes: Parameter: CoveragePercentage = 100.00
  // After the changes:  Parameter: CoveragePercentage = 100.00
  EXPECT_FLOAT_EQ(Parameter->getCoveragePercentage(), 100.00f);

  // Before the changes: Variable: CoveragePercentage = 1000.00
  // After the changes:  Variable: CoveragePercentage = 56.83
  EXPECT_FLOAT_EQ(Variable->getCoveragePercentage(), 56.83f);
}

TEST(LogicalViewTest, LocationRanges) {
  ScopedPrinter W(outs());
  ReaderTestLocations Reader(W);

  // Reader options.
  LVOptions ReaderOptions;
  ReaderOptions.setAttributeOffset();
  ReaderOptions.setPrintAll();
  ReaderOptions.resolveDependencies();
  options().setOptions(&ReaderOptions);

  Reader.createElements();
  Reader.addElements();
  Reader.initElements();
}

TEST(LogicalViewTest, LocationCoverage) {
  ScopedPrinter W(outs());
  ReaderTestCoverage Reader(W);

  // Reader options.
  LVOptions ReaderOptions;
  ReaderOptions.setAttributeOffset();
  ReaderOptions.setAttributeRange();
  ReaderOptions.setAttributeLocation();
  ReaderOptions.setPrintAll();
  ReaderOptions.resolveDependencies();
  options().setOptions(&ReaderOptions);

  Reader.createElements();
  Reader.addElements();
  Reader.initElements();
}

} // namespace
