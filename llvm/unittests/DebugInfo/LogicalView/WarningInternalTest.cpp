//===- llvm/unittest/DebugInfo/LogicalView/WarningInternalTest.cpp --------===//
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

class MyLocation : public LVLocation {
public:
  bool validateRanges();
};

// This code emulates the work done by the Readers when processing the
// binary files and the creation of the AddressToLine mapping is done
// automatically, using the text sections.
class MyAddressToLine {
  using LVAddressToLine = std::map<LVAddress, LVLine *>;
  LVAddressToLine AddressToLineData;

public:
  MyAddressToLine() = default;

  void insert(LVLine *Line) {
    AddressToLineData.emplace(Line->getOffset(), Line);
  }

  LVLine *lineLowerBound(LVAddress Address);
  LVLine *lineUpperBound(LVAddress Address);
};

LVLine *MyAddressToLine::lineLowerBound(LVAddress Address) {
  LVAddressToLine::const_iterator Iter = AddressToLineData.lower_bound(Address);
  return (Iter != AddressToLineData.end()) ? Iter->second : nullptr;
}

LVLine *MyAddressToLine::lineUpperBound(LVAddress Address) {
  if (AddressToLineData.empty())
    return nullptr;
  LVAddressToLine::const_iterator Iter = AddressToLineData.upper_bound(Address);
  if (Iter != AddressToLineData.begin())
    Iter = std::prev(Iter);
  return Iter->second;
}

MyAddressToLine AddressToLine;

class ReaderTestWarningInternal : public LVReader {
#define CREATE(VARIABLE, CREATE_FUNCTION)                                      \
  VARIABLE = CREATE_FUNCTION();                                                \
  EXPECT_NE(VARIABLE, nullptr);

#define CREATE_CUSTOM(VARIABLE, CREATE_FUNCTION)                               \
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
  MyLocation *LocationOne = nullptr;
  MyLocation *LocationTwo = nullptr;
  MyLocation *LocationThree = nullptr;
  MyLocation *LocationFour = nullptr;
  MyLocation *LocationFive = nullptr;
  MyLocation *LocationSix = nullptr;

  llvm::SpecificBumpPtrAllocator<MyLocation> AllocatedLocations;

protected:
  MyLocation *createCustomLocation() {
    return new (AllocatedLocations.Allocate()) MyLocation();
  }

  void add(LVSymbol *Symbol, LVLine *LowerLine, LVLine *UpperLine);
  void add(LVScope *Parent, LVElement *Element);
  void set(LVElement *Element, StringRef Name, LVOffset Offset,
           uint32_t LineNumber = 0, LVElement *Type = nullptr);
  void set(MyLocation *Location, LVLine *LowerLine, LVLine *UpperLine,
           LVAddress LowerAddress, LVAddress UpperAddress);

public:
  ReaderTestWarningInternal(ScopedPrinter &W) : LVReader("", "", W) {
    setInstance(this);
  }

  Error createScopes() { return LVReader::createScopes(); }

  void setMapping();
  void createElements();
  void addElements();
  void initElements();
  void resolveElements();
  void checkWarnings();
};

bool MyLocation::validateRanges() {
  // Traverse the locations and validate them against the address to line
  // mapping in the current compile unit. Record those invalid ranges.
  // A valid range must meet the following conditions:
  // a) line(lopc) <= line(hipc)
  // b) line(lopc) and line(hipc) are valid.

  LVLine *LowLine = AddressToLine.lineLowerBound(getLowerAddress());
  LVLine *HighLine = AddressToLine.lineUpperBound(getUpperAddress());
  if (LowLine)
    setLowerLine(LowLine);
  else {
    setIsInvalidLower();
    return false;
  }
  if (HighLine)
    setUpperLine(HighLine);
  else {
    setIsInvalidUpper();
    return false;
  }
  // Check for a valid interval.
  if (LowLine->getLineNumber() > HighLine->getLineNumber()) {
    setIsInvalidRange();
    return false;
  }

  return true;
}

// Map all logical lines with their addresses.
void ReaderTestWarningInternal::setMapping() {
  AddressToLine.insert(LineOne);
  AddressToLine.insert(LineTwo);
  AddressToLine.insert(LineThree);
  AddressToLine.insert(LineFour);
  AddressToLine.insert(LineFive);
  AddressToLine.insert(LineSix);
}

// Helper function to add a logical element to a given scope.
void ReaderTestWarningInternal::add(LVScope *Parent, LVElement *Child) {
  Parent->addElement(Child);
  EXPECT_EQ(Child->getParent(), Parent);
  EXPECT_EQ(Child->getLevel(), Parent->getLevel() + 1);
}

// Helper function to set the initial values for a given logical element.
void ReaderTestWarningInternal::set(LVElement *Element, StringRef Name,
                                    LVOffset Offset, uint32_t LineNumber,
                                    LVElement *Type) {
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
void ReaderTestWarningInternal::set(MyLocation *Location, LVLine *LowerLine,
                                    LVLine *UpperLine, LVAddress LowerAddress,
                                    LVAddress UpperAddress) {
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
void ReaderTestWarningInternal::add(LVSymbol *Symbol, LVLine *LowerLine,
                                    LVLine *UpperLine) {
  dwarf::Attribute Attr = dwarf::DW_AT_location;

  Symbol->addLocation(Attr, LowerLine->getAddress(), UpperLine->getAddress(),
                      /*SectionOffset=*/0, /*LocDesOffset=*/0);
}

// Create the logical elements.
void ReaderTestWarningInternal::createElements() {
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
  CREATE_CUSTOM(LocationOne, createCustomLocation);
  CREATE_CUSTOM(LocationTwo, createCustomLocation);
  CREATE_CUSTOM(LocationThree, createCustomLocation);
  CREATE_CUSTOM(LocationFour, createCustomLocation);
  CREATE_CUSTOM(LocationFive, createCustomLocation);
  CREATE_CUSTOM(LocationSix, createCustomLocation);
}

// Create the logical view adding the created logical elements.
void ReaderTestWarningInternal::addElements() {
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

void ReaderTestWarningInternal::resolveElements() {
  // Traverse the given scope and its children checking for any warnings.
  std::function<void(LVScope * Parent)> TraverseScope = [&](LVScope *Parent) {
    auto Warnings = [&](auto *Entry) {
      if (Entry->getIsLine()) {
        LVLine *Line = (LVLine *)Entry;
        if (options().getWarningLines() && Line->getIsLineDebug() &&
            !Line->getLineNumber())
          CompileUnit->addLineZero(Line);
      }
    };
    auto Traverse = [&](const auto *Set) {
      if (Set)
        for (const auto &Entry : *Set) {
          Warnings(Entry);
        }
    };

    Warnings(Parent);

    Traverse(Parent->getSymbols());
    Traverse(Parent->getTypes());
    Traverse(Parent->getLines());

    if (const LVScopes *Scopes = Parent->getScopes())
      for (LVScope *Scope : *Scopes) {
        Warnings(Scope);
        TraverseScope(Scope);
      }
  };

  // Start traversing the scopes root and resolve the elements.
  TraverseScope(Root);
}

// Set initial values to logical elements.
void ReaderTestWarningInternal::initElements() {
  // Types.
  set(IntegerType, "int", 0x1000);

  // Scopes.
  set(CompileUnit, "foo.cpp", 0x2000);
  set(Function, "foo", 0x2010, 100, IntegerType);
  set(NestedScope, "", 0x2020, 300);

  // Symbols.
  set(Parameter, "Param", 0x3000, 110, IntegerType);
  set(LocalVariable, "LocalVariable", 0x3020, 120, IntegerType);
  set(NestedVariable, "NestedVariable", 0x3010, 310, IntegerType);

  // Lines.
  set(LineOne, "", 0x5000, 100);
  LineOne->setIsLineDebug();
  set(LineTwo, "", 0x5200, 000);
  LineTwo->setIsLineDebug();
  set(LineThree, "", 0x5400, 300);
  LineThree->setIsLineDebug();
  set(LineFour, "", 0x5600, 000);
  LineFour->setIsLineDebug();
  set(LineFive, "", 0x5800, 500);
  LineOne->setIsLineDebug();
  set(LineSix, "", 0x6000, 600);
  LineSix->setIsLineDebug();

  // Locations.
  set(LocationOne, LineOne, LineOne, 0x5000, 0x5100);
  EXPECT_STREQ(LocationOne->getIntervalInfo().c_str(),
               " Lines 100:100 [0x0000005000:0x0000005100]");

  // Uses a Line zero.
  set(LocationTwo, LineTwo, LineTwo, 0x5200, 0x5300);
  EXPECT_STREQ(LocationTwo->getIntervalInfo().c_str(),
               " Lines -:- [0x0000005200:0x0000005300]");

  set(LocationThree, LineThree, LineThree, 0x5400, 0x5500);
  EXPECT_STREQ(LocationThree->getIntervalInfo().c_str(),
               " Lines 300:300 [0x0000005400:0x0000005500]");

  // Uses a Line zero.
  set(LocationFour, LineFour, LineFour, 0x5600, 0x5700);
  LocationFour->setIsAddressRange();
  EXPECT_STREQ(LocationFour->getIntervalInfo().c_str(),
               "{Range} Lines -:- [0x0000005600:0x0000005700]");

  // Invalid range.
  set(LocationFive, LineFive, LineFive, 0x7800, 0x5900);
  LocationFive->setIsAddressRange();
  EXPECT_STREQ(LocationFive->getIntervalInfo().c_str(),
               "{Range} Lines 500:500 [0x0000007800:0x0000005900]");

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
  EXPECT_EQ(Ranges.getEntry(0x5850), nullptr);
  EXPECT_EQ(Ranges.getEntry(0x5010, 0x5090), Function);
  EXPECT_EQ(Ranges.getEntry(0x5210, 0x5290), Function);
  EXPECT_EQ(Ranges.getEntry(0x5810, 0x5890), nullptr);
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
  add(NestedVariable, LineOne, LineSix);
}

// Check logical elements warnigs.
void ReaderTestWarningInternal::checkWarnings() {
  // Map all lines with their addresses.
  setMapping();

  // Check for lines with line zero.
  resolveElements();

  // Check invalid locations and ranges using a customized validation.
  CompileUnit->processRangeLocationCoverage(
      (LVValidLocation)(&MyLocation::validateRanges));

  // Get lines with line zero. [Parent, Line]
  //   Function, LineTwo
  //   NestedScope, LineFour
  LVOffsetLinesMap LinesZero = CompileUnit->getLinesZero();
  ASSERT_EQ(LinesZero.size(), 2u);

  LVOffsetLinesMap::iterator IterZero = LinesZero.begin();
  EXPECT_EQ(IterZero->first, Function->getOffset());
  LVLines *Lines = &IterZero->second;
  EXPECT_NE(Lines, nullptr);
  ASSERT_EQ(Lines->size(), 1u);
  LVLine *Line = *(Lines->begin());
  EXPECT_NE(Line, nullptr);
  EXPECT_EQ(Line, LineTwo);

  ++IterZero;
  EXPECT_EQ(IterZero->first, NestedScope->getOffset());
  Lines = &IterZero->second;
  EXPECT_NE(Lines, nullptr);
  ASSERT_EQ(Lines->size(), 1u);
  Line = *(Lines->begin());
  EXPECT_NE(Line, nullptr);
  EXPECT_EQ(Line, LineFour);

  // Elements with invalid offsets.
  //   Function (line zero)
  //   NestedScope (line zero)
  //   NestedVariable (invalid location)
  LVOffsetElementMap InvalidOffsets = CompileUnit->getWarningOffsets();
  ASSERT_EQ(InvalidOffsets.size(), 3u);

  LVOffsetElementMap::iterator IterOffset = InvalidOffsets.begin();
  EXPECT_EQ(IterOffset->second, Function);
  ++IterOffset;
  EXPECT_EQ(IterOffset->second, NestedScope);
  ++IterOffset;
  EXPECT_EQ(IterOffset->second, NestedVariable);

  // Invalid ranges.
  //   Function
  LVOffsetLocationsMap InvalidRanges = CompileUnit->getInvalidRanges();
  ASSERT_EQ(InvalidRanges.size(), 1u);

  LVOffsetLocationsMap::iterator IterRange = InvalidRanges.begin();
  EXPECT_EQ(IterRange->first, Function->getOffset());
  LVLocations *Locations = &IterRange->second;
  EXPECT_NE(Locations, nullptr);
  ASSERT_EQ(Locations->size(), 1u);
  LVLocation *Location = *(Locations->begin());
  EXPECT_NE(Location, nullptr);
  EXPECT_EQ(Location, LocationFive);

  // Invalid location.
  //   NestedVariable
  LVOffsetLocationsMap InvalidLocations = CompileUnit->getInvalidLocations();
  ASSERT_EQ(InvalidLocations.size(), 1u);

  LVOffsetLocationsMap::iterator IterLocations = InvalidLocations.begin();
  EXPECT_EQ(IterLocations->first, NestedVariable->getOffset());
  Locations = &IterLocations->second;
  EXPECT_NE(Locations, nullptr);
  ASSERT_EQ(Locations->size(), 1u);
  Location = *(Locations->begin());
  EXPECT_NE(Location, nullptr);
  EXPECT_EQ(Location->getLowerAddress(), LocationThree->getLowerAddress());
  EXPECT_EQ(Location->getUpperAddress(), LocationFour->getLowerAddress());
  EXPECT_EQ(Location->getLowerLine()->getLineNumber(),
            LineThree->getLineNumber());
  EXPECT_EQ(Location->getUpperLine()->getLineNumber(), 0u);

  // Invalid coverages.
  //   NestedVariable
  LVOffsetSymbolMap InvalidCoverages = CompileUnit->getInvalidCoverages();
  ASSERT_EQ(InvalidCoverages.size(), 1u);

  LVOffsetSymbolMap::iterator IterCoverages = InvalidCoverages.begin();
  EXPECT_EQ(IterCoverages->first, NestedVariable->getOffset());
  EXPECT_EQ(IterCoverages->second, NestedVariable);
  EXPECT_GE((int)NestedVariable->getCoveragePercentage(), 100);
  EXPECT_EQ((int)NestedVariable->getCoveragePercentage(), 900);
  EXPECT_EQ(NestedVariable->getCoverageFactor(), 0x1200u);

  EXPECT_EQ((unsigned)Parameter->getCoveragePercentage(), 100u);
  EXPECT_EQ(Parameter->getCoverageFactor(), 100u);

  EXPECT_EQ((unsigned)LocalVariable->getCoveragePercentage(), 47u);
  EXPECT_EQ(LocalVariable->getCoverageFactor(),
            LineSix->getAddress() - LineOne->getAddress());
}

TEST(LogicalViewTest, WarningInternal) {
  ScopedPrinter W(outs());
  ReaderTestWarningInternal Reader(W);

  // Reader options.
  LVOptions ReaderOptions;
  ReaderOptions.setAttributeOffset();
  ReaderOptions.setAttributeRange();
  ReaderOptions.setAttributeLocation();
  ReaderOptions.setPrintAll();
  ReaderOptions.setWarningCoverages();
  ReaderOptions.setWarningLines();
  ReaderOptions.setWarningLocations();
  ReaderOptions.setWarningRanges();
  ReaderOptions.resolveDependencies();
  options().setOptions(&ReaderOptions);

  Reader.createElements();
  Reader.addElements();
  Reader.initElements();
  Reader.checkWarnings();
}

} // namespace
