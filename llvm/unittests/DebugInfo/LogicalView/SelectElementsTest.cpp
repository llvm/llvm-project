//===- llvm/unittest/DebugInfo/LogicalView/SelectElementsTest.cpp ---------===//
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

class ReaderTestSelection : public LVReader {
#define CREATE(VARIABLE, CREATE_FUNCTION, SET_FUNCTION)                        \
  VARIABLE = CREATE_FUNCTION();                                                \
  EXPECT_NE(VARIABLE, nullptr);                                                \
  VARIABLE->SET_FUNCTION();

  // Types.
  LVType *IntegerType = nullptr;

  // Scopes.
  LVScope *NestedScope = nullptr;
  LVScopeAggregate *Aggregate = nullptr;
  LVScopeFunction *Function = nullptr;
  LVScopeNamespace *Namespace = nullptr;

  // Symbols.
  LVSymbol *ClassMember = nullptr;
  LVSymbol *LocalVariable = nullptr;
  LVSymbol *NestedVariable = nullptr;
  LVSymbol *Parameter = nullptr;

  // Lines.
  LVLine *LineOne = nullptr;
  LVLine *LineTwo = nullptr;
  LVLine *LineThree = nullptr;
  LVLine *LineFour = nullptr;
  LVLine *LineFive = nullptr;

protected:
  void add(LVScope *Parent, LVElement *Element);
  void set(LVElement *Element, StringRef Name, LVOffset Offset,
           uint32_t LineNumber = 0, LVElement *Type = nullptr);

public:
  ReaderTestSelection(ScopedPrinter &W) : LVReader("", "", W) {
    setInstance(this);
  }

  Error createScopes() { return LVReader::createScopes(); }

  void createElements();
  void addElements();
  void initElements();
  void resolvePatterns(LVPatterns &Patterns);
  void checkFlexiblePatterns();
  void checkGenericPatterns();
  void checkKindPatterns();
};

// Helper function to add a logical element to a given scope.
void ReaderTestSelection::add(LVScope *Parent, LVElement *Child) {
  Parent->addElement(Child);
  EXPECT_EQ(Child->getParent(), Parent);
  EXPECT_EQ(Child->getLevel(), Parent->getLevel() + 1);
}

// Helper function to set the initial values for a given logical element.
void ReaderTestSelection::set(LVElement *Element, StringRef Name,
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

// Create the logical elements.
void ReaderTestSelection::createElements() {
  // Create scope root.
  Error Err = createScopes();
  ASSERT_THAT_ERROR(std::move(Err), Succeeded());
  Root = getScopesRoot();
  EXPECT_NE(Root, nullptr);

  // Create the logical types.
  CREATE(IntegerType, createType, setIsBase);

  // Create the logical scopes.
  CREATE(CompileUnit, createScopeCompileUnit, setIsCompileUnit);
  CREATE(Function, createScopeFunction, setIsFunction);
  CREATE(NestedScope, createScope, setIsLexicalBlock);
  CREATE(Namespace, createScopeNamespace, setIsNamespace);
  CREATE(Aggregate, createScopeAggregate, setIsAggregate);

  // Create the logical symbols.
  CREATE(ClassMember, createSymbol, setIsMember);
  CREATE(LocalVariable, createSymbol, setIsVariable);
  CREATE(NestedVariable, createSymbol, setIsVariable);
  CREATE(Parameter, createSymbol, setIsParameter);

  // Create the logical lines.
  CREATE(LineOne, createLine, setIsLineDebug);
  CREATE(LineTwo, createLine, setIsBasicBlock);
  CREATE(LineThree, createLine, setIsNewStatement);
  CREATE(LineFour, createLine, setIsPrologueEnd);
  CREATE(LineFive, createLine, setIsLineAssembler);
}

// Create the logical view adding the created logical elements.
void ReaderTestSelection::addElements() {
  setCompileUnit(CompileUnit);

  // Root
  //   CompileUnit
  //     IntegerType
  //     Namespace
  //       Aggregate
  //         ClassMember
  //     Function
  //       Parameter
  //       LocalVariable
  //       LineOne
  //       LineTwo
  //       NestedScope
  //         NestedVariable
  //         LineThree
  //         LineFour
  //       LineFive

  // Add elements to Root.
  add(Root, CompileUnit);

  // Add elements to CompileUnit.
  add(CompileUnit, IntegerType);
  add(CompileUnit, Namespace);
  add(CompileUnit, Function);

  // Add elements to Namespace.
  add(Namespace, Aggregate);

  // Add elements to Function.
  add(Function, Parameter);
  add(Function, LocalVariable);
  add(Function, LineOne);
  add(Function, LineTwo);
  add(Function, LineFive);
  add(Function, NestedScope);

  // Add elements to Aggregate.
  add(Aggregate, ClassMember);

  // Add elements to NestedScope.
  add(NestedScope, NestedVariable);
  add(NestedScope, LineThree);
  add(NestedScope, LineFour);
}

void ReaderTestSelection::resolvePatterns(LVPatterns &Patterns) {
  // Traverse the given scope and its children applying the pattern match.
  // Before applying the pattern, reset previous matched state.
  std::function<void(LVScope * Parent)> TraverseScope = [&](LVScope *Parent) {
    auto Traverse = [&](const auto *Set) {
      if (Set)
        for (const auto &Entry : *Set) {
          Entry->resetIsMatched();
          Patterns.resolvePatternMatch(Entry);
        }
    };

    Parent->resetIsMatched();
    Patterns.resolvePatternMatch(Parent);

    Traverse(Parent->getSymbols());
    Traverse(Parent->getTypes());
    Traverse(Parent->getLines());

    if (const LVScopes *Scopes = Parent->getScopes())
      for (LVScope *Scope : *Scopes) {
        Scope->resetIsMatched();
        Patterns.resolvePatternMatch(Scope);
        TraverseScope(Scope);
      }
  };

  // Start traversing the scopes root and apply any matching pattern.
  TraverseScope(Root);
}

// Set initial values to logical elements.
void ReaderTestSelection::initElements() {
  // Types.
  set(IntegerType, "int", 0x1000);

  // Scopes.
  set(CompileUnit, "test.cpp", 0x2000);
  set(Namespace, "anyNamespace", 0x3000, 300);
  set(Aggregate, "anyClass", 0x4000, 400);
  set(Function, "anyFunction", 0x5000, 500, IntegerType);
  set(NestedScope, "", 0x6000, 600);

  // Symbols.
  set(Parameter, "Param", 0x5100, 510, IntegerType);
  set(LocalVariable, "LocalVariable", 0x5200, 520, IntegerType);
  set(NestedVariable, "NestedVariable", 0x6200, 620, IntegerType);

  // Lines.
  set(LineOne, "", 0x5110, 511);
  set(LineTwo, "", 0x5210, 521);
  set(LineThree, "", 0x6110, 611);
  set(LineFour, "", 0x6210, 621);
  set(LineFive, "", 0x7110, 711);
}

// Check logical elements kind patterns.
void ReaderTestSelection::checkKindPatterns() {
  // Add patterns.
  LVPatterns &Patterns = patterns();
  Patterns.clear();

  LVElementKindSet KindElements; // --select-elements=<Kind>
  LVLineKindSet KindLines;       // --select-lines=<Kind>
  LVScopeKindSet KindScopes;     // --select-scopes=<Kind>
  LVSymbolKindSet KindSymbols;   // --select-symbols=<Kind>
  LVTypeKindSelection KindTypes; // --select-types=<Kind>

  KindElements.insert(LVElementKind::Global);
  KindLines.insert(LVLineKind::IsLineDebug);
  KindLines.insert(LVLineKind::IsNewStatement);
  KindLines.insert(LVLineKind::IsLineAssembler);
  KindScopes.insert(LVScopeKind::IsLexicalBlock);
  KindSymbols.insert(LVSymbolKind::IsMember);
  KindSymbols.insert(LVSymbolKind::IsParameter);
  KindTypes.insert(LVTypeKind::IsBase);

  // Add requests based on the element kind.
  Patterns.addRequest(KindElements);
  Patterns.addRequest(KindLines);
  Patterns.addRequest(KindScopes);
  Patterns.addRequest(KindSymbols);
  Patterns.addRequest(KindTypes);

  // Apply the collected patterns.
  resolvePatterns(Patterns);

  EXPECT_FALSE(CompileUnit->getIsMatched());
  EXPECT_FALSE(Namespace->getIsMatched());
  EXPECT_FALSE(Aggregate->getIsMatched());
  EXPECT_FALSE(Function->getIsMatched());
  EXPECT_TRUE(NestedScope->getIsMatched());

  EXPECT_TRUE(IntegerType->getIsMatched());

  EXPECT_TRUE(ClassMember->getIsMatched());
  EXPECT_TRUE(Parameter->getIsMatched());
  EXPECT_FALSE(LocalVariable->getIsMatched());
  EXPECT_FALSE(NestedVariable->getIsMatched());

  EXPECT_TRUE(LineOne->getIsMatched());
  EXPECT_FALSE(LineTwo->getIsMatched());
  EXPECT_TRUE(LineThree->getIsMatched());
  EXPECT_FALSE(LineFour->getIsMatched());
  EXPECT_TRUE(LineFive->getIsMatched());
}

// Check logical elements generic patterns (Case sensitive).
void ReaderTestSelection::checkGenericPatterns() {
  // Add patterns.
  LVPatterns &Patterns = patterns();
  Patterns.clear();

  StringSet<> Generic;                      // --select=<Pattern>
  Generic.insert(Function->getName());      // anyFunction
  Generic.insert(Namespace->getName());     // anyNamespace
  Generic.insert(LocalVariable->getName()); // LocalVariable

  LVOffsetSet Offsets; // --select-offset=<Offset>
  Offsets.insert(IntegerType->getOffset());
  Offsets.insert(LineOne->getOffset());
  Offsets.insert(LineTwo->getOffset());

  // Add requests based on the generic string and offset.
  Patterns.addGenericPatterns(Generic);
  Patterns.addOffsetPatterns(Offsets);

  // Apply the collected patterns.
  resolvePatterns(Patterns);

  EXPECT_FALSE(CompileUnit->getIsMatched());
  EXPECT_TRUE(Namespace->getIsMatched());
  EXPECT_FALSE(Aggregate->getIsMatched());
  EXPECT_TRUE(Function->getIsMatched());
  EXPECT_FALSE(NestedScope->getIsMatched());

  EXPECT_TRUE(IntegerType->getIsMatched());

  EXPECT_FALSE(ClassMember->getIsMatched());
  EXPECT_FALSE(Parameter->getIsMatched());
  EXPECT_TRUE(LocalVariable->getIsMatched());
  EXPECT_FALSE(NestedVariable->getIsMatched());

  EXPECT_TRUE(LineOne->getIsMatched());
  EXPECT_TRUE(LineTwo->getIsMatched());
  EXPECT_FALSE(LineThree->getIsMatched());
  EXPECT_FALSE(LineFour->getIsMatched());
  EXPECT_FALSE(LineFive->getIsMatched());
}

// Check logical elements flexible patterns (case insensitive, RegEx).
void ReaderTestSelection::checkFlexiblePatterns() {
  options().setSelectIgnoreCase();
  options().setSelectUseRegex();

  // Add patterns.
  LVPatterns &Patterns = patterns();
  Patterns.clear();

  StringSet<> Generic; // --select=<Pattern>
  Generic.insert("function");
  Generic.insert("NaMeSpAcE");
  Generic.insert("[a-z]*Variable");
  Generic.insert("[0-9]21");

  // Add requests based on the flexible string.
  Patterns.addGenericPatterns(Generic);

  // Apply the collected patterns.
  resolvePatterns(Patterns);

  EXPECT_FALSE(CompileUnit->getIsMatched());
  EXPECT_TRUE(Namespace->getIsMatched()); // anyNamespace
  EXPECT_FALSE(Aggregate->getIsMatched());
  EXPECT_TRUE(Function->getIsMatched()); // anyFunction
  EXPECT_FALSE(NestedScope->getIsMatched());

  EXPECT_FALSE(IntegerType->getIsMatched());

  EXPECT_FALSE(ClassMember->getIsMatched());
  EXPECT_FALSE(Parameter->getIsMatched());
  EXPECT_TRUE(LocalVariable->getIsMatched());  // LocalVariable
  EXPECT_TRUE(NestedVariable->getIsMatched()); // NestedVariable

  EXPECT_FALSE(LineOne->getIsMatched());
  EXPECT_TRUE(LineTwo->getIsMatched()); // 521
  EXPECT_FALSE(LineThree->getIsMatched());
  EXPECT_TRUE(LineFour->getIsMatched()); // 621
  EXPECT_FALSE(LineFive->getIsMatched());
}

TEST(LogicalViewTest, SelectElements) {
  ScopedPrinter W(outs());
  ReaderTestSelection Reader(W);

  // Reader options.
  LVOptions ReaderOptions;
  ReaderOptions.setAttributeOffset();
  ReaderOptions.setPrintAll();
  ReaderOptions.setReportList();
  ReaderOptions.setReportAnyView();

  ReaderOptions.resolveDependencies();
  options().setOptions(&ReaderOptions);

  Reader.createElements();
  Reader.addElements();
  Reader.initElements();
  Reader.checkKindPatterns();
  Reader.checkGenericPatterns();
  Reader.checkFlexiblePatterns();
}

} // namespace
