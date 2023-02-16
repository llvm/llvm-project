//===- llvm/unittest/DebugInfo/LogicalView/LogicalElementsTest.cpp --------===//
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

class ReaderTestElements : public LVReader {
#define CREATE(VARIABLE, CREATE_FUNCTION)                                      \
  VARIABLE = CREATE_FUNCTION();                                                \
  EXPECT_NE(VARIABLE, nullptr);

  // Types.
  LVType *IntegerType = nullptr;
  LVType *UnsignedType = nullptr;
  LVType *GlobalType = nullptr;
  LVType *LocalType = nullptr;
  LVType *NestedType = nullptr;
  LVTypeDefinition *TypeDefinitionOne = nullptr;
  LVTypeDefinition *TypeDefinitionTwo = nullptr;
  LVTypeEnumerator *EnumeratorOne = nullptr;
  LVTypeEnumerator *EnumeratorTwo = nullptr;
  LVTypeImport *TypeImport = nullptr;
  LVTypeParam *TypeParam = nullptr;
  LVTypeSubrange *TypeSubrange = nullptr;

  // Scopes.
  LVScope *NestedScope = nullptr;
  LVScopeAggregate *Aggregate = nullptr;
  LVScopeArray *Array = nullptr;
  LVScopeEnumeration *Enumeration = nullptr;
  LVScopeFunction *Function = nullptr;
  LVScopeFunction *ClassFunction = nullptr;
  LVScopeFunctionInlined *InlinedFunction = nullptr;
  LVScopeNamespace *Namespace = nullptr;

  // Symbols.
  LVSymbol *GlobalVariable = nullptr;
  LVSymbol *LocalVariable = nullptr;
  LVSymbol *ClassMember = nullptr;
  LVSymbol *NestedVariable = nullptr;
  LVSymbol *Parameter = nullptr;

  // Lines.
  LVLine *LocalLine = nullptr;
  LVLine *NestedLine = nullptr;

protected:
  void add(LVScope *Parent, LVElement *Element);
  void set(LVElement *Element, StringRef Name, LVOffset Offset,
           uint32_t LineNumber = 0, LVElement *Type = nullptr);

public:
  ReaderTestElements(ScopedPrinter &W) : LVReader("", "", W) {
    setInstance(this);
  }

  Error createScopes() { return LVReader::createScopes(); }
  Error printScopes() { return LVReader::printScopes(); }

  void createElements();
  void addElements();
  void initElements();
};

// Helper function to add a logical element to a given scope.
void ReaderTestElements::add(LVScope *Parent, LVElement *Child) {
  Parent->addElement(Child);
  EXPECT_EQ(Child->getParent(), Parent);
  EXPECT_EQ(Child->getLevel(), Parent->getLevel() + 1);
}

// Helper function to set the initial values for a given logical element.
void ReaderTestElements::set(LVElement *Element, StringRef Name,
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
void ReaderTestElements::createElements() {
  // Create scope root.
  Error Err = createScopes();
  ASSERT_THAT_ERROR(std::move(Err), Succeeded());
  Root = getScopesRoot();
  ASSERT_NE(Root, nullptr);

  // Create the logical types.
  CREATE(IntegerType, createType);
  CREATE(UnsignedType, createType);
  CREATE(GlobalType, createType);
  CREATE(LocalType, createType);
  CREATE(NestedType, createType);
  CREATE(EnumeratorOne, createTypeEnumerator);
  CREATE(EnumeratorTwo, createTypeEnumerator);
  CREATE(TypeDefinitionOne, createTypeDefinition);
  CREATE(TypeDefinitionTwo, createTypeDefinition);
  CREATE(TypeSubrange, createTypeSubrange);
  CREATE(TypeParam, createTypeParam);
  CREATE(TypeImport, createTypeImport);

  // Create the logical scopes.
  CREATE(NestedScope, createScope);
  CREATE(Aggregate, createScopeAggregate);
  CREATE(Array, createScopeArray);
  CREATE(CompileUnit, createScopeCompileUnit);
  CREATE(Enumeration, createScopeEnumeration);
  CREATE(Function, createScopeFunction);
  CREATE(ClassFunction, createScopeFunction);
  CREATE(InlinedFunction, createScopeFunctionInlined);
  CREATE(Namespace, createScopeNamespace);

  // Create the logical symbols.
  CREATE(GlobalVariable, createSymbol);
  CREATE(LocalVariable, createSymbol);
  CREATE(ClassMember, createSymbol);
  CREATE(NestedVariable, createSymbol);
  CREATE(Parameter, createSymbol);

  // Create the logical lines.
  CREATE(LocalLine, createLine);
  CREATE(NestedLine, createLine);
}

// Create the logical view adding the created logical elements.
void ReaderTestElements::addElements() {
  setCompileUnit(CompileUnit);

  // Root
  //   CompileUnit
  //     IntegerType
  //     UnsignedType
  //     Array
  //       TypeSubrange
  //     Function
  //       Parameter
  //       LocalVariable
  //       LocalType
  //       LocalLine
  //       InlinedFunction
  //       TypeImport
  //       TypeParam
  //       NestedScope
  //         NestedVariable
  //         NestedType
  //         NestedLine
  //     GlobalVariable
  //     GlobalType
  //     Namespace
  //       Aggregate
  //         ClassMember
  //         ClassFunction
  //       Enumeration
  //         EnumeratorOne
  //         EnumeratorTwo
  //       TypeDefinitionOne
  //       TypeDefinitionTwo

  add(Root, CompileUnit);
  EXPECT_EQ(Root->lineCount(), 0u);
  EXPECT_EQ(Root->scopeCount(), 1u);
  EXPECT_EQ(Root->symbolCount(), 0u);
  EXPECT_EQ(Root->typeCount(), 0u);

  // Add elements to CompileUnit.
  add(CompileUnit, IntegerType);
  add(CompileUnit, UnsignedType);
  add(CompileUnit, Array);
  add(CompileUnit, Function);
  add(CompileUnit, GlobalVariable);
  add(CompileUnit, GlobalType);
  add(CompileUnit, Namespace);
  EXPECT_EQ(CompileUnit->lineCount(), 0u);
  EXPECT_EQ(CompileUnit->scopeCount(), 3u);
  EXPECT_EQ(CompileUnit->symbolCount(), 1u);
  EXPECT_EQ(CompileUnit->typeCount(), 3u);

  // Add elements to Namespace.
  add(Namespace, Aggregate);
  add(Namespace, Enumeration);
  add(Namespace, TypeDefinitionOne);
  add(Namespace, TypeDefinitionTwo);
  EXPECT_EQ(Namespace->lineCount(), 0u);
  EXPECT_EQ(Namespace->scopeCount(), 2u);
  EXPECT_EQ(Namespace->symbolCount(), 0u);
  EXPECT_EQ(Namespace->typeCount(), 2u);

  // Add elements to Function.
  add(Function, Parameter);
  add(Function, LocalVariable);
  add(Function, LocalType);
  add(Function, LocalLine);
  add(Function, InlinedFunction);
  add(Function, TypeImport);
  add(Function, TypeParam);
  add(Function, NestedScope);
  EXPECT_EQ(Function->lineCount(), 1u);
  EXPECT_EQ(Function->scopeCount(), 2u);
  EXPECT_EQ(Function->symbolCount(), 2u);
  EXPECT_EQ(Function->typeCount(), 3u);

  // Add elements to NestedScope.
  add(NestedScope, NestedVariable);
  add(NestedScope, NestedType);
  add(NestedScope, NestedLine);
  EXPECT_EQ(NestedScope->lineCount(), 1u);
  EXPECT_EQ(NestedScope->scopeCount(), 0u);
  EXPECT_EQ(NestedScope->symbolCount(), 1u);
  EXPECT_EQ(NestedScope->typeCount(), 1u);

  // Add elements to Enumeration.
  add(Enumeration, EnumeratorOne);
  add(Enumeration, EnumeratorTwo);
  EXPECT_EQ(Enumeration->lineCount(), 0u);
  EXPECT_EQ(Enumeration->scopeCount(), 0u);
  EXPECT_EQ(Enumeration->symbolCount(), 0u);
  EXPECT_EQ(Enumeration->typeCount(), 2u);

  // Add elements to Aggregate.
  add(Aggregate, ClassMember);
  add(Aggregate, ClassFunction);
  EXPECT_EQ(Aggregate->lineCount(), 0u);
  EXPECT_EQ(Aggregate->scopeCount(), 1u);
  EXPECT_EQ(Aggregate->symbolCount(), 1u);
  EXPECT_EQ(Aggregate->typeCount(), 0u);

  // Add elements to Array.
  add(Array, TypeSubrange);
  EXPECT_EQ(Array->lineCount(), 0u);
  EXPECT_EQ(Array->scopeCount(), 0u);
  EXPECT_EQ(Array->symbolCount(), 0u);
  EXPECT_EQ(Array->typeCount(), 1u);
}

// Set initial values to logical elements.
void ReaderTestElements::initElements() {
  setFilename("LogicalElements.obj");
  EXPECT_EQ(getFilename(), "LogicalElements.obj");

  Root->setFileFormatName("FileFormat");
  EXPECT_EQ(Root->getFileFormatName(), "FileFormat");

  // Types.
  set(IntegerType, "int", 0x1000);
  set(UnsignedType, "unsigned", 0x1010);
  set(GlobalType, "GlobalType", 0x1020, 1020);
  set(LocalType, "LocalType", 0x1030, 1030);
  set(NestedType, "NestedType", 0x1040, 1040);

  set(TypeDefinitionOne, "INTEGER", 0x1040, 1040, IntegerType);
  set(TypeDefinitionTwo, "INT", 0x1050, 1050, TypeDefinitionOne);
  EXPECT_EQ(TypeDefinitionOne->getUnderlyingType(), IntegerType);
  EXPECT_EQ(TypeDefinitionTwo->getUnderlyingType(), IntegerType);

  set(EnumeratorOne, "one", 0x1060, 1060);
  EnumeratorOne->setValue("blue");
  EXPECT_EQ(EnumeratorOne->getValue(), "blue");

  set(EnumeratorTwo, "two", 0x1070, 1070);
  EnumeratorTwo->setValue("red");
  EXPECT_EQ(EnumeratorTwo->getValue(), "red");

  set(TypeSubrange, "", 0x1080, 1080, IntegerType);
  TypeSubrange->setCount(5);
  EXPECT_EQ(TypeSubrange->getCount(), 5);

  TypeSubrange->setLowerBound(10);
  TypeSubrange->setUpperBound(15);
  EXPECT_EQ(TypeSubrange->getLowerBound(), 10);
  EXPECT_EQ(TypeSubrange->getUpperBound(), 15);

  TypeSubrange->setBounds(20, 25);
  std::pair<unsigned, unsigned> Pair;
  Pair = TypeSubrange->getBounds();
  EXPECT_EQ(Pair.first, 20u);
  EXPECT_EQ(Pair.second, 25u);

  set(TypeParam, "INTEGER", 0x1090, 1090, UnsignedType);
  TypeParam->setValue("10");
  EXPECT_EQ(TypeParam->getValue(), "10");

  set(TypeImport, "", 0x1090, 1090, Aggregate);
  EXPECT_EQ(TypeImport->getType(), Aggregate);

  // Scopes.
  set(Aggregate, "Class", 0x2000, 2000);
  set(Enumeration, "Colors", 0x2010, 2010);
  set(Function, "function", 0x2020, 2020, GlobalType);
  set(ClassFunction, "foo", 0x2030, 2030, TypeDefinitionTwo);
  set(Namespace, "nsp", 0x2040, 2040);
  set(NestedScope, "", 0x2050, 2050);
  set(Array, "", 0x2060, 2060, UnsignedType);
  set(InlinedFunction, "bar", 0x2070, 2070, TypeDefinitionOne);
  set(CompileUnit, "test.cpp", 0x2080, 2080);

  // Symbols.
  set(GlobalVariable, "GlobalVariable", 0x3000, 3000);
  set(LocalVariable, "LocalVariable", 0x3010, 3010, TypeDefinitionOne);
  set(ClassMember, "Member", 0x3020, 3020, IntegerType);
  set(Parameter, "Param", 0x3030, 3030, UnsignedType);
  set(NestedVariable, "NestedVariable", 0x3040, 3040);

  // Lines.
  set(LocalLine, "", 0x4000, 4000);
  set(NestedLine, "", 0x4010, 4010);
}

TEST(LogicalViewTest, LogicalElements) {
  ScopedPrinter W(outs());
  ReaderTestElements Reader(W);

  Reader.createElements();
  Reader.addElements();
  Reader.initElements();
}

} // namespace
