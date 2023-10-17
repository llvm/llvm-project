//===- llvm/unittest/DebugInfo/LogicalView/CompareElementsTest.cpp --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/LogicalView/Core/LVCompare.h"
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

//===----------------------------------------------------------------------===//
// Basic Reader functionality.
class ReaderTestCompare : public LVReader {
#define CREATE(VARIABLE, CREATE_FUNCTION, SET_FUNCTION)                        \
  VARIABLE = CREATE_FUNCTION();                                                \
  EXPECT_NE(VARIABLE, nullptr);                                                \
  VARIABLE->SET_FUNCTION();

public:
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

  // Scopes.
  LVScope *NestedScope = nullptr;
  LVScope *InnerScope = nullptr;
  LVScopeAggregate *Aggregate = nullptr;
  LVScopeEnumeration *Enumeration = nullptr;
  LVScopeFunction *FunctionOne = nullptr;
  LVScopeFunction *FunctionTwo = nullptr;
  LVScopeNamespace *Namespace = nullptr;

  // Symbols.
  LVSymbol *GlobalVariable = nullptr;
  LVSymbol *LocalVariable = nullptr;
  LVSymbol *ClassMember = nullptr;
  LVSymbol *NestedVariable = nullptr;
  LVSymbol *ParameterOne = nullptr;
  LVSymbol *ParameterTwo = nullptr;

  // Lines.
  LVLine *LineOne = nullptr;
  LVLine *LineTwo = nullptr;
  LVLine *LineThree = nullptr;

protected:
  void add(LVScope *Parent, LVElement *Element);
  void set(LVElement *Element, StringRef Name, LVOffset Offset,
           uint32_t LineNumber = 0, LVElement *Type = nullptr);

public:
  ReaderTestCompare(ScopedPrinter &W) : LVReader("", "", W) {
    setInstance(this);
  }

  Error createScopes() { return LVReader::createScopes(); }
  Error printScopes() { return LVReader::printScopes(); }

  void createElements();
  void addElements(bool IsReference, bool IsTarget);
  void initElements();
};

// Helper function to add a logical element to a given scope.
void ReaderTestCompare::add(LVScope *Parent, LVElement *Child) {
  Parent->addElement(Child);
  EXPECT_EQ(Child->getParent(), Parent);
  EXPECT_EQ(Child->getLevel(), Parent->getLevel() + 1);
}

// Helper function to set the initial values for a given logical element.
void ReaderTestCompare::set(LVElement *Element, StringRef Name, LVOffset Offset,
                            uint32_t LineNumber, LVElement *Type) {
  Element->setName(Name);
  Element->setOffset(Offset);
  Element->setLineNumber(LineNumber);
  Element->setType(Type);
}

//===----------------------------------------------------------------------===//
// Create the logical elements.
void ReaderTestCompare::createElements() {
  // Create scope root.
  Error Err = createScopes();
  ASSERT_THAT_ERROR(std::move(Err), Succeeded());
  Root = getScopesRoot();
  ASSERT_NE(Root, nullptr);

  // Create the logical types.
  CREATE(IntegerType, createType, setIsBase);
  CREATE(UnsignedType, createType, setIsBase);
  CREATE(GlobalType, createType, setIsBase);
  CREATE(LocalType, createType, setIsBase);
  CREATE(NestedType, createType, setIsBase);
  CREATE(EnumeratorOne, createTypeEnumerator, setIsEnumerator);
  CREATE(EnumeratorTwo, createTypeEnumerator, setIsEnumerator);
  CREATE(TypeDefinitionOne, createTypeDefinition, setIsTypedef);
  CREATE(TypeDefinitionTwo, createTypeDefinition, setIsTypedef);

  // Create the logical scopes.
  CREATE(NestedScope, createScope, setIsLexicalBlock);
  CREATE(InnerScope, createScope, setIsLexicalBlock);
  CREATE(Aggregate, createScopeAggregate, setIsAggregate);
  CREATE(CompileUnit, createScopeCompileUnit, setIsCompileUnit);
  CREATE(Enumeration, createScopeEnumeration, setIsEnumeration);
  CREATE(FunctionOne, createScopeFunction, setIsFunction);
  CREATE(FunctionTwo, createScopeFunction, setIsFunction);
  CREATE(Namespace, createScopeNamespace, setIsNamespace);

  // Create the logical symbols.
  CREATE(GlobalVariable, createSymbol, setIsVariable);
  CREATE(LocalVariable, createSymbol, setIsVariable);
  CREATE(ClassMember, createSymbol, setIsMember);
  CREATE(NestedVariable, createSymbol, setIsVariable);
  CREATE(ParameterOne, createSymbol, setIsParameter);
  CREATE(ParameterTwo, createSymbol, setIsParameter);

  // Create the logical lines.
  CREATE(LineOne, createLine, setIsLineDebug);
  CREATE(LineTwo, createLine, setIsLineDebug);
  CREATE(LineThree, createLine, setIsLineDebug);
}

// Reference Reader:              Target Reader:
// ----------------------         ----------------------
// Root                           Root
//   CompileUnit                    CompileUnit
//     IntegerType                    IntegerType
//     UnsignedType                   UnsignedType
//     FunctionOne                    FunctionOne
//       ParameterOne                   ParameterOne
//       LocalVariable                  ---
//       LocalType                      LocalType
//       LineOne                        LineOne
//       NestedScope                    NestedScope
//         NestedVariable                 NestedVariable
//         NestedType                     NestedType
//         LineTwo                        ---
//         InnerScope                     InnerScope
//           ---                            LineThree
//     ---                            FunctionTwo
//     ---                              ParameterTwo
//     GlobalVariable                 GlobalVariable
//     GlobalType                     GlobalType
//     Namespace                      Namespace
//       Aggregate                      Aggregate
//         ClassMember                    ClassMember
//       Enumeration                    Enumeration
//         EnumeratorOne                  EnumeratorOne
//         EnumeratorTwo                  EnumeratorTwo
//       TypeDefinitionOne              ---
//       ---                            TypeDefinitionTwo

// Create the logical view adding the created logical elements.
void ReaderTestCompare::addElements(bool IsReference, bool IsTarget) {
  Root->setName(IsReference ? "Reference-Reader" : "Target-Reader");

  auto Insert = [&](bool Insert, auto *Parent, auto *Child) {
    if (Insert)
      add(Parent, Child);
  };

  setCompileUnit(CompileUnit);
  add(Root, CompileUnit);

  // Add elements to CompileUnit.
  Insert(true, CompileUnit, IntegerType);
  Insert(true, CompileUnit, UnsignedType);
  Insert(true, CompileUnit, FunctionOne);
  Insert(IsTarget, CompileUnit, FunctionTwo);
  Insert(true, CompileUnit, GlobalVariable);
  Insert(true, CompileUnit, GlobalType);
  Insert(true, CompileUnit, Namespace);

  // Add elements to Namespace.
  Insert(true, Namespace, Aggregate);
  Insert(true, Namespace, Enumeration);
  Insert(IsReference, Namespace, TypeDefinitionOne);
  Insert(IsTarget, Namespace, TypeDefinitionTwo);

  // Add elements to FunctionOne.
  Insert(true, FunctionOne, ParameterOne);
  Insert(IsReference, FunctionOne, LocalVariable);
  Insert(true, FunctionOne, LocalType);
  Insert(true, FunctionOne, LineOne);
  Insert(true, FunctionOne, NestedScope);

  // Add elements to FunctionTwo.
  Insert(IsTarget, FunctionTwo, ParameterTwo);

  // Add elements to NestedScope.
  Insert(true, NestedScope, NestedVariable);
  Insert(true, NestedScope, NestedType);
  Insert(IsReference, NestedScope, LineTwo);
  Insert(true, NestedScope, InnerScope);

  // Add elements to Enumeration.
  Insert(true, Enumeration, EnumeratorOne);
  Insert(true, Enumeration, EnumeratorTwo);

  // Add elements to Aggregate.
  Insert(true, Aggregate, ClassMember);

  Insert(IsTarget, InnerScope, LineThree);
}

// Set initial values to logical elements.
void ReaderTestCompare::initElements() {
  setFilename("LogicalElements.obj");

  Root->setFileFormatName("FileFormat");

  // Types.
  set(IntegerType, "int", 0x1000);
  set(UnsignedType, "unsigned", 0x1010);
  set(GlobalType, "GlobalType", 0x1020, 1020);
  set(LocalType, "LocalType", 0x1030, 1030);
  set(NestedType, "NestedType", 0x1040, 1040);

  set(TypeDefinitionOne, "INTEGER", 0x1050, 1050, IntegerType);
  set(TypeDefinitionTwo, "INT", 0x1060, 1060, TypeDefinitionOne);

  set(EnumeratorOne, "One", 0x1070, 1070);
  EnumeratorOne->setValue("Blue");

  set(EnumeratorTwo, "Two", 0x1080, 1080);
  EnumeratorTwo->setValue("Red");

  // Scopes.
  set(Aggregate, "Class", 0x2000, 2000);
  set(Enumeration, "Colors", 0x2010, 2010);
  set(FunctionOne, "FunctionOne", 0x2020, 2020, GlobalType);
  set(FunctionTwo, "FunctionTwo", 0x2030, 2030, GlobalType);
  set(Namespace, "Namespace", 0x2040, 2040);
  set(NestedScope, "", 0x2050, 2050);
  set(InnerScope, "", 0x2060, 2060);
  set(CompileUnit, "test.cpp", 0x2070, 2070);

  // Symbols.
  set(GlobalVariable, "GlobalVariable", 0x3000, 3000);
  set(LocalVariable, "LocalVariable", 0x3010, 3010, UnsignedType);
  set(ClassMember, "ClassMember", 0x3020, 3020, IntegerType);
  set(ParameterOne, "ParameterOne", 0x3030, 3030, UnsignedType);
  set(ParameterTwo, "ParameterTwo", 0x3040, 3040, UnsignedType);
  set(NestedVariable, "NestedVariable", 0x3050, 3050);

  // Lines.
  set(LineOne, "", 0x4000, 4000);
  set(LineTwo, "", 0x4010, 4010);
  set(LineThree, "", 0x4020, 4020);
}

// Compare the logical views.
void compareReadersViews(ReaderTestCompare *ReferenceReader,
                         ReaderTestCompare *TargetReader) {
  LVCompare Compare(nulls());
  Error Err = Compare.execute(ReferenceReader, TargetReader);
  ASSERT_THAT_ERROR(std::move(Err), Succeeded());

  // Get comparison table.
  LVPassTable PassTable = Compare.getPassTable();
  ASSERT_EQ(PassTable.size(), 5u);

  LVReader *Reader;
  LVElement *Element;
  LVComparePass Pass;

  // Reference: Missing 'FunctionOne'
  std::tie(Reader, Element, Pass) = PassTable[0];
  EXPECT_EQ(Reader, ReferenceReader);
  EXPECT_EQ(Element, ReferenceReader->FunctionOne);
  EXPECT_EQ(Pass, LVComparePass::Missing);

  // Reference: Missing 'TypeDefinitionOne'
  std::tie(Reader, Element, Pass) = PassTable[1];
  EXPECT_EQ(Reader, ReferenceReader);
  EXPECT_EQ(Element, ReferenceReader->TypeDefinitionOne);
  EXPECT_EQ(Pass, LVComparePass::Missing);

  // Target: Added 'FunctionOne'
  std::tie(Reader, Element, Pass) = PassTable[2];
  EXPECT_EQ(Reader, TargetReader);
  EXPECT_EQ(Element, TargetReader->FunctionOne);
  EXPECT_EQ(Pass, LVComparePass::Added);

  // Target: Added 'FunctionTwo'
  std::tie(Reader, Element, Pass) = PassTable[3];
  EXPECT_EQ(Reader, TargetReader);
  EXPECT_EQ(Element, TargetReader->FunctionTwo);
  EXPECT_EQ(Pass, LVComparePass::Added);

  // Target: Added 'TypeDefinitionTwo'
  std::tie(Reader, Element, Pass) = PassTable[4];
  EXPECT_EQ(Reader, TargetReader);
  EXPECT_EQ(Element, TargetReader->TypeDefinitionTwo);
  EXPECT_EQ(Pass, LVComparePass::Added);
}

// Compare the logical elements.
void compareReadersElements(ReaderTestCompare *ReferenceReader,
                            ReaderTestCompare *TargetReader) {
  LVCompare Compare(nulls());
  Error Err = Compare.execute(ReferenceReader, TargetReader);
  ASSERT_THAT_ERROR(std::move(Err), Succeeded());

  // Get comparison table.
  LVPassTable PassTable = Compare.getPassTable();
  ASSERT_EQ(PassTable.size(), 7u);

  LVReader *Reader;
  LVElement *Element;
  LVComparePass Pass;

  // Reference: Missing 'LocalVariable'
  std::tie(Reader, Element, Pass) = PassTable[0];
  EXPECT_EQ(Reader, ReferenceReader);
  EXPECT_EQ(Element, ReferenceReader->LocalVariable);
  EXPECT_EQ(Pass, LVComparePass::Missing);

  // Reference: Missing 'TypeDefinitionOne'
  std::tie(Reader, Element, Pass) = PassTable[1];
  EXPECT_EQ(Reader, ReferenceReader);
  EXPECT_EQ(Element, ReferenceReader->TypeDefinitionOne);
  EXPECT_EQ(Pass, LVComparePass::Missing);

  // Reference: Missing 'LineTwo'
  std::tie(Reader, Element, Pass) = PassTable[2];
  EXPECT_EQ(Reader, ReferenceReader);
  EXPECT_EQ(Element, ReferenceReader->LineTwo);
  EXPECT_EQ(Pass, LVComparePass::Missing);

  // Target: Added 'FunctionTwo'
  std::tie(Reader, Element, Pass) = PassTable[3];
  EXPECT_EQ(Reader, TargetReader);
  EXPECT_EQ(Element, TargetReader->FunctionTwo);
  EXPECT_EQ(Pass, LVComparePass::Added);

  // Target: Added 'ParameterTwo'
  std::tie(Reader, Element, Pass) = PassTable[4];
  EXPECT_EQ(Reader, TargetReader);
  EXPECT_EQ(Element, TargetReader->ParameterTwo);
  EXPECT_EQ(Pass, LVComparePass::Added);

  // Target: Added 'TypeDefinitionTwo'
  std::tie(Reader, Element, Pass) = PassTable[5];
  EXPECT_EQ(Reader, TargetReader);
  EXPECT_EQ(Element, TargetReader->TypeDefinitionTwo);
  EXPECT_EQ(Pass, LVComparePass::Added);

  // Target: Added 'LineThree'
  std::tie(Reader, Element, Pass) = PassTable[6];
  EXPECT_EQ(Reader, TargetReader);
  EXPECT_EQ(Element, TargetReader->LineThree);
  EXPECT_EQ(Pass, LVComparePass::Added);
}

TEST(LogicalViewTest, CompareElements) {
  ScopedPrinter W(outs());

  // Reader options.
  LVOptions ReaderOptions;
  ReaderOptions.setCompareLines();
  ReaderOptions.setCompareScopes();
  ReaderOptions.setCompareSymbols();
  ReaderOptions.setCompareTypes();

  // The next set-ups are very similar. The only difference is the
  // comparison type, which must be set before the readers are created.
  //   Views: setCompareContext()
  //   Elements: resetCompareContext()
  {
    // Compare the logical views as whole unit (--compare-context).
    ReaderOptions.setCompareContext();
    ReaderOptions.resolveDependencies();
    options().setOptions(&ReaderOptions);

    ReaderTestCompare ReferenceReader(W);
    ReferenceReader.createElements();
    ReferenceReader.addElements(/*IsReference=*/true, /*IsTarget=*/false);
    ReferenceReader.initElements();

    ReaderTestCompare TargetReader(W);
    TargetReader.createElements();
    TargetReader.addElements(/*IsReference=*/false, /*IsTarget=*/true);
    TargetReader.initElements();

    compareReadersViews(&ReferenceReader, &TargetReader);
  }
  {
    // Compare the logical elements.
    ReaderOptions.resetCompareContext();
    ReaderOptions.resolveDependencies();
    options().setOptions(&ReaderOptions);

    ReaderTestCompare ReferenceReader(W);
    ReferenceReader.createElements();
    ReferenceReader.addElements(/*IsReference=*/true, /*IsTarget=*/false);
    ReferenceReader.initElements();

    ReaderTestCompare TargetReader(W);
    TargetReader.createElements();
    TargetReader.addElements(/*IsReference=*/false, /*IsTarget=*/true);
    TargetReader.initElements();

    compareReadersElements(&ReferenceReader, &TargetReader);
  }
}

} // namespace
