//===-- RecordTests.cpp - TextAPI Record Type Test-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===//

#include "llvm/TargetParser/Triple.h"
#include "llvm/TextAPI/RecordsSlice.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::MachO;

namespace TAPIRecord {

TEST(TAPIRecord, Simple) {
  GlobalRecord API{"_sym", RecordLinkage::Rexported,
                   SymbolFlags::Rexported | SymbolFlags::Text |
                       SymbolFlags::ThreadLocalValue,
                   GlobalRecord::Kind::Function, /*Inlined=*/false};
  EXPECT_TRUE(API.isExported());
  EXPECT_TRUE(API.isText());
  EXPECT_TRUE(API.isRexported());
  EXPECT_TRUE(API.isFunction());
  EXPECT_TRUE(API.isThreadLocalValue());
  EXPECT_FALSE(API.isInternal());
  EXPECT_FALSE(API.isUndefined());
  EXPECT_FALSE(API.isWeakDefined());
  EXPECT_FALSE(API.isWeakReferenced());
  EXPECT_FALSE(API.isVariable());
  EXPECT_FALSE(API.isInlined());
}

TEST(TAPIRecord, SimpleObjC) {
  const ObjCIFSymbolKind CompleteInterface =
      ObjCIFSymbolKind::Class | ObjCIFSymbolKind::MetaClass;
  ObjCInterfaceRecord Class{"NSObject", RecordLinkage::Exported,
                            CompleteInterface};
  ObjCInterfaceRecord ClassEH{"NSObject", RecordLinkage::Exported,
                              CompleteInterface | ObjCIFSymbolKind::EHType};

  EXPECT_TRUE(Class.isExported());
  EXPECT_EQ(Class.isExported(), ClassEH.isExported());
  EXPECT_FALSE(Class.hasExceptionAttribute());
  EXPECT_TRUE(ClassEH.hasExceptionAttribute());
  EXPECT_EQ(ObjCIVarRecord::createScopedName("NSObject", "var"),
            "NSObject.var");
  EXPECT_TRUE(Class.isCompleteInterface());
  EXPECT_TRUE(ClassEH.isCompleteInterface());
  EXPECT_TRUE(Class.isExportedSymbol(ObjCIFSymbolKind::MetaClass));
  EXPECT_EQ(ClassEH.getLinkageForSymbol(ObjCIFSymbolKind::EHType),
            RecordLinkage::Exported);
}

TEST(TAPIRecord, IncompleteObjC) {
  ObjCInterfaceRecord Class{"NSObject", RecordLinkage::Rexported,
                            ObjCIFSymbolKind::MetaClass};
  EXPECT_EQ(Class.getLinkageForSymbol(ObjCIFSymbolKind::EHType),
            RecordLinkage::Unknown);
  EXPECT_EQ(Class.getLinkageForSymbol(ObjCIFSymbolKind::MetaClass),
            RecordLinkage::Rexported);
  EXPECT_TRUE(Class.isExportedSymbol(ObjCIFSymbolKind::MetaClass));
  EXPECT_FALSE(Class.isCompleteInterface());
  EXPECT_TRUE(Class.isExported());

  Class.updateLinkageForSymbols(ObjCIFSymbolKind::Class,
                                RecordLinkage::Internal);
  EXPECT_TRUE(Class.isExported());
  EXPECT_FALSE(Class.isCompleteInterface());
  EXPECT_FALSE(Class.isExportedSymbol(ObjCIFSymbolKind::Class));
  EXPECT_EQ(Class.getLinkageForSymbol(ObjCIFSymbolKind::Class),
            RecordLinkage::Internal);
}

TEST(TAPIRecord, SimpleSlice) {
  Triple T("arm64-apple-macosx13.3");
  RecordsSlice Slice(T);
  EXPECT_TRUE(Slice.empty());
  Slice.addRecord("_OBJC_CLASS_$_NSObject", SymbolFlags::None,
                  GlobalRecord::Kind::Unknown, RecordLinkage::Rexported);
  Slice.addRecord("_OBJC_METACLASS_$_NSObject", SymbolFlags::None,
                  GlobalRecord::Kind::Unknown, RecordLinkage::Rexported);
  Slice.addRecord("_OBJC_IVAR_$_NSConcreteValue.typeInfo", SymbolFlags::None,
                  GlobalRecord::Kind::Unknown, RecordLinkage::Exported);
  Slice.addRecord("_OBJC_IVAR_$_NSObject.objInfo", SymbolFlags::None,
                  GlobalRecord::Kind::Unknown, RecordLinkage::Exported);
  Slice.addRecord("_foo", SymbolFlags::WeakDefined | SymbolFlags::Rexported,
                  GlobalRecord::Kind::Variable, RecordLinkage::Rexported);
  EXPECT_FALSE(Slice.empty());

  // Check global.
  EXPECT_FALSE(Slice.findGlobal("_foo", GlobalRecord::Kind::Function));
  auto *Global = Slice.findGlobal("_foo");
  ASSERT_TRUE(Global);
  EXPECT_TRUE(Global->isVariable());
  EXPECT_TRUE(Global->isWeakDefined());
  EXPECT_TRUE(Global->isRexported());
  EXPECT_TRUE(Global->isData());

  // Check class.
  auto *Class = Slice.findObjCInterface("NSObject");
  ASSERT_TRUE(Class);
  EXPECT_TRUE(Class->isRexported());
  EXPECT_TRUE(Class->isData());
  EXPECT_FALSE(Class->hasExceptionAttribute());
  auto ClassIVar = Class->findObjCIVar("objInfo");
  ASSERT_TRUE(ClassIVar);
  EXPECT_TRUE(ClassIVar->isExported());
  EXPECT_FALSE(ClassIVar->isRexported());

  // Check fall-back extension.
  auto *Cat = Slice.findObjCCategory("NSConcreteValue", "");
  ASSERT_TRUE(Cat);
  // There is not linkage information for categories.
  EXPECT_FALSE(Cat->isExported());
  EXPECT_FALSE(Cat->isInternal());
  auto CatIVar = Cat->findObjCIVar("typeInfo");
  EXPECT_TRUE(CatIVar);
  EXPECT_TRUE(CatIVar->isExported());
  EXPECT_FALSE(CatIVar->isRexported());

  // Find IVars directly.
  auto TIIVar =
      Slice.findObjCIVar(/*IsScopedName=*/true, "NSConcreteValue.typeInfo");
  ASSERT_TRUE(TIIVar);
  EXPECT_EQ(CatIVar->getName(), TIIVar->getName());

  auto OIIVar = Slice.findObjCIVar(/*IsScopedName=*/false, "objInfo");
  ASSERT_TRUE(OIIVar);
  EXPECT_EQ(ClassIVar->getName(), OIIVar->getName());

  EXPECT_FALSE(Slice.findObjCIVar(/*IsScopedName=*/true, "typeInfo"));
}

TEST(TAPIRecord, LibraryAttrs) {
  Triple T("arm64-apple-ios15.1");
  RecordsSlice Slice(T);
  EXPECT_TRUE(Slice.empty());

  auto BA = Slice.getBinaryAttrs();
  EXPECT_TRUE(Slice.hasBinaryAttrs());
}

} // namespace TAPIRecord
