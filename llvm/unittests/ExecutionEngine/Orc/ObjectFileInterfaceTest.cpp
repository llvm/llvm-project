//===- ObjectFileInterfaceTest.cpp - ObjectFileInterface tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/ObjectFileInterface.h"
#include "OrcTestCommon.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

// wdata: weak external with a function type and a data default.
// a: weak external whose default b is a weak external of a function.
const char *COFFWeakExternals = R"(
--- !COFF
header:
  Machine:         IMAGE_FILE_MACHINE_AMD64
  Characteristics: [  ]
sections:
  - Name:            .text
    Characteristics: [ IMAGE_SCN_CNT_CODE, IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ ]
    Alignment:       16
    SectionData:     C3C3
  - Name:            .data
    Characteristics: [ IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_MEM_READ, IMAGE_SCN_MEM_WRITE ]
    Alignment:       4
    SectionData:     2A000000
symbols:
  - Name:            main
    Value:           0
    SectionNumber:   1
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_FUNCTION
    StorageClass:    IMAGE_SYM_CLASS_EXTERNAL
  - Name:            wdata
    Value:           0
    SectionNumber:   0
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_FUNCTION
    StorageClass:    IMAGE_SYM_CLASS_WEAK_EXTERNAL
    WeakExternal:
      TagIndex:        3
      Characteristics: IMAGE_WEAK_EXTERN_SEARCH_ALIAS
  - Name:            .weak.wdata.default.main
    Value:           0
    SectionNumber:   2
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_EXTERNAL
  - Name:            b
    Value:           0
    SectionNumber:   0
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_WEAK_EXTERNAL
    WeakExternal:
      TagIndex:        6
      Characteristics: IMAGE_WEAK_EXTERN_SEARCH_ALIAS
  - Name:            .weak.b.default.main
    Value:           1
    SectionNumber:   1
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_FUNCTION
    StorageClass:    IMAGE_SYM_CLASS_EXTERNAL
  - Name:            a
    Value:           0
    SectionNumber:   0
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_WEAK_EXTERNAL
    WeakExternal:
      TagIndex:        4
      Characteristics: IMAGE_WEAK_EXTERN_SEARCH_ALIAS
...
)";

TEST(ObjectFileInterfaceTest, COFFWeakExternalCallableFromDefault) {
  SmallString<0> Storage;
  auto Obj = yaml::yaml2ObjectFile(Storage, COFFWeakExternals,
                                   [](const Twine &Msg) { FAIL() << Msg; });
  ASSERT_TRUE(Obj);

  ExecutionSession ES(std::make_unique<UnsupportedExecutorProcessControl>());
  auto I = getObjectFileInterface(ES, Obj->getMemoryBufferRef());
  ASSERT_THAT_EXPECTED(I, Succeeded());

  JITSymbolFlags WeakExported = JITSymbolFlags::Weak | JITSymbolFlags::Exported;
  EXPECT_EQ(I->SymbolFlags.lookup(ES.intern("wdata")), WeakExported);
  EXPECT_EQ(I->SymbolFlags.lookup(ES.intern("b")),
            WeakExported | JITSymbolFlags::Callable);
  EXPECT_EQ(I->SymbolFlags.lookup(ES.intern("a")),
            WeakExported | JITSymbolFlags::Callable);

  cantFail(ES.endSession());
}

} // namespace
