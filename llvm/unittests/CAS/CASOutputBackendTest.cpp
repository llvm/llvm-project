//===- CASOutputBackendTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASOutputBackend.h"
#include "llvm/CAS/CASFileSystem.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::vfs;

template <class T>
static std::unique_ptr<T>
errorOrToPointer(ErrorOr<std::unique_ptr<T>> ErrorOrPointer) {
  if (ErrorOrPointer)
    return std::move(*ErrorOrPointer);
  return nullptr;
}

TEST(CASOutputBackendTest, createFiles) {
  std::unique_ptr<ObjectStore> CAS = createInMemoryCAS();
  ASSERT_TRUE(CAS);

  auto Outputs = makeIntrusiveRefCnt<CASOutputBackend>(*CAS);

  Optional<ObjectProxy> Content1;
  Optional<ObjectProxy> Content2;
  ASSERT_THAT_ERROR(CAS->createProxy(None, "content1").moveInto(Content1),
                    Succeeded());
  ASSERT_THAT_ERROR(CAS->createProxy(None, "content2").moveInto(Content2),
                    Succeeded());
  std::string AbsolutePath1 = "/absolute/path/1";
  std::string AbsolutePath2 = "/absolute/path/2";
  std::string RelativePath = "relative/path/./2";
  std::string WindowsPath = "c:\\windows/path";

  // FIXME: Add test of duplicate paths. Maybe could error at createFile()...
  struct OutputDescription {
    ObjectProxy Content;
    std::string Path;
  };
  OutputDescription OutputDescriptions[] = {
      {*Content1, AbsolutePath1}, {*Content2, AbsolutePath2},
      {*Content1, AbsolutePath1}, {*Content1, RelativePath},
      {*Content1, WindowsPath},
  };
  for (const OutputDescription &OD : OutputDescriptions) {
    // Use consumeDiscardOnDestroy() so that early exits from
    // ASSERT_THAT_ERROR do not crash the unit test suite.
    Optional<vfs::OutputFile> O;
    ASSERT_THAT_ERROR(
        consumeDiscardOnDestroy(Outputs->createFile(OD.Path)).moveInto(O),
        Succeeded());
    *O << OD.Content.getData();
    ASSERT_THAT_ERROR(O->keep(), Succeeded());
  }

  SmallVector<CASOutputBackend::OutputFile> OFs = Outputs->takeOutputs();

  auto Array = makeArrayRef(OutputDescriptions);
  ASSERT_EQ(OFs.size(), Array.size());
  for (size_t I = 0, E = Array.size(); I != E; ++I) {
    EXPECT_EQ(Array[I].Path, OFs[I].Path);
    EXPECT_EQ(Array[I].Content, OFs[I].Object);
  }
}
