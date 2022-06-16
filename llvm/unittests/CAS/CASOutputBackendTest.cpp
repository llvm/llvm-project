//===- CASOutputBackendTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASOutputBackend.h"
#include "llvm/CAS/CASDB.h"
#include "llvm/CAS/CASFileSystem.h"
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
  std::unique_ptr<CASDB> CAS = createInMemoryCAS();
  ASSERT_TRUE(CAS);

  auto Outputs = makeIntrusiveRefCnt<CASOutputBackend>(*CAS);

  Optional<BlobProxy> Content1;
  Optional<BlobProxy> Content2;
  Optional<BlobProxy> AbsolutePath1;
  Optional<BlobProxy> AbsolutePath2;
  Optional<BlobProxy> RelativePath;
  Optional<BlobProxy> WindowsPath;
  ASSERT_THAT_ERROR(CAS->createBlob("content1").moveInto(Content1),
                    Succeeded());
  ASSERT_THAT_ERROR(CAS->createBlob("content2").moveInto(Content2),
                    Succeeded());
  ASSERT_THAT_ERROR(CAS->createBlob("/absolute/path/1").moveInto(AbsolutePath1),
                    Succeeded());
  ASSERT_THAT_ERROR(CAS->createBlob("/absolute/path/2").moveInto(AbsolutePath2),
                    Succeeded());
  ASSERT_THAT_ERROR(CAS->createBlob("relative/path/./2").moveInto(RelativePath),
                    Succeeded());
  ASSERT_THAT_ERROR(CAS->createBlob("c:\\windows/path").moveInto(WindowsPath),
                    Succeeded());

  // FIXME: Add test of duplicate paths. Maybe could error at createFile()...
  struct OutputDescription {
    BlobProxy Content;
    BlobProxy Path;
  };
  OutputDescription OutputDescriptions[] = {
      {*Content1, *AbsolutePath1}, {*Content2, *AbsolutePath2},
      {*Content1, *AbsolutePath1}, {*Content1, *RelativePath},
      {*Content1, *WindowsPath},
  };
  for (OutputDescription OD : OutputDescriptions) {
    // Use consumeDiscardOnDestroy() so that early exits from
    // ASSERT_THAT_ERROR do not crash the unit test suite.
    Optional<vfs::OutputFile> O;
    ASSERT_THAT_ERROR(
        consumeDiscardOnDestroy(Outputs->createFile(OD.Path.getData()))
            .moveInto(O),
        Succeeded());
    *O << OD.Content.getData();
    ASSERT_THAT_ERROR(O->keep(), Succeeded());
  }

  Optional<NodeProxy> Root;
  ASSERT_THAT_ERROR(Outputs->createNode().moveInto(Root), Succeeded());

  auto Array = makeArrayRef(OutputDescriptions);
  ASSERT_EQ(Array.size() * 2, Root->getNumReferences());
  for (size_t I = 0, E = Array.size(); I != E; ++I) {
    ASSERT_EQ(Array[I].Path, Root->getReference(I * 2));
    ASSERT_EQ(Array[I].Content, Root->getReference(I * 2 + 1));
  }
}
