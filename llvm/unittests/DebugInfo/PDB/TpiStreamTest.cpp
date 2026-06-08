//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/PDB.h"
#include "llvm/Support/Path.h"

#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::pdb;
using namespace llvm::codeview;

extern const char *TestMainArgv0;

// FIXME: This should use ObjectYAML to create a PDB from YAML.
static std::string getPdbPath() {
  SmallString<128> InputsDir = unittest::getInputFileDirectory(TestMainArgv0);
  llvm::sys::path::append(InputsDir, "SimpleTest.pdb");
  return std::string(InputsDir);
}

// Types:
//  0x1000 | LF_ARGLIST [size = 8]
//  0x1001 | LF_PROCEDURE [size = 16]
static Expected<std::unique_ptr<PDBFile>>
openSimplePdb(BumpPtrAllocator &Allocator) {
  std::string PdbPath = getPdbPath();
  ErrorOr<std::unique_ptr<MemoryBuffer>> ErrorOrBuffer =
      MemoryBuffer::getFile(PdbPath, /*IsText=*/false,
                            /*RequiresNullTerminator=*/false);
  EXPECT_TRUE(ErrorOrBuffer);
  std::unique_ptr<llvm::MemoryBuffer> Buffer = std::move(*ErrorOrBuffer);

  auto Stream = std::make_unique<MemoryBufferByteStream>(
      std::move(Buffer), llvm::endianness::little);

  auto File = std::make_unique<PDBFile>(PdbPath, std::move(Stream), Allocator);
  if (Error Err = File->parseFileHeaders())
    return Err;
  if (Error Err = File->parseStreamData())
    return Err;
  return File;
}

TEST(TpiStreamTest, getType) {
  BumpPtrAllocator Allocator;
  Expected<std::unique_ptr<PDBFile>> FileOrErr = openSimplePdb(Allocator);
  ASSERT_TRUE(!!FileOrErr);
  std::unique_ptr<PDBFile> File = std::move(*FileOrErr);

  Expected<TpiStream &> Tpi = File->getPDBTpiStream();
  ASSERT_TRUE(!!Tpi) << Tpi.takeError();

  ASSERT_EQ(Tpi->getType(TypeIndex(0x1000)).kind(), LF_ARGLIST);
  ASSERT_EQ(Tpi->getType(TypeIndex(0x1001)).kind(), LF_PROCEDURE);
}

TEST(TpiStreamTest, getTypeOrEmpty) {
  BumpPtrAllocator Allocator;
  Expected<std::unique_ptr<PDBFile>> FileOrErr = openSimplePdb(Allocator);
  ASSERT_TRUE(!!FileOrErr);
  std::unique_ptr<PDBFile> File = std::move(*FileOrErr);

  Expected<TpiStream &> Tpi = File->getPDBTpiStream();
  ASSERT_TRUE(!!Tpi);

  ASSERT_FALSE(Tpi->getTypeOrEmpty(TypeIndex(0)).valid());
  ASSERT_FALSE(Tpi->getTypeOrEmpty(TypeIndex(0xFFF)).valid());
  ASSERT_FALSE(Tpi->getTypeOrEmpty(TypeIndex(0x1002)).valid());
  ASSERT_FALSE(Tpi->getTypeOrEmpty(TypeIndex(0x1003)).valid());
  ASSERT_FALSE(Tpi->getTypeOrEmpty(TypeIndex(0x1234567)).valid());

  ASSERT_EQ(Tpi->getTypeOrEmpty(TypeIndex(0x1000)).kind(), LF_ARGLIST);
  ASSERT_EQ(Tpi->getTypeOrEmpty(TypeIndex(0x1001)).kind(), LF_PROCEDURE);
}

TEST(TpiStreamTest, tryGetType) {
  BumpPtrAllocator Allocator;
  Expected<std::unique_ptr<PDBFile>> FileOrErr = openSimplePdb(Allocator);
  ASSERT_TRUE(!!FileOrErr);
  std::unique_ptr<PDBFile> File = std::move(*FileOrErr);

  Expected<TpiStream &> Tpi = File->getPDBTpiStream();
  ASSERT_TRUE(!!Tpi);

  ASSERT_FALSE(Tpi->tryGetType(TypeIndex(0)).has_value());
  ASSERT_FALSE(Tpi->tryGetType(TypeIndex(0xFFF)).has_value());
  ASSERT_FALSE(Tpi->tryGetType(TypeIndex(0x1002)).has_value());
  ASSERT_FALSE(Tpi->tryGetType(TypeIndex(0x1003)).has_value());
  ASSERT_FALSE(Tpi->tryGetType(TypeIndex(0x1234567)).has_value());

  std::optional<CVType> T0 = Tpi->tryGetType(TypeIndex(0x1000));
  std::optional<CVType> T1 = Tpi->tryGetType(TypeIndex(0x1001));
  ASSERT_TRUE(T0.has_value());
  ASSERT_EQ(T0->kind(), LF_ARGLIST);
  ASSERT_TRUE(T1.has_value());
  ASSERT_EQ(T1->kind(), LF_PROCEDURE);
}

TEST(TpiStreamTest, getTypeOrErr) {
  BumpPtrAllocator Allocator;
  Expected<std::unique_ptr<PDBFile>> FileOrErr = openSimplePdb(Allocator);
  ASSERT_TRUE(!!FileOrErr);
  std::unique_ptr<PDBFile> File = std::move(*FileOrErr);

  Expected<TpiStream &> Tpi = File->getPDBTpiStream();
  ASSERT_TRUE(!!Tpi);

  Expected<CVType> E0 = Tpi->getTypeOrError(TypeIndex(0));
  ASSERT_FALSE(E0);
  ASSERT_EQ(toString(E0.takeError()), "Type index too low (0)");
  Expected<CVType> E1 = Tpi->getTypeOrError(TypeIndex(0xFFF));
  ASSERT_FALSE(E1);
  ASSERT_EQ(toString(E1.takeError()), "Type index too low (4095)");
  Expected<CVType> E2 = Tpi->getTypeOrError(TypeIndex(0x1002));
  ASSERT_FALSE(E2);
  ASSERT_EQ(toString(E2.takeError()), "Type index too high (4098)");
  Expected<CVType> E3 = Tpi->getTypeOrError(TypeIndex(0x1003));
  ASSERT_FALSE(E3);
  ASSERT_EQ(toString(E3.takeError()), "Invalid type index");
  Expected<CVType> E4 = Tpi->getTypeOrError(TypeIndex(0x123456));
  ASSERT_FALSE(E4);
  ASSERT_EQ(toString(E4.takeError()), "Invalid type index");

  Expected<CVType> T0 = Tpi->getTypeOrError(TypeIndex(0x1000));
  ASSERT_TRUE(!!T0);
  ASSERT_EQ(T0->kind(), LF_ARGLIST);

  Expected<CVType> T1 = Tpi->getTypeOrError(TypeIndex(0x1001));
  ASSERT_TRUE(!!T1);
  ASSERT_EQ(T1->kind(), LF_PROCEDURE);
}
