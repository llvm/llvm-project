//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/BuiltinObjectHasher.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;

using HasherT = BLAKE3;
using HashType = BuiltinObjectHasher<HasherT>::HashT;

TEST(BuiltinObjectHasherTest, Basic) {
  unittest::TempFile TmpFile("somefile.o", /*Suffix=*/"", /*Contents=*/"",
                             /*Unique=*/true);
  {
    std::error_code EC;
    raw_fd_stream Out(TmpFile.path(), EC);
    ASSERT_FALSE(EC);
    SmallVector<char, 200> Data;
    for (unsigned i = 1; i != 201; ++i) {
      Data.push_back(i);
    }
    for (unsigned i = 0; i != 1000; ++i) {
      Out.write(Data.data(), Data.size());
    }
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFile(TmpFile.path());
  ASSERT_TRUE(!!MB);
  ASSERT_NE(*MB, nullptr);

  HashType Hash1 =
      BuiltinObjectHasher<HasherT>::hashObject({}, (*MB)->getBuffer());
  std::optional<HashType> Hash2;
  ASSERT_THAT_ERROR(
      BuiltinObjectHasher<HasherT>::hashFile(TmpFile.path()).moveInto(Hash2),
      Succeeded());
  EXPECT_EQ(Hash1, *Hash2);
}
