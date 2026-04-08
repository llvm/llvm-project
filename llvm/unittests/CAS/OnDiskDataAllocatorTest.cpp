//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/OnDiskDataAllocator.h"
#include "llvm/CAS/MappedFileRegionArena.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"

#if LLVM_ENABLE_ONDISK_CAS

using namespace llvm;
using namespace llvm::cas;

TEST(OnDiskDataAllocatorTest, Allocate) {
  unittest::TempDir Temp("data-allocator", /*Unique=*/true);
  constexpr size_t MB = 1024u * 1024u;

  std::optional<OnDiskDataAllocator> Allocator;
  ASSERT_THAT_ERROR(OnDiskDataAllocator::create(
                        Temp.path("allocator"), "data", /*MaxFileSize=*/MB,
                        /*NewFileInitialSize=*/std::nullopt)
                        .moveInto(Allocator),
                    Succeeded());

  // Allocate.
  {
    for (size_t Size = 1; Size < 16; ++Size) {
      OnDiskDataAllocator::OnDiskPtr P;
      ASSERT_THAT_ERROR(Allocator->allocate(Size).moveInto(P), Succeeded());
      EXPECT_TRUE(
          isAligned(MappedFileRegionArena::getAlign(), P.getOffset().get()));
    }
  }

  // Out of space.
  {
    OnDiskDataAllocator::OnDiskPtr P;
    ASSERT_THAT_ERROR(Allocator->allocate(MB).moveInto(P), Failed());
  }

  // Check size and capacity.
  {
    ASSERT_EQ(Allocator->capacity(), MB);
    ASSERT_LE(Allocator->size(), MB);
  }

  // Get.
  {
    OnDiskDataAllocator::OnDiskPtr P;
    ASSERT_THAT_ERROR(Allocator->allocate(32).moveInto(P), Succeeded());
    ArrayRef<char> Data;
    ASSERT_THAT_ERROR(Allocator->get(P.getOffset(), 16).moveInto(Data),
                      Succeeded());
    ASSERT_THAT_ERROR(Allocator->get(P.getOffset(), 1025).moveInto(Data),
                      Failed());
  }
}

#endif // LLVM_ENABLE_ONDISK_CAS
