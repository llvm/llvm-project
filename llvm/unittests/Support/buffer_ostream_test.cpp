//===- buffer_ostream_test.cpp - buffer_ostream tests ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

/// Naive version of raw_svector_ostream that is buffered (by default) and
/// doesn't support pwrite.
class NaiveSmallVectorStream : public raw_ostream {
public:
  uint64_t current_pos() const override { return Vector.size(); }
  void write_impl(const char *Ptr, size_t Size) override {
    Vector.append(Ptr, Ptr + Size);
  }

  explicit NaiveSmallVectorStream(SmallVectorImpl<char> &Vector)
      : Vector(Vector) {}
  ~NaiveSmallVectorStream() override { flush(); }

  SmallVectorImpl<char> &Vector;
};

TEST(buffer_ostreamTest, Reference) {
  SmallString<128> Dest;
  {
    NaiveSmallVectorStream DestOS(Dest);
    buffer_ostream BufferOS(DestOS);

    // Writing and flushing should have no effect on Dest.
    BufferOS << "abcd";
    static_cast<raw_ostream &>(BufferOS).flush();
    EXPECT_EQ("", Dest);
    DestOS.flush();
    EXPECT_EQ("", Dest);
  }

  // Write should land when constructor is called.
  EXPECT_EQ("abcd", Dest);
}

TEST(buffer_ostreamTest, Owned) {
  SmallString<128> Dest;
  {
    auto DestOS = std::make_unique<NaiveSmallVectorStream>(Dest);

    // Confirm that NaiveSmallVectorStream is buffered by default.
    EXPECT_NE(0u, DestOS->GetBufferSize());

    // Confirm that passing ownership to buffer_unique_ostream sets it to
    // unbuffered. Also steal a reference to DestOS.
    NaiveSmallVectorStream &DestOSRef = *DestOS;
    buffer_unique_ostream BufferOS(std::move(DestOS));
    EXPECT_EQ(0u, DestOSRef.GetBufferSize());

    // Writing and flushing should have no effect on Dest.
    BufferOS << "abcd";
    static_cast<raw_ostream &>(BufferOS).flush();
    EXPECT_EQ("", Dest);
    DestOSRef.flush();
    EXPECT_EQ("", Dest);
  }

  // Write should land when constructor is called.
  EXPECT_EQ("abcd", Dest);
}

} // end namespace
