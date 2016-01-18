//=== - llvm/unittest/Support/TrailingObjectsTest.cpp ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TrailingObjects.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
// This class, beyond being used by the test case, a nice
// demonstration of the intended usage of TrailingObjects, with a
// single trailing array.
class Class1 final : protected TrailingObjects<Class1, short> {
  friend TrailingObjects;

  unsigned NumShorts;

protected:
  size_t numTrailingObjects(OverloadToken<short>) const { return NumShorts; }

  Class1(int *ShortArray, unsigned NumShorts) : NumShorts(NumShorts) {
    std::uninitialized_copy(ShortArray, ShortArray + NumShorts,
                            getTrailingObjects<short>());
  }

public:
  static Class1 *create(int *ShortArray, unsigned NumShorts) {
    void *Mem = ::operator new(totalSizeToAlloc<short>(NumShorts));
    return new (Mem) Class1(ShortArray, NumShorts);
  }

  short get(unsigned Num) const { return getTrailingObjects<short>()[Num]; }

  unsigned numShorts() const { return NumShorts; }

  // Pull some protected members in as public, for testability.
  using TrailingObjects::totalSizeToAlloc;
  using TrailingObjects::additionalSizeToAlloc;
  using TrailingObjects::getTrailingObjects;
};

// Here, there are two singular optional object types appended.
// Note that it fails to compile without the alignment spec.
class LLVM_ALIGNAS(8) Class2 final : protected TrailingObjects<Class2, double, short> {
  friend TrailingObjects;

  bool HasShort, HasDouble;

protected:
  size_t numTrailingObjects(OverloadToken<short>) const {
    return HasShort ? 1 : 0;
  }
  size_t numTrailingObjects(OverloadToken<double>) const {
    return HasDouble ? 1 : 0;
  }

  Class2(bool HasShort, bool HasDouble)
      : HasShort(HasShort), HasDouble(HasDouble) {}

public:
  static Class2 *create(short S = 0, double D = 0.0) {
    bool HasShort = S != 0;
    bool HasDouble = D != 0.0;

    void *Mem =
        ::operator new(totalSizeToAlloc<double, short>(HasDouble, HasShort));
    Class2 *C = new (Mem) Class2(HasShort, HasDouble);
    if (HasShort)
      *C->getTrailingObjects<short>() = S;
    if (HasDouble)
      *C->getTrailingObjects<double>() = D;
    return C;
  }

  short getShort() const {
    if (!HasShort)
      return 0;
    return *getTrailingObjects<short>();
  }

  double getDouble() const {
    if (!HasDouble)
      return 0.0;
    return *getTrailingObjects<double>();
  }

  // Pull some protected members in as public, for testability.
  using TrailingObjects::totalSizeToAlloc;
  using TrailingObjects::additionalSizeToAlloc;
  using TrailingObjects::getTrailingObjects;
};

TEST(TrailingObjects, OneArg) {
  int arr[] = {1, 2, 3};
  Class1 *C = Class1::create(arr, 3);
  EXPECT_EQ(sizeof(Class1), sizeof(unsigned));
  EXPECT_EQ(Class1::additionalSizeToAlloc<short>(1), sizeof(short));
  EXPECT_EQ(Class1::additionalSizeToAlloc<short>(3), sizeof(short) * 3);

  EXPECT_EQ(Class1::totalSizeToAlloc<short>(1), sizeof(Class1) + sizeof(short));
  EXPECT_EQ(Class1::totalSizeToAlloc<short>(3),
            sizeof(Class1) + sizeof(short) * 3);

  EXPECT_EQ(C->getTrailingObjects<short>(), reinterpret_cast<short *>(C + 1));
  EXPECT_EQ(C->get(0), 1);
  EXPECT_EQ(C->get(2), 3);
  delete C;
}

TEST(TrailingObjects, TwoArg) {
  Class2 *C1 = Class2::create(4);
  Class2 *C2 = Class2::create(0, 4.2);

  EXPECT_EQ(sizeof(Class2), 8u); // due to alignment

  EXPECT_EQ((Class2::additionalSizeToAlloc<double, short>(1, 0)),
            sizeof(double));
  EXPECT_EQ((Class2::additionalSizeToAlloc<double, short>(0, 1)),
            sizeof(short));
  EXPECT_EQ((Class2::additionalSizeToAlloc<double, short>(3, 1)),
            sizeof(double) * 3 + sizeof(short));

  EXPECT_EQ((Class2::totalSizeToAlloc<double, short>(1, 1)),
            sizeof(Class2) + sizeof(double) + sizeof(short));

  EXPECT_EQ(C1->getDouble(), 0);
  EXPECT_EQ(C1->getShort(), 4);
  EXPECT_EQ(C1->getTrailingObjects<double>(),
            reinterpret_cast<double *>(C1 + 1));
  EXPECT_EQ(C1->getTrailingObjects<short>(), reinterpret_cast<short *>(C1 + 1));

  EXPECT_EQ(C2->getDouble(), 4.2);
  EXPECT_EQ(C2->getShort(), 0);
  EXPECT_EQ(C2->getTrailingObjects<double>(),
            reinterpret_cast<double *>(C2 + 1));
  EXPECT_EQ(C2->getTrailingObjects<short>(),
            reinterpret_cast<short *>(reinterpret_cast<double *>(C2 + 1) + 1));
  delete C1;
  delete C2;
}
}
