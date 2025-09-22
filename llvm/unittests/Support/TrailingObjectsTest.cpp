//=== - llvm/unittest/Support/TrailingObjectsTest.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TrailingObjects.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
// This class, beyond being used by the test case, a nice
// demonstration of the intended usage of TrailingObjects, with a
// single trailing array.
class Class1 final : private TrailingObjects<Class1, short> {
  friend TrailingObjects;

  unsigned NumShorts;

protected:
  Class1(ArrayRef<int> ShortArray) : NumShorts(ShortArray.size()) {
    // This tests the non-templated getTrailingObjects() that returns a pointer
    // when using a single trailing type.
    llvm::copy(ShortArray, getTrailingObjects());
  }

public:
  static Class1 *create(ArrayRef<int> ShortArray) {
    void *Mem = ::operator new(totalSizeToAlloc<short>(ShortArray.size()));
    return new (Mem) Class1(ShortArray);
  }
  void operator delete(void *Ptr) { ::operator delete(Ptr); }

  // This indexes into the ArrayRef<> returned by `getTrailingObjects`.
  short get(unsigned Num) const { return getTrailingObjects(NumShorts)[Num]; }

  unsigned numShorts() const { return NumShorts; }

  // Pull some protected members in as public, for testability.
  template <typename... Ty>
  using FixedSizeStorage = TrailingObjects::FixedSizeStorage<Ty...>;

  using TrailingObjects::additionalSizeToAlloc;
  using TrailingObjects::getTrailingObjects;
  using TrailingObjects::getTrailingObjectsNonStrict;
  using TrailingObjects::totalSizeToAlloc;
};

// Here, there are two singular optional object types appended. Note
// that the alignment of Class2 is automatically increased to account
// for the alignment requirements of the trailing objects.
class Class2 final : private TrailingObjects<Class2, double, short> {
  friend TrailingObjects;

  bool HasShort, HasDouble;

protected:
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
  void operator delete(void *Ptr) { ::operator delete(Ptr); }

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
  template <typename... Ty>
  using FixedSizeStorage = TrailingObjects::FixedSizeStorage<Ty...>;

  using TrailingObjects::totalSizeToAlloc;
  using TrailingObjects::additionalSizeToAlloc;
  using TrailingObjects::getTrailingObjects;
};

TEST(TrailingObjects, OneArg) {
  int arr[] = {1, 2, 3};
  Class1 *C = Class1::create(arr);
  EXPECT_EQ(sizeof(Class1), sizeof(unsigned));
  EXPECT_EQ(Class1::additionalSizeToAlloc<short>(1), sizeof(short));
  EXPECT_EQ(Class1::additionalSizeToAlloc<short>(3), sizeof(short) * 3);

  EXPECT_EQ(alignof(Class1),
            alignof(Class1::FixedSizeStorage<short>::with_counts<1>::type));
  EXPECT_EQ(sizeof(Class1::FixedSizeStorage<short>::with_counts<1>::type),
            llvm::alignTo(Class1::totalSizeToAlloc<short>(1), alignof(Class1)));
  EXPECT_EQ(Class1::totalSizeToAlloc<short>(1), sizeof(Class1) + sizeof(short));

  EXPECT_EQ(alignof(Class1),
            alignof(Class1::FixedSizeStorage<short>::with_counts<3>::type));
  EXPECT_EQ(sizeof(Class1::FixedSizeStorage<short>::with_counts<3>::type),
            llvm::alignTo(Class1::totalSizeToAlloc<short>(3), alignof(Class1)));
  EXPECT_EQ(Class1::totalSizeToAlloc<short>(3),
            sizeof(Class1) + sizeof(short) * 3);

  EXPECT_EQ(C->getTrailingObjects(), reinterpret_cast<short *>(C + 1));
  EXPECT_EQ(C->get(0), 1);
  EXPECT_EQ(C->get(2), 3);

  EXPECT_EQ(C->getTrailingObjects(), C->getTrailingObjectsNonStrict<short>());

  delete C;
}

TEST(TrailingObjects, TwoArg) {
  Class2 *C1 = Class2::create(4);
  Class2 *C2 = Class2::create(0, 4.2);

  EXPECT_EQ(sizeof(Class2), llvm::alignTo(sizeof(bool) * 2, alignof(double)));
  EXPECT_EQ(alignof(Class2), alignof(double));

  EXPECT_EQ((Class2::additionalSizeToAlloc<double, short>(1, 0)),
            sizeof(double));
  EXPECT_EQ((Class2::additionalSizeToAlloc<double, short>(0, 1)),
            sizeof(short));
  EXPECT_EQ((Class2::additionalSizeToAlloc<double, short>(3, 1)),
            sizeof(double) * 3 + sizeof(short));

  EXPECT_EQ(
      alignof(Class2),
      (alignof(
          Class2::FixedSizeStorage<double, short>::with_counts<1, 1>::type)));
  EXPECT_EQ(
      sizeof(Class2::FixedSizeStorage<double, short>::with_counts<1, 1>::type),
      llvm::alignTo(Class2::totalSizeToAlloc<double, short>(1, 1),
                    alignof(Class2)));
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

// This test class is not trying to be a usage demo, just asserting
// that three args does actually work too (it's the same code that
// handles the second arg, so it's basically covered by the above, but
// just in case..)
class Class3 final : private TrailingObjects<Class3, double, short, bool> {
  friend TrailingObjects;

  size_t numTrailingObjects(OverloadToken<double>) const { return 1; }
  size_t numTrailingObjects(OverloadToken<short>) const { return 1; }

public:
  // Pull some protected members in as public, for testability.
  template <typename... Ty>
  using FixedSizeStorage = TrailingObjects::FixedSizeStorage<Ty...>;

  using TrailingObjects::additionalSizeToAlloc;
  using TrailingObjects::getTrailingObjects;
  using TrailingObjects::totalSizeToAlloc;
};

TEST(TrailingObjects, ThreeArg) {
  EXPECT_EQ((Class3::additionalSizeToAlloc<double, short, bool>(1, 1, 3)),
            sizeof(double) + sizeof(short) + 3 * sizeof(bool));
  EXPECT_EQ(sizeof(Class3), llvm::alignTo(1, alignof(double)));

  EXPECT_EQ(
      alignof(Class3),
      (alignof(Class3::FixedSizeStorage<double, short,
                                        bool>::with_counts<1, 1, 3>::type)));
  EXPECT_EQ(
      sizeof(Class3::FixedSizeStorage<double, short,
                                      bool>::with_counts<1, 1, 3>::type),
      llvm::alignTo(Class3::totalSizeToAlloc<double, short, bool>(1, 1, 3),
                    alignof(Class3)));

  std::unique_ptr<char[]> P(new char[1000]);
  Class3 *C = reinterpret_cast<Class3 *>(P.get());
  EXPECT_EQ(C->getTrailingObjects<double>(), reinterpret_cast<double *>(C + 1));
  EXPECT_EQ(C->getTrailingObjects<short>(),
            reinterpret_cast<short *>(reinterpret_cast<double *>(C + 1) + 1));
  EXPECT_EQ(
      C->getTrailingObjects<bool>(),
      reinterpret_cast<bool *>(
          reinterpret_cast<short *>(reinterpret_cast<double *>(C + 1) + 1) +
          1));
}

class Class4 final : private TrailingObjects<Class4, char, long> {
  friend TrailingObjects;
  size_t numTrailingObjects(OverloadToken<char>) const { return 1; }

public:
  // Pull some protected members in as public, for testability.
  template <typename... Ty>
  using FixedSizeStorage = TrailingObjects::FixedSizeStorage<Ty...>;

  using TrailingObjects::additionalSizeToAlloc;
  using TrailingObjects::getTrailingObjects;
  using TrailingObjects::totalSizeToAlloc;
};

TEST(TrailingObjects, Realignment) {
  EXPECT_EQ((Class4::additionalSizeToAlloc<char, long>(1, 1)),
            llvm::alignTo(sizeof(long) + 1, alignof(long)));
  EXPECT_EQ(sizeof(Class4), llvm::alignTo(1, alignof(long)));

  EXPECT_EQ(
      alignof(Class4),
      (alignof(Class4::FixedSizeStorage<char, long>::with_counts<1, 1>::type)));
  EXPECT_EQ(
      sizeof(Class4::FixedSizeStorage<char, long>::with_counts<1, 1>::type),
      llvm::alignTo(Class4::totalSizeToAlloc<char, long>(1, 1),
                    alignof(Class4)));

  std::unique_ptr<char[]> P(new char[1000]);
  Class4 *C = reinterpret_cast<Class4 *>(P.get());
  EXPECT_EQ(C->getTrailingObjects<char>(), reinterpret_cast<char *>(C + 1));
  EXPECT_EQ(C->getTrailingObjects<long>(),
            reinterpret_cast<long *>(llvm::alignAddr(
                reinterpret_cast<char *>(C + 1) + 1, Align::Of<long>())));
}
}

// Test the use of TrailingObjects with a template class. This
// previously failed to compile due to a bug in MSVC's member access
// control/lookup handling for OverloadToken.
template <typename Derived>
class Class5Tmpl : private llvm::TrailingObjects<Derived, float, int> {
  using TrailingObjects = typename llvm::TrailingObjects<Derived, float>;
  friend TrailingObjects;

  size_t numTrailingObjects(
      typename TrailingObjects::template OverloadToken<float>) const {
    return 1;
  }
};

class Class5 : public Class5Tmpl<Class5> {};
