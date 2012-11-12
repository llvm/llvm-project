//===- llvm/unittest/ADT/SmallVectorTest.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// SmallVector unit tests.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include <stdarg.h>
#include <list>

using namespace llvm;

namespace {

/// A helper class that counts the total number of constructor and
/// destructor calls.
class Constructable {
private:
  static int numConstructorCalls;
  static int numDestructorCalls;
  static int numAssignmentCalls;

  int value;

public:
  Constructable() : value(0) {
    ++numConstructorCalls;
  }

  Constructable(int val) : value(val) {
    ++numConstructorCalls;
  }

  Constructable(const Constructable & src) {
    value = src.value;
    ++numConstructorCalls;
  }

  ~Constructable() {
    ++numDestructorCalls;
  }

  Constructable & operator=(const Constructable & src) {
    value = src.value;
    ++numAssignmentCalls;
    return *this;
  }

  int getValue() const {
    return abs(value);
  }

  static void reset() {
    numConstructorCalls = 0;
    numDestructorCalls = 0;
    numAssignmentCalls = 0;
  }

  static int getNumConstructorCalls() {
    return numConstructorCalls;
  }

  static int getNumDestructorCalls() {
    return numDestructorCalls;
  }

  friend bool operator==(const Constructable & c0, const Constructable & c1) {
    return c0.getValue() == c1.getValue();
  }

  friend bool LLVM_ATTRIBUTE_UNUSED
  operator!=(const Constructable & c0, const Constructable & c1) {
    return c0.getValue() != c1.getValue();
  }
};

int Constructable::numConstructorCalls;
int Constructable::numDestructorCalls;
int Constructable::numAssignmentCalls;

// Test fixture class
template <typename VectorT>
class SmallVectorTest : public testing::Test {
protected:
  VectorT theVector;
  VectorT otherVector;

  void SetUp() {
    Constructable::reset();
  }

  void assertEmpty(VectorT & v) {
    // Size tests
    EXPECT_EQ(0u, v.size());
    EXPECT_TRUE(v.empty());

    // Iterator tests
    EXPECT_TRUE(v.begin() == v.end());
  }

  // Assert that theVector contains the specified values, in order.
  void assertValuesInOrder(VectorT & v, size_t size, ...) {
    EXPECT_EQ(size, v.size());

    va_list ap;
    va_start(ap, size);
    for (size_t i = 0; i < size; ++i) {
      int value = va_arg(ap, int);
      EXPECT_EQ(value, v[i].getValue());
    }

    va_end(ap);
  }

  // Generate a sequence of values to initialize the vector.
  void makeSequence(VectorT & v, int start, int end) {
    for (int i = start; i <= end; ++i) {
      v.push_back(Constructable(i));
    }
  }
};

typedef ::testing::Types<SmallVector<Constructable, 0>,
                         SmallVector<Constructable, 1>,
                         SmallVector<Constructable, 2>,
                         SmallVector<Constructable, 4>
                         > SmallVectorTestTypes;
TYPED_TEST_CASE(SmallVectorTest, SmallVectorTestTypes);

// New vector test.
TYPED_TEST(SmallVectorTest, EmptyVectorTest) {
  SCOPED_TRACE("EmptyVectorTest");
  this->assertEmpty(this->theVector);
  EXPECT_TRUE(this->theVector.rbegin() == this->theVector.rend());
  EXPECT_EQ(0, Constructable::getNumConstructorCalls());
  EXPECT_EQ(0, Constructable::getNumDestructorCalls());
}

// Simple insertions and deletions.
TYPED_TEST(SmallVectorTest, PushPopTest) {
  SCOPED_TRACE("PushPopTest");

  // Track whether the vector will potentially have to grow.
  bool RequiresGrowth = this->theVector.capacity() < 3;

  // Push an element
  this->theVector.push_back(Constructable(1));

  // Size tests
  this->assertValuesInOrder(this->theVector, 1u, 1);
  EXPECT_FALSE(this->theVector.begin() == this->theVector.end());
  EXPECT_FALSE(this->theVector.empty());

  // Push another element
  this->theVector.push_back(Constructable(2));
  this->assertValuesInOrder(this->theVector, 2u, 1, 2);

  // Insert at beginning
  this->theVector.insert(this->theVector.begin(), this->theVector[1]);
  this->assertValuesInOrder(this->theVector, 3u, 2, 1, 2);

  // Pop one element
  this->theVector.pop_back();
  this->assertValuesInOrder(this->theVector, 2u, 2, 1);

  // Pop remaining elements
  this->theVector.pop_back();
  this->theVector.pop_back();
  this->assertEmpty(this->theVector);

  // Check number of constructor calls. Should be 2 for each list element,
  // one for the argument to push_back, one for the argument to insert,
  // and one for the list element itself.
  if (!RequiresGrowth) {
    EXPECT_EQ(5, Constructable::getNumConstructorCalls());
    EXPECT_EQ(5, Constructable::getNumDestructorCalls());
  } else {
    // If we had to grow the vector, these only have a lower bound, but should
    // always be equal.
    EXPECT_LE(5, Constructable::getNumConstructorCalls());
    EXPECT_EQ(Constructable::getNumConstructorCalls(),
              Constructable::getNumDestructorCalls());
  }
}

// Clear test.
TYPED_TEST(SmallVectorTest, ClearTest) {
  SCOPED_TRACE("ClearTest");

  this->theVector.reserve(2);
  this->makeSequence(this->theVector, 1, 2);
  this->theVector.clear();

  this->assertEmpty(this->theVector);
  EXPECT_EQ(4, Constructable::getNumConstructorCalls());
  EXPECT_EQ(4, Constructable::getNumDestructorCalls());
}

// Resize smaller test.
TYPED_TEST(SmallVectorTest, ResizeShrinkTest) {
  SCOPED_TRACE("ResizeShrinkTest");

  this->theVector.reserve(3);
  this->makeSequence(this->theVector, 1, 3);
  this->theVector.resize(1);

  this->assertValuesInOrder(this->theVector, 1u, 1);
  EXPECT_EQ(6, Constructable::getNumConstructorCalls());
  EXPECT_EQ(5, Constructable::getNumDestructorCalls());
}

// Resize bigger test.
TYPED_TEST(SmallVectorTest, ResizeGrowTest) {
  SCOPED_TRACE("ResizeGrowTest");

  this->theVector.resize(2);

  // The extra constructor/destructor calls come from the temporary object used
  // to initialize the contents of the resized array (via copy construction).
  EXPECT_EQ(3, Constructable::getNumConstructorCalls());
  EXPECT_EQ(1, Constructable::getNumDestructorCalls());
  EXPECT_EQ(2u, this->theVector.size());
}

// Resize with fill value.
TYPED_TEST(SmallVectorTest, ResizeFillTest) {
  SCOPED_TRACE("ResizeFillTest");

  this->theVector.resize(3, Constructable(77));
  this->assertValuesInOrder(this->theVector, 3u, 77, 77, 77);
}

// Overflow past fixed size.
TYPED_TEST(SmallVectorTest, OverflowTest) {
  SCOPED_TRACE("OverflowTest");

  // Push more elements than the fixed size.
  this->makeSequence(this->theVector, 1, 10);

  // Test size and values.
  EXPECT_EQ(10u, this->theVector.size());
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(i+1, this->theVector[i].getValue());
  }

  // Now resize back to fixed size.
  this->theVector.resize(1);

  this->assertValuesInOrder(this->theVector, 1u, 1);
}

// Iteration tests.
TYPED_TEST(SmallVectorTest, IterationTest) {
  this->makeSequence(this->theVector, 1, 2);

  // Forward Iteration
  typename TypeParam::iterator it = this->theVector.begin();
  EXPECT_TRUE(*it == this->theVector.front());
  EXPECT_TRUE(*it == this->theVector[0]);
  EXPECT_EQ(1, it->getValue());
  ++it;
  EXPECT_TRUE(*it == this->theVector[1]);
  EXPECT_TRUE(*it == this->theVector.back());
  EXPECT_EQ(2, it->getValue());
  ++it;
  EXPECT_TRUE(it == this->theVector.end());
  --it;
  EXPECT_TRUE(*it == this->theVector[1]);
  EXPECT_EQ(2, it->getValue());
  --it;
  EXPECT_TRUE(*it == this->theVector[0]);
  EXPECT_EQ(1, it->getValue());

  // Reverse Iteration
  typename TypeParam::reverse_iterator rit = this->theVector.rbegin();
  EXPECT_TRUE(*rit == this->theVector[1]);
  EXPECT_EQ(2, rit->getValue());
  ++rit;
  EXPECT_TRUE(*rit == this->theVector[0]);
  EXPECT_EQ(1, rit->getValue());
  ++rit;
  EXPECT_TRUE(rit == this->theVector.rend());
  --rit;
  EXPECT_TRUE(*rit == this->theVector[0]);
  EXPECT_EQ(1, rit->getValue());
  --rit;
  EXPECT_TRUE(*rit == this->theVector[1]);
  EXPECT_EQ(2, rit->getValue());
}

// Swap test.
TYPED_TEST(SmallVectorTest, SwapTest) {
  SCOPED_TRACE("SwapTest");

  this->makeSequence(this->theVector, 1, 2);
  std::swap(this->theVector, this->otherVector);

  this->assertEmpty(this->theVector);
  this->assertValuesInOrder(this->otherVector, 2u, 1, 2);
}

// Append test
TYPED_TEST(SmallVectorTest, AppendTest) {
  SCOPED_TRACE("AppendTest");

  this->makeSequence(this->otherVector, 2, 3);

  this->theVector.push_back(Constructable(1));
  this->theVector.append(this->otherVector.begin(), this->otherVector.end());

  this->assertValuesInOrder(this->theVector, 3u, 1, 2, 3);
}

// Append repeated test
TYPED_TEST(SmallVectorTest, AppendRepeatedTest) {
  SCOPED_TRACE("AppendRepeatedTest");

  this->theVector.push_back(Constructable(1));
  this->theVector.append(2, Constructable(77));
  this->assertValuesInOrder(this->theVector, 3u, 1, 77, 77);
}

// Assign test
TYPED_TEST(SmallVectorTest, AssignTest) {
  SCOPED_TRACE("AssignTest");

  this->theVector.push_back(Constructable(1));
  this->theVector.assign(2, Constructable(77));
  this->assertValuesInOrder(this->theVector, 2u, 77, 77);
}

// Erase a single element
TYPED_TEST(SmallVectorTest, EraseTest) {
  SCOPED_TRACE("EraseTest");

  this->makeSequence(this->theVector, 1, 3);
  this->theVector.erase(this->theVector.begin());
  this->assertValuesInOrder(this->theVector, 2u, 2, 3);
}

// Erase a range of elements
TYPED_TEST(SmallVectorTest, EraseRangeTest) {
  SCOPED_TRACE("EraseRangeTest");

  this->makeSequence(this->theVector, 1, 3);
  this->theVector.erase(this->theVector.begin(), this->theVector.begin() + 2);
  this->assertValuesInOrder(this->theVector, 1u, 3);
}

// Insert a single element.
TYPED_TEST(SmallVectorTest, InsertTest) {
  SCOPED_TRACE("InsertTest");

  this->makeSequence(this->theVector, 1, 3);
  typename TypeParam::iterator I =
    this->theVector.insert(this->theVector.begin() + 1, Constructable(77));
  EXPECT_EQ(this->theVector.begin() + 1, I);
  this->assertValuesInOrder(this->theVector, 4u, 1, 77, 2, 3);
}

// Insert repeated elements.
TYPED_TEST(SmallVectorTest, InsertRepeatedTest) {
  SCOPED_TRACE("InsertRepeatedTest");

  this->makeSequence(this->theVector, 10, 15);
  typename TypeParam::iterator I =
    this->theVector.insert(this->theVector.begin() + 1, 2, Constructable(16));
  EXPECT_EQ(this->theVector.begin() + 1, I);
  this->assertValuesInOrder(this->theVector, 8u,
                      10, 16, 16, 11, 12, 13, 14, 15);

  // Insert at end.
  I = this->theVector.insert(this->theVector.end(), 2, Constructable(16));
  EXPECT_EQ(this->theVector.begin() + 8, I);
  this->assertValuesInOrder(this->theVector, 10u,
                      10, 16, 16, 11, 12, 13, 14, 15, 16, 16);

  // Empty insert.
  EXPECT_EQ(this->theVector.end(),
            this->theVector.insert(this->theVector.end(),
                                   0, Constructable(42)));
  EXPECT_EQ(this->theVector.begin() + 1,
            this->theVector.insert(this->theVector.begin() + 1,
                                   0, Constructable(42)));
}

// Insert range.
TYPED_TEST(SmallVectorTest, InsertRangeTest) {
  SCOPED_TRACE("InsertRangeTest");

  Constructable Arr[3] =
    { Constructable(77), Constructable(77), Constructable(77) };

  this->makeSequence(this->theVector, 1, 3);
  typename TypeParam::iterator I =
    this->theVector.insert(this->theVector.begin() + 1, Arr, Arr+3);
  EXPECT_EQ(this->theVector.begin() + 1, I);
  this->assertValuesInOrder(this->theVector, 6u, 1, 77, 77, 77, 2, 3);

  // Insert at end.
  I = this->theVector.insert(this->theVector.end(), Arr, Arr+3);
  EXPECT_EQ(this->theVector.begin() + 6, I);
  this->assertValuesInOrder(this->theVector, 9u,
                            1, 77, 77, 77, 2, 3, 77, 77, 77);

  // Empty insert.
  EXPECT_EQ(this->theVector.end(),
            this->theVector.insert(this->theVector.end(),
                                   this->theVector.begin(),
                                   this->theVector.begin()));
  EXPECT_EQ(this->theVector.begin() + 1,
            this->theVector.insert(this->theVector.begin() + 1,
                                   this->theVector.begin(),
                                   this->theVector.begin()));
}

// Comparison tests.
TYPED_TEST(SmallVectorTest, ComparisonTest) {
  SCOPED_TRACE("ComparisonTest");

  this->makeSequence(this->theVector, 1, 3);
  this->makeSequence(this->otherVector, 1, 3);

  EXPECT_TRUE(this->theVector == this->otherVector);
  EXPECT_FALSE(this->theVector != this->otherVector);

  this->otherVector.clear();
  this->makeSequence(this->otherVector, 2, 4);

  EXPECT_FALSE(this->theVector == this->otherVector);
  EXPECT_TRUE(this->theVector != this->otherVector);
}

// Constant vector tests.
TYPED_TEST(SmallVectorTest, ConstVectorTest) {
  const TypeParam constVector;

  EXPECT_EQ(0u, constVector.size());
  EXPECT_TRUE(constVector.empty());
  EXPECT_TRUE(constVector.begin() == constVector.end());
}

// Direct array access.
TYPED_TEST(SmallVectorTest, DirectVectorTest) {
  EXPECT_EQ(0u, this->theVector.size());
  this->theVector.reserve(4);
  EXPECT_LE(4u, this->theVector.capacity());
  EXPECT_EQ(0, Constructable::getNumConstructorCalls());
  this->theVector.end()[0] = 1;
  this->theVector.end()[1] = 2;
  this->theVector.end()[2] = 3;
  this->theVector.end()[3] = 4;
  this->theVector.set_size(4);
  EXPECT_EQ(4u, this->theVector.size());
  EXPECT_EQ(4, Constructable::getNumConstructorCalls());
  EXPECT_EQ(1, this->theVector[0].getValue());
  EXPECT_EQ(2, this->theVector[1].getValue());
  EXPECT_EQ(3, this->theVector[2].getValue());
  EXPECT_EQ(4, this->theVector[3].getValue());
}

TYPED_TEST(SmallVectorTest, IteratorTest) {
  std::list<int> L;
  this->theVector.insert(this->theVector.end(), L.begin(), L.end());
}

struct notassignable {
  int &x;
  notassignable(int &x) : x(x) {}
};

TEST(SmallVectorCustomTest, NoAssignTest) {
  int x = 0;
  SmallVector<notassignable, 2> vec;
  vec.push_back(notassignable(x));
  x = 42;
  EXPECT_EQ(42, vec.pop_back_val().x);
}

}
