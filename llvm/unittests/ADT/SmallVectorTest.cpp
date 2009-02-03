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
#include <stdarg.h>

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

  friend bool operator!=(const Constructable & c0, const Constructable & c1) {
    return c0.getValue() != c1.getValue();
  }
};

int Constructable::numConstructorCalls;
int Constructable::numDestructorCalls;
int Constructable::numAssignmentCalls;

// Test fixture class
class SmallVectorTest : public testing::Test {
protected:
  typedef SmallVector<Constructable, 4> VectorType;
  
  VectorType theVector;
  VectorType otherVector;
  
  void SetUp() {
    Constructable::reset();
  }

  void assertEmpty(VectorType & v) {
    // Size tests
    EXPECT_EQ(0u, v.size());
    EXPECT_TRUE(v.empty());

    // Iterator tests
    EXPECT_TRUE(v.begin() == v.end());
  }

  // Assert that theVector contains the specified values, in order.
  void assertValuesInOrder(VectorType & v, size_t size, ...) {
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
  void makeSequence(VectorType & v, int start, int end) {
    for (int i = start; i <= end; ++i) {
      v.push_back(Constructable(i));
    }
  }
};

// New vector test.
TEST_F(SmallVectorTest, EmptyVectorTest) {
  SCOPED_TRACE("EmptyVectorTest");
  assertEmpty(theVector);
  EXPECT_TRUE(theVector.rbegin() == theVector.rend());
  EXPECT_EQ(0, Constructable::getNumConstructorCalls());
  EXPECT_EQ(0, Constructable::getNumDestructorCalls());
}

// Simple insertions and deletions.
TEST_F(SmallVectorTest, PushPopTest) {
  SCOPED_TRACE("PushPopTest");

  // Push an element
  theVector.push_back(Constructable(1));

  // Size tests
  assertValuesInOrder(theVector, 1u, 1);
  EXPECT_FALSE(theVector.begin() == theVector.end());
  EXPECT_FALSE(theVector.empty());

  // Push another element
  theVector.push_back(Constructable(2));
  assertValuesInOrder(theVector, 2u, 1, 2);

  // Pop one element
  theVector.pop_back();
  assertValuesInOrder(theVector, 1u, 1);

  // Pop another element
  theVector.pop_back();
  assertEmpty(theVector);
  
  // Check number of constructor calls. Should be 2 for each list element,
  // one for the argument to push_back, and one for the list element itself.
  EXPECT_EQ(4, Constructable::getNumConstructorCalls());
  EXPECT_EQ(4, Constructable::getNumDestructorCalls());
}

// Clear test.
TEST_F(SmallVectorTest, ClearTest) {
  SCOPED_TRACE("ClearTest");

  makeSequence(theVector, 1, 2);
  theVector.clear();

  assertEmpty(theVector);
  EXPECT_EQ(4, Constructable::getNumConstructorCalls());
  EXPECT_EQ(4, Constructable::getNumDestructorCalls());
}

// Resize smaller test.
TEST_F(SmallVectorTest, ResizeShrinkTest) {
  SCOPED_TRACE("ResizeShrinkTest");

  makeSequence(theVector, 1, 3);
  theVector.resize(1);

  assertValuesInOrder(theVector, 1u, 1);
  EXPECT_EQ(6, Constructable::getNumConstructorCalls());
  EXPECT_EQ(5, Constructable::getNumDestructorCalls());
}

// Resize bigger test.
TEST_F(SmallVectorTest, ResizeGrowTest) {
  SCOPED_TRACE("ResizeGrowTest");

  theVector.resize(2);
  
  // XXX: I don't know where the extra construct/destruct is coming from.
  EXPECT_EQ(3, Constructable::getNumConstructorCalls());
  EXPECT_EQ(1, Constructable::getNumDestructorCalls());
  EXPECT_EQ(2u, theVector.size());
}

// Resize with fill value.
TEST_F(SmallVectorTest, ResizeFillTest) {
  SCOPED_TRACE("ResizeFillTest");

  theVector.resize(3, Constructable(77));
  assertValuesInOrder(theVector, 3u, 77, 77, 77);
}

// Overflow past fixed size.
TEST_F(SmallVectorTest, OverflowTest) {
  SCOPED_TRACE("OverflowTest");

  // Push more elements than the fixed size
  makeSequence(theVector, 1, 10);

  // test size and values
  EXPECT_EQ(10u, theVector.size());
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(i+1, theVector[i].getValue());
  }
  
  // Now resize back to fixed size
  theVector.resize(1);
  
  assertValuesInOrder(theVector, 1u, 1);
}

// Iteration tests.
TEST_F(SmallVectorTest, IterationTest) {
  makeSequence(theVector, 1, 2);

  // Forward Iteration
  VectorType::iterator it = theVector.begin();
  EXPECT_TRUE(*it == theVector.front());
  EXPECT_TRUE(*it == theVector[0]);
  EXPECT_EQ(1, it->getValue());
  ++it;
  EXPECT_TRUE(*it == theVector[1]);
  EXPECT_TRUE(*it == theVector.back());
  EXPECT_EQ(2, it->getValue());
  ++it;
  EXPECT_TRUE(it == theVector.end());
  --it;
  EXPECT_TRUE(*it == theVector[1]);
  EXPECT_EQ(2, it->getValue());
  --it;
  EXPECT_TRUE(*it == theVector[0]);
  EXPECT_EQ(1, it->getValue());

  // Reverse Iteration
  VectorType::reverse_iterator rit = theVector.rbegin();
  EXPECT_TRUE(*rit == theVector[1]);
  EXPECT_EQ(2, rit->getValue());
  ++rit;
  EXPECT_TRUE(*rit == theVector[0]);
  EXPECT_EQ(1, rit->getValue());
  ++rit;
  EXPECT_TRUE(rit == theVector.rend());
  --rit;
  EXPECT_TRUE(*rit == theVector[0]);
  EXPECT_EQ(1, rit->getValue());
  --rit;
  EXPECT_TRUE(*rit == theVector[1]);
  EXPECT_EQ(2, rit->getValue());
}

// Swap test.
TEST_F(SmallVectorTest, SwapTest) {
  SCOPED_TRACE("SwapTest");

  makeSequence(theVector, 1, 2);
  std::swap(theVector, otherVector);

  assertEmpty(theVector);
  assertValuesInOrder(otherVector, 2u, 1, 2);
}

// Append test
TEST_F(SmallVectorTest, AppendTest) {
  SCOPED_TRACE("AppendTest");

  makeSequence(otherVector, 2, 3);

  theVector.push_back(Constructable(1));
  theVector.append(otherVector.begin(), otherVector.end());

  assertValuesInOrder(theVector, 3u, 1, 2, 3);
}

// Append repeated test
TEST_F(SmallVectorTest, AppendRepeatedTest) {
  SCOPED_TRACE("AppendRepeatedTest");

  theVector.push_back(Constructable(1));
  theVector.append(2, Constructable(77));
  assertValuesInOrder(theVector, 3u, 1, 77, 77);
}

// Assign test
TEST_F(SmallVectorTest, AssignTest) {
  SCOPED_TRACE("AssignTest");

  theVector.push_back(Constructable(1));
  theVector.assign(2, Constructable(77));
  assertValuesInOrder(theVector, 2u, 77, 77);
}

// Erase a single element
TEST_F(SmallVectorTest, EraseTest) {
  SCOPED_TRACE("EraseTest");

  makeSequence(theVector, 1, 3);
  theVector.erase(theVector.begin());
  assertValuesInOrder(theVector, 2u, 2, 3);
}

// Erase a range of elements
TEST_F(SmallVectorTest, EraseRangeTest) {
  SCOPED_TRACE("EraseRangeTest");

  makeSequence(theVector, 1, 3);
  theVector.erase(theVector.begin(), theVector.begin() + 2);
  assertValuesInOrder(theVector, 1u, 3);
}

// Insert a single element.
TEST_F(SmallVectorTest, InsertTest) {
  SCOPED_TRACE("InsertTest");

  makeSequence(theVector, 1, 3);
  theVector.insert(theVector.begin() + 1, Constructable(77));
  assertValuesInOrder(theVector, 4u, 1, 77, 2, 3);
}

// Insert repeated elements.
TEST_F(SmallVectorTest, InsertRepeatedTest) {
  SCOPED_TRACE("InsertRepeatedTest");

  makeSequence(theVector, 1, 3);
  theVector.insert(theVector.begin() + 1, 3, Constructable(77));
  assertValuesInOrder(theVector, 6u, 1, 77, 77, 77, 2, 3);
}

// Insert range.
TEST_F(SmallVectorTest, InsertRangeTest) {
  SCOPED_TRACE("InsertRepeatedTest");

  makeSequence(theVector, 1, 3);
  theVector.insert(theVector.begin() + 1, 3, Constructable(77));
  assertValuesInOrder(theVector, 6u, 1, 77, 77, 77, 2, 3);
}

// Comparison tests.
TEST_F(SmallVectorTest, ComparisonTest) {
  SCOPED_TRACE("ComparisonTest");

  makeSequence(theVector, 1, 3);
  makeSequence(otherVector, 1, 3);
  
  EXPECT_TRUE(theVector == otherVector);
  EXPECT_FALSE(theVector != otherVector);

  otherVector.clear();
  makeSequence(otherVector, 2, 4);
  
  EXPECT_FALSE(theVector == otherVector);
  EXPECT_TRUE(theVector != otherVector);
}

// Constant vector tests.
TEST_F(SmallVectorTest, ConstVectorTest) {
  const VectorType constVector;

  EXPECT_EQ(0u, constVector.size());
  EXPECT_TRUE(constVector.empty());
  EXPECT_TRUE(constVector.begin() == constVector.end());
}

}
