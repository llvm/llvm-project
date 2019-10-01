//===- llvm/unittest/ADT/OptionalTest.cpp - Optional unit tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"

#include <array>


using namespace llvm;

static_assert(is_trivially_copyable<Optional<int>>::value,
          "trivially copyable");

static_assert(is_trivially_copyable<Optional<std::array<int, 3>>>::value,
              "trivially copyable");

namespace {

struct NonDefaultConstructible {
  static unsigned CopyConstructions;
  static unsigned Destructions;
  static unsigned CopyAssignments;
  explicit NonDefaultConstructible(int) {
  }
  NonDefaultConstructible(const NonDefaultConstructible&) {
    ++CopyConstructions;
  }
  NonDefaultConstructible &operator=(const NonDefaultConstructible&) {
    ++CopyAssignments;
    return *this;
  }
  ~NonDefaultConstructible() {
    ++Destructions;
  }
  static void ResetCounts() {
    CopyConstructions = 0;
    Destructions = 0;
    CopyAssignments = 0;
  }
};

unsigned NonDefaultConstructible::CopyConstructions = 0;
unsigned NonDefaultConstructible::Destructions = 0;
unsigned NonDefaultConstructible::CopyAssignments = 0;

static_assert(
      !is_trivially_copyable<Optional<NonDefaultConstructible>>::value,
      "not trivially copyable");

// Test fixture
class OptionalTest : public testing::Test {
};

TEST_F(OptionalTest, NonDefaultConstructibleTest) {
  Optional<NonDefaultConstructible> O;
  EXPECT_FALSE(O);
}

TEST_F(OptionalTest, ResetTest) {
  NonDefaultConstructible::ResetCounts();
  Optional<NonDefaultConstructible> O(NonDefaultConstructible(3));
  EXPECT_EQ(1u, NonDefaultConstructible::CopyConstructions);
  EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
  EXPECT_EQ(1u, NonDefaultConstructible::Destructions);
  NonDefaultConstructible::ResetCounts();
  O.reset();
  EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
  EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
  EXPECT_EQ(1u, NonDefaultConstructible::Destructions);
}

TEST_F(OptionalTest, InitializationLeakTest) {
  NonDefaultConstructible::ResetCounts();
  Optional<NonDefaultConstructible>(NonDefaultConstructible(3));
  EXPECT_EQ(1u, NonDefaultConstructible::CopyConstructions);
  EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
  EXPECT_EQ(2u, NonDefaultConstructible::Destructions);
}

TEST_F(OptionalTest, CopyConstructionTest) {
  NonDefaultConstructible::ResetCounts();
  {
    Optional<NonDefaultConstructible> A(NonDefaultConstructible(3));
    EXPECT_EQ(1u, NonDefaultConstructible::CopyConstructions);
    EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
    EXPECT_EQ(1u, NonDefaultConstructible::Destructions);
    NonDefaultConstructible::ResetCounts();
    Optional<NonDefaultConstructible> B(A);
    EXPECT_EQ(1u, NonDefaultConstructible::CopyConstructions);
    EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
    EXPECT_EQ(0u, NonDefaultConstructible::Destructions);
    NonDefaultConstructible::ResetCounts();
  }
  EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
  EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
  EXPECT_EQ(2u, NonDefaultConstructible::Destructions);
}

TEST_F(OptionalTest, ConstructingCopyAssignmentTest) {
  NonDefaultConstructible::ResetCounts();
  {
    Optional<NonDefaultConstructible> A(NonDefaultConstructible(3));
    Optional<NonDefaultConstructible> B;
    EXPECT_EQ(1u, NonDefaultConstructible::CopyConstructions);
    EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
    EXPECT_EQ(1u, NonDefaultConstructible::Destructions);
    NonDefaultConstructible::ResetCounts();
    B = A;
    EXPECT_EQ(1u, NonDefaultConstructible::CopyConstructions);
    EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
    EXPECT_EQ(0u, NonDefaultConstructible::Destructions);
    NonDefaultConstructible::ResetCounts();
  }
  EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
  EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
  EXPECT_EQ(2u, NonDefaultConstructible::Destructions);
}

TEST_F(OptionalTest, CopyingCopyAssignmentTest) {
  NonDefaultConstructible::ResetCounts();
  {
    Optional<NonDefaultConstructible> A(NonDefaultConstructible(3));
    Optional<NonDefaultConstructible> B(NonDefaultConstructible(4));
    EXPECT_EQ(2u, NonDefaultConstructible::CopyConstructions);
    EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
    EXPECT_EQ(2u, NonDefaultConstructible::Destructions);
    NonDefaultConstructible::ResetCounts();
    B = A;
    EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
    EXPECT_EQ(1u, NonDefaultConstructible::CopyAssignments);
    EXPECT_EQ(0u, NonDefaultConstructible::Destructions);
    NonDefaultConstructible::ResetCounts();
  }
  EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
  EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
  EXPECT_EQ(2u, NonDefaultConstructible::Destructions);
}

TEST_F(OptionalTest, DeletingCopyAssignmentTest) {
  NonDefaultConstructible::ResetCounts();
  {
    Optional<NonDefaultConstructible> A;
    Optional<NonDefaultConstructible> B(NonDefaultConstructible(3));
    EXPECT_EQ(1u, NonDefaultConstructible::CopyConstructions);
    EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
    EXPECT_EQ(1u, NonDefaultConstructible::Destructions);
    NonDefaultConstructible::ResetCounts();
    B = A;
    EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
    EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
    EXPECT_EQ(1u, NonDefaultConstructible::Destructions);
    NonDefaultConstructible::ResetCounts();
  }
  EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
  EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
  EXPECT_EQ(0u, NonDefaultConstructible::Destructions);
}

TEST_F(OptionalTest, NullCopyConstructionTest) {
  NonDefaultConstructible::ResetCounts();
  {
    Optional<NonDefaultConstructible> A;
    Optional<NonDefaultConstructible> B;
    EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
    EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
    EXPECT_EQ(0u, NonDefaultConstructible::Destructions);
    NonDefaultConstructible::ResetCounts();
    B = A;
    EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
    EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
    EXPECT_EQ(0u, NonDefaultConstructible::Destructions);
    NonDefaultConstructible::ResetCounts();
  }
  EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
  EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
  EXPECT_EQ(0u, NonDefaultConstructible::Destructions);
}

TEST_F(OptionalTest, GetValueOr) {
  Optional<int> A;
  EXPECT_EQ(42, A.getValueOr(42));

  A = 5;
  EXPECT_EQ(5, A.getValueOr(42));
}

struct MultiArgConstructor {
  int x, y;
  MultiArgConstructor(int x, int y) : x(x), y(y) {}
  explicit MultiArgConstructor(int x, bool positive)
    : x(x), y(positive ? x : -x) {}

  MultiArgConstructor(const MultiArgConstructor &) = delete;
  MultiArgConstructor(MultiArgConstructor &&) = delete;
  MultiArgConstructor &operator=(const MultiArgConstructor &) = delete;
  MultiArgConstructor &operator=(MultiArgConstructor &&) = delete;

  static unsigned Destructions;
  ~MultiArgConstructor() {
    ++Destructions;
  }
  static void ResetCounts() {
    Destructions = 0;
  }
};
unsigned MultiArgConstructor::Destructions = 0;

static_assert(
  !is_trivially_copyable<Optional<MultiArgConstructor>>::value,
  "not trivially copyable");

TEST_F(OptionalTest, Emplace) {
  MultiArgConstructor::ResetCounts();
  Optional<MultiArgConstructor> A;
  
  A.emplace(1, 2);
  EXPECT_TRUE(A.hasValue());
  EXPECT_EQ(1, A->x);
  EXPECT_EQ(2, A->y);
  EXPECT_EQ(0u, MultiArgConstructor::Destructions);

  A.emplace(5, false);
  EXPECT_TRUE(A.hasValue());
  EXPECT_EQ(5, A->x);
  EXPECT_EQ(-5, A->y);
  EXPECT_EQ(1u, MultiArgConstructor::Destructions);
}

struct MoveOnly {
  static unsigned MoveConstructions;
  static unsigned Destructions;
  static unsigned MoveAssignments;
  int val;
  explicit MoveOnly(int val) : val(val) {
  }
  MoveOnly(MoveOnly&& other) {
    val = other.val;
    ++MoveConstructions;
  }
  MoveOnly &operator=(MoveOnly&& other) {
    val = other.val;
    ++MoveAssignments;
    return *this;
  }
  ~MoveOnly() {
    ++Destructions;
  }
  static void ResetCounts() {
    MoveConstructions = 0;
    Destructions = 0;
    MoveAssignments = 0;
  }
};

unsigned MoveOnly::MoveConstructions = 0;
unsigned MoveOnly::Destructions = 0;
unsigned MoveOnly::MoveAssignments = 0;

static_assert(!is_trivially_copyable<Optional<MoveOnly>>::value,
              "not trivially copyable");

TEST_F(OptionalTest, MoveOnlyNull) {
  MoveOnly::ResetCounts();
  Optional<MoveOnly> O;
  EXPECT_EQ(0u, MoveOnly::MoveConstructions);
  EXPECT_EQ(0u, MoveOnly::MoveAssignments);
  EXPECT_EQ(0u, MoveOnly::Destructions);
}

TEST_F(OptionalTest, MoveOnlyConstruction) {
  MoveOnly::ResetCounts();
  Optional<MoveOnly> O(MoveOnly(3));
  EXPECT_TRUE((bool)O);
  EXPECT_EQ(3, O->val);
  EXPECT_EQ(1u, MoveOnly::MoveConstructions);
  EXPECT_EQ(0u, MoveOnly::MoveAssignments);
  EXPECT_EQ(1u, MoveOnly::Destructions);
}

TEST_F(OptionalTest, MoveOnlyMoveConstruction) {
  Optional<MoveOnly> A(MoveOnly(3));
  MoveOnly::ResetCounts();
  Optional<MoveOnly> B(std::move(A));
  EXPECT_TRUE((bool)A);
  EXPECT_TRUE((bool)B);
  EXPECT_EQ(3, B->val);
  EXPECT_EQ(1u, MoveOnly::MoveConstructions);
  EXPECT_EQ(0u, MoveOnly::MoveAssignments);
  EXPECT_EQ(0u, MoveOnly::Destructions);
}

TEST_F(OptionalTest, MoveOnlyAssignment) {
  MoveOnly::ResetCounts();
  Optional<MoveOnly> O;
  O = MoveOnly(3);
  EXPECT_TRUE((bool)O);
  EXPECT_EQ(3, O->val);
  EXPECT_EQ(1u, MoveOnly::MoveConstructions);
  EXPECT_EQ(0u, MoveOnly::MoveAssignments);
  EXPECT_EQ(1u, MoveOnly::Destructions);
}

TEST_F(OptionalTest, MoveOnlyInitializingAssignment) {
  Optional<MoveOnly> A(MoveOnly(3));
  Optional<MoveOnly> B;
  MoveOnly::ResetCounts();
  B = std::move(A);
  EXPECT_TRUE((bool)A);
  EXPECT_TRUE((bool)B);
  EXPECT_EQ(3, B->val);
  EXPECT_EQ(1u, MoveOnly::MoveConstructions);
  EXPECT_EQ(0u, MoveOnly::MoveAssignments);
  EXPECT_EQ(0u, MoveOnly::Destructions);
}

TEST_F(OptionalTest, MoveOnlyNullingAssignment) {
  Optional<MoveOnly> A;
  Optional<MoveOnly> B(MoveOnly(3));
  MoveOnly::ResetCounts();
  B = std::move(A);
  EXPECT_FALSE((bool)A);
  EXPECT_FALSE((bool)B);
  EXPECT_EQ(0u, MoveOnly::MoveConstructions);
  EXPECT_EQ(0u, MoveOnly::MoveAssignments);
  EXPECT_EQ(1u, MoveOnly::Destructions);
}

TEST_F(OptionalTest, MoveOnlyAssigningAssignment) {
  Optional<MoveOnly> A(MoveOnly(3));
  Optional<MoveOnly> B(MoveOnly(4));
  MoveOnly::ResetCounts();
  B = std::move(A);
  EXPECT_TRUE((bool)A);
  EXPECT_TRUE((bool)B);
  EXPECT_EQ(3, B->val);
  EXPECT_EQ(0u, MoveOnly::MoveConstructions);
  EXPECT_EQ(1u, MoveOnly::MoveAssignments);
  EXPECT_EQ(0u, MoveOnly::Destructions);
}

struct Immovable {
  static unsigned Constructions;
  static unsigned Destructions;
  int val;
  explicit Immovable(int val) : val(val) {
    ++Constructions;
  }
  ~Immovable() {
    ++Destructions;
  }
  static void ResetCounts() {
    Constructions = 0;
    Destructions = 0;
  }
private:
  // This should disable all move/copy operations.
  Immovable(Immovable&& other) = delete;
};

unsigned Immovable::Constructions = 0;
unsigned Immovable::Destructions = 0;

static_assert(!is_trivially_copyable<Optional<Immovable>>::value,
              "not trivially copyable");

TEST_F(OptionalTest, ImmovableEmplace) {
  Optional<Immovable> A;
  Immovable::ResetCounts();
  A.emplace(4);
  EXPECT_TRUE((bool)A);
  EXPECT_EQ(4, A->val);
  EXPECT_EQ(1u, Immovable::Constructions);
  EXPECT_EQ(0u, Immovable::Destructions);
}

#if LLVM_HAS_RVALUE_REFERENCE_THIS

TEST_F(OptionalTest, MoveGetValueOr) {
  Optional<MoveOnly> A;

  MoveOnly::ResetCounts();
  EXPECT_EQ(42, std::move(A).getValueOr(MoveOnly(42)).val);
  EXPECT_EQ(1u, MoveOnly::MoveConstructions);
  EXPECT_EQ(0u, MoveOnly::MoveAssignments);
  EXPECT_EQ(2u, MoveOnly::Destructions);

  A = MoveOnly(5);
  MoveOnly::ResetCounts();
  EXPECT_EQ(5, std::move(A).getValueOr(MoveOnly(42)).val);
  EXPECT_EQ(1u, MoveOnly::MoveConstructions);
  EXPECT_EQ(0u, MoveOnly::MoveAssignments);
  EXPECT_EQ(2u, MoveOnly::Destructions);
}

#endif // LLVM_HAS_RVALUE_REFERENCE_THIS

struct EqualTo {
  template <typename T, typename U> static bool apply(const T &X, const U &Y) {
    return X == Y;
  }
};

struct NotEqualTo {
  template <typename T, typename U> static bool apply(const T &X, const U &Y) {
    return X != Y;
  }
};

struct Less {
  template <typename T, typename U> static bool apply(const T &X, const U &Y) {
    return X < Y;
  }
};

struct Greater {
  template <typename T, typename U> static bool apply(const T &X, const U &Y) {
    return X > Y;
  }
};

struct LessEqual {
  template <typename T, typename U> static bool apply(const T &X, const U &Y) {
    return X <= Y;
  }
};

struct GreaterEqual {
  template <typename T, typename U> static bool apply(const T &X, const U &Y) {
    return X >= Y;
  }
};

template <typename OperatorT, typename T>
void CheckRelation(const Optional<T> &Lhs, const Optional<T> &Rhs,
                   bool Expected) {
  EXPECT_EQ(Expected, OperatorT::apply(Lhs, Rhs));

  if (Lhs)
    EXPECT_EQ(Expected, OperatorT::apply(*Lhs, Rhs));
  else
    EXPECT_EQ(Expected, OperatorT::apply(None, Rhs));

  if (Rhs)
    EXPECT_EQ(Expected, OperatorT::apply(Lhs, *Rhs));
  else
    EXPECT_EQ(Expected, OperatorT::apply(Lhs, None));
}

struct EqualityMock {};
const Optional<EqualityMock> NoneEq, EqualityLhs((EqualityMock())),
    EqualityRhs((EqualityMock()));
bool IsEqual;

bool operator==(const EqualityMock &Lhs, const EqualityMock &Rhs) {
  EXPECT_EQ(&*EqualityLhs, &Lhs);
  EXPECT_EQ(&*EqualityRhs, &Rhs);
  return IsEqual;
}

TEST_F(OptionalTest, OperatorEqual) {
  CheckRelation<EqualTo>(NoneEq, NoneEq, true);
  CheckRelation<EqualTo>(NoneEq, EqualityRhs, false);
  CheckRelation<EqualTo>(EqualityLhs, NoneEq, false);

  IsEqual = false;
  CheckRelation<EqualTo>(EqualityLhs, EqualityRhs, IsEqual);
  IsEqual = true;
  CheckRelation<EqualTo>(EqualityLhs, EqualityRhs, IsEqual);
}

TEST_F(OptionalTest, OperatorNotEqual) {
  CheckRelation<NotEqualTo>(NoneEq, NoneEq, false);
  CheckRelation<NotEqualTo>(NoneEq, EqualityRhs, true);
  CheckRelation<NotEqualTo>(EqualityLhs, NoneEq, true);

  IsEqual = false;
  CheckRelation<NotEqualTo>(EqualityLhs, EqualityRhs, !IsEqual);
  IsEqual = true;
  CheckRelation<NotEqualTo>(EqualityLhs, EqualityRhs, !IsEqual);
}

struct InequalityMock {};
const Optional<InequalityMock> NoneIneq, InequalityLhs((InequalityMock())),
    InequalityRhs((InequalityMock()));
bool IsLess;

bool operator<(const InequalityMock &Lhs, const InequalityMock &Rhs) {
  EXPECT_EQ(&*InequalityLhs, &Lhs);
  EXPECT_EQ(&*InequalityRhs, &Rhs);
  return IsLess;
}

TEST_F(OptionalTest, OperatorLess) {
  CheckRelation<Less>(NoneIneq, NoneIneq, false);
  CheckRelation<Less>(NoneIneq, InequalityRhs, true);
  CheckRelation<Less>(InequalityLhs, NoneIneq, false);

  IsLess = false;
  CheckRelation<Less>(InequalityLhs, InequalityRhs, IsLess);
  IsLess = true;
  CheckRelation<Less>(InequalityLhs, InequalityRhs, IsLess);
}

TEST_F(OptionalTest, OperatorGreater) {
  CheckRelation<Greater>(NoneIneq, NoneIneq, false);
  CheckRelation<Greater>(NoneIneq, InequalityRhs, false);
  CheckRelation<Greater>(InequalityLhs, NoneIneq, true);

  IsLess = false;
  CheckRelation<Greater>(InequalityRhs, InequalityLhs, IsLess);
  IsLess = true;
  CheckRelation<Greater>(InequalityRhs, InequalityLhs, IsLess);
}

TEST_F(OptionalTest, OperatorLessEqual) {
  CheckRelation<LessEqual>(NoneIneq, NoneIneq, true);
  CheckRelation<LessEqual>(NoneIneq, InequalityRhs, true);
  CheckRelation<LessEqual>(InequalityLhs, NoneIneq, false);

  IsLess = false;
  CheckRelation<LessEqual>(InequalityRhs, InequalityLhs, !IsLess);
  IsLess = true;
  CheckRelation<LessEqual>(InequalityRhs, InequalityLhs, !IsLess);
}

TEST_F(OptionalTest, OperatorGreaterEqual) {
  CheckRelation<GreaterEqual>(NoneIneq, NoneIneq, true);
  CheckRelation<GreaterEqual>(NoneIneq, InequalityRhs, false);
  CheckRelation<GreaterEqual>(InequalityLhs, NoneIneq, true);

  IsLess = false;
  CheckRelation<GreaterEqual>(InequalityLhs, InequalityRhs, !IsLess);
  IsLess = true;
  CheckRelation<GreaterEqual>(InequalityLhs, InequalityRhs, !IsLess);
}

struct ComparableAndStreamable {
  friend bool operator==(ComparableAndStreamable,
                         ComparableAndStreamable) LLVM_ATTRIBUTE_USED {
    return true;
  }

  friend raw_ostream &operator<<(raw_ostream &OS, ComparableAndStreamable) {
    return OS << "ComparableAndStreamable";
  }

  static Optional<ComparableAndStreamable> get() {
    return ComparableAndStreamable();
  }
};

TEST_F(OptionalTest, StreamOperator) {
  auto to_string = [](Optional<ComparableAndStreamable> O) {
    SmallString<16> S;
    raw_svector_ostream OS(S);
    OS << O;
    return S;
  };
  EXPECT_EQ("ComparableAndStreamable",
            to_string(ComparableAndStreamable::get()));
  EXPECT_EQ("None", to_string(None));
}

struct Comparable {
  friend bool operator==(Comparable, Comparable) LLVM_ATTRIBUTE_USED {
    return true;
  }
  static Optional<Comparable> get() { return Comparable(); }
};

TEST_F(OptionalTest, UseInUnitTests) {
  // Test that we invoke the streaming operators when pretty-printing values in
  // EXPECT macros.
  EXPECT_NONFATAL_FAILURE(EXPECT_EQ(llvm::None, ComparableAndStreamable::get()),
                          "Expected: llvm::None\n"
                          "      Which is: None\n"
                          "To be equal to: ComparableAndStreamable::get()\n"
                          "      Which is: ComparableAndStreamable");

  // Test that it is still possible to compare objects which do not have a
  // custom streaming operator.
  EXPECT_NONFATAL_FAILURE(EXPECT_EQ(llvm::None, Comparable::get()), "object");
}

} // end anonymous namespace
