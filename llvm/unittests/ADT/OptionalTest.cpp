//===- llvm/unittest/ADT/OptionalTest.cpp - Optional unit tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"

#include <array>


using namespace llvm;

static_assert(std::is_trivially_copyable<Optional<int>>::value,
              "trivially copyable");

static_assert(std::is_trivially_copyable<Optional<std::array<int, 3>>>::value,
              "trivially copyable");

void OptionalWorksInConstexpr() {
  constexpr auto x1 = Optional<int>();
  constexpr Optional<int> x2{};
  static_assert(!x1.has_value() && !x2.has_value(),
                "Default construction and hasValue() are contexpr");
  static_assert(!x1.hasValue() && !x2.hasValue(),
                "Default construction and hasValue() are contexpr");
  constexpr auto y1 = Optional<int>(3);
  constexpr Optional<int> y2{3};
  static_assert(y1.value() == y2.value() && y1.value() == 3,
                "Construction with value and getValue() are constexpr");
  static_assert(y1.getValue() == y2.getValue() && y1.getValue() == 3,
                "Construction with value and getValue() are constexpr");
  static_assert(Optional<int>{3} >= 2 && Optional<int>{1} < Optional<int>{2},
                "Comparisons work in constexpr");
}

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
    !std::is_trivially_copyable<Optional<NonDefaultConstructible>>::value,
    "not trivially copyable");

TEST(OptionalTest, NonDefaultConstructibleTest) {
  Optional<NonDefaultConstructible> O;
  EXPECT_FALSE(O);
}

TEST(OptionalTest, ResetTest) {
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

TEST(OptionalTest, InitializationLeakTest) {
  NonDefaultConstructible::ResetCounts();
  Optional<NonDefaultConstructible>(NonDefaultConstructible(3));
  EXPECT_EQ(1u, NonDefaultConstructible::CopyConstructions);
  EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
  EXPECT_EQ(2u, NonDefaultConstructible::Destructions);
}

TEST(OptionalTest, CopyConstructionTest) {
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

TEST(OptionalTest, ConstructingCopyAssignmentTest) {
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

TEST(OptionalTest, CopyingCopyAssignmentTest) {
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

TEST(OptionalTest, DeletingCopyAssignmentTest) {
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

TEST(OptionalTest, NullCopyConstructionTest) {
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

TEST(OptionalTest, InPlaceConstructionNonDefaultConstructibleTest) {
  NonDefaultConstructible::ResetCounts();
  { Optional<NonDefaultConstructible> A{in_place, 1}; }
  EXPECT_EQ(0u, NonDefaultConstructible::CopyConstructions);
  EXPECT_EQ(0u, NonDefaultConstructible::CopyAssignments);
  EXPECT_EQ(1u, NonDefaultConstructible::Destructions);
}

TEST(OptionalTest, GetValueOr) {
  Optional<int> A;
  EXPECT_EQ(42, A.value_or(42));
  EXPECT_EQ(42, A.getValueOr(42));

  A = 5;
  EXPECT_EQ(5, A.value_or(42));
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

  friend bool operator==(const MultiArgConstructor &LHS,
                         const MultiArgConstructor &RHS) {
    return LHS.x == RHS.x && LHS.y == RHS.y;
  }

  static unsigned Destructions;
  ~MultiArgConstructor() {
    ++Destructions;
  }
  static void ResetCounts() {
    Destructions = 0;
  }
};
unsigned MultiArgConstructor::Destructions = 0;

static_assert(!std::is_trivially_copyable<Optional<MultiArgConstructor>>::value,
              "not trivially copyable");

TEST(OptionalTest, Emplace) {
  MultiArgConstructor::ResetCounts();
  Optional<MultiArgConstructor> A;
  
  A.emplace(1, 2);
  EXPECT_TRUE(A.has_value());
  EXPECT_TRUE(A.hasValue());
  EXPECT_EQ(1, A->x);
  EXPECT_EQ(2, A->y);
  EXPECT_EQ(0u, MultiArgConstructor::Destructions);

  A.emplace(5, false);
  EXPECT_TRUE(A.has_value());
  EXPECT_TRUE(A.hasValue());
  EXPECT_EQ(5, A->x);
  EXPECT_EQ(-5, A->y);
  EXPECT_EQ(1u, MultiArgConstructor::Destructions);
}

TEST(OptionalTest, InPlaceConstructionMultiArgConstructorTest) {
  MultiArgConstructor::ResetCounts();
  {
    Optional<MultiArgConstructor> A{in_place, 1, 2};
    EXPECT_TRUE(A.has_value());
    EXPECT_TRUE(A.hasValue());
    EXPECT_EQ(1, A->x);
    EXPECT_EQ(2, A->y);
    Optional<MultiArgConstructor> B{in_place, 5, false};
    EXPECT_TRUE(B.has_value());
    EXPECT_TRUE(B.hasValue());
    EXPECT_EQ(5, B->x);
    EXPECT_EQ(-5, B->y);
    EXPECT_EQ(0u, MultiArgConstructor::Destructions);
  }
  EXPECT_EQ(2u, MultiArgConstructor::Destructions);
}

TEST(OptionalTest, InPlaceConstructionAndEmplaceEquivalentTest) {
  MultiArgConstructor::ResetCounts();
  {
    Optional<MultiArgConstructor> A{in_place, 1, 2};
    Optional<MultiArgConstructor> B;
    B.emplace(1, 2);
    EXPECT_EQ(0u, MultiArgConstructor::Destructions);
    ASSERT_EQ(A, B);
  }
  EXPECT_EQ(2u, MultiArgConstructor::Destructions);
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

static_assert(!std::is_trivially_copyable<Optional<MoveOnly>>::value,
              "not trivially copyable");

TEST(OptionalTest, MoveOnlyNull) {
  MoveOnly::ResetCounts();
  Optional<MoveOnly> O;
  EXPECT_EQ(0u, MoveOnly::MoveConstructions);
  EXPECT_EQ(0u, MoveOnly::MoveAssignments);
  EXPECT_EQ(0u, MoveOnly::Destructions);
}

TEST(OptionalTest, MoveOnlyConstruction) {
  MoveOnly::ResetCounts();
  Optional<MoveOnly> O(MoveOnly(3));
  EXPECT_TRUE((bool)O);
  EXPECT_EQ(3, O->val);
  EXPECT_EQ(1u, MoveOnly::MoveConstructions);
  EXPECT_EQ(0u, MoveOnly::MoveAssignments);
  EXPECT_EQ(1u, MoveOnly::Destructions);
}

TEST(OptionalTest, MoveOnlyMoveConstruction) {
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

TEST(OptionalTest, MoveOnlyAssignment) {
  MoveOnly::ResetCounts();
  Optional<MoveOnly> O;
  O = MoveOnly(3);
  EXPECT_TRUE((bool)O);
  EXPECT_EQ(3, O->val);
  EXPECT_EQ(1u, MoveOnly::MoveConstructions);
  EXPECT_EQ(0u, MoveOnly::MoveAssignments);
  EXPECT_EQ(1u, MoveOnly::Destructions);
}

TEST(OptionalTest, MoveOnlyInitializingAssignment) {
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

TEST(OptionalTest, MoveOnlyNullingAssignment) {
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

TEST(OptionalTest, MoveOnlyAssigningAssignment) {
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

static_assert(!std::is_trivially_copyable<Optional<Immovable>>::value,
              "not trivially copyable");

TEST(OptionalTest, ImmovableEmplace) {
  Optional<Immovable> A;
  Immovable::ResetCounts();
  A.emplace(4);
  EXPECT_TRUE((bool)A);
  EXPECT_EQ(4, A->val);
  EXPECT_EQ(1u, Immovable::Constructions);
  EXPECT_EQ(0u, Immovable::Destructions);
}

TEST(OptionalTest, ImmovableInPlaceConstruction) {
  Immovable::ResetCounts();
  Optional<Immovable> A{in_place, 4};
  EXPECT_TRUE((bool)A);
  EXPECT_EQ(4, A->val);
  EXPECT_EQ(1u, Immovable::Constructions);
  EXPECT_EQ(0u, Immovable::Destructions);
}

// Craft a class which is_trivially_copyable, but not
// is_trivially_copy_constructible.
struct NonTCopy {
  NonTCopy() = default;

  // Delete the volatile copy constructor to engage the "rule of 3" and delete
  // any unspecified copy assignment or constructor.
  NonTCopy(volatile NonTCopy const &) = delete;

  // Leave the non-volatile default copy constructor unspecified (deleted by
  // rule of 3)

  // This template can serve as the copy constructor, but isn't chosen
  // by =default in a class with a 'NonTCopy' member.
  template <typename Self = NonTCopy>
  NonTCopy(Self const &Other) : Val(Other.Val) {}

  NonTCopy &operator=(NonTCopy const &) = default;

  int Val{0};
};

#if defined(_MSC_VER) && _MSC_VER >= 1927 && !defined(__clang__)
// Currently only true on recent MSVC releases.
static_assert(std::is_trivially_copyable<NonTCopy>::value,
              "Expect NonTCopy to be trivially copyable");

static_assert(!std::is_trivially_copy_constructible<NonTCopy>::value,
              "Expect NonTCopy not to be trivially copy constructible.");
#endif // defined(_MSC_VER) && _MSC_VER >= 1927

TEST(OptionalTest, DeletedCopyConstructor) {

  // Expect compile to fail if 'trivial' version of
  // optional_detail::OptionalStorage is chosen.
  using NonTCopyOptT = Optional<NonTCopy>;
  NonTCopyOptT NonTCopy1;

  // Check that the Optional can be copy constructed.
  NonTCopyOptT NonTCopy2{NonTCopy1};

  // Check that the Optional can be copy assigned.
  NonTCopy1 = NonTCopy2;
}

// Craft a class which is_trivially_copyable, but not
// is_trivially_copy_assignable.
class NonTAssign {
public:
  NonTAssign() = default;
  NonTAssign(NonTAssign const &) = default;

  // Delete the volatile copy assignment to engage the "rule of 3" and delete
  // any unspecified copy assignment or constructor.
  NonTAssign &operator=(volatile NonTAssign const &) = delete;

  // Leave the non-volatile default copy assignment unspecified (deleted by rule
  // of 3).

  // This template can serve as the copy assignment, but isn't chosen
  // by =default in a class with a 'NonTAssign' member.
  template <typename Self = NonTAssign>
  NonTAssign &operator=(Self const &Other) {
    A = Other.A;
    return *this;
  }

  int A{0};
};

#if defined(_MSC_VER) && _MSC_VER >= 1927 && !defined(__clang__)
// Currently only true on recent MSVC releases.
static_assert(std::is_trivially_copyable<NonTAssign>::value,
              "Expect NonTAssign to be trivially copyable");

static_assert(!std::is_trivially_copy_assignable<NonTAssign>::value,
              "Expect NonTAssign not to be trivially assignable.");
#endif // defined(_MSC_VER) && _MSC_VER >= 1927

TEST(OptionalTest, DeletedCopyAssignment) {

  // Expect compile to fail if 'trivial' version of
  // optional_detail::OptionalStorage is chosen.
  using NonTAssignOptT = Optional<NonTAssign>;
  NonTAssignOptT NonTAssign1;

  // Check that the Optional can be copy constructed.
  NonTAssignOptT NonTAssign2{NonTAssign1};

  // Check that the Optional can be copy assigned.
  NonTAssign1 = NonTAssign2;
}

struct NoTMove {
  NoTMove() = default;
  NoTMove(NoTMove const &) = default;
  NoTMove &operator=(NoTMove const &) = default;

  // Delete move constructor / assignment.  Compiler should fall-back to the
  // trivial copy constructor / assignment in the trivial OptionalStorage
  // specialization.
  NoTMove(NoTMove &&) = delete;
  NoTMove &operator=(NoTMove &&) = delete;

  int Val{0};
};

TEST(OptionalTest, DeletedMoveConstructor) {
  using NoTMoveOptT = Optional<NoTMove>;

  NoTMoveOptT NonTMove1;
  NoTMoveOptT NonTMove2{std::move(NonTMove1)};

  NonTMove1 = std::move(NonTMove2);

  static_assert(
      std::is_trivially_copyable<NoTMoveOptT>::value,
      "Expect Optional<NoTMove> to still use the trivial specialization "
      "of OptionalStorage despite the deleted move constructor / assignment.");
}

class NoCopyStringMap {
public:
  NoCopyStringMap() = default;

private:
  llvm::StringMap<std::unique_ptr<int>> Map;
};

TEST(OptionalTest, DeletedCopyStringMap) {
  // Old versions of gcc (7.3 and prior) instantiate the copy constructor when
  // std::is_trivially_copyable is instantiated.  This test will fail
  // compilation if std::is_trivially_copyable is used in the OptionalStorage
  // specialization condition by gcc <= 7.3.
  Optional<NoCopyStringMap> TestInstantiation;
}

TEST(OptionalTest, MoveValueOr) {
  Optional<MoveOnly> A;

  MoveOnly::ResetCounts();
  EXPECT_EQ(42, std::move(A).value_or(MoveOnly(42)).val);
  EXPECT_EQ(1u, MoveOnly::MoveConstructions);
  EXPECT_EQ(0u, MoveOnly::MoveAssignments);
  EXPECT_EQ(2u, MoveOnly::Destructions);

  A = MoveOnly(5);
  MoveOnly::ResetCounts();
  EXPECT_EQ(5, std::move(A).value_or(MoveOnly(42)).val);
  EXPECT_EQ(1u, MoveOnly::MoveConstructions);
  EXPECT_EQ(0u, MoveOnly::MoveAssignments);
  EXPECT_EQ(2u, MoveOnly::Destructions);
}

TEST(OptionalTest, MoveGetValueOr) {
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

TEST(OptionalTest, OperatorEqual) {
  CheckRelation<EqualTo>(NoneEq, NoneEq, true);
  CheckRelation<EqualTo>(NoneEq, EqualityRhs, false);
  CheckRelation<EqualTo>(EqualityLhs, NoneEq, false);

  IsEqual = false;
  CheckRelation<EqualTo>(EqualityLhs, EqualityRhs, IsEqual);
  IsEqual = true;
  CheckRelation<EqualTo>(EqualityLhs, EqualityRhs, IsEqual);
}

TEST(OptionalTest, OperatorNotEqual) {
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

TEST(OptionalTest, OperatorLess) {
  CheckRelation<Less>(NoneIneq, NoneIneq, false);
  CheckRelation<Less>(NoneIneq, InequalityRhs, true);
  CheckRelation<Less>(InequalityLhs, NoneIneq, false);

  IsLess = false;
  CheckRelation<Less>(InequalityLhs, InequalityRhs, IsLess);
  IsLess = true;
  CheckRelation<Less>(InequalityLhs, InequalityRhs, IsLess);
}

TEST(OptionalTest, OperatorGreater) {
  CheckRelation<Greater>(NoneIneq, NoneIneq, false);
  CheckRelation<Greater>(NoneIneq, InequalityRhs, false);
  CheckRelation<Greater>(InequalityLhs, NoneIneq, true);

  IsLess = false;
  CheckRelation<Greater>(InequalityRhs, InequalityLhs, IsLess);
  IsLess = true;
  CheckRelation<Greater>(InequalityRhs, InequalityLhs, IsLess);
}

TEST(OptionalTest, OperatorLessEqual) {
  CheckRelation<LessEqual>(NoneIneq, NoneIneq, true);
  CheckRelation<LessEqual>(NoneIneq, InequalityRhs, true);
  CheckRelation<LessEqual>(InequalityLhs, NoneIneq, false);

  IsLess = false;
  CheckRelation<LessEqual>(InequalityRhs, InequalityLhs, !IsLess);
  IsLess = true;
  CheckRelation<LessEqual>(InequalityRhs, InequalityLhs, !IsLess);
}

TEST(OptionalTest, OperatorGreaterEqual) {
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

TEST(OptionalTest, StreamOperator) {
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

TEST(OptionalTest, UseInUnitTests) {
  // Test that we invoke the streaming operators when pretty-printing values in
  // EXPECT macros.
  EXPECT_NONFATAL_FAILURE(EXPECT_EQ(llvm::None, ComparableAndStreamable::get()),
                          "Expected equality of these values:\n"
                          "  llvm::None\n"
                          "    Which is: None\n"
                          "  ComparableAndStreamable::get()\n"
                          "    Which is: ComparableAndStreamable");

  // Test that it is still possible to compare objects which do not have a
  // custom streaming operator.
  EXPECT_NONFATAL_FAILURE(EXPECT_EQ(llvm::None, Comparable::get()), "object");
}

TEST(OptionalTest, HashValue) {
  // Check that None, false, and true all hash differently.
  Optional<bool> B, B0 = false, B1 = true;
  EXPECT_NE(hash_value(B0), hash_value(B));
  EXPECT_NE(hash_value(B1), hash_value(B));
  EXPECT_NE(hash_value(B1), hash_value(B0));

  // Check that None, 0, and 1 all hash differently.
  Optional<int> I, I0 = 0, I1 = 1;
  EXPECT_NE(hash_value(I0), hash_value(I));
  EXPECT_NE(hash_value(I1), hash_value(I));
  EXPECT_NE(hash_value(I1), hash_value(I0));

  // Check None hash the same way regardless of type.
  EXPECT_EQ(hash_value(B), hash_value(I));
}

struct NotTriviallyCopyable {
  NotTriviallyCopyable(); // Constructor out-of-line.
  virtual ~NotTriviallyCopyable() = default;
  Optional<MoveOnly> MO;
};

TEST(OptionalTest, GCCIsTriviallyMoveConstructibleCompat) {
  Optional<NotTriviallyCopyable> V;
  EXPECT_FALSE(V);
}

} // end anonymous namespace
