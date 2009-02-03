// Copyright 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Authors: wan@google.com (Zhanyong Wan), eefacm@gmail.com (Sean Mcafee)
//
// The Google C++ Testing Framework (Google Test)
//
// This header file declares functions and macros used internally by
// Google Test.  They are subject to change without notice.

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_INTERNAL_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_INTERNAL_H_

#include <gtest/internal/gtest-port.h>

#ifdef GTEST_OS_LINUX
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif  // GTEST_OS_LINUX

#include <ctype.h>
#include <string.h>
#include <iomanip>
#include <limits>
#include <set>

#include <gtest/internal/gtest-string.h>
#include <gtest/internal/gtest-filepath.h>
#include <gtest/internal/gtest-type-util.h>

// Due to C++ preprocessor weirdness, we need double indirection to
// concatenate two tokens when one of them is __LINE__.  Writing
//
//   foo ## __LINE__
//
// will result in the token foo__LINE__, instead of foo followed by
// the current line number.  For more details, see
// http://www.parashift.com/c++-faq-lite/misc-technical-issues.html#faq-39.6
#define GTEST_CONCAT_TOKEN_(foo, bar) GTEST_CONCAT_TOKEN_IMPL_(foo, bar)
#define GTEST_CONCAT_TOKEN_IMPL_(foo, bar) foo ## bar

// Google Test defines the testing::Message class to allow construction of
// test messages via the << operator.  The idea is that anything
// streamable to std::ostream can be streamed to a testing::Message.
// This allows a user to use his own types in Google Test assertions by
// overloading the << operator.
//
// util/gtl/stl_logging-inl.h overloads << for STL containers.  These
// overloads cannot be defined in the std namespace, as that will be
// undefined behavior.  Therefore, they are defined in the global
// namespace instead.
//
// C++'s symbol lookup rule (i.e. Koenig lookup) says that these
// overloads are visible in either the std namespace or the global
// namespace, but not other namespaces, including the testing
// namespace which Google Test's Message class is in.
//
// To allow STL containers (and other types that has a << operator
// defined in the global namespace) to be used in Google Test assertions,
// testing::Message must access the custom << operator from the global
// namespace.  Hence this helper function.
//
// Note: Jeffrey Yasskin suggested an alternative fix by "using
// ::operator<<;" in the definition of Message's operator<<.  That fix
// doesn't require a helper function, but unfortunately doesn't
// compile with MSVC.
template <typename T>
inline void GTestStreamToHelper(std::ostream* os, const T& val) {
  *os << val;
}

namespace testing {

// Forward declaration of classes.

class Message;                         // Represents a failure message.
class Test;                            // Represents a test.
class TestCase;                        // A collection of related tests.
class TestPartResult;                  // Result of a test part.
class TestInfo;                        // Information about a test.
class UnitTest;                        // A collection of test cases.
class UnitTestEventListenerInterface;  // Listens to Google Test events.
class AssertionResult;                 // Result of an assertion.

namespace internal {

struct TraceInfo;                      // Information about a trace point.
class ScopedTrace;                     // Implements scoped trace.
class TestInfoImpl;                    // Opaque implementation of TestInfo
class TestResult;                      // Result of a single Test.
class UnitTestImpl;                    // Opaque implementation of UnitTest

template <typename E> class List;      // A generic list.
template <typename E> class ListNode;  // A node in a generic list.

// How many times InitGoogleTest() has been called.
extern int g_init_gtest_count;

// The text used in failure messages to indicate the start of the
// stack trace.
extern const char kStackTraceMarker[];

// A secret type that Google Test users don't know about.  It has no
// definition on purpose.  Therefore it's impossible to create a
// Secret object, which is what we want.
class Secret;

// Two overloaded helpers for checking at compile time whether an
// expression is a null pointer literal (i.e. NULL or any 0-valued
// compile-time integral constant).  Their return values have
// different sizes, so we can use sizeof() to test which version is
// picked by the compiler.  These helpers have no implementations, as
// we only need their signatures.
//
// Given IsNullLiteralHelper(x), the compiler will pick the first
// version if x can be implicitly converted to Secret*, and pick the
// second version otherwise.  Since Secret is a secret and incomplete
// type, the only expression a user can write that has type Secret* is
// a null pointer literal.  Therefore, we know that x is a null
// pointer literal if and only if the first version is picked by the
// compiler.
char IsNullLiteralHelper(Secret* p);
char (&IsNullLiteralHelper(...))[2];  // NOLINT

// A compile-time bool constant that is true if and only if x is a
// null pointer literal (i.e. NULL or any 0-valued compile-time
// integral constant).
#ifdef GTEST_ELLIPSIS_NEEDS_COPY_
// Passing non-POD classes through ellipsis (...) crashes the ARM
// compiler.  The Nokia Symbian and the IBM XL C/C++ compiler try to
// instantiate a copy constructor for objects passed through ellipsis
// (...), failing for uncopyable objects.  Hence we define this to
// false (and lose support for NULL detection).
#define GTEST_IS_NULL_LITERAL_(x) false
#else
#define GTEST_IS_NULL_LITERAL_(x) \
    (sizeof(::testing::internal::IsNullLiteralHelper(x)) == 1)
#endif  // GTEST_ELLIPSIS_NEEDS_COPY_

// Appends the user-supplied message to the Google-Test-generated message.
String AppendUserMessage(const String& gtest_msg,
                         const Message& user_msg);

// A helper class for creating scoped traces in user programs.
class ScopedTrace {
 public:
  // The c'tor pushes the given source file location and message onto
  // a trace stack maintained by Google Test.
  ScopedTrace(const char* file, int line, const Message& message);

  // The d'tor pops the info pushed by the c'tor.
  //
  // Note that the d'tor is not virtual in order to be efficient.
  // Don't inherit from ScopedTrace!
  ~ScopedTrace();

 private:
  GTEST_DISALLOW_COPY_AND_ASSIGN_(ScopedTrace);
} GTEST_ATTRIBUTE_UNUSED_;  // A ScopedTrace object does its job in its
                            // c'tor and d'tor.  Therefore it doesn't
                            // need to be used otherwise.

// Converts a streamable value to a String.  A NULL pointer is
// converted to "(null)".  When the input value is a ::string,
// ::std::string, ::wstring, or ::std::wstring object, each NUL
// character in it is replaced with "\\0".
// Declared here but defined in gtest.h, so that it has access
// to the definition of the Message class, required by the ARM
// compiler.
template <typename T>
String StreamableToString(const T& streamable);

// Formats a value to be used in a failure message.

#ifdef GTEST_NEEDS_IS_POINTER_

// These are needed as the Nokia Symbian and IBM XL C/C++ compilers
// cannot decide between const T& and const T* in a function template.
// These compilers _can_ decide between class template specializations
// for T and T*, so a tr1::type_traits-like is_pointer works, and we
// can overload on that.

// This overload makes sure that all pointers (including
// those to char or wchar_t) are printed as raw pointers.
template <typename T>
inline String FormatValueForFailureMessage(internal::true_type dummy,
                                           T* pointer) {
  return StreamableToString(static_cast<const void*>(pointer));
}

template <typename T>
inline String FormatValueForFailureMessage(internal::false_type dummy,
                                           const T& value) {
  return StreamableToString(value);
}

template <typename T>
inline String FormatForFailureMessage(const T& value) {
  return FormatValueForFailureMessage(
      typename internal::is_pointer<T>::type(), value);
}

#else

// These are needed as the above solution using is_pointer has the
// limitation that T cannot be a type without external linkage, when
// compiled using MSVC.

template <typename T>
inline String FormatForFailureMessage(const T& value) {
  return StreamableToString(value);
}

// This overload makes sure that all pointers (including
// those to char or wchar_t) are printed as raw pointers.
template <typename T>
inline String FormatForFailureMessage(T* pointer) {
  return StreamableToString(static_cast<const void*>(pointer));
}

#endif  // GTEST_NEEDS_IS_POINTER_

// These overloaded versions handle narrow and wide characters.
String FormatForFailureMessage(char ch);
String FormatForFailureMessage(wchar_t wchar);

// When this operand is a const char* or char*, and the other operand
// is a ::std::string or ::string, we print this operand as a C string
// rather than a pointer.  We do the same for wide strings.

// This internal macro is used to avoid duplicated code.
#define GTEST_FORMAT_IMPL_(operand2_type, operand1_printer)\
inline String FormatForComparisonFailureMessage(\
    operand2_type::value_type* str, const operand2_type& /*operand2*/) {\
  return operand1_printer(str);\
}\
inline String FormatForComparisonFailureMessage(\
    const operand2_type::value_type* str, const operand2_type& /*operand2*/) {\
  return operand1_printer(str);\
}

#if GTEST_HAS_STD_STRING
GTEST_FORMAT_IMPL_(::std::string, String::ShowCStringQuoted)
#endif  // GTEST_HAS_STD_STRING
#if GTEST_HAS_STD_WSTRING
GTEST_FORMAT_IMPL_(::std::wstring, String::ShowWideCStringQuoted)
#endif  // GTEST_HAS_STD_WSTRING

#if GTEST_HAS_GLOBAL_STRING
GTEST_FORMAT_IMPL_(::string, String::ShowCStringQuoted)
#endif  // GTEST_HAS_GLOBAL_STRING
#if GTEST_HAS_GLOBAL_WSTRING
GTEST_FORMAT_IMPL_(::wstring, String::ShowWideCStringQuoted)
#endif  // GTEST_HAS_GLOBAL_WSTRING

#undef GTEST_FORMAT_IMPL_

// Constructs and returns the message for an equality assertion
// (e.g. ASSERT_EQ, EXPECT_STREQ, etc) failure.
//
// The first four parameters are the expressions used in the assertion
// and their values, as strings.  For example, for ASSERT_EQ(foo, bar)
// where foo is 5 and bar is 6, we have:
//
//   expected_expression: "foo"
//   actual_expression:   "bar"
//   expected_value:      "5"
//   actual_value:        "6"
//
// The ignoring_case parameter is true iff the assertion is a
// *_STRCASEEQ*.  When it's true, the string " (ignoring case)" will
// be inserted into the message.
AssertionResult EqFailure(const char* expected_expression,
                          const char* actual_expression,
                          const String& expected_value,
                          const String& actual_value,
                          bool ignoring_case);


// This template class represents an IEEE floating-point number
// (either single-precision or double-precision, depending on the
// template parameters).
//
// The purpose of this class is to do more sophisticated number
// comparison.  (Due to round-off error, etc, it's very unlikely that
// two floating-points will be equal exactly.  Hence a naive
// comparison by the == operation often doesn't work.)
//
// Format of IEEE floating-point:
//
//   The most-significant bit being the leftmost, an IEEE
//   floating-point looks like
//
//     sign_bit exponent_bits fraction_bits
//
//   Here, sign_bit is a single bit that designates the sign of the
//   number.
//
//   For float, there are 8 exponent bits and 23 fraction bits.
//
//   For double, there are 11 exponent bits and 52 fraction bits.
//
//   More details can be found at
//   http://en.wikipedia.org/wiki/IEEE_floating-point_standard.
//
// Template parameter:
//
//   RawType: the raw floating-point type (either float or double)
template <typename RawType>
class FloatingPoint {
 public:
  // Defines the unsigned integer type that has the same size as the
  // floating point number.
  typedef typename TypeWithSize<sizeof(RawType)>::UInt Bits;

  // Constants.

  // # of bits in a number.
  static const size_t kBitCount = 8*sizeof(RawType);

  // # of fraction bits in a number.
  static const size_t kFractionBitCount =
    std::numeric_limits<RawType>::digits - 1;

  // # of exponent bits in a number.
  static const size_t kExponentBitCount = kBitCount - 1 - kFractionBitCount;

  // The mask for the sign bit.
  static const Bits kSignBitMask = static_cast<Bits>(1) << (kBitCount - 1);

  // The mask for the fraction bits.
  static const Bits kFractionBitMask =
    ~static_cast<Bits>(0) >> (kExponentBitCount + 1);

  // The mask for the exponent bits.
  static const Bits kExponentBitMask = ~(kSignBitMask | kFractionBitMask);

  // How many ULP's (Units in the Last Place) we want to tolerate when
  // comparing two numbers.  The larger the value, the more error we
  // allow.  A 0 value means that two numbers must be exactly the same
  // to be considered equal.
  //
  // The maximum error of a single floating-point operation is 0.5
  // units in the last place.  On Intel CPU's, all floating-point
  // calculations are done with 80-bit precision, while double has 64
  // bits.  Therefore, 4 should be enough for ordinary use.
  //
  // See the following article for more details on ULP:
  // http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm.
  static const size_t kMaxUlps = 4;

  // Constructs a FloatingPoint from a raw floating-point number.
  //
  // On an Intel CPU, passing a non-normalized NAN (Not a Number)
  // around may change its bits, although the new value is guaranteed
  // to be also a NAN.  Therefore, don't expect this constructor to
  // preserve the bits in x when x is a NAN.
  explicit FloatingPoint(const RawType& x) : value_(x) {}

  // Static methods

  // Reinterprets a bit pattern as a floating-point number.
  //
  // This function is needed to test the AlmostEquals() method.
  static RawType ReinterpretBits(const Bits bits) {
    FloatingPoint fp(0);
    fp.bits_ = bits;
    return fp.value_;
  }

  // Returns the floating-point number that represent positive infinity.
  static RawType Infinity() {
    return ReinterpretBits(kExponentBitMask);
  }

  // Non-static methods

  // Returns the bits that represents this number.
  const Bits &bits() const { return bits_; }

  // Returns the exponent bits of this number.
  Bits exponent_bits() const { return kExponentBitMask & bits_; }

  // Returns the fraction bits of this number.
  Bits fraction_bits() const { return kFractionBitMask & bits_; }

  // Returns the sign bit of this number.
  Bits sign_bit() const { return kSignBitMask & bits_; }

  // Returns true iff this is NAN (not a number).
  bool is_nan() const {
    // It's a NAN if the exponent bits are all ones and the fraction
    // bits are not entirely zeros.
    return (exponent_bits() == kExponentBitMask) && (fraction_bits() != 0);
  }

  // Returns true iff this number is at most kMaxUlps ULP's away from
  // rhs.  In particular, this function:
  //
  //   - returns false if either number is (or both are) NAN.
  //   - treats really large numbers as almost equal to infinity.
  //   - thinks +0.0 and -0.0 are 0 DLP's apart.
  bool AlmostEquals(const FloatingPoint& rhs) const {
    // The IEEE standard says that any comparison operation involving
    // a NAN must return false.
    if (is_nan() || rhs.is_nan()) return false;

    return DistanceBetweenSignAndMagnitudeNumbers(bits_, rhs.bits_) <= kMaxUlps;
  }

 private:
  // Converts an integer from the sign-and-magnitude representation to
  // the biased representation.  More precisely, let N be 2 to the
  // power of (kBitCount - 1), an integer x is represented by the
  // unsigned number x + N.
  //
  // For instance,
  //
  //   -N + 1 (the most negative number representable using
  //          sign-and-magnitude) is represented by 1;
  //   0      is represented by N; and
  //   N - 1  (the biggest number representable using
  //          sign-and-magnitude) is represented by 2N - 1.
  //
  // Read http://en.wikipedia.org/wiki/Signed_number_representations
  // for more details on signed number representations.
  static Bits SignAndMagnitudeToBiased(const Bits &sam) {
    if (kSignBitMask & sam) {
      // sam represents a negative number.
      return ~sam + 1;
    } else {
      // sam represents a positive number.
      return kSignBitMask | sam;
    }
  }

  // Given two numbers in the sign-and-magnitude representation,
  // returns the distance between them as an unsigned number.
  static Bits DistanceBetweenSignAndMagnitudeNumbers(const Bits &sam1,
                                                     const Bits &sam2) {
    const Bits biased1 = SignAndMagnitudeToBiased(sam1);
    const Bits biased2 = SignAndMagnitudeToBiased(sam2);
    return (biased1 >= biased2) ? (biased1 - biased2) : (biased2 - biased1);
  }

  union {
    RawType value_;  // The raw floating-point number.
    Bits bits_;      // The bits that represent the number.
  };
};

// Typedefs the instances of the FloatingPoint template class that we
// care to use.
typedef FloatingPoint<float> Float;
typedef FloatingPoint<double> Double;

// In order to catch the mistake of putting tests that use different
// test fixture classes in the same test case, we need to assign
// unique IDs to fixture classes and compare them.  The TypeId type is
// used to hold such IDs.  The user should treat TypeId as an opaque
// type: the only operation allowed on TypeId values is to compare
// them for equality using the == operator.
typedef const void* TypeId;

template <typename T>
class TypeIdHelper {
 public:
  // dummy_ must not have a const type.  Otherwise an overly eager
  // compiler (e.g. MSVC 7.1 & 8.0) may try to merge
  // TypeIdHelper<T>::dummy_ for different Ts as an "optimization".
  static bool dummy_;
};

template <typename T>
bool TypeIdHelper<T>::dummy_ = false;

// GetTypeId<T>() returns the ID of type T.  Different values will be
// returned for different types.  Calling the function twice with the
// same type argument is guaranteed to return the same ID.
template <typename T>
TypeId GetTypeId() {
  // The compiler is required to allocate a different
  // TypeIdHelper<T>::dummy_ variable for each T used to instantiate
  // the template.  Therefore, the address of dummy_ is guaranteed to
  // be unique.
  return &(TypeIdHelper<T>::dummy_);
}

// Returns the type ID of ::testing::Test.  Always call this instead
// of GetTypeId< ::testing::Test>() to get the type ID of
// ::testing::Test, as the latter may give the wrong result due to a
// suspected linker bug when compiling Google Test as a Mac OS X
// framework.
TypeId GetTestTypeId();

// Defines the abstract factory interface that creates instances
// of a Test object.
class TestFactoryBase {
 public:
  virtual ~TestFactoryBase() {}

  // Creates a test instance to run. The instance is both created and destroyed
  // within TestInfoImpl::Run()
  virtual Test* CreateTest() = 0;

 protected:
  TestFactoryBase() {}

 private:
  GTEST_DISALLOW_COPY_AND_ASSIGN_(TestFactoryBase);
};

// This class provides implementation of TeastFactoryBase interface.
// It is used in TEST and TEST_F macros.
template <class TestClass>
class TestFactoryImpl : public TestFactoryBase {
 public:
  virtual Test* CreateTest() { return new TestClass; }
};

#ifdef GTEST_OS_WINDOWS

// Predicate-formatters for implementing the HRESULT checking macros
// {ASSERT|EXPECT}_HRESULT_{SUCCEEDED|FAILED}
// We pass a long instead of HRESULT to avoid causing an
// include dependency for the HRESULT type.
AssertionResult IsHRESULTSuccess(const char* expr, long hr);  // NOLINT
AssertionResult IsHRESULTFailure(const char* expr, long hr);  // NOLINT

#endif  // GTEST_OS_WINDOWS

// Formats a source file path and a line number as they would appear
// in a compiler error message.
inline String FormatFileLocation(const char* file, int line) {
  const char* const file_name = file == NULL ? "unknown file" : file;
  if (line < 0) {
    return String::Format("%s:", file_name);
  }
#ifdef _MSC_VER
  return String::Format("%s(%d):", file_name, line);
#else
  return String::Format("%s:%d:", file_name, line);
#endif  // _MSC_VER
}

// Types of SetUpTestCase() and TearDownTestCase() functions.
typedef void (*SetUpTestCaseFunc)();
typedef void (*TearDownTestCaseFunc)();

// Creates a new TestInfo object and registers it with Google Test;
// returns the created object.
//
// Arguments:
//
//   test_case_name:   name of the test case
//   name:             name of the test
//   test_case_comment: a comment on the test case that will be included in
//                      the test output
//   comment:          a comment on the test that will be included in the
//                     test output
//   fixture_class_id: ID of the test fixture class
//   set_up_tc:        pointer to the function that sets up the test case
//   tear_down_tc:     pointer to the function that tears down the test case
//   factory:          pointer to the factory that creates a test object.
//                     The newly created TestInfo instance will assume
//                     ownership of the factory object.
TestInfo* MakeAndRegisterTestInfo(
    const char* test_case_name, const char* name,
    const char* test_case_comment, const char* comment,
    TypeId fixture_class_id,
    SetUpTestCaseFunc set_up_tc,
    TearDownTestCaseFunc tear_down_tc,
    TestFactoryBase* factory);

#if defined(GTEST_HAS_TYPED_TEST) || defined(GTEST_HAS_TYPED_TEST_P)

// State of the definition of a type-parameterized test case.
class TypedTestCasePState {
 public:
  TypedTestCasePState() : registered_(false) {}

  // Adds the given test name to defined_test_names_ and return true
  // if the test case hasn't been registered; otherwise aborts the
  // program.
  bool AddTestName(const char* file, int line, const char* case_name,
                   const char* test_name) {
    if (registered_) {
      fprintf(stderr, "%s Test %s must be defined before "
              "REGISTER_TYPED_TEST_CASE_P(%s, ...).\n",
              FormatFileLocation(file, line).c_str(), test_name, case_name);
      abort();
    }
    defined_test_names_.insert(test_name);
    return true;
  }

  // Verifies that registered_tests match the test names in
  // defined_test_names_; returns registered_tests if successful, or
  // aborts the program otherwise.
  const char* VerifyRegisteredTestNames(
      const char* file, int line, const char* registered_tests);

 private:
  bool registered_;
  ::std::set<const char*> defined_test_names_;
};

// Skips to the first non-space char after the first comma in 'str';
// returns NULL if no comma is found in 'str'.
inline const char* SkipComma(const char* str) {
  const char* comma = strchr(str, ',');
  if (comma == NULL) {
    return NULL;
  }
  while (isspace(*(++comma))) {}
  return comma;
}

// Returns the prefix of 'str' before the first comma in it; returns
// the entire string if it contains no comma.
inline String GetPrefixUntilComma(const char* str) {
  const char* comma = strchr(str, ',');
  return comma == NULL ? String(str) : String(str, comma - str);
}

// TypeParameterizedTest<Fixture, TestSel, Types>::Register()
// registers a list of type-parameterized tests with Google Test.  The
// return value is insignificant - we just need to return something
// such that we can call this function in a namespace scope.
//
// Implementation note: The GTEST_TEMPLATE_ macro declares a template
// template parameter.  It's defined in gtest-type-util.h.
template <GTEST_TEMPLATE_ Fixture, class TestSel, typename Types>
class TypeParameterizedTest {
 public:
  // 'index' is the index of the test in the type list 'Types'
  // specified in INSTANTIATE_TYPED_TEST_CASE_P(Prefix, TestCase,
  // Types).  Valid values for 'index' are [0, N - 1] where N is the
  // length of Types.
  static bool Register(const char* prefix, const char* case_name,
                       const char* test_names, int index) {
    typedef typename Types::Head Type;
    typedef Fixture<Type> FixtureClass;
    typedef typename GTEST_BIND_(TestSel, Type) TestClass;

    // First, registers the first type-parameterized test in the type
    // list.
    MakeAndRegisterTestInfo(
        String::Format("%s%s%s/%d", prefix, prefix[0] == '\0' ? "" : "/",
                       case_name, index).c_str(),
        GetPrefixUntilComma(test_names).c_str(),
        String::Format("TypeParam = %s", GetTypeName<Type>().c_str()).c_str(),
        "",
        GetTypeId<FixtureClass>(),
        TestClass::SetUpTestCase,
        TestClass::TearDownTestCase,
        new TestFactoryImpl<TestClass>);

    // Next, recurses (at compile time) with the tail of the type list.
    return TypeParameterizedTest<Fixture, TestSel, typename Types::Tail>
        ::Register(prefix, case_name, test_names, index + 1);
  }
};

// The base case for the compile time recursion.
template <GTEST_TEMPLATE_ Fixture, class TestSel>
class TypeParameterizedTest<Fixture, TestSel, Types0> {
 public:
  static bool Register(const char* /*prefix*/, const char* /*case_name*/,
                       const char* /*test_names*/, int /*index*/) {
    return true;
  }
};

// TypeParameterizedTestCase<Fixture, Tests, Types>::Register()
// registers *all combinations* of 'Tests' and 'Types' with Google
// Test.  The return value is insignificant - we just need to return
// something such that we can call this function in a namespace scope.
template <GTEST_TEMPLATE_ Fixture, typename Tests, typename Types>
class TypeParameterizedTestCase {
 public:
  static bool Register(const char* prefix, const char* case_name,
                       const char* test_names) {
    typedef typename Tests::Head Head;

    // First, register the first test in 'Test' for each type in 'Types'.
    TypeParameterizedTest<Fixture, Head, Types>::Register(
        prefix, case_name, test_names, 0);

    // Next, recurses (at compile time) with the tail of the test list.
    return TypeParameterizedTestCase<Fixture, typename Tests::Tail, Types>
        ::Register(prefix, case_name, SkipComma(test_names));
  }
};

// The base case for the compile time recursion.
template <GTEST_TEMPLATE_ Fixture, typename Types>
class TypeParameterizedTestCase<Fixture, Templates0, Types> {
 public:
  static bool Register(const char* prefix, const char* case_name,
                       const char* test_names) {
    return true;
  }
};

#endif  // GTEST_HAS_TYPED_TEST || GTEST_HAS_TYPED_TEST_P

// Returns the current OS stack trace as a String.
//
// The maximum number of stack frames to be included is specified by
// the gtest_stack_trace_depth flag.  The skip_count parameter
// specifies the number of top frames to be skipped, which doesn't
// count against the number of frames to be included.
//
// For example, if Foo() calls Bar(), which in turn calls
// GetCurrentOsStackTraceExceptTop(..., 1), Foo() will be included in
// the trace but Bar() and GetCurrentOsStackTraceExceptTop() won't.
String GetCurrentOsStackTraceExceptTop(UnitTest* unit_test, int skip_count);

// Returns the number of failed test parts in the given test result object.
int GetFailedPartCount(const TestResult* result);

}  // namespace internal
}  // namespace testing

#define GTEST_MESSAGE_(message, result_type) \
  ::testing::internal::AssertHelper(result_type, __FILE__, __LINE__, message) \
    = ::testing::Message()

#define GTEST_FATAL_FAILURE_(message) \
  return GTEST_MESSAGE_(message, ::testing::TPRT_FATAL_FAILURE)

#define GTEST_NONFATAL_FAILURE_(message) \
  GTEST_MESSAGE_(message, ::testing::TPRT_NONFATAL_FAILURE)

#define GTEST_SUCCESS_(message) \
  GTEST_MESSAGE_(message, ::testing::TPRT_SUCCESS)

#define GTEST_TEST_THROW_(statement, expected_exception, fail) \
  GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
  if (const char* gtest_msg = "") { \
    bool gtest_caught_expected = false; \
    try { \
      statement; \
    } \
    catch (expected_exception const&) { \
      gtest_caught_expected = true; \
    } \
    catch (...) { \
      gtest_msg = "Expected: " #statement " throws an exception of type " \
                  #expected_exception ".\n  Actual: it throws a different " \
                  "type."; \
      goto GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__); \
    } \
    if (!gtest_caught_expected) { \
      gtest_msg = "Expected: " #statement " throws an exception of type " \
                  #expected_exception ".\n  Actual: it throws nothing."; \
      goto GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__); \
    } \
  } else \
    GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__): \
      fail(gtest_msg)

#define GTEST_TEST_NO_THROW_(statement, fail) \
  GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
  if (const char* gtest_msg = "") { \
    try { \
      statement; \
    } \
    catch (...) { \
      gtest_msg = "Expected: " #statement " doesn't throw an exception.\n" \
                  "  Actual: it throws."; \
      goto GTEST_CONCAT_TOKEN_(gtest_label_testnothrow_, __LINE__); \
    } \
  } else \
    GTEST_CONCAT_TOKEN_(gtest_label_testnothrow_, __LINE__): \
      fail(gtest_msg)

#define GTEST_TEST_ANY_THROW_(statement, fail) \
  GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
  if (const char* gtest_msg = "") { \
    bool gtest_caught_any = false; \
    try { \
      statement; \
    } \
    catch (...) { \
      gtest_caught_any = true; \
    } \
    if (!gtest_caught_any) { \
      gtest_msg = "Expected: " #statement " throws an exception.\n" \
                  "  Actual: it doesn't."; \
      goto GTEST_CONCAT_TOKEN_(gtest_label_testanythrow_, __LINE__); \
    } \
  } else \
    GTEST_CONCAT_TOKEN_(gtest_label_testanythrow_, __LINE__): \
      fail(gtest_msg)


#define GTEST_TEST_BOOLEAN_(boolexpr, booltext, actual, expected, fail) \
  GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
  if (boolexpr) \
    ; \
  else \
    fail("Value of: " booltext "\n  Actual: " #actual "\nExpected: " #expected)

#define GTEST_TEST_NO_FATAL_FAILURE_(statement, fail) \
  GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
  if (const char* gtest_msg = "") { \
    ::testing::internal::HasNewFatalFailureHelper gtest_fatal_failure_checker; \
    { statement; } \
    if (gtest_fatal_failure_checker.has_new_fatal_failure()) { \
      gtest_msg = "Expected: " #statement " doesn't generate new fatal " \
                  "failures in the current thread.\n" \
                  "  Actual: it does."; \
      goto GTEST_CONCAT_TOKEN_(gtest_label_testnofatal_, __LINE__); \
    } \
  } else \
    GTEST_CONCAT_TOKEN_(gtest_label_testnofatal_, __LINE__): \
      fail(gtest_msg)

// Expands to the name of the class that implements the given test.
#define GTEST_TEST_CLASS_NAME_(test_case_name, test_name) \
  test_case_name##_##test_name##_Test

// Helper macro for defining tests.
#define GTEST_TEST_(test_case_name, test_name, parent_class, parent_id)\
class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) : public parent_class {\
 public:\
  GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {}\
 private:\
  virtual void TestBody();\
  static ::testing::TestInfo* const test_info_;\
  GTEST_DISALLOW_COPY_AND_ASSIGN_(\
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name));\
};\
\
::testing::TestInfo* const GTEST_TEST_CLASS_NAME_(test_case_name, test_name)\
  ::test_info_ =\
    ::testing::internal::MakeAndRegisterTestInfo(\
        #test_case_name, #test_name, "", "", \
        (parent_id), \
        parent_class::SetUpTestCase, \
        parent_class::TearDownTestCase, \
        new ::testing::internal::TestFactoryImpl<\
            GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>);\
void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBody()

#endif  // GTEST_INCLUDE_GTEST_INTERNAL_GTEST_INTERNAL_H_
