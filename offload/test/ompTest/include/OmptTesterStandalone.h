#ifndef OFFLOAD_TEST_OMPTEST_INCLUDE_OMPTTESTERSTANDALONE_H
#define OFFLOAD_TEST_OMPTEST_INCLUDE_OMPTTESTERSTANDALONE_H

#include "OmptAssertEvent.h"
#include "OmptAsserter.h"
#include "OmptTesterGlobals.h"

#include <unordered_map>
#include <vector>

// Forward declarations.
namespace omptest {
struct OmptEventAsserter;
class OmptEventReporter;
struct OmptSequencedAsserter;
} // namespace omptest

struct Error {
  operator bool() { return Fail; }
  bool Fail;
};

/// A pretty crude test case abstraction
class TestCase : public omptest::OmptEventGroupInterface {
public:
  TestCase(const std::string &name)
      : IsDisabled(name.rfind("DISABLED_", 0) == 0), Name(name) {}
  TestCase(const std::string &name, const omptest::AssertState &expected)
      : IsDisabled(name.rfind("DISABLED_", 0) == 0), Name(name),
        ExpectedState(expected) {}
  virtual ~TestCase() = default;
  Error exec();
  virtual void execImpl() { assert(false && "Allocating base class"); }

  bool IsDisabled{false};
  std::string Name;
  omptest::AssertState ExpectedState{omptest::AssertState::pass};
  omptest::AssertState ResultState{omptest::AssertState::pass};
};
/// A pretty crude test suite abstraction
struct TestSuite {
  using TestCaseVec = std::vector<std::unique_ptr<TestCase>>;
  std::string Name;
  TestSuite() = default;
  TestSuite(const TestSuite &O) = delete;
  TestSuite(TestSuite &&O);
  void setup();
  void teardown();
  TestCaseVec::iterator begin();
  TestCaseVec::iterator end();
  TestCaseVec TestCases;
};
/// Static class used to register all test cases and provide them to the driver
struct TestRegistrar {
  static TestRegistrar &get();
  static std::vector<TestSuite> getTestSuites();
  static void addCaseToSuite(TestCase *TC, std::string TSName);

private:
  TestRegistrar() = default;
  TestRegistrar(const TestRegistrar &o) = delete;
  TestRegistrar operator=(const TestRegistrar &o) = delete;
  static std::unordered_map<std::string, TestSuite> Tests;
};
/// Hack to register test cases
struct Registerer {
  Registerer(TestCase *TC, const std::string SuiteName);
};
/// Eventually executes all test suites and cases, should contain logic to skip
/// stuff if needed
struct Runner {
  Runner() : TestSuites(TestRegistrar::get().getTestSuites()) {}
  int run();
  void reportError(const Error &Err);
  void abortOrKeepGoing();
  // Print an execution summary of all testsuites and their corresponding
  // testcases.
  void printSummary();
  std::vector<TestSuite> TestSuites;
};

/// MACROS TO DEFINE A TESTSUITE + TESTCASE (like GoogleTest does)
#define XQUOTE(str) QUOTE(str)
#define QUOTE(str) #str

#define TEST_TEMPLATE(SuiteName, CaseName, ExpectedState)                      \
  struct SuiteName##_##CaseName : public TestCase {                            \
    SuiteName##_##CaseName()                                                   \
        : TestCase(XQUOTE(CaseName), omptest::AssertState::ExpectedState) {}   \
    virtual void execImpl() override;                                          \
  };                                                                           \
  static Registerer R_##SuiteName##CaseName(new SuiteName##_##CaseName(),      \
                                            #SuiteName);                       \
  void SuiteName##_##CaseName::execImpl()

#define TEST(SuiteName, CaseName)                                              \
  TEST_TEMPLATE(SuiteName, CaseName, /*ExpectedState=*/pass)
#define TEST_XFAIL(SuiteName, CaseName)                                        \
  TEST_TEMPLATE(SuiteName, CaseName, /*ExpectedState=*/fail)

#endif
