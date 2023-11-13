#ifndef OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTTESTERSTANDALONE_H
#define OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTTESTERSTANDALONE_H

#include "AssertMacros.h"
#include "OmptAliases.h"
#include "OmptAssertEvent.h"
#include "OmptAsserter.h"
#include "OmptCallbackHandler.h"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct Error {
  operator bool() { return Fail; }
  bool Fail;
};

/// A pretty crude test case abstraction
struct TestCase {
  TestCase(const std::string &name) : Name(name) {}
  TestCase(const std::string &name, const omptest::AssertState &expected)
      : Name(name), ExpectedState(expected) {}
  virtual ~TestCase() {}
  std::string Name;
  omptest::AssertState ExpectedState{omptest::AssertState::pass};
  omptest::AssertState ResultState{omptest::AssertState::pass};
  Error exec() {
    OmptCallbackHandler::get().subscribe(&SequenceAsserter);
    OmptCallbackHandler::get().subscribe(&SetAsserter);
    OmptCallbackHandler::get().subscribe(&EventReporter);
    Error E;
    E.Fail = false;
    execImpl();
    // We remove subscribers to not be notified of events after our test case
    // finished.
    OmptCallbackHandler::get().clearSubscribers();
    omptest::AssertState SequenceResultState = SequenceAsserter.getState();
    omptest::AssertState SetResultState = SetAsserter.getState();
    bool AnyFail = SequenceResultState == omptest::AssertState::fail ||
                   SetResultState == omptest::AssertState::fail;
    bool AllPass = SequenceResultState == omptest::AssertState::pass &&
                   SetResultState == omptest::AssertState::pass;
    if (ExpectedState == omptest::AssertState::pass && AnyFail)
      E.Fail = true;
    else if (ExpectedState == omptest::AssertState::fail && AllPass)
      E.Fail = true;
    if (AnyFail)
      ResultState = omptest::AssertState::fail;
    return E;
  };
  virtual void execImpl() { assert(false && "Allocating base class"); }
  // TODO: Should the asserter get a pointer to "their" test case? Probably, as
  // that would allow them to manipulate the Error or AssertState.
  OmptSequencedAsserter SequenceAsserter;
  OmptEventAsserter SetAsserter;
  OmptEventReporter EventReporter;
};
/// A pretty crude test suite abstraction
struct TestSuite {
  using C = std::vector<std::unique_ptr<TestCase>>;
  std::string Name;
  TestSuite() = default;
  TestSuite(const TestSuite &O) = delete;
  TestSuite(TestSuite &&O) {
    Name = O.Name;
    TestCases.swap(O.TestCases);
  }
  void setup() {}
  void teardown() {}
  C::iterator begin() { return TestCases.begin(); }
  C::iterator end() { return TestCases.end(); }
  C TestCases;
};
/// Static class used to register all test cases and provide them to the driver
struct TestRegistrar {
  static TestRegistrar &get() {
    static TestRegistrar TR;
    return TR;
  }
  static std::vector<TestSuite> getTestSuites() {
    std::vector<TestSuite> TSs;
    for (auto &[k, v] : Tests)
      TSs.emplace_back(std::move(v));
    return TSs;
  }
  static void addCaseToSuite(TestCase *TC, std::string TSName) {
    auto &TS = Tests[TSName];
    if (TS.Name.empty())
      TS.Name = TSName;
    TS.TestCases.emplace_back(TC);
  }

private:
  TestRegistrar() = default;
  TestRegistrar(const TestRegistrar &o) = delete;
  TestRegistrar operator=(const TestRegistrar &o) = delete;
  static std::unordered_map<std::string, TestSuite> Tests;
};
/// Hack to register test cases
struct Registerer {
  Registerer(TestCase *TC, const std::string SuiteName) {
    std::cout << "Adding " << TC->Name << " to " << SuiteName << std::endl;
    TestRegistrar::get().addCaseToSuite(TC, SuiteName);
  }
};
/// Eventually executes all test suites and cases, should contain logic to skip
/// stuff if needed
struct Runner {
  Runner() : TestSuites(TestRegistrar::get().getTestSuites()) {}
  int run() {
    for (auto &TS : TestSuites) {
      std::cout << "\n======\nExecuting for " << TS.Name << std::endl;
      TS.setup();
      for (auto &TC : TS) {
        std::cout << "\nExecuting " << TC->Name << std::endl;
        if (Error Err = TC->exec()) {
          reportError(Err);
          abortOrKeepGoing();
        }
      }
      TS.teardown();
    }
    printSummary();
    return 0;
  }
  void reportError(const Error &Err) {}
  void abortOrKeepGoing() {}
  // Print an execution summary of all testsuites and their corresponding
  // testcases.
  void printSummary() {
    std::cout << "\n====== SUMMARY\n";
    for (auto &TS : TestSuites) {
      std::cout << "  - " << TS.Name;
      for (auto &TC : TS) {
        std::string Result =
            (TC->ResultState == TC->ExpectedState)
                ? (TC->ResultState == omptest::AssertState::pass) ? "PASS"
                                                                  : "XFAIL"
                : "FAIL";
        std::cout << "\n      " << std::setw(5) << Result << " : " << TC->Name;
      }
      std::cout << std::endl;
    }
  }
  std::vector<TestSuite> TestSuites;
};

/// MACROS TO DEFINE A TESTSUITE + TESTCASE (like GoogleTest does)
#define XQUOTE(str) QUOTE(str)
#define QUOTE(str) #str

#define TEST(SuiteName, CaseName)                                              \
  struct SuiteName##_##CaseName : public TestCase {                            \
    SuiteName##_##CaseName() : TestCase(XQUOTE(CaseName)) {}                   \
    virtual void execImpl() override;                                          \
  };                                                                           \
  static Registerer R_##SuiteName##CaseName(new SuiteName##_##CaseName(),      \
                                            #SuiteName);                       \
  void SuiteName##_##CaseName::execImpl()
#define TEST_XFAIL(SuiteName, CaseName)                                        \
  struct SuiteName##_##CaseName : public TestCase {                            \
    SuiteName##_##CaseName()                                                   \
        : TestCase(XQUOTE(CaseName), omptest::AssertState::fail) {}            \
    virtual void execImpl() override;                                          \
  };                                                                           \
  static Registerer R_##SuiteName##CaseName(new SuiteName##_##CaseName(),      \
                                            #SuiteName);                       \
  void SuiteName##_##CaseName::execImpl()

#endif // include guard
