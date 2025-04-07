//===-- DILTests.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Runner.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBType.h"
#include "lldb/lldb-enumerations.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::MakeMatcher;
using ::testing::Matcher;
using ::testing::MatcherInterface;
using ::testing::MatchResultListener;

struct EvalResult {
  lldb::SBError lldb_DIL_error;
  mutable lldb::SBValue lldb_DIL_value;
  mutable std::optional<lldb::SBValue> lldb_value;
};

class EvaluatorHelper {
public:
  EvaluatorHelper(lldb::SBFrame frame, bool compare_with_frame_var)
      : frame_(frame), compare_with_frame_var_(compare_with_frame_var) {}

public:
  EvalResult Eval(const std::string &expr) {
    EvalResult ret;
    ret.lldb_DIL_value = frame_.TestGetValueForVariablePath(
        expr.c_str(), lldb::eNoDynamicValues, true);
    if (!ret.lldb_DIL_value.GetError().Success())
      ret.lldb_DIL_error = ret.lldb_DIL_value.GetError();
    if (compare_with_frame_var_) {
      ret.lldb_value = frame_.TestGetValueForVariablePath(
          expr.c_str(), lldb::eNoDynamicValues, false);
    }
    return ret;
  }

private:
  lldb::SBFrame frame_;
  bool compare_with_frame_var_;
};

void PrintError(::testing::MatchResultListener *listener,
                const std::string &error) {
  *listener << "error:";
  // Print multiline errors on a separate line.
  if (error.find('\n') != std::string::npos) {
    *listener << "\n";
  } else {
    *listener << " ";
  }
  *listener << error;
}

class IsOkMatcher : public MatcherInterface<EvalResult> {
public:
  explicit IsOkMatcher(bool compare_types) : compare_types_(compare_types) {}

  bool MatchAndExplain(EvalResult result,
                       MatchResultListener *listener) const override {
    if (result.lldb_DIL_error.GetError()) {
      PrintError(listener, result.lldb_DIL_error.GetCString());
      return false;
    }

    std::string actual = result.lldb_DIL_value.GetValue();
    // Compare only if we tried to evaluate with LLDB.
    if (result.lldb_value.has_value()) {
      if (result.lldb_value.value().GetError().GetError()) {
        *listener << "values produced by DIL and 'frame var' don't match\n"
                  << "DIL      : " << actual << "\n"
                  << "frame var: "
                  << result.lldb_value.value().GetError().GetCString();
        return false;

      } else if (actual != result.lldb_value.value().GetValue()) {
        *listener << "values produced by DIL and 'frame var' don't match\n"
                  << "DIL      : " << actual << "\n"
                  << "frame var: " << result.lldb_value.value().GetValue();
        return false;
      }

      if (compare_types_) {
        const char *lldb_DIL_type =
            result.lldb_DIL_value.GetType().GetUnqualifiedType().GetName();
        const char *lldb_type =
            result.lldb_value.value().GetType().GetUnqualifiedType().GetName();
        if (strcmp(lldb_DIL_type, lldb_type) != 0) {
          *listener << "types produced by DIL and 'frame var' don't match\n"
                    << "DIL      : " << lldb_DIL_type << "\n"
                    << "frame var: " << lldb_type;
          return false;
        }
      }
    }

    return true;
  }

  void DescribeTo(std::ostream *os) const override {
    *os << "evaluates without an error and equals to LLDB";
  }

private:
  bool compare_types_;
};

Matcher<EvalResult> IsOk(bool compare_types = true) {
  return MakeMatcher(new IsOkMatcher(compare_types));
}

class IsEqualMatcher : public MatcherInterface<EvalResult> {
public:
  IsEqualMatcher(std::string value, bool compare_types)
      : value_(std::move(value)), compare_types_(compare_types) {}

public:
  bool MatchAndExplain(EvalResult result,
                       MatchResultListener *listener) const override {
    if (result.lldb_DIL_error.GetError()) {
      PrintError(listener, result.lldb_DIL_error.GetCString());
      return false;
    }

    std::string actual = result.lldb_DIL_value.GetValue();
    if (actual != value_) {
      *listener << "evaluated to '" << actual << "'";
      return false;
    }

    // Compare only if we tried to evaluate with LLDB.
    if (result.lldb_value.has_value()) {
      if (result.lldb_value.value().GetError().GetError()) {
        *listener << "values produced by DIL and 'frame var' don't match\n"
                  << "DIL      : " << actual << "\n"
                  << "frame var: "
                  << result.lldb_value.value().GetError().GetCString();
        return false;

      } else if (actual != result.lldb_value.value().GetValue()) {
        *listener << "values produced by DIL and 'frame var' don't match\n"
                  << "DIL      : " << actual << "\n"
                  << "frame var: " << result.lldb_value.value().GetValue();
        return false;
      }

      if (compare_types_) {
        const char *lldb_DIL_type =
            result.lldb_DIL_value.GetType().GetUnqualifiedType().GetName();
        const char *lldb_type =
            result.lldb_value.value().GetType().GetUnqualifiedType().GetName();
        if (strcmp(lldb_DIL_type, lldb_type) != 0) {
          *listener << "types produced by DIL and 'frame var' don't match\n"
                    << "DIL      : " << lldb_DIL_type << "\n"
                    << "frame var: " << lldb_type;
          return false;
        }
      }
    }
    return true;
  }

  void DescribeTo(std::ostream *os) const override {
    *os << "evaluates to '" << value_ << "'";
  }

private:
  std::string value_;
  bool compare_types_;
};

Matcher<EvalResult> IsEqual(std::string value, bool compare_types = true) {
  return MakeMatcher(new IsEqualMatcher(std::move(value), compare_types));
}

class IsErrorMatcher : public MatcherInterface<EvalResult> {
public:
  explicit IsErrorMatcher(std::string value) : value_(std::move(value)) {}

public:
  bool MatchAndExplain(EvalResult result,
                       MatchResultListener *listener) const override {
    if (!result.lldb_DIL_error.GetError()) {
      *listener << "evaluated to '" << result.lldb_DIL_value.GetValue() << "'";
      return false;
    }
    std::string message = result.lldb_DIL_error.GetCString();
    if (message.find(value_) == std::string::npos) {
      PrintError(listener, message);
      return false;
    }

    return true;
  }

  void DescribeTo(std::ostream *os) const override {
    *os << "evaluates with an error: '" << value_ << "'";
  }

private:
  std::string value_;
};

Matcher<EvalResult> IsError(std::string value) {
  return MakeMatcher(new IsErrorMatcher(std::move(value)));
}

class EvalTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { lldb::SBDebugger::Initialize(); }

  static void TearDownTestSuite() { lldb::SBDebugger::Terminate(); }

  void SetUp() override {
    std::string test_name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::string break_line = "// BREAK(" + test_name + ")";

    std::string binary_path =
        lldb_private::GetInputFilePath("test_binary.bin");
    std::string source_path = lldb_private::GetInputFilePath("test_binary.cpp");

    debugger_ = lldb::SBDebugger::Create(false);
    process_ =
        LaunchTestProgram(debugger_, source_path, binary_path, break_line);
    frame_ = process_.GetSelectedThread().GetSelectedFrame();
  }

  void TearDown() override {
    process_.Destroy();
    lldb::SBDebugger::Destroy(debugger_);
  }

  EvalResult Eval(const std::string &expr) {
    return EvaluatorHelper(frame_, compare_with_frame_var_).Eval(expr);
  }

  bool Is32Bit() const {
    if (process_.GetAddressByteSize() == 4) {
      return true;
    }
    return false;
  }

protected:
  lldb::SBDebugger debugger_;
  lldb::SBProcess process_;
  lldb::SBFrame frame_;

  // Evaluate with both DIL and LLDB by default.
  bool compare_with_frame_var_ = true;
};

TEST_F(EvalTest, TestSymbols) {
  EXPECT_GT(frame_.GetModule().GetNumSymbols(), (size_t)0)
      << "No symbols might indicate that the test binary was built incorrectly";
}

TEST_F(EvalTest, TestPointerDereference) {
  EXPECT_THAT(Eval("*p_int0"), IsEqual("0"));
  EXPECT_THAT(Eval("*cp_int5"), IsEqual("5"));
  EXPECT_THAT(Eval("*rcp_int0"), IsOk());

  EXPECT_THAT(Eval("&*p_void"),
              IsError("indirection not permitted on operand of type"
                            " 'void *'"));

  this->compare_with_frame_var_ = false;
  EXPECT_THAT(Eval("*array"), IsEqual("0"));
  EXPECT_THAT(Eval("&*p_null"),
              IsEqual(Is32Bit() ? "0x00000000" : "0x0000000000000000"));
  EXPECT_THAT(Eval("**pp_int0"), IsEqual("0"));
  EXPECT_THAT(Eval("&**pp_int0"), IsOk());
}

TEST_F(EvalTest, TestAddressOf) {
  EXPECT_THAT(Eval("&x"), IsOk());
  EXPECT_THAT(Eval("r"), IsOk());
  EXPECT_THAT(Eval("&r"), IsOk());
  EXPECT_THAT(Eval("pr"), IsOk());
  EXPECT_THAT(Eval("&pr"), IsOk());
  EXPECT_THAT(Eval("my_pr"), IsOk());
  EXPECT_THAT(Eval("&my_pr"), IsOk());

  EXPECT_THAT(Eval("&globalVar"), IsOk());
  EXPECT_THAT(Eval("&s_str"), IsOk());
  EXPECT_THAT(Eval("&param"), IsOk());
}
