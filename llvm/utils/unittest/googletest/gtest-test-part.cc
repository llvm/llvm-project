// Copyright 2008, Google Inc.
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
// Author: mheule@google.com (Markus Heule)
//
// The Google C++ Testing Framework (Google Test)

#include <gtest/gtest-test-part.h>

// Indicates that this translation unit is part of Google Test's
// implementation.  It must come before gtest-internal-inl.h is
// included, or there will be a compiler error.  This trick is to
// prevent a user from accidentally including gtest-internal-inl.h in
// his code.
#define GTEST_IMPLEMENTATION
#include "gtest/internal/gtest-internal-inl.h"
#undef GTEST_IMPLEMENTATION

namespace testing {

// Gets the summary of the failure message by omitting the stack trace
// in it.
internal::String TestPartResult::ExtractSummary(const char* message) {
  const char* const stack_trace = strstr(message, internal::kStackTraceMarker);
  return stack_trace == NULL ? internal::String(message) :
      internal::String(message, stack_trace - message);
}

// Prints a TestPartResult object.
std::ostream& operator<<(std::ostream& os, const TestPartResult& result) {
  return os << result.file_name() << ":"
            << result.line_number() << ": "
            << (result.type() == TPRT_SUCCESS ? "Success" :
                result.type() == TPRT_FATAL_FAILURE ? "Fatal failure" :
                "Non-fatal failure") << ":\n"
            << result.message() << std::endl;
}

// Constructs an empty TestPartResultArray.
TestPartResultArray::TestPartResultArray()
    : list_(new internal::List<TestPartResult>) {
}

// Destructs a TestPartResultArray.
TestPartResultArray::~TestPartResultArray() {
  delete list_;
}

// Appends a TestPartResult to the array.
void TestPartResultArray::Append(const TestPartResult& result) {
  list_->PushBack(result);
}

// Returns the TestPartResult at the given index (0-based).
const TestPartResult& TestPartResultArray::GetTestPartResult(int index) const {
  if (index < 0 || index >= size()) {
    printf("\nInvalid index (%d) into TestPartResultArray.\n", index);
    internal::abort();
  }

  const internal::ListNode<TestPartResult>* p = list_->Head();
  for (int i = 0; i < index; i++) {
    p = p->next();
  }

  return p->element();
}

// Returns the number of TestPartResult objects in the array.
int TestPartResultArray::size() const {
  return list_->size();
}

namespace internal {

HasNewFatalFailureHelper::HasNewFatalFailureHelper()
    : has_new_fatal_failure_(false),
      original_reporter_(UnitTest::GetInstance()->impl()->
                         GetTestPartResultReporterForCurrentThread()) {
  UnitTest::GetInstance()->impl()->SetTestPartResultReporterForCurrentThread(
      this);
}

HasNewFatalFailureHelper::~HasNewFatalFailureHelper() {
  UnitTest::GetInstance()->impl()->SetTestPartResultReporterForCurrentThread(
      original_reporter_);
}

void HasNewFatalFailureHelper::ReportTestPartResult(
    const TestPartResult& result) {
  if (result.fatally_failed())
    has_new_fatal_failure_ = true;
  original_reporter_->ReportTestPartResult(result);
}

}  // namespace internal

}  // namespace testing
