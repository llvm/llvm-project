// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==--- program_manager.cpp - SYCL program manager test --------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <CL/sycl/detail/program_manager/program_manager.hpp>

#include <cassert>

using namespace cl::sycl;

int main() {
  context ContextFirst;
  context ContextSecond;

  auto &PM = detail::ProgramManager::getInstance();

  const cl_program ClProgramFirst = PM.getBuiltOpenCLProgram(ContextFirst);
  const cl_program ClProgramSecond = PM.getBuiltOpenCLProgram(ContextSecond);
  // The check what getBuiltOpenCLProgram returns unique cl_program for unique
  // context
  assert(ClProgramFirst != ClProgramSecond);
  for (size_t i = 0; i < 10; ++i) {
    const cl_program ClProgramFirstNew = PM.getBuiltOpenCLProgram(ContextFirst);
    const cl_program ClProgramSecondNew =
        PM.getBuiltOpenCLProgram(ContextSecond);
    // The check what getBuiltOpenCLProgram returns the same program for the
    // same context each time
    assert(ClProgramFirst == ClProgramFirstNew);
    assert(ClProgramSecond == ClProgramSecondNew);
  }

  return 0;
}
