//==--------------- Dump.cpp - Test SYCL scheduler graph dumping -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %clang -std=c++11 -g %s -o %t.out -lstdc++ -lOpenCL -lsycl
// RUN: %t.out
// RUN: env SS_DUMP_TEXT=1 %t.out
// RUN: env SS_DUMP_WHOLE_GRAPH=1 %t.out
// RUN: env SS_DUMP_RUN_GRAPH=1 %t.out

#include <CL/sycl.hpp>

#include <cassert>
#include <cstdlib>

using namespace cl::sycl::simple_scheduler;

int main() {
  const bool TextFlag = Scheduler::getInstance().getDumpFlagValue(
      Scheduler::DumpOptions::Text);
  const bool TextEnv = std::getenv("SS_DUMP_TEXT");
  assert(TextFlag == TextEnv);

  const bool WholeGraphFlag = Scheduler::getInstance().getDumpFlagValue(
      Scheduler::DumpOptions::WholeGraph);
  const bool WholeGraphEnv = std::getenv("SS_DUMP_WHOLE_GRAPH");
  assert(WholeGraphFlag == WholeGraphEnv);

  const bool RunGraphFlag = Scheduler::getInstance().getDumpFlagValue(
      Scheduler::DumpOptions::RunGraph);
  const bool RunGraphEnv = std::getenv("SS_DUMP_RUN_GRAPH");
  assert(RunGraphFlag == RunGraphEnv);

  return 0;
}
