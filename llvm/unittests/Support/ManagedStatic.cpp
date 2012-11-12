//===- llvm/unittest/Support/ManagedStatic.cpp - ManagedStatic tests ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Threading.h"
#include "llvm/Config/config.h"
#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif

#include "gtest/gtest.h"

using namespace llvm;

namespace {

#ifdef HAVE_PTHREAD_H
namespace test1 {
  llvm::ManagedStatic<int> ms;
  void *helper(void*) {
    *ms;
    return NULL;
  }
}

TEST(Initialize, MultipleThreads) {
  // Run this test under tsan: http://code.google.com/p/data-race-test/

  llvm_start_multithreaded();
  pthread_t t1, t2;
  pthread_create(&t1, NULL, test1::helper, NULL);
  pthread_create(&t2, NULL, test1::helper, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  llvm_stop_multithreaded();
}
#endif

} // anonymous namespace
