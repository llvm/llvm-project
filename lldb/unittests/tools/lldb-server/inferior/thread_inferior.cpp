//===-- thread_inferior.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char* argv[]) {
  int thread_count = 2;
  if (argc > 1) {
    thread_count = std::stoi(argv[1], nullptr, 10);
  }

  std::atomic<bool> delay(true);
  std::vector<std::thread> threads;
  for (int i = 0; i < thread_count; i++) {
    threads.push_back(std::thread([&delay] {
      while (delay.load())
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }));
  }

  // Cause a break.
  volatile char *p = nullptr;
  *p = 'a';

  delay.store(false);
  for (std::thread& t : threads) {
    t.join();
  }

  return 0;
}
