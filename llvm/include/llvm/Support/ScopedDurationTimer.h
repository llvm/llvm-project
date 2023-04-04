//===-- llvm/Support/ScopedDurationTimer.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SCOPEDDURATIONTIMER_H
#define LLVM_SUPPORT_SCOPEDDURATIONTIMER_H

#include <chrono>
#include <ratio>
#include <utility>

namespace llvm {

/// RAII timer that captures the duration between construction and destruction
/// and passes it to the provided \p ElapsedHandler at destruction.
///
/// Example use:
/// \code
///   {
///     llvm::ScopedDurationTimer ScopedTime([](double Seconds) {
///       llvm::outs() << "duration: " << Seconds << "\n";
///     });
///     // Actions to get duration for.
///     <...>
///   }
/// \endcode
template <typename Callable> class ScopedDurationTimer {
public:
  explicit ScopedDurationTimer(Callable &&ElapsedHandler)
      : ElapsedHandler(std::move(ElapsedHandler)) {}

  Callable ElapsedHandler;
  std::chrono::steady_clock::time_point StartTime =
      std::chrono::steady_clock::now();

  ~ScopedDurationTimer() {
    using Seconds = std::chrono::duration<double, std::ratio<1>>;
    ElapsedHandler(
        Seconds(std::chrono::steady_clock::now() - StartTime).count());
  }
};

template <typename Callable>
ScopedDurationTimer(Callable &&) -> ScopedDurationTimer<Callable>;

} // end namespace llvm

#endif
