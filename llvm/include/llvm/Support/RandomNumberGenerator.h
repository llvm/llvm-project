//==- llvm/Support/RandomNumberGenerator.h - RNG for diversity ---*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an abstraction for deterministic random number
// generation (RNG).  Note that the current implementation is not
// cryptographically secure as it uses the C++11 <random> facilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_RANDOMNUMBERGENERATOR_H_
#define LLVM_SUPPORT_RANDOMNUMBERGENERATOR_H_

#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h" // Needed for uint64_t on Windows.
#include <random>
#include <system_error>

namespace llvm {
class StringRef;

/// A random number generator.
///
/// Instances of this class should not be shared across threads. The
/// seed should be set by passing the -rng-seed=<uint64> option. Use
/// Module::createRNG to create a new RNG instance for use with that
/// module.
class RandomNumberGenerator {

  // 64-bit Mersenne Twister by Matsumoto and Nishimura, 2000
  // http://en.cppreference.com/w/cpp/numeric/random/mersenne_twister_engine
  // This RNG is deterministically portable across C++11
  // implementations.
  using generator_type = std::mt19937_64;

public:
  using result_type = generator_type::result_type;

  /// Returns a random number in the range [0, Max).
  result_type operator()();

  // We can only make min/max constexpr if generator_type::min/max are
  // constexpr.  The MSVC 2013 STL does not make these constexpr, so we have to
  // avoid declaring them as constexpr even if the compiler, like clang-cl,
  // supports it.
#if defined(_MSC_VER) && _MSC_VER < 1900
#define STL_CONSTEXPR
#else
#define STL_CONSTEXPR LLVM_CONSTEXPR
#endif

  static STL_CONSTEXPR result_type min() { return generator_type::min(); }
  static STL_CONSTEXPR result_type max() { return generator_type::max(); }

private:
  /// Seeds and salts the underlying RNG engine.
  ///
  /// This constructor should not be used directly. Instead use
  /// Module::createRNG to create a new RNG salted with the Module ID.
  RandomNumberGenerator(StringRef Salt);

  generator_type Generator;

  // Noncopyable.
  RandomNumberGenerator(const RandomNumberGenerator &other) = delete;
  RandomNumberGenerator &operator=(const RandomNumberGenerator &other) = delete;

  friend class Module;
};

// Get random vector of specified size
std::error_code getRandomBytes(void *Buffer, size_t Size);
}

#endif
