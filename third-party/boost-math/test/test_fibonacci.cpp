//  Copyright Madhur Chauhan 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/policies/error_handling.hpp>
#include <boost/math/special_functions/fibonacci.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/test/tools/old/interface.hpp>
#include <cstdint>
#include <exception>
#define BOOST_TEST_MODULE Fibonacci_Test_Module
#include <boost/test/included/unit_test.hpp>

using boost::math::fibonacci;
using boost::math::unchecked_fibonacci;
using namespace boost::multiprecision;
typedef cpp_int BST;
typedef number<backends::cpp_int_backend<2048, 2048, unsigned_magnitude, unchecked>> BST_2048;

// Some sanity checks using OEIS A000045
BOOST_AUTO_TEST_CASE(sanity_checks) {
    BOOST_TEST(fibonacci<char>(0) == 0);    // Base case
    BOOST_TEST(fibonacci<int>(1) == 1);     // Base case
    BOOST_TEST(fibonacci<int16_t>(2) == 1); // Base Case
    BOOST_TEST(fibonacci<int16_t>(3) == 2); // First computation
    BOOST_TEST(fibonacci<int16_t>(10) == 55);
    BOOST_TEST(fibonacci<int16_t>(15) == 610);
    BOOST_TEST(fibonacci<int16_t>(18) == 2584);
    BOOST_TEST(fibonacci<int32_t>(40) == 102334155);
}

// Tests unchecked_fibonacci by computing naively
BOOST_AUTO_TEST_CASE(big_integer_check) {
    // GMP is used as type for fibonacci and cpp_int is for naive computation
    BST val = 0, a = 0, b = 1;
    for (int i = 0; i <= 1e4; ++i) {
        BOOST_TEST(fibonacci<BST>(i) == a);
        val = b, b += a, a = val;
    }
}

// Check for overflow throw using magic constants
BOOST_AUTO_TEST_CASE(overflow_check) {

    // 1. check for unsigned integer overflow
    BOOST_CHECK_NO_THROW(fibonacci<uint64_t>(93));
    BOOST_CHECK_THROW(fibonacci<uint64_t>(94), std::exception);
    BOOST_CHECK_NO_THROW(unchecked_fibonacci<uint64_t>(94));

    // 2. check for signed integer overflow
    BOOST_CHECK_NO_THROW(fibonacci<int64_t>(91));
    // BOOST_CHECK_THROW(fibonacci<int64_t>(92), std::exception); // this should be the correct value but imprecisions
    BOOST_CHECK_THROW(fibonacci<int64_t>(93), std::exception);
    // In UBSAN this will error out, but it is expected
    BOOST_CHECK_NO_THROW(unchecked_fibonacci<int64_t>(93));

    // 3. check for floating point (double)
    BOOST_CHECK_NO_THROW(fibonacci<double>(78));
    BOOST_CHECK_THROW(fibonacci<double>(79), std::exception);
    BOOST_CHECK_NO_THROW(unchecked_fibonacci<double>(79));

    // 4. check using boost's multiprecision unchecked integer
    BOOST_CHECK_NO_THROW(fibonacci<BST_2048>(2950));
    // BOOST_CHECK_THROW(fibonacci<T>(2951), std::exception); // this should be the correct value but imprecisions
    BOOST_CHECK_THROW(fibonacci<BST_2048>(2952), std::exception);
    BOOST_CHECK_NO_THROW(unchecked_fibonacci<BST_2048>(2952));
}

BOOST_AUTO_TEST_CASE(generator_check) {
    // first 5 values
    boost::math::fibonacci_generator<BST_2048> gen;
    for (int i : {0, 1, 1, 2, 3, 5, 8, 13, 21}) {
        BOOST_TEST(gen() == i);
    }

    // test whether the generator is set correctly to the given index --- next
    const int next = 1000; // next <=2950 (checked from test above)
    gen.set(next);
    BST_2048 a = fibonacci<BST_2048>(next), b = fibonacci<BST_2048>(next + 1);
    for (int i = next; i < next + 50; ++i) {
        BOOST_TEST(gen() == a);
        swap(a, b);
        b += a;
    }

    // shift the generator back to next and check again
    a = fibonacci<BST_2048>(next), b = fibonacci<BST_2048>(next + 1);
    gen.set(next);
    for (int i = next; i < next + 50; ++i) {
        BOOST_TEST(gen() == a);
        swap(a, b);
        b += a;
    }
}

#ifndef BOOST_NO_CXX14_CONSTEXPR

BOOST_AUTO_TEST_CASE(constexpr_check) {
    constexpr int x = boost::math::unchecked_fibonacci<int>(32);
    static_assert(x == 2178309, "constexpr check 1");

    constexpr double y = boost::math::unchecked_fibonacci<double>(40);
    static_assert(y == 102334155.0, "constexpr check 2");

    constexpr int xx = boost::math::fibonacci<int>(32);
    static_assert(xx == 2178309, "constexpr check 3");

    constexpr double yy = boost::math::fibonacci<double>(40);
    static_assert(yy == 102334155.0, "constexpr check 4");

}

#endif
