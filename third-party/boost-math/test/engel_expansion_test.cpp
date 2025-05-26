/*
 * Copyright Nick Thompson, 2020
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include <boost/math/tools/engel_expansion.hpp>
#include <boost/core/demangle.hpp>
#include <boost/math/constants/constants.hpp>
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif
#include <boost/multiprecision/cpp_bin_float.hpp>

using boost::math::tools::engel_expansion;
using boost::multiprecision::cpp_bin_float_100;
using boost::math::constants::pi;

template<class Real>
void test_rationals()
{
    for (int64_t i = 1; i < 20; ++i)
    {
        Real x = Real(1)/Real(i);
        auto engel = engel_expansion<Real>(x);
        auto const & a = engel.digits();
        CHECK_EQUAL(size_t(1), a.size());
        CHECK_EQUAL(i, a.front());
    }

    Real x = Real(3)/Real(8);
    auto engel = engel_expansion<Real>(x);
    auto const & a = engel.digits();
    if (!CHECK_EQUAL(size_t(2), a.size())) {
        std::cerr << "  Wrong number of digits for x = 3/8 on type " << boost::core::demangle(typeid(Real).name()) << "\n";
        std::cerr << "  engel = " << engel << "\n";
    }
    CHECK_EQUAL(int64_t(3), a.front());
    CHECK_EQUAL(int64_t(8), a.back());
}

template<typename Real>
void test_well_known()
{
    std::cout << "Testing well-known Engel expansions on type " << boost::core::demangle(typeid(Real).name()) << "\n";
    using boost::math::constants::pi;
    using boost::math::constants::e;
    auto engel = engel_expansion(pi<Real>());
    // See: http://oeis.org/A006784/list
    std::vector<int64_t> expected{1,1,1,8,8,17,19,300,1991,2492,7236,10586,34588,
                                   63403,70637,1236467,5417668,5515697,5633167,
                                   7458122,9637848,9805775,41840855,58408380,
                                   213130873,424342175,2366457522,4109464489,
                                   21846713216,27803071890,31804388758,32651669133};
    auto a = engel.digits();
    // The last digit might be off somewhat, so don't test it:
    size_t n = (std::min)(a.size() - 1, expected.size());
    for(size_t i = 0; i < n; ++i)
    {
        if (!CHECK_EQUAL(expected[i], a[i]))
        {
            std::cerr << "  Engel expansion of pi differs from expected at digit " << i << " of " << a.size() - 1 << "\n";
        }
    }

    // The Engel expansion of e is {1, 1, 2, 3, 4, 5, 6, ...}
    engel = engel_expansion(e<Real>());
    a = engel.digits();
    CHECK_EQUAL(int64_t(1), a.front());
    int64_t m = a.size() - 1;
    for (int64_t i = 1; i < m; ++i)
    {
        CHECK_EQUAL(i, a[i]);
    }
}


int main()
{
    test_rationals<float>();
    test_rationals<double>();
    test_rationals<long double>();
    test_rationals<cpp_bin_float_100>();

    test_well_known<float>();
    test_well_known<double>();
    test_well_known<long double>();
    test_well_known<cpp_bin_float_100>();
    
    #ifdef BOOST_HAS_FLOAT128
    test_rationals<float128>();
    test_well_known<float128>();
    #endif
    return boost::math::test::report_errors();
}
