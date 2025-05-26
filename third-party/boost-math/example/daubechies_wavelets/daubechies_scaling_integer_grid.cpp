/*
 * Copyright Nick Thompson, John Maddock 2020
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 * 
 * We want to use c-style asserts in examples:
 * 
 * boost-no-inspect
 */

#define BOOST_MATH_GENERATE_DAUBECHIES_GRID

#include <iostream>
#include <vector>
#include <numeric>
#include <list>
#include <cmath>
#include <cassert>
#include <fstream>
#include <Eigen/Eigenvalues>
#include <boost/hana/for_each.hpp>
#include <boost/hana/ext/std/integer_sequence.hpp>
#include <boost/core/demangle.hpp>
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
#endif
#include <boost/math/constants/constants.hpp>
#include <boost/math/filters/daubechies.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

typedef boost::multiprecision::number<boost::multiprecision::cpp_bin_float<237, boost::multiprecision::backends::digit_base_2, std::allocator<char>, std::int32_t, -262142, 262143>,  boost::multiprecision::et_off> octuple_type;

#ifdef BOOST_HAS_FLOAT128
typedef boost::multiprecision::float128 float128_t;
#else
typedef boost::multiprecision::cpp_bin_float_quad float128_t;
#endif

template<class Real, int p>
std::list<std::vector<Real>> integer_grid()
{
    std::cout << std::setprecision(std::numeric_limits<Real>::digits10 + 3);
    using std::abs;
    using std::sqrt;
    using std::pow;
    std::list<std::vector<Real>> grids;

    auto c = boost::math::filters::daubechies_scaling_filter<Real, p>();
    for (auto & x : c)
    {
        x *= boost::math::constants::root_two<Real>();
    }
    std::cout << "\n\nTaps in filter = " << c.size() << "\n";


    Eigen::Matrix<Real, 2*p - 2, 2*p-2> A;
    for (int j = 0; j < 2*p-2; ++j) {
        for (int k = 0; k < 2*p-2; ++k) {
            if ( (2*j-k + 1) < 0 || (2*j - k  + 1) >= 2*p)
            {
                A(j,k) = 0;
            }
            else {
                A(j,k) = c[2*j - k + 1];
            }
        }
    }

    Eigen::EigenSolver<decltype(A)> es(A);

    auto complex_eigs = es.eigenvalues();

    std::vector<Real> eigs(complex_eigs.size(), std::numeric_limits<Real>::quiet_NaN());

    std::cout << "Eigenvalues = {";
    for (long i = 0; i < complex_eigs.size(); ++i) {
        assert(abs(complex_eigs[i].imag()) < std::numeric_limits<Real>::epsilon());
        eigs[i] = complex_eigs[i].real();
        std::cout << eigs[i] << ", ";
    }
    std::cout << "}\n";

    // Eigen does not sort the eigenpairs by any criteria on the eigenvalues.
    // In any case, even if it did, some of the eigenpairs do not correspond to derivatives anyway.
    for (size_t j = 0; j < eigs.size(); ++j) {
        auto f = [&](Real x) {
                 return abs(x - Real(1)/Real(1 << j) ) < sqrt(std::numeric_limits<Real>::epsilon());
                 };
        auto it = std::find_if(eigs.begin(), eigs.end(), f);
        if (it == eigs.end()) {
            std::cout << "couldn't find eigenvalue " << Real(1)/Real(1 << j) << "\n";
            continue;
        }
        size_t idx = std::distance(eigs.begin(), it);
        std::cout << "Eigenvector for derivative " << j << " is at index " << idx << "\n";
        auto eigenvector_matrix = es.eigenvectors();
        auto complex_eigenvec = eigenvector_matrix.col(idx);

        std::vector<Real> eigenvec(complex_eigenvec.size() + 2, std::numeric_limits<Real>::quiet_NaN());
        eigenvec[0] = 0;
        eigenvec[eigenvec.size()-1] = 0;
        for (size_t i = 0; i < eigenvec.size() - 2; ++i) {
            assert(abs(complex_eigenvec[i].imag()) < std::numeric_limits<Real>::epsilon());
            eigenvec[i+1] = complex_eigenvec[i].real();
        }

        Real sum = 0;
        for(size_t k = 1; k < eigenvec.size(); ++k) {
            sum += pow(k, j)*eigenvec[k];
        }

        Real alpha = pow(-1, j)*boost::math::factorial<Real>(j)/sum;

        for (size_t i = 1; i < eigenvec.size(); ++i) {
            eigenvec[i] *= alpha;
        }


        std::cout << "Eigenvector = {";
        for (size_t i = 0; i < eigenvec.size() -1; ++i) {
            std::cout << eigenvec[i] << ", ";
        }
        std::cout << eigenvec[eigenvec.size()-1] << "}\n";

        sum = 0;
        for(size_t k = 1; k < eigenvec.size(); ++k) {
            sum += pow(k, j)*eigenvec[k];
        }

        std::cout << "Moment sum = " << sum << ", expected = " << pow(-1, j)*boost::math::factorial<Real>(j) << "\n";

        assert(abs(sum - pow(-1, j)*boost::math::factorial<Real>(j))/abs(pow(-1, j)*boost::math::factorial<Real>(j)) < sqrt(std::numeric_limits<Real>::epsilon()));

        grids.push_back(eigenvec);
    }


    return grids;
}

template<class Real, int p>
void write_grid(std::ofstream & fs)
{
    auto grids = integer_grid<Real, p>();
    size_t j = 0;
    fs << std::setprecision(std::numeric_limits< boost::multiprecision::cpp_bin_float_quad>::max_digits10);
    for (auto it = grids.begin(); it != grids.end(); ++it) 
    {
       auto const& grid = *it;
       fs << "template <typename Real> struct daubechies_scaling_integer_grid_imp <Real, " << p << ", ";
      fs << j << "> { static inline constexpr std::array<Real, " << grid.size() << "> value = { ";
      for (size_t i = 0; i < grid.size() -1; ++i){
        fs << "C_(" << static_cast<float128_t>(grid[i]) << "), ";
      }
      fs << "C_(" << static_cast<float128_t>(grid[grid.size()-1]) << ") }; };\n";
      ++j;
    }
}

int main()
{
    constexpr const size_t p_max = 18;
    std::ofstream fs{"daubechies_scaling_integer_grid.hpp"};
    fs << "/*\n"
       << " * Copyright Nick Thompson, John Maddock 2020\n"
       << " * Use, modification and distribution are subject to the\n"
       << " * Boost Software License, Version 1.0. (See accompanying file\n"
       << " * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)\n"
       << " */\n"
       << "// THIS FILE GENERATED BY EXAMPLE/DAUBECHIES_SCALING_INTEGER_GRID.CPP, DO NOT EDIT.\n"
       << "#ifndef BOOST_MATH_DAUBECHIES_SCALING_INTEGER_GRID_HPP\n"
       << "#define BOOST_MATH_DAUBECHIES_SCALING_INTEGER_GRID_HPP\n"
       << "#include <array>\n"
       << "#include <cfloat>\n"
       << "#include <boost/config.hpp>\n"
       << "/*\n"
       << "In order to keep the character count as small as possible and speed up\n"
       << "compiler parsing times, we define a macro C_ which appends an appropriate\n"
       << "suffix to each literal, and then casts it to type Real.\n"
       << "The suffix is as follows:\n\n"
       << "* Q, when we have __float128 support.\n"
       << "* L, when we have either 80 or 128 bit long doubles.\n"
       << "* Nothing otherwise.\n"
       << "*/\n\n"
       << "#ifdef BOOST_HAS_FLOAT128\n"
       << "#  define C_(x) static_cast<Real>(x##Q)\n"
       << "#elif (LDBL_MANT_DIG > DBL_MANT_DIG)\n"
       << "#  define C_(x) static_cast<Real>(x##L)\n"
       << "#else\n"
       << "#  define C_(x) static_cast<Real>(x)\n"
       << "#endif\n\n"
       << "namespace boost::math::detail {\n\n"
       << "template <typename Real, int p, int order> struct daubechies_scaling_integer_grid_imp;\n\n";

    fs << std::hexfloat << std::setprecision(std::numeric_limits<boost::multiprecision::cpp_bin_float_quad>::max_digits10);

    boost::hana::for_each(std::make_index_sequence<p_max>(), [&](auto idx){
        write_grid<octuple_type, idx+2>(fs);
    });

    fs << "\n\ntemplate <typename Real, unsigned p, unsigned order>\n"
       << "constexpr inline std::array<Real, 2*p> daubechies_scaling_integer_grid()\n"
       << "{\n"
       << "    static_assert(sizeof(Real) <= 16, \"Integer grids only computed up to 128 bits of precision.\");\n"
       << "    static_assert(p <= " << p_max + 1 << ", \"Integer grids only implemented up to " << p_max + 1 << ".\");\n"
       << "    static_assert(p > 1, \"Integer grids only implemented for p >= 2.\");\n"
       << "    return daubechies_scaling_integer_grid_imp<Real, p, order>::value;\n"
       << "}\n\n";

    fs << "} // namespaces\n";
    fs << "#endif\n";
    fs.close();

    return 0;
}
