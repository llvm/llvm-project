//  (C) Copyright Nick Thompson 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <boost/math/special_functions/daubechies_scaling.hpp>
#include <boost/core/demangle.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/ext/std/integer_sequence.hpp>

int main()
{
    boost::hana::for_each(std::make_index_sequence<18>(),
    [](auto i) {
        std::cout << std::right;
        auto daub = boost::math::daubechies_scaling<float, i+2>();
        std::cout << "The Daubechies " << std::setw(2) <<  i + 2 << " scaling function occupies " 
                  << std::setw(12) << daub.bytes()/1000.0 << " kilobytes in relative accuracy mode in "
                  << boost::core::demangle(typeid(float).name()) << " precision\n";
    });

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

    boost::hana::for_each(std::make_index_sequence<18>(),
    [](auto i) {
        std::cout << std::right;
        auto daub = boost::math::daubechies_scaling<float, i+2>(-2);
        std::cout << "The Daubechies " << std::setw(2) <<  i + 2 << " scaling function occupies " 
                  << std::setw(12) << daub.bytes()/1000.0 << " kilobytes in absolute accuracy mode in "
                  << boost::core::demangle(typeid(float).name()) << " precision\n";
    });

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;


    boost::hana::for_each(std::make_index_sequence<18>(),
    [](auto i) {
        std::cout << std::right;
        auto daub = boost::math::daubechies_scaling<double, i+2>();
        std::cout << "The Daubechies " << std::setw(2) <<  i + 2 << " scaling function occupies " 
                  << std::setw(12) << daub.bytes()/1000.0 << " kilobytes in relative accuracy mode in "
                  << boost::core::demangle(typeid(double).name()) << " precision\n";
    });

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

    boost::hana::for_each(std::make_index_sequence<18>(),
    [](auto i) {
        std::cout << std::right;
        auto daub = boost::math::daubechies_scaling<double, i+2>(-2);
        std::cout << "The Daubechies " << std::setw(2) <<  i + 2 << " scaling function occupies " 
                  << std::setw(12) << daub.bytes()/1000.0 << " kilobytes in absolute accuracy mode in "
                  << boost::core::demangle(typeid(double).name()) << " precision\n";
    });


}
