// inspect.cpp

// Copyright (c) 2006 Johan Rade

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//-------------------------------------

#include <cstring>

#include <iomanip>
#include <iostream>
#include <limits>

#ifndef BOOST_MATH_STANDALONE
#include <boost/endian.hpp>
#else
#include <boost/math/tools/config.hpp>
#endif

#include <boost/math/special_functions/next.hpp>  // for has_denorm_now

//------------------------------------------------------------------------------

bool is_big_endian()
{
    float x = 1.0f;
    unsigned char first_byte;
  memcpy(&first_byte, &x, 1);
    return first_byte != 0;
}

//------------------------------------------------------------------------------

void print_processor();
void print_endianness();
template<class T> void print_table();
template<class T> void print_row(const char* name, T val, bool ok = true);

//------------------------------------------------------------------------------

int main()
{
    std::cout << '\n';

  print_processor();

  print_endianness();

    std::cout << "---------- float --------------------\n\n";
    print_table<float>();

    std::cout << "---------- double -------------------\n\n";
    print_table<double>();

    std::cout << "---------- long double --------------\n\n";
    print_table<long double>();

    return 0;
}

//------------------------------------------------------------------------------

void print_processor()
{
#if defined(__i386) || defined(__i386__) || defined(_M_IX86) \
    || defined(__amd64) || defined(__amd64__)  || defined(_M_AMD64) \
    || defined(__x86_64) || defined(__x86_64__) || defined(_M_X64)

  std::cout << "Processor: x86 or x64\n\n";

#elif defined(__ia64) || defined(__ia64__) || defined(_M_IA64)

  std::cout << "Processor: ia64\n\n";

#elif defined(__powerpc) || defined(__powerpc__) || defined(__POWERPC__) \
    || defined(__ppc) || defined(__ppc__) || defined(__PPC__)

  std::cout << "Processor: PowerPC\n\n";

#elif defined(__m68k) || defined(__m68k__) \
    || defined(__mc68000) || defined(__mc68000__) \

  std::cout << "Processor: Motorola 68K\n\n";

#else

  std::cout << "Processor: Unknown\n\n";

#endif
}

void print_endianness()
{
    if(is_big_endian())
        std::cout << "This platform is big-endian.\n";
    else
        std::cout << "This platform is little-endian.\n";

#ifdef BOOST_BIG_ENDIAN
    std::cout << "BOOST_BIG_ENDIAN is defined.\n\n";
#endif
#if defined BOOST_LITTLE_ENDIAN
    std::cout << "BOOST_LITTTLE_ENDIAN is defined.\n\n";
#endif
}

//..............................................................................

template<class T> void print_table()
{
    print_row("0", (T)0);
    print_row("sn.min", std::numeric_limits<T>::denorm_min(),
          boost::math::detail::has_denorm_now<T>());
    print_row("-sn.min", -std::numeric_limits<T>::denorm_min(),
          boost::math::detail::has_denorm_now<T>());
    print_row("n.min/256", (std::numeric_limits<T>::min)()/256);
    print_row("n.min/2", (std::numeric_limits<T>::min)()/2);
    print_row("-n.min/2", -(std::numeric_limits<T>::min)()/2);
    print_row("n.min", (std::numeric_limits<T>::min)());
    print_row("1", (T)1);
    print_row("3/4", (T)3/(T)4);
    print_row("4/3", (T)4/(T)3);
    print_row("max", (std::numeric_limits<T>::max)());
    print_row("inf", std::numeric_limits<T>::infinity(),
          std::numeric_limits<T>::has_infinity);
    print_row("q.nan", std::numeric_limits<T>::quiet_NaN(),
          std::numeric_limits<T>::has_quiet_NaN);
    print_row("s.nan", std::numeric_limits<T>::signaling_NaN(),
          std::numeric_limits<T>::has_signaling_NaN);

    std::cout << "\n\n";
}

template<class T>
void print_row(const char* name, T val, bool ok)
{
    std::cout << std::left << std::setw(10) << name << std::right;

    std::cout << std::hex << std::setfill('0');

    if(ok) {
        if(is_big_endian()) {
      for(size_t i = 0; i < sizeof(T); ++i) {
        unsigned char c = *(reinterpret_cast<unsigned char*>(&val) + i);
                std::cout << std::setw(2)
                    << static_cast<unsigned int>(c) << ' ';
      }
        }
        else {
      for(size_t i = sizeof(T) - 1; i < sizeof(T); --i) {
        unsigned char c = *(reinterpret_cast<unsigned char*>(&val) + i);
                std::cout << std::setw(2)
                    << static_cast<unsigned int>(c) << ' ';
      }
        }
    }
    else {
        for(size_t i = 0; i < sizeof(T); ++i)
            std::cout << "-- ";
    }

    std::cout << '\n';
    std::cout << std::dec << std::setfill(' ');
}

/*

Sample output on an AMD Quadcore running MSVC 10

  Processor: x86 or x64

  This platform is little-endian.
  BOOST_LITTTLE_ENDIAN is defined.

  ---------- float --------------------

  0         00 00 00 00
  sn.min    00 00 00 01
  -sn.min   80 00 00 01
  n.min/256 00 00 80 00
  n.min/2   00 40 00 00
  -n.min/2  80 40 00 00
  n.min     00 80 00 00
  1         3f 80 00 00
  3/4       3f 40 00 00
  4/3       3f aa aa ab
  max       7f 7f ff ff
  inf       7f 80 00 00
  q.nan     7f c0 00 00
  s.nan     7f c0 00 01


  ---------- double -------------------

  0         00 00 00 00 00 00 00 00
  sn.min    00 00 00 00 00 00 00 01
  -sn.min   80 00 00 00 00 00 00 01
  n.min/256 00 00 10 00 00 00 00 00
  n.min/2   00 08 00 00 00 00 00 00
  -n.min/2  80 08 00 00 00 00 00 00
  n.min     00 10 00 00 00 00 00 00
  1         3f f0 00 00 00 00 00 00
  3/4       3f e8 00 00 00 00 00 00
  4/3       3f f5 55 55 55 55 55 55
  max       7f ef ff ff ff ff ff ff
  inf       7f f0 00 00 00 00 00 00
  q.nan     7f f8 00 00 00 00 00 00
  s.nan     7f f8 00 00 00 00 00 01


  ---------- long double --------------

  0         00 00 00 00 00 00 00 00
  sn.min    00 00 00 00 00 00 00 01
  -sn.min   80 00 00 00 00 00 00 01
  n.min/256 00 00 10 00 00 00 00 00
  n.min/2   00 08 00 00 00 00 00 00
  -n.min/2  80 08 00 00 00 00 00 00
  n.min     00 10 00 00 00 00 00 00
  1         3f f0 00 00 00 00 00 00
  3/4       3f e8 00 00 00 00 00 00
  4/3       3f f5 55 55 55 55 55 55
  max       7f ef ff ff ff ff ff ff
  inf       7f f0 00 00 00 00 00 00
  q.nan     7f f8 00 00 00 00 00 00
  s.nan     7f f8 00 00 00 00 00 01

  */
