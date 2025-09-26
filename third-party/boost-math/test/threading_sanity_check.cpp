// Copyright John Maddock, 2021
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/tools/config.hpp>

#if !defined(BOOST_MATH_NO_THREAD_LOCAL_WITH_NON_TRIVIAL_TYPES) && defined(BOOST_HAS_THREADS)

#include <vector>
#include <atomic>
#include <thread>
#include <random>
#include <algorithm>
#include <iostream>

std::atomic<int> counter{ 0 };


void thread_proc()
{
   static thread_local std::vector<double> list;

   std::default_random_engine rnd;
   std::uniform_real_distribution<> dist;

   for (unsigned i = 0; i < 1000; ++i)
   {
      list.push_back(dist(rnd));
      ++counter;
   }
   std::sort(list.begin(), list.end());
}

int main()
{
   std::thread f1(thread_proc);
   std::thread f2(thread_proc);
   std::thread f3(thread_proc);
   std::thread f4(thread_proc);
   std::thread f5(thread_proc);
   std::thread f6(thread_proc);

   f1.join();
   f2.join();
   f3.join();
   f4.join();
   f5.join();
   f6.join();

   std::cout << "Counter value was: " << counter << std::endl;

   return counter - 6000;
}

#else

int main() { return 0; }

#endif
