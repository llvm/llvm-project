//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <iterator>
#include <utility>
#include <algorithm>
#include <type_traits>
#include <initializer_list>
#include <boost/math/special_functions/logaddexp.hpp>

namespace boost { namespace math {

// https://nhigham.com/2021/01/05/what-is-the-log-sum-exp-function/
// See equation (#)
template <typename ForwardIterator, typename Real = typename std::iterator_traits<ForwardIterator>::value_type>
Real logsumexp(ForwardIterator first, ForwardIterator last)
{
    using std::exp;
    using std::log1p;
    
    const auto elem = std::max_element(first, last);
    const Real max_val = *elem;

    Real arg = 0;
    while (first != last)
    {
        if (first != elem) 
        {
            arg += exp(*first - max_val);
        }

        ++first;
    }

    return max_val + log1p(arg);
}

template <typename Container, typename Real = typename Container::value_type>
inline Real logsumexp(const Container& c)
{
    return logsumexp(std::begin(c), std::end(c));
}

template <typename... Args, typename Real = typename std::common_type<Args...>::type, 
          typename std::enable_if<std::is_floating_point<Real>::value, bool>::type = true>
inline Real logsumexp(Args&& ...args)
{
    std::initializer_list<Real> list {std::forward<Args>(args)...};
    
    if(list.size() == 2)
    {
        return logaddexp(*list.begin(), *std::next(list.begin()));
    }
    return logsumexp(list.begin(), list.end());
}

}} // Namespace boost::math
