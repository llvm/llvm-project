//  (C) Copyright Nick Thompson 2018.
//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstddef>
#include <random>
#include <type_traits>
#include <vector>

namespace boost { namespace math {

// To stress test, set global_seed = 0, global_size = huge.
static constexpr std::size_t global_seed = 0;
static constexpr std::size_t global_size = 128;

template<typename T, typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
std::vector<T> generate_random_vector(std::size_t size, std::size_t seed)
{
    if (seed == 0)
    {
        std::random_device rd;
        seed = rd();
    }
    std::vector<T> v(size);

    std::mt19937_64 gen(seed);

    std::normal_distribution<T> dis(0, 1);
    for (auto& x : v)
    {
        x = dis(gen);
    }
    return v;
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
std::vector<T> generate_random_uniform_vector(std::size_t size, std::size_t seed, T lower_bound = T(0), T upper_bound = T(1))
{
    if (seed == 0)
    {
        std::random_device rd;
        seed = rd();
    }
    std::vector<T> v(size);

    std::mt19937 gen(seed);

    std::uniform_real_distribution<T> dis(lower_bound, upper_bound);

    for (auto& x : v)
    {
        x = dis(gen);
    }
    
    return v;
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
std::vector<T> generate_random_vector(std::size_t size, std::size_t seed, T mean, T stddev)
{
    if (seed == 0)
    {
        std::random_device rd;
        seed = rd();
    }
    std::vector<T> v(size);

    std::mt19937 gen(seed);

    std::normal_distribution<T> dis(mean, stddev);
    for (auto& x : v)
    {
        x = dis(gen);
    }
    return v;
}

template<typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
std::vector<T> generate_random_vector(std::size_t size, std::size_t seed)
{
    if (seed == 0)
    {
        std::random_device rd;
        seed = rd();
    }
    std::vector<T> v(size);

    std::mt19937_64 gen(seed);

    // Rescaling by larger than 2 is UB!
    std::uniform_int_distribution<T> dis(std::numeric_limits<T>::lowest()/2, (std::numeric_limits<T>::max)()/2);
    for (auto& x : v)
    {
        x = dis(gen);
    }
    return v;
}

}} // Namespaces
