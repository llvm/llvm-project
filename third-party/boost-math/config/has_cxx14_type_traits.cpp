// Copyright 2025 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <type_traits>
#include <cstdint>

using big_int = std::conditional_t<(sizeof(long) > sizeof(std::uint32_t)), long, std::uint32_t>;
static_assert(sizeof(big_int) >= sizeof(std::uint32_t), "big_int is too small");

int main()
{
    return 0;
}
