#ifndef MATHTEST_DIM_HPP
#define MATHTEST_DIM_HPP

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace mathtest {

class Dim {
public:
  Dim() = delete;

  constexpr Dim(uint32_t X, uint32_t Y = 1, uint32_t Z = 1) noexcept
      : Data{X, Y, Z} {
    assert(X > 0 && Y > 0 && Z > 0 && "Dimensions must be positive");
  }

  constexpr Dim(std::initializer_list<uint32_t> Dimensions) noexcept
      : Data{1, 1, 1} {
    assert(Dimensions.size() <= 3 &&
           "The number of dimensions must be less than or equal to 3");

    std::size_t Index = 0;
    for (uint32_t DimValue : Dimensions) {
      Data[Index++] = DimValue;
    }

    assert(Data[0] > 0 && Data[1] > 0 && Data[2] > 0 &&
           "Dimensions must be positive");
  }

  [[nodiscard]] constexpr uint32_t
  operator[](std::size_t Index) const noexcept {
    assert(Index < 3 && "Index is out of range");
    return Data[Index];
  }

private:
  uint32_t Data[3];
};
} // namespace mathtest

#endif // MATHTEST_DIM_HPP
