#ifndef MATHTEST_INPUTGENERATOR_HPP
#define MATHTEST_INPUTGENERATOR_HPP

#include "llvm/ADT/ArrayRef.h"

namespace mathtest {

template <typename... InTypes> class InputGenerator {
public:
  virtual ~InputGenerator() noexcept = default;

  [[nodiscard]] virtual size_t
  fill(llvm::MutableArrayRef<InTypes>... Buffers) noexcept = 0;
};
} // namespace mathtest

#endif // MATHTEST_INPUTGENERATOR_HPP
