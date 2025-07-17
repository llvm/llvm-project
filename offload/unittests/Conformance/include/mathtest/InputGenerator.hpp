#pragma once

#include "llvm/ADT/ArrayRef.h"

namespace mathtest {

template <typename... InTypes> class InputGenerator {
public:
  virtual ~InputGenerator() noexcept = default;

  [[nodiscard]] virtual size_t
  fill(llvm::MutableArrayRef<InTypes>... Buffers) noexcept = 0;
};
} // namespace mathtest
