// RUN: %libomptarget-compile-run-and-check-generic
// UNSUPPORTED: apu

// Use the runtime API for descriptors that Clang cannot emit from valid C/C++:
// zero physical strides from zero-sized Fortran dimensions, and synthetic
// oversized counts without an allocation or a data-motion flag. The ordinary
// array-section cases are covered by pragmas in strided_update/ and
// mapping/present/zero_length_array_section_update.c.

#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdio>

// Data structure and flags from the OpenMP runtime ABI.
struct Dim {
  uint64_t Offset;
  uint64_t Count;
  uint64_t Stride;
};

constexpr int64_t To = 0x1;
constexpr int64_t From = 0x2;
constexpr int64_t Present = 0x1000;
constexpr int64_t NonContiguous = 0x100000000000;

extern "C" void __tgt_target_data_update(int64_t Device, int32_t NumArgs,
                                         void **Bases, void **Begins,
                                         int64_t *Sizes,
                                         int64_t *Types) noexcept;

template <unsigned N>
static void update(void *Base, Dim (&Dims)[N], int64_t Flags) noexcept {
  void *Begin = Dims;
  int64_t Size = N;
  Flags |= NonContiguous;
  __tgt_target_data_update(-1, 1, &Base, &Begin, &Size, &Flags);
}

int main() {
  // A zero innermost extent also makes all outer physical strides zero.
  // Merging these dimensions must not result in a zero-byte runtime copy.
  Dim EmptyInner[] = {{0, 2, 0}, {0, 3, 0}, {0, 5, 0}, {0, 0, 4}, {0, 4, 1}};

  int Data[120];
  for (int &Value : Data)
    Value = -1;
#pragma omp target data map(to : Data)
  {
    for (int &Value : Data)
      Value = 7;
    update(Data, EmptyInner, To);
#pragma omp target update from(Data)
    for (int Value : Data)
      assert(Value == -1);

    for (int &Value : Data)
      Value = 7;
    update(Data, EmptyInner, From);
    for (int Value : Data)
      assert(Value == 7);
  }

  // Empty sections must not look up an unmapped address, even with PRESENT.
  static int Unmapped;
  update(&Unmapped, EmptyInner, To | Present);
  update(&Unmapped, EmptyInner, From | Present);

  // An already contiguous final dimension needs one visit regardless of its
  // count. No TO/FROM flag or allocation is needed to check the traversal.
  Dim Large[] = {{0, uint64_t(UINT_MAX) + 1, 1}};
  update(&Unmapped, Large, 0);

  // Skip an empty section before traversing a potentially enormous outer
  // dimension. This also checks that an inner zero suppresses PRESENT checks.
  Dim LargeEmpty[] = {{0, uint64_t(UINT_MAX) + 1, 8}, {0, 0, 4}, {0, 4, 1}};
  update(&Unmapped, LargeEmpty, To | Present);

  std::puts("PASS");
  // CHECK: PASS
}
