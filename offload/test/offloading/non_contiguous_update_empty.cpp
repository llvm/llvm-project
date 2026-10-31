// RUN: %libomptarget-compile-run-and-check-generic
// UNSUPPORTED: apu

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

static void emptyContiguousUpdate(void *Base, int64_t Flags) noexcept {
  void *Begin = Base;
  int64_t Size = 0;
  __tgt_target_data_update(-1, 1, &Base, &Begin, &Size, &Flags);
}

int main() {
  // Fortran integer(4) array(4,5,3,2), selecting array(:,3:5,:,:).
  Dim Section[] = {{0, 2, 240}, {0, 3, 80}, {32, 3, 16}, {0, 4, 4}, {0, 4, 1}};
  // A zero innermost extent also makes all outer physical strides zero.
  // Merging these dimensions must not result in a zero-byte runtime copy.
  Dim EmptyInner[] = {{0, 2, 0}, {0, 3, 0}, {0, 5, 0}, {0, 0, 4}, {0, 4, 1}};
  Dim EmptyOuter[] = {
      {0, 0, 240}, {0, 3, 80}, {32, 3, 16}, {0, 4, 4}, {0, 4, 1}};

  int Data[120];
  for (int &Value : Data)
    Value = -1;
#pragma omp target data map(to : Data)
  {
    for (int &Value : Data)
      Value = 7;
    emptyContiguousUpdate(Data, To);
    emptyContiguousUpdate(Data, To | Present);
    update(Data, EmptyInner, To);
    update(Data, EmptyOuter, To);
#pragma omp target update from(Data)
    for (int Value : Data)
      assert(Value == -1);

    for (int &Value : Data)
      Value = 7;
    update(Data, Section, To);
#pragma omp target update from(Data)
    for (int I = 0; I < 120; ++I)
      assert(Data[I] == (I % 20 >= 8 ? 7 : -1));

    for (int I = 0; I < 120; ++I)
      Data[I] = I + 100;
#pragma omp target update to(Data)
    for (int &Value : Data)
      Value = -9;
    emptyContiguousUpdate(Data, From);
    emptyContiguousUpdate(Data, From | Present);
    update(Data, EmptyInner, From);
    update(Data, EmptyOuter, From);
    for (int Value : Data)
      assert(Value == -9);
    update(Data, Section, From);
    for (int I = 0; I < 120; ++I)
      assert(Data[I] == (I % 20 >= 8 ? I + 100 : -9));
  }

  // Empty sections must not look up an unmapped address, even with PRESENT.
  static int Unmapped;
  emptyContiguousUpdate(&Unmapped, To);
  emptyContiguousUpdate(&Unmapped, From);
  emptyContiguousUpdate(&Unmapped, To | Present);
  emptyContiguousUpdate(&Unmapped, From | Present);
  update(&Unmapped, EmptyInner, To | Present);
  update(&Unmapped, EmptyOuter, From | Present);

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
