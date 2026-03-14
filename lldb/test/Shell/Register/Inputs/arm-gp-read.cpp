#include <cstdint>

struct alignas(16) vec_t {
  uint64_t a, b;
};

int main() {
  constexpr uint32_t gprs[] = {
    0x00010203,
    0x10111213,
    0x20212223,
    0x30313233,
    0x40414243,
    0x50515253,
    0x60616263,
    0x70717273,
  };

  constexpr vec_t vecs[] = {
    { 0x0F0E0D0C0B0A0908, 0x1716151413121110, },
    { 0x100F0E0D0C0B0A09, 0x1817161514131211, },
    { 0x11100F0E0D0C0B0A, 0x1918171615141312, },
    { 0x1211100F0E0D0C0B, 0x1A19181716151413, },
  };
  const vec_t *vec_ptr = vecs;

  asm volatile(
    "ldrd     r0,  r1,  [%1]\n\t"
    "ldrd     r2,  r3,  [%1, #8]\n\t"
    "ldrd     r4,  r5,  [%1, #16]\n\t"
    "ldrd     r6,  r7,  [%1, #24]\n\t"
    "\n\t"
    "vld1.64  {q0, q1}, [%0]!\n\t"
    "vld1.64  {q2, q3}, [%0]!\n\t"
    "\n\t"
    "bkpt     #0\n\t"
    : "+r"(vec_ptr)
    : "r"(gprs)
    : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",
      "q0", "q1", "q2", "q3"
  );

  return 0;
}
