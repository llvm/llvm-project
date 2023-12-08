#include <cstdint>

int main() {
  constexpr uint64_t rax = 0x0102030405060708;
  constexpr uint64_t rbx = 0x1112131415161718;
  constexpr uint64_t rcx = 0x2122232425262728;
  constexpr uint64_t rdx = 0x3132333435363738;
  constexpr uint64_t rsp = 0x4142434445464748;
  constexpr uint64_t rbp = 0x5152535455565758;
  constexpr uint64_t rsi = 0x6162636465666768;
  constexpr uint64_t rdi = 0x7172737475767778;

  asm volatile(
    // save rsp & rbp
    "movq    %%rsp, %%r8\n\t"
    "movq    %%rbp, %%r9\n\t"
    "\n\t"
    "movq    %4, %%rsp\n\t"
    "movq    %5, %%rbp\n\t"
    "\n\t"
    "int3\n\t"
    "\n\t"
    // restore rsp & rbp
    "movq    %%r8, %%rsp\n\t"
    "movq    %%r9, %%rbp"
    :
    : "a"(rax), "b"(rbx), "c"(rcx), "d"(rdx), "i"(rsp), "i"(rbp), "S"(rsi),
      "D"(rdi)
    : "%r8", "%r9"
  );

  return 0;
}
