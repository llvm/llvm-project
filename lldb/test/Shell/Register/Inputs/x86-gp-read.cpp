#include <cstdint>

int main() {
  constexpr uint32_t eax = 0x05060708;
  constexpr uint32_t ebx = 0x15161718;
  constexpr uint32_t ecx = 0x25262728;
  constexpr uint32_t edx = 0x35363738;
  constexpr uint32_t esp = 0x45464748;
  constexpr uint32_t ebp = 0x55565758;
  constexpr uint32_t esi = 0x65666768;
  constexpr uint32_t edi = 0x75767778;

  asm volatile(
    // save esp & ebp
    "movd    %%esp, %%mm0\n\t"
    "movd    %%ebp, %%mm1\n\t"
    "\n\t"
    "movl    %4, %%esp\n\t"
    "movl    %5, %%ebp\n\t"
    "\n\t"
    "int3\n\t"
    "\n\t"
    // restore esp & ebp
    "movd    %%mm0, %%esp\n\t"
    "movd    %%mm1, %%ebp\n\t"
    :
    : "a"(eax), "b"(ebx), "c"(ecx), "d"(edx), "i"(esp), "i"(ebp), "S"(esi),
      "D"(edi)
    : "%mm0", "%mm1"
  );

  return 0;
}
