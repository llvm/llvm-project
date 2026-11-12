#include <cinttypes>
#include <cstdint>
#include <cstdio>

int main() {
  constexpr uint32_t fill = 0x0F0F0F0F;

  uint32_t eax, ebx, ecx, edx, esi, edi;
  // need to use 64-bit types due to bug in clang
  // https://bugs.llvm.org/show_bug.cgi?id=41748
  uint64_t esp, ebp;

  asm volatile(
    // save esp & ebp
    "movd    %%esp, %%mm0\n\t"
    "movd    %%ebp, %%mm1\n\t"
    "\n\t"
    "movl    %8, %%eax\n\t"
    "movl    %8, %%ebx\n\t"
    "movl    %8, %%ecx\n\t"
    "movl    %8, %%edx\n\t"
    "movl    %8, %%esp\n\t"
    "movl    %8, %%ebp\n\t"
    "movl    %8, %%esi\n\t"
    "movl    %8, %%edi\n\t"
    "\n\t"
    "int3\n\t"
    "\n\t"
    // copy new values of esp & ebp
    "movd    %%esp, %4\n\t"
    "movd    %%ebp, %5\n\t"
    // restore saved esp & ebp
    "movd    %%mm0, %%esp\n\t"
    "movd    %%mm1, %%ebp\n\t"
    : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx), "=y"(esp), "=y"(ebp),
      "=S"(esi), "=D"(edi)
    : "i"(fill)
    : "%mm0", "%mm1"
  );

  printf("eax = 0x%08" PRIx32 "\n", eax);
  printf("ebx = 0x%08" PRIx32 "\n", ebx);
  printf("ecx = 0x%08" PRIx32 "\n", ecx);
  printf("edx = 0x%08" PRIx32 "\n", edx);
  printf("esp = 0x%08" PRIx32 "\n", static_cast<uint32_t>(esp));
  printf("ebp = 0x%08" PRIx32 "\n", static_cast<uint32_t>(ebp));
  printf("esi = 0x%08" PRIx32 "\n", esi);
  printf("edi = 0x%08" PRIx32 "\n", edi);

  return 0;
}
