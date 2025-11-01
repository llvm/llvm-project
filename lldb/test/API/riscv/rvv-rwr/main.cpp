#include <limits.h>
#include <stdlib.h>

void do_vector_stuff() {
  unsigned vl;
  asm volatile("vsetvli %[new_vl], x0, e8, m8, ta, ma\n\t"

               "vxor.vv v0,  v0,  v0\n\t"
               "vxor.vv v8,  v8,  v8\n\t"
               "vxor.vv v16, v16, v16\n\t"
               "vxor.vv v24, v24, v24\n\t"

               "vadd.vi v0,  v0,  15\n\t"
               "vadd.vi v8,  v8,  15\n\t"
               "vadd.vi v16, v16, 15\n\t"
               "vadd.vi v24, v24, 15\n\t"

               "csrrsi zero, vxrm, 3\n\t"
               "csrrsi zero, vxsat, 1\n\t"
               : [new_vl] "=r"(vl)
               :
               : "memory");

  asm volatile("nop"); /* do_vector_stuff_end */
}

int main() {
  for (int i = 0; i < 777; ++i)
    do_vector_stuff();
  return 0;
}
