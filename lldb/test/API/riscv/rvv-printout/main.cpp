#include <limits.h>
#include <stdlib.h>

unsigned do_vlenb_read() {
  unsigned vlenb;
  asm volatile("csrr %[vlenb], vlenb" : [vlenb] "=r"(vlenb) : :);
  return vlenb;
}

unsigned do_vsetvli() {
  unsigned vl;
  asm volatile("vsetvli %[new_vl], x0, e8, m8, tu, mu" : [new_vl] "=r"(vl) : :);
  return vl;
}

char *STORAGE;

void do_vector_stuff() {
  unsigned vlenb_value = do_vlenb_read();
  STORAGE = (char *)calloc(1, vlenb_value * CHAR_BIT);
  do_vsetvli();
  asm volatile("vxor.vv v0,  v0,  v0\n\t"
               "vxor.vv v8,  v8,  v8\n\t"
               "vxor.vv v16, v16, v16\n\t"
               "vxor.vv v24, v24, v24\n\t"

               "vsetvli t0, x0, e8, m1, tu, mu\n\t"

               "vadd.vi v1, v1, 0x1\n\t"
               "vadd.vi v2, v1, 0x2\n\t"

               "vs1r.v  v1, (%[mem])\n\t"
               :
               : [mem] "r"(STORAGE)
               : "t0", "memory"); /* pre_vect_mem */
  asm volatile("vl1re8.v v2, (%0)" : : "r"(STORAGE) : "memory");
}

int main() {
  do_vector_stuff();
  return 0;
}
