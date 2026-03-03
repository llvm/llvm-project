#include <limits.h>
#include <stdlib.h>

void zero_out_vec_ctx() {
  unsigned vl;
  asm volatile("vsetvli %[new_vl], x0, e8, m8, tu, mu\n\t"
               "vxor.vv v0,  v0,  v0\n\t"
               "vxor.vv v8,  v8,  v8\n\t"
               "vxor.vv v16, v16, v16\n\t"
               "vxor.vv v24, v24, v24\n\t"
               : [new_vl] "=r"(vl)
               :
               : "memory");
}

void do_wide_operations() {
  unsigned vl;
  asm volatile("vsetvli %[new_vl], x0, e8, m8, tu, mu\n\t"
               "vadd.vi v0,  v0,  0x1\n\t"
               : [new_vl] "=r"(vl)
               :
               : "memory");

  asm volatile("vadd.vi v24, v0, 0x2"); /* vect_op_v0_add1 */
  asm volatile("vadd.vi v16, v8, 0x2"); /* vect_op_v24_v0_add2 */
  asm volatile("vadd.vi v10, v9, 0x3"); /* vect_op_v16_v8_add2 */
  asm volatile("nop");                  /* vect_wide_op_end */
}

void do_controlled_vadd() {
  unsigned vl;
  asm volatile("vsetvli %[new_vl], x0, e8, m1, tu, mu" : [new_vl] "=r"(vl) : :);
  asm volatile("vadd.vv v2, v1, v0"); /* vect_control_vadd_start */
  asm volatile("nop");                /* controlled_vadd_done */
}

int main() {
  zero_out_vec_ctx();
  do_controlled_vadd();
  zero_out_vec_ctx();
  do_wide_operations();
  return 0;
}
