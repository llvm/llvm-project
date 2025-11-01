#include <limits.h>
#include <stdint.h>
#include <stdlib.h>

unsigned do_vsetvli() {
  unsigned vl;
  asm volatile("vsetvli %[new_vl], x0, e8, m1, ta, ma" : [new_vl] "=r"(vl) : :);
  return vl;
}

void do_workload() {
  unsigned long long app_vtype;
  unsigned app_vl;
  unsigned app_vlenb;
  asm volatile(
      "csrr %[vtype], vtype\n\t"
      "csrr %[vl], vl\n\t"
      "csrr %[vlenb], vlenb\n\t"

      "vxor.vv v24, v16, v8\n\t"
      : [vtype] "=r"(app_vtype), [vl] "=r"(app_vl), [vlenb] "=r"(app_vlenb)
      :
      : "memory");

  asm volatile("nop\n\t"); /* workload_end */
}

int main() {
  do_vsetvli();
  for (int i = 0; i < 777; ++i)
    do_workload();
  return 0;
}
