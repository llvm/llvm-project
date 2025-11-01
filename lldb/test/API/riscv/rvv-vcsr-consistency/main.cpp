#include <vector>

enum VLMUL {
  LMUL1 = 0,
  LMUL2 = 1,
  LMUL4 = 2,
  LMUL8 = 3,
  LMUL_F8 = 5,
  LMUL_F4 = 6,
  LMUL_F2 = 7
};

enum SEW {
  SEW8 = 0,
  SEW16 = 1,
  SEW32 = 2,
  SEW64 = 3,
};

unsigned do_vsetvli() {
  unsigned vl;
  asm volatile("vsetvli %[new_vl], x0, e8, m1, ta, ma" : [new_vl] "=r"(vl) : :);
  return vl;
}

unsigned do_vsetv(unsigned vl, VLMUL vlmul, SEW vsew, unsigned vta,
                  unsigned vma) {
  unsigned vtype =
      (unsigned)vlmul | ((unsigned)vsew << 3) | (vta << 6) | (vma << 7);
  asm volatile("vsetvl %[new_vl], %[new_vl], %[vtype]"
               : [new_vl] "+r"(vl)
               : [vtype] "r"(vtype)
               :);
  return vl; /* vsetvl_done */
}

void do_vsetv_test() {
  std::vector<VLMUL> vlmul = {
      VLMUL::LMUL1,   VLMUL::LMUL2,   VLMUL::LMUL4,   VLMUL::LMUL8,
      VLMUL::LMUL_F8, VLMUL::LMUL_F4, VLMUL::LMUL_F2,
  };
  std::vector<SEW> vsew = {
      SEW::SEW8,
      SEW::SEW16,
      SEW::SEW32,
      SEW::SEW64,
  };

  for (auto vlmul : vlmul)
    for (auto sew : vsew)
      for (int vta = 0; vta < 2; ++vta)
        for (int vma = 0; vma < 2; ++vma)
          for (int vl = 1; vl < 3; ++vl)
            do_vsetv(vl, vlmul, sew, vta, vma);
}

void do_vcsr_test() {
  asm volatile("csrw vxrm, %[rnd_m]" : : [rnd_m] "i"(0) :);
  asm volatile("csrw vxrm, %[rnd_m]" : : [rnd_m] "i"(1) :);  /* vxrm_0 */
  asm volatile("csrw vxrm, %[rnd_m]" : : [rnd_m] "i"(2) :);  /* vxrm_1 */
  asm volatile("csrw vxrm, %[rnd_m]" : : [rnd_m] "i"(3) :);  /* vxrm_2 */
  asm volatile("csrw vxsat, %[vxsat]" : : [vxsat] "i"(1) :); /* vxrm_3 */
  asm volatile("csrw vxrm, %[rnd_m]" : : [rnd_m] "i"(0) :);  /* vxrm_0_again */
  unsigned vtype = -1;
  unsigned vl = -1;
  asm volatile("vsetvl %[new_vl], %[new_vl], %[vtype]" /* vcsr_done */
               : [new_vl] "+r"(vl), [vtype] "=r"(vtype)
               :
               :);
}

int main() {
  do_vsetvli();
  /* rvv_initialized */
  do_vsetv_test();
  do_vcsr_test();
  /* do_vsetv_test_end */
  return 0;
}
