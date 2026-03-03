unsigned do_vlenb_read() {
  unsigned vlenb;
  asm volatile("csrr %[vlenb], vlenb" : [vlenb] "=r"(vlenb) : :);
  return vlenb;
}

unsigned do_vsetvli() {
  unsigned vl;
  asm volatile("vsetvli %[new_vl], x0, e8, m1, ta, ma" : [new_vl] "=r"(vl) : :);
  return vl;
}

#ifdef READ_VLENB_BEFORE_MAIN
unsigned VLENB = do_vlenb_read();
#endif // READ_VLENB_BEFORE_MAIN

#ifdef SET_VSETVLI_BEFORE_MAIN
unsigned VL = do_vsetvli();
#endif // SET_VSETVLI_BEFORE_MAIN

int main() { return 0; }
