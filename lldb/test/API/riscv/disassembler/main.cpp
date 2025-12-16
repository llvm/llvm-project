void do_zbb_stuff() {
  asm volatile("andn a2, a0, a1\n"
               "orn a2, a0, a1\n"
               "xnor a2, a0, a1\n"
               "rol a2, a0, a1\n"
               "ror a2, a0, a1\n");
}

int main() {
  do_zbb_stuff();
  return 0;
}
