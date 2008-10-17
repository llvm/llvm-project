// Test frontend handling of __sync builtins.
// Modified from a gcc testcase.
// RUN: %llvmgcc -S %s -o - | grep atomic | count 200
// RUN: %llvmgcc -S %s -o - | grep p0i8 | count 50
// RUN: %llvmgcc -S %s -o - | grep p0i16 | count 50
// RUN: %llvmgcc -S %s -o - | grep p0i32 | count 50
// RUN: %llvmgcc -S %s -o - | grep volatile | count 8

// Currently this is implemented only for Alpha, X86, PowerPC.
// Add your target here if it doesn't work.
// PPC32 does not translate the long long variants, so fails this test.
// XFAIL: sparc|arm|ia64|powerpc

signed char sc;
unsigned char uc;
signed short ss;
unsigned short us;
signed int si;
unsigned int ui;
signed long long sll;
unsigned long long ull;

void test_op_ignore (void)
{
  (void) __sync_fetch_and_add (&sc, 1);
  (void) __sync_fetch_and_add (&uc, 1);
  (void) __sync_fetch_and_add (&ss, 1);
  (void) __sync_fetch_and_add (&us, 1);
  (void) __sync_fetch_and_add (&si, 1);
  (void) __sync_fetch_and_add (&ui, 1);
  (void) __sync_fetch_and_add (&sll, 1);
  (void) __sync_fetch_and_add (&ull, 1);

  (void) __sync_fetch_and_sub (&sc, 1);
  (void) __sync_fetch_and_sub (&uc, 1);
  (void) __sync_fetch_and_sub (&ss, 1);
  (void) __sync_fetch_and_sub (&us, 1);
  (void) __sync_fetch_and_sub (&si, 1);
  (void) __sync_fetch_and_sub (&ui, 1);
  (void) __sync_fetch_and_sub (&sll, 1);
  (void) __sync_fetch_and_sub (&ull, 1);

  (void) __sync_fetch_and_or (&sc, 1);
  (void) __sync_fetch_and_or (&uc, 1);
  (void) __sync_fetch_and_or (&ss, 1);
  (void) __sync_fetch_and_or (&us, 1);
  (void) __sync_fetch_and_or (&si, 1);
  (void) __sync_fetch_and_or (&ui, 1);
  (void) __sync_fetch_and_or (&sll, 1);
  (void) __sync_fetch_and_or (&ull, 1);

  (void) __sync_fetch_and_xor (&sc, 1);
  (void) __sync_fetch_and_xor (&uc, 1);
  (void) __sync_fetch_and_xor (&ss, 1);
  (void) __sync_fetch_and_xor (&us, 1);
  (void) __sync_fetch_and_xor (&si, 1);
  (void) __sync_fetch_and_xor (&ui, 1);
  (void) __sync_fetch_and_xor (&sll, 1);
  (void) __sync_fetch_and_xor (&ull, 1);

  (void) __sync_fetch_and_and (&sc, 1);
  (void) __sync_fetch_and_and (&uc, 1);
  (void) __sync_fetch_and_and (&ss, 1);
  (void) __sync_fetch_and_and (&us, 1);
  (void) __sync_fetch_and_and (&si, 1);
  (void) __sync_fetch_and_and (&ui, 1);
  (void) __sync_fetch_and_and (&sll, 1);
  (void) __sync_fetch_and_and (&ull, 1);

  (void) __sync_fetch_and_nand (&sc, 1);
  (void) __sync_fetch_and_nand (&uc, 1);
  (void) __sync_fetch_and_nand (&ss, 1);
  (void) __sync_fetch_and_nand (&us, 1);
  (void) __sync_fetch_and_nand (&si, 1);
  (void) __sync_fetch_and_nand (&ui, 1);
  (void) __sync_fetch_and_nand (&sll, 1);
  (void) __sync_fetch_and_nand (&ull, 1);
}

void test_fetch_and_op (void)
{
  sc = __sync_fetch_and_add (&sc, 11);
  uc = __sync_fetch_and_add (&uc, 11);
  ss = __sync_fetch_and_add (&ss, 11);
  us = __sync_fetch_and_add (&us, 11);
  si = __sync_fetch_and_add (&si, 11);
  ui = __sync_fetch_and_add (&ui, 11);
  sll = __sync_fetch_and_add (&sll, 11);
  ull = __sync_fetch_and_add (&ull, 11);

  sc = __sync_fetch_and_sub (&sc, 11);
  uc = __sync_fetch_and_sub (&uc, 11);
  ss = __sync_fetch_and_sub (&ss, 11);
  us = __sync_fetch_and_sub (&us, 11);
  si = __sync_fetch_and_sub (&si, 11);
  ui = __sync_fetch_and_sub (&ui, 11);
  sll = __sync_fetch_and_sub (&sll, 11);
  ull = __sync_fetch_and_sub (&ull, 11);

  sc = __sync_fetch_and_or (&sc, 11);
  uc = __sync_fetch_and_or (&uc, 11);
  ss = __sync_fetch_and_or (&ss, 11);
  us = __sync_fetch_and_or (&us, 11);
  si = __sync_fetch_and_or (&si, 11);
  ui = __sync_fetch_and_or (&ui, 11);
  sll = __sync_fetch_and_or (&sll, 11);
  ull = __sync_fetch_and_or (&ull, 11);

  sc = __sync_fetch_and_xor (&sc, 11);
  uc = __sync_fetch_and_xor (&uc, 11);
  ss = __sync_fetch_and_xor (&ss, 11);
  us = __sync_fetch_and_xor (&us, 11);
  si = __sync_fetch_and_xor (&si, 11);
  ui = __sync_fetch_and_xor (&ui, 11);
  sll = __sync_fetch_and_xor (&sll, 11);
  ull = __sync_fetch_and_xor (&ull, 11);

  sc = __sync_fetch_and_and (&sc, 11);
  uc = __sync_fetch_and_and (&uc, 11);
  ss = __sync_fetch_and_and (&ss, 11);
  us = __sync_fetch_and_and (&us, 11);
  si = __sync_fetch_and_and (&si, 11);
  ui = __sync_fetch_and_and (&ui, 11);
  sll = __sync_fetch_and_and (&sll, 11);
  ull = __sync_fetch_and_and (&ull, 11);

  sc = __sync_fetch_and_nand (&sc, 11);
  uc = __sync_fetch_and_nand (&uc, 11);
  ss = __sync_fetch_and_nand (&ss, 11);
  us = __sync_fetch_and_nand (&us, 11);
  si = __sync_fetch_and_nand (&si, 11);
  ui = __sync_fetch_and_nand (&ui, 11);
  sll = __sync_fetch_and_nand (&sll, 11);
  ull = __sync_fetch_and_nand (&ull, 11);
}

void test_op_and_fetch (void)
{
  sc = __sync_add_and_fetch (&sc, uc);
  uc = __sync_add_and_fetch (&uc, uc);
  ss = __sync_add_and_fetch (&ss, uc);
  us = __sync_add_and_fetch (&us, uc);
  si = __sync_add_and_fetch (&si, uc);
  ui = __sync_add_and_fetch (&ui, uc);
  sll = __sync_add_and_fetch (&sll, uc);
  ull = __sync_add_and_fetch (&ull, uc);

  sc = __sync_sub_and_fetch (&sc, uc);
  uc = __sync_sub_and_fetch (&uc, uc);
  ss = __sync_sub_and_fetch (&ss, uc);
  us = __sync_sub_and_fetch (&us, uc);
  si = __sync_sub_and_fetch (&si, uc);
  ui = __sync_sub_and_fetch (&ui, uc);
  sll = __sync_sub_and_fetch (&sll, uc);
  ull = __sync_sub_and_fetch (&ull, uc);

  sc = __sync_or_and_fetch (&sc, uc);
  uc = __sync_or_and_fetch (&uc, uc);
  ss = __sync_or_and_fetch (&ss, uc);
  us = __sync_or_and_fetch (&us, uc);
  si = __sync_or_and_fetch (&si, uc);
  ui = __sync_or_and_fetch (&ui, uc);
  sll = __sync_or_and_fetch (&sll, uc);
  ull = __sync_or_and_fetch (&ull, uc);

  sc = __sync_xor_and_fetch (&sc, uc);
  uc = __sync_xor_and_fetch (&uc, uc);
  ss = __sync_xor_and_fetch (&ss, uc);
  us = __sync_xor_and_fetch (&us, uc);
  si = __sync_xor_and_fetch (&si, uc);
  ui = __sync_xor_and_fetch (&ui, uc);
  sll = __sync_xor_and_fetch (&sll, uc);
  ull = __sync_xor_and_fetch (&ull, uc);

  sc = __sync_and_and_fetch (&sc, uc);
  uc = __sync_and_and_fetch (&uc, uc);
  ss = __sync_and_and_fetch (&ss, uc);
  us = __sync_and_and_fetch (&us, uc);
  si = __sync_and_and_fetch (&si, uc);
  ui = __sync_and_and_fetch (&ui, uc);
  sll = __sync_and_and_fetch (&sll, uc);
  ull = __sync_and_and_fetch (&ull, uc);

  sc = __sync_nand_and_fetch (&sc, uc);
  uc = __sync_nand_and_fetch (&uc, uc);
  ss = __sync_nand_and_fetch (&ss, uc);
  us = __sync_nand_and_fetch (&us, uc);
  si = __sync_nand_and_fetch (&si, uc);
  ui = __sync_nand_and_fetch (&ui, uc);
  sll = __sync_nand_and_fetch (&sll, uc);
  ull = __sync_nand_and_fetch (&ull, uc);
}

void test_compare_and_swap (void)
{
  sc = __sync_val_compare_and_swap (&sc, uc, sc);
  uc = __sync_val_compare_and_swap (&uc, uc, sc);
  ss = __sync_val_compare_and_swap (&ss, uc, sc);
  us = __sync_val_compare_and_swap (&us, uc, sc);
  si = __sync_val_compare_and_swap (&si, uc, sc);
  ui = __sync_val_compare_and_swap (&ui, uc, sc);
  sll = __sync_val_compare_and_swap (&sll, uc, sc);
  ull = __sync_val_compare_and_swap (&ull, uc, sc);

  ui = __sync_bool_compare_and_swap (&sc, uc, sc);
  ui = __sync_bool_compare_and_swap (&uc, uc, sc);
  ui = __sync_bool_compare_and_swap (&ss, uc, sc);
  ui = __sync_bool_compare_and_swap (&us, uc, sc);
  ui = __sync_bool_compare_and_swap (&si, uc, sc);
  ui = __sync_bool_compare_and_swap (&ui, uc, sc);
  ui = __sync_bool_compare_and_swap (&sll, uc, sc);
  ui = __sync_bool_compare_and_swap (&ull, uc, sc);
}

void test_lock (void)
{
  sc = __sync_lock_test_and_set (&sc, 1);
  uc = __sync_lock_test_and_set (&uc, 1);
  ss = __sync_lock_test_and_set (&ss, 1);
  us = __sync_lock_test_and_set (&us, 1);
  si = __sync_lock_test_and_set (&si, 1);
  ui = __sync_lock_test_and_set (&ui, 1);
  sll = __sync_lock_test_and_set (&sll, 1);
  ull = __sync_lock_test_and_set (&ull, 1);

  __sync_synchronize ();

  __sync_lock_release (&sc);
  __sync_lock_release (&uc);
  __sync_lock_release (&ss);
  __sync_lock_release (&us);
  __sync_lock_release (&si);
  __sync_lock_release (&ui);
  __sync_lock_release (&sll);
  __sync_lock_release (&ull);
}
