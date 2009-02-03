// RUN: %llvmgcc -S -O1 -o - %s | grep icmp
// PR1678
extern void B (void);
static __typeof(B) A __attribute__ ((__weakref__("B")));
int active (void)
{
  static void *const p = __extension__ (void *) &A;
  return p != 0;
}
