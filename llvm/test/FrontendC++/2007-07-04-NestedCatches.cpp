// RUN: %llvmgxx %s -S -O2 -o - | \
// RUN:   ignore grep {eh\.selector.*One.*Two.*Three.*Four.*Five.*Six.*null} | \
// RUN:     wc -l | grep {\[01\]}

extern void X(void);

struct One   {};
struct Two   {};
struct Three {};
struct Four  {};
struct Five  {};
struct Six   {};

static void A(void) throw ()
{
  X();
}

static void B(void) throw (Two)
{
  try { A(); } catch (One) {}
}

static void C(void) throw (Six, Five)
{
  try { B(); } catch (Three) {} catch (Four) {}
}

int main ()
{
  try { C(); } catch (...) {}
}
