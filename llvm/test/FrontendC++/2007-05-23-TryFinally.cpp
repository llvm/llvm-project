// RUN: %llvmgxx %s -S -emit-llvm -O2 -o - | ignore grep _Unwind_Resume | \
// RUN:   wc -l | grep {\[23\]}

struct One { };
struct Two { };

void handle_unexpected () {
  try
  {
    throw;
  }
  catch (One &)
  {
    throw Two ();
  }
}
