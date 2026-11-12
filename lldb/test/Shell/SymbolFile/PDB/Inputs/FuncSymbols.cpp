// Static function
namespace {
static long StaticFunction(int a)
{
  return 2;
}
}

// Inlined function
static inline int InlinedFunction(long a) { return 10; }

void FunctionCall()
{
  StaticFunction(1);
  InlinedFunction(1);
}
