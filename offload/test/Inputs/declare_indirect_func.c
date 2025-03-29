
int func() { return 42; }
#pragma omp declare target indirect to(func)
