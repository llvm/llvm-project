#include <cstdio>

template <typename Ty>
bool ab(Ty a, Ty b) {
  return (a && b);
}
// CHECK: [[@LINE-2]]| 4| return

// CHECK: MC/DC Coverage for Decision{{[:]}}  50.00%
// CHECK: MC/DC Coverage for Decision{{[:]}}  50.00%
// CHECK: MC/DC Coverage for Decision{{[:]}}   0.00%
// CHECK-NOT: MC/DC Coverage for Decision{{[:]}}

// CHECK: _Z2abIbEbT_S0_{{[:]}}
// CHECK: | | MC/DC Coverage for Decision{{[:]}}  50.00%

// CHECK: _Z2abIxEbT_S0_{{[:]}}
// CHECK: | | MC/DC Coverage for Decision{{[:]}}  50.00%

// CHECK: Unexecuted instantiation{{[:]}} _Z2abIdEbT_S0_

template <bool C>
bool Cab(bool a, bool b) {
  return (a && b && C);
}
// CHECK: [[@LINE-2]]| 4| return

// CHECK: MC/DC Coverage for Decision{{[:]}}   0.00%
// CHECK: MC/DC Coverage for Decision{{[:]}}  50.00%
// CHECK-NOT: MC/DC Coverage for Decision{{[:]}}

// CHECK: _Z3CabILb0EEbbb{{[:]}}
// CHECK: | | MC/DC Coverage for Decision{{[:]}}   0.00%

// CHECK: _Z3CabILb1EEbbb{{[:]}}
// CHECK: | | MC/DC Coverage for Decision{{[:]}}  50.00%

// CHECK: [[@LINE+1]]| 1|int main
int main(int argc, char **argv) {
  printf("%d\n", Cab<false>(false, false));
  printf("%d\n", Cab<false>(true, true));
  printf("%d\n", Cab<true>(true, false));
  printf("%d\n", Cab<true>(true, true));
  printf("%d\n", ab(false, false));
  printf("%d\n", ab(true, true));
  printf("%d\n", ab(1LL, 0LL));
  printf("%d\n", ab(1LL, 1LL));
  if (argc == 2)
    printf("%d\n", ab(0.0, 0.0));
  return 0;
}
