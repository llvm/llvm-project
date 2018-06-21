//===---------------------------------------------------------------------===//
// Testcases that crash the X86 backend because they aren't implemented
//===---------------------------------------------------------------------===//

These are cases we know the X86 backend doesn't handle.  Patches are welcome
and appreciated, because no one has signed up to implemented these yet.
Implementing these would allow elimination of the corresponding intrinsics,
which would be great.

1) vector shifts
2) vector comparisons
3) vector fp<->int conversions: PR2683, PR2684, PR2685, PR2686, PR2688
4) bitcasts from vectors to scalars: PR2804
5) llvm.atomic.cmp.swap.i128.p0i128: PR3462
