# RUN: llvm-mca -mtriple=armv7 -mcpu=cortex-a9 -all-views=false -resource-pressure -iterations=2 < %s | FileCheck %s
# RUN: llvm-mca -mtriple=armv7 -mcpu=cortex-a9 -all-views=false -resource-pressure -instruction-tables < %s | FileCheck %s

.syntax unified
vld1.8 {d0}, [r0]
vld1.8 {d0-d1}, [r0]
vld1.8 {d0-d2}, [r0]
vld1.8 {d0-d3}, [r0]
vld1.8 {d0-d1}, [r0]!
vld1.8 {d0-d1}, [r0], r1
vld2.8 {d0-d1}, [r0]
vld2.8 {d0-d3}, [r0]
vld3.8 {d0-d2}, [r0]
vld4.8 {d0-d3}, [r0]
vld1.8 {d0[1]}, [r0]
vld2.8 {d0[1], d1[1]}, [r0]
vld3.8 {d0[1], d1[1], d2[1]}, [r0]
vld4.8 {d0[1], d1[1], d2[1], d3[1]}, [r0]
vld1.8 {d0[]}, [r0]
vld2.8 {d0[], d1[]}, [r0]
vld3.8 {d0[], d1[], d2[]}, [r0]
vld4.8 {d0[], d1[], d2[], d3[]}, [r0]
vst1.8 {d0}, [r0]
vst1.8 {d0-d1}, [r0]
vst1.8 {d0-d2}, [r0]
vst1.8 {d0-d3}, [r0]
vst1.8 {d0-d1}, [r0]!
vst1.8 {d0-d1}, [r0], r1
vst2.8 {d0-d1}, [r0]
vst2.8 {d0-d3}, [r0]
vst3.8 {d0-d2}, [r0]
vst4.8 {d0-d3}, [r0]
vst1.8 {d0[1]}, [r0]
vst2.8 {d0[1], d1[1]}, [r0]
vst3.8 {d0[1], d1[1], d2[1]}, [r0]
vst4.8 {d0[1], d1[1], d2[1], d3[1]}, [r0]
vldr d0, [r0]
vstr d0, [r0]

# CHECK: Resources:
# CHECK-NEXT: [0]   - A9UnitAGU
# CHECK-NEXT: [1.0] - A9UnitALU
# CHECK-NEXT: [1.1] - A9UnitALU
# CHECK-NEXT: [2]   - A9UnitB
# CHECK-NEXT: [3]   - A9UnitFP
# CHECK-NEXT: [4]   - A9UnitLS
# CHECK-NEXT: [5]   - A9UnitMul
# CHECK: Resource pressure per iteration:
# CHECK-NEXT: [0]    [1.0]  [1.1]  [2]    [3]    [4]    [5]
# CHECK-NEXT: 68.00   -      -      -     71.00  60.00   -
# CHECK: Resource pressure by instruction:
# CHECK-NEXT: [0]    [1.0]  [1.1]  [2]    [3]    [4]    [5]    Instructions:
# CHECK-NEXT: 1.00    -      -      -     1.00   1.00    -     vld1.8	{d0}, [r0]
# CHECK-NEXT: 1.00    -      -      -     1.00   1.00    -     vld1.8	{d0, d1}, [r0]
# CHECK-NEXT: 2.00    -      -      -     2.00   2.00    -     vld1.8	{d0, d1, d2}, [r0]
# CHECK-NEXT: 2.00    -      -      -     2.00   2.00    -     vld1.8	{d0, d1, d2, d3}, [r0]
# CHECK-NEXT: 1.00    -      -      -     1.00   1.00    -     vld1.8	{d0, d1}, [r0]!
# CHECK-NEXT: 1.00    -      -      -     1.00   1.00    -     vld1.8	{d0, d1}, [r0], r1
# CHECK-NEXT: 2.00    -      -      -     2.00   1.00    -     vld2.8	{d0, d1}, [r0]
# CHECK-NEXT: 2.00    -      -      -     3.00   2.00    -     vld2.8	{d0, d1, d2, d3}, [r0]
# CHECK-NEXT: 4.00    -      -      -     4.00   3.00    -     vld3.8	{d0, d1, d2}, [r0]
# CHECK-NEXT: 5.00    -      -      -     5.00   4.00    -     vld4.8	{d0, d1, d2, d3}, [r0]
# CHECK-NEXT: 2.00    -      -      -     3.00   2.00    -     vld1.8	{d0[1]}, [r0]
# CHECK-NEXT: 2.00    -      -      -     3.00   2.00    -     vld2.8	{d0[1], d1[1]}, [r0]
# CHECK-NEXT: 6.00    -      -      -     6.00   5.00    -     vld3.8	{d0[1], d1[1], d2[1]}, [r0]
# CHECK-NEXT: 5.00    -      -      -     5.00   4.00    -     vld4.8	{d0[1], d1[1], d2[1], d3[1]}, [r0]
# CHECK-NEXT: 2.00    -      -      -     2.00   1.00    -     vld1.8	{d0[]}, [r0]
# CHECK-NEXT: 2.00    -      -      -     2.00   1.00    -     vld2.8	{d0[], d1[]}, [r0]
# CHECK-NEXT: 4.00    -      -      -     4.00   3.00    -     vld3.8	{d0[], d1[], d2[]}, [r0]
# CHECK-NEXT: 2.00    -      -      -     2.00   2.00    -     vld4.8	{d0[], d1[], d2[], d3[]}, [r0]
# CHECK-NEXT: 1.00    -      -      -     1.00   1.00    -     vst1.8	{d0}, [r0]
# CHECK-NEXT: 1.00    -      -      -     1.00   1.00    -     vst1.8	{d0, d1}, [r0]
# CHECK-NEXT: 2.00    -      -      -     2.00   2.00    -     vst1.8	{d0, d1, d2}, [r0]
# CHECK-NEXT: 2.00    -      -      -     2.00   2.00    -     vst1.8	{d0, d1, d2, d3}, [r0]
# CHECK-NEXT: 1.00    -      -      -     1.00   1.00    -     vst1.8	{d0, d1}, [r0]!
# CHECK-NEXT: 1.00    -      -      -     1.00   1.00    -     vst1.8	{d0, d1}, [r0], r1
# CHECK-NEXT: 1.00    -      -      -     1.00   1.00    -     vst2.8	{d0, d1}, [r0]
# CHECK-NEXT: 1.00    -      -      -     1.00   1.00    -     vst2.8	{d0, d1, d2, d3}, [r0]
# CHECK-NEXT: 2.00    -      -      -     2.00   2.00    -     vst3.8	{d0, d1, d2}, [r0]
# CHECK-NEXT: 2.00    -      -      -     2.00   2.00    -     vst4.8	{d0, d1, d2, d3}, [r0]
# CHECK-NEXT: 1.00    -      -      -     1.00   1.00    -     vst1.8	{d0[1]}, [r0]
# CHECK-NEXT: 1.00    -      -      -     1.00   1.00    -     vst2.8	{d0[1], d1[1]}, [r0]
# CHECK-NEXT: 2.00    -      -      -     2.00   2.00    -     vst3.8	{d0[1], d1[1], d2[1]}, [r0]
# CHECK-NEXT: 2.00    -      -      -     2.00   2.00    -     vst4.8	{d0[1], d1[1], d2[1], d3[1]}, [r0]
# CHECK-NEXT: 1.00    -      -      -     1.00   1.00    -     vldr	d0, [r0]
# CHECK-NEXT: 1.00    -      -      -     1.00   1.00    -     vstr	d0, [r0]
