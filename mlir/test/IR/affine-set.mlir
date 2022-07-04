// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s

// CHECK-DAG: #set{{[0-9]+}} = affine_set<() : (0 == 0)>
#set0 = affine_set<() : ()>

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0) : (d0 == 0)>
#set1 = affine_set<(i) : (i == 0)>

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0) : (d0 >= 0)>
#set2 = affine_set<(i) : (i >= 0)>

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0, d1) : (d0 >= 0, d1 >= 0)>
#set3 = affine_set<(i, j) : (i >= 0, j >= 0)>

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0, d1) : (d0 == 0, d1 >= 0)>
#set4 = affine_set<(i, j) : (i == 0, j >= 0)>

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0, d1)[s0, s1] : (d0 * 2 + s0 >= 0, d1 * 2 + s1 >= 0, d0 >= 0, d1 >= 0, s0 == 0, s1 == 0)>
#set5 = affine_set<(i, j)[N, M] : (i * 2 + N >= 0, j * 2 + M >= 0, i >= 0, j >= 0, N == 0, M == 0)>

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0, d1)[s0, s1] : (-d0 + s0 >= 0, -d1 + s1 >= 0, d0 >= 0, d1 >= 0)>
#set6 = affine_set<(i, j)[N, M] : (N - i >= 0, M - j >= 0, i >= 0, j >= 0)>

// Check if affine constraints with affine exprs on RHS can be parsed.

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0) : (d0 - 1 == 0)>
#set7 = affine_set<(i) : (i == 1)>

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0)[s0, s1] : (d0 >= 0, -d0 + s0 >= 0, s0 - 5 == 0, -d0 + s1 + 1 >= 0)>
#set8 = affine_set<(i)[N, M] : (i >= 0, N >= i, N == 5, M + 1 >= i)>

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 >= 0, d1 >= 0, d0 - d1 >= 0)>
#set9 = affine_set<(i, j)[N] : (i >= 0, N >= i, j >= 0, i >= j)>

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0, d1)[s0, s1] : (-(d0 + d1 + s0 + s1) == 0, d0 + d1 - (s0 + s1) == 0)>
#set10 = affine_set<(i0, i1)[N, M] : (0 == i0 + i1 + N + M, i0 + i1 == N + M)>

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0, d1)[s0, s1] : (-(d0 + d1 + s0 + s1) >= 0, d0 + d1 - (s0 + s1) >= 0)>
#set11 = affine_set<(i0, i1)[N, M] : (0 >= i0 + i1 + N + M, i0 + i1 >= N + M)>

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0, d1, d2, d3) : ((d0 + d1) mod 2 - (d2 + d3) floordiv 2 == 0, d0 mod 2 + d1 mod 2 - (d2 + d3 + d2) >= 0)>
#set12 = affine_set<(d0, d1, r0, r1) : ((d0 + d1) mod 2 == (r0 + r1) floordiv 2, ((d0) mod 2) + ((d1) mod 2) >= (r0 + r1) + r0)>

// CHECK-DAG: "testset0"() {set = #set{{[0-9]+}}} : () -> ()
"testset0"() {set = #set0} : () -> ()

// CHECK-DAG: "testset1"() {set = #set{{[0-9]+}}} : () -> ()
"testset1"() {set = #set1} : () -> ()

// CHECK-DAG: "testset2"() {set = #set{{[0-9]+}}} : () -> ()
"testset2"() {set = #set2} : () -> ()

// CHECK-DAG: "testset3"() {set = #set{{[0-9]+}}} : () -> ()
"testset3"() {set = #set3} : () -> ()

// CHECK-DAG: "testset4"() {set = #set{{[0-9]+}}} : () -> ()
"testset4"() {set = #set4} : () -> ()

// CHECK-DAG: "testset5"() {set = #set{{[0-9]+}}} : () -> ()
"testset5"() {set = #set5} : () -> ()

// CHECK-DAG: "testset6"() {set = #set{{[0-9]+}}} : () -> ()
"testset6"() {set = #set6} : () -> ()

// CHECK-DAG: "testset7"() {set = #set{{[0-9]+}}} : () -> ()
"testset7"() {set = #set7} : () -> ()

// CHECK-DAG: "testset8"() {set = #set{{[0-9]+}}} : () -> ()
"testset8"() {set = #set8} : () -> ()

// CHECK-DAG: "testset9"() {set = #set{{[0-9]+}}} : () -> ()
"testset9"() {set = #set9} : () -> ()

// CHECK-DAG: "testset10"() {set = #set{{[0-9]+}}} : () -> ()
"testset10"() {set = #set10} : () -> ()

// CHECK-DAG: "testset11"() {set = #set{{[0-9]+}}} : () -> ()
"testset11"() {set = #set11} : () -> ()

// CHECK-DAG: "testset12"() {set = #set{{[0-9]+}}} : () -> ()
"testset12"() {set = #set12} : () -> ()
