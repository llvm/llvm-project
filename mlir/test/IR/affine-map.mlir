// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s

// Identity maps used in trivial compositions in MemRefs are optimized away.
// CHECK-NOT: #map{{[0-9]+}} = affine_map<(d0, d1) -> (d0, d1)>
#map0 = affine_map<(i, j) -> (i, j)>

// CHECK-NOT: #map{{[0-9]+}} = affine_map<(d0, d1)[s0] -> (d0, d1)>
#map1 = affine_map<(i, j)[s0] -> (i, j)>

// CHECK: #map{{[0-9]+}} = affine_map<() -> (0)>
// A map may have 0 inputs.
// However, an affine.apply always takes at least one input.
#map2 = affine_map<() -> (0)>

// All the maps in the following block are equivalent and are unique'd as one
// map. Therefore there should be only one output and we explicitly CHECK-NOT
// for the others.
// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1) -> (d0 + 1, d1 * 4 + 2)>
#map3  = affine_map<(i, j) -> (i+1, 4*j + 2)>
// CHECK-NOT: #map3{{[a-z]}}
#map3a = affine_map<(i, j) -> (1+i, 4*j + 2)>
#map3b = affine_map<(i, j) -> (2 + 3 - 2*2 + i, 4*j + 2)>
#map3c = affine_map<(i, j) -> (i +1 + 0, 4*j + 2)>
#map3d = affine_map<(i, j) -> (i + 3 + 2 - 4, 4*j + 2)>
#map3e = affine_map<(i, j) -> (1*i+3*2-2*2-1, 4*j + 2)>
#map3f = affine_map<(i, j) -> (i + 1, 4*j*1 + 2)>
#map3g = affine_map<(i, j) -> (i + 1, 2*2*j + 2)>
#map3h = affine_map<(i, j) -> (i + 1, 2*j*2 + 2)>
#map3i = affine_map<(i, j) -> (i + 1, j*2*2 + 2)>
#map3j = affine_map<(i, j) -> (i + 1, j*1*4 + 2)>
#map3k = affine_map<(i, j) -> (i + 1, j*4*1 + 2)>

// The following reduction should be unique'd out too but such expression
// simplification is not performed for IR parsing, but only through analyses
// and transforms.
// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1) -> (d1 - d0 + (d0 - d1 + 1) * 2 + d1 - 1, d1 * 4 + 2)>
#map3l = affine_map<(i, j) -> ((j - i) + 2*(i - j + 1) + j - 1 + 0, j + j + 1 + j + j + 1)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1) -> (d0 + 2, d1)>
#map4  = affine_map<(i, j) -> (3+3-2*2+i, j)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>
#map5 = affine_map<(i, j)[s0] -> (i + s0, j)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0] -> (d0 + s0, d1 + 5)>
#map6 = affine_map<(i, j)[s0] -> (i + s0, j + 5)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0] -> (d0 + d1 + s0, d1)>
#map7 = affine_map<(i, j)[s0] -> (i + j + s0, j)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0] -> (d0 + d1 + s0 + 5, d1)>
#map8 = affine_map<(i, j)[s0] -> (5 + i + j + s0, j)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0] -> (d0 + d1 + 5, d1)>
#map9 = affine_map<(i, j)[s0] -> ((i + j) + 5, j)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0] -> (d0 + d1 + 5, d1)>
#map10 = affine_map<(i, j)[s0] -> (i + (j + 5), j)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0] -> (d0 * 2, d1 * 3)>
#map11 = affine_map<(i, j)[s0] -> (2*i, 3*j)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0] -> (d0 + (d1 + s0 * 3) * 5 + 12, d1)>
#map12 = affine_map<(i, j)[s0] -> (i + 2*6 + 5*(j+s0*3), j)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0] -> (d0 * 5 + d1, d1)>
#map13 = affine_map<(i, j)[s0] -> (5*i + j, j)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0] -> (d0 + d1, d1)>
#map14 = affine_map<(i, j)[s0] -> ((i + j), (j))>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0] -> (d0 + d1 + 7, d1 + 3)>
#map15 = affine_map<(i, j)[s0] -> ((i + j + 2) + 5, (j)+3)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0] -> (d0, 0)>
#map16 = affine_map<(i, j)[s1] -> (i, 0)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0] -> (d0, d1 * s0)>
#map17 = affine_map<(i, j)[s0] -> (i, s0*j)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1) -> (d0, d0 * 3 + d1)>
#map19 = affine_map<(i, j) -> (i, 3*i + j)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1) -> (d0, d0 + d1 * 3)>
#map20 = affine_map<(i, j)  -> (i, i + 3*j)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0] -> (d0, d0 * ((s0 * s0) * 9) + 3)>
#map18 = affine_map<(i, j)[N] -> (i, 2 + N*N*9*i + 1)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1) -> (1, d0 + d1 * 3 + 5)>
#map21 = affine_map<(i, j)  -> (1, i + 3*j + 5)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0] -> (s0 * 5, d0 + d1 * 3 + d0 * 5)>
#map22 = affine_map<(i, j)[s0] -> (5*s0, i + 3*j + 5*i)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0, s1] -> (d0 * (s0 * s1), d1)>
#map23 = affine_map<(i, j)[s0, s1] -> (i*(s0*s1), j)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0, s1] -> (d0, d1 mod 5)>
#map24 = affine_map<(i, j)[s0, s1] -> (i, j mod 5)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0, s1] -> (d0, d1 floordiv 5)>
#map25 = affine_map<(i, j)[s0, s1] -> (i, j floordiv 5)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0, s1] -> (d0, d1 ceildiv 5)>
#map26 = affine_map<(i, j)[s0, s1] -> (i, j ceildiv 5)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0, s1] -> (d0, d0 - d1 - 5)>
#map29 = affine_map<(i, j)[s0, s1] -> (i, i - j - 5)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0, s1] -> (d0, d0 - d1 * s1 + 2)>
#map30 = affine_map<(i, j)[M, N] -> (i, i - N*j + 2)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0, s1] -> (d0 * -5, d1 * -3, -2, -(d0 + d1), -s0)>
#map32 = affine_map<(i, j)[s0, s1] -> (-5*i, -3*j, -2, -1*(i+j), -1*s0)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1) -> (-4, -d0)>
#map33 = affine_map<(i, j) -> (-2+-5-(-3), -1*i)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0, s1] -> (d0, d1 floordiv s0, d1 mod s0)>
#map34 = affine_map<(i, j)[s0, s1] -> (i, j floordiv s0, j mod s0)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1, d2)[s0, s1, s2] -> ((d0 * s1) * s2 + d1 * s1 + d2)>
#map35 = affine_map<(i, j, k)[s0, s1, s2] -> (i*s1*s2 + j*s1 + k)>

// Constant folding.
// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1) -> (8, 4, 1, 3, 2, 4)>
#map36 = affine_map<(i, j) -> (5+3, 2*2, 8-7, 100 floordiv 32, 5 mod 3, 10 ceildiv 3)>
// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1) -> (4, 11, 512, 15)>
#map37 = affine_map<(i, j) -> (5 mod 3 + 2, 5*3 - 4, 128 * (500 ceildiv 128), 40 floordiv 7 * 3)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1) -> (d0 * 2 + 1, d1 + 2)>
#map38 = affine_map<(i, j) -> (1 + i*2, 2 + j)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0, s1] -> (d0 * s0, d0 + s0, d0 + 2, d1 * 2, s1 * 2, s0 + 2)>
#map39 = affine_map<(i, j)[M, N] -> (i*M, M + i, 2+i, j*2, N*2, 2 + M)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0] -> ((d0 * 5) floordiv 4, (d1 ceildiv 7) mod s0)>
#map43 = affine_map<(i, j) [s0] -> ( i * 5 floordiv 4, j ceildiv 7 mod s0)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1) -> (d0 - d1 * 2, (d1 * 6) floordiv 4)>
#map44 = affine_map<(i, j) -> (i - 2*j, j * 6 floordiv 4)>

// Simplifications
// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1, d2)[s0] -> (d0 + d1 + d2 + 1, d2 + d1, (d0 * s0) * 8)>
#map45 = affine_map<(i, j, k) [N] -> (1 + i + 3 + j - 3 + k, k + 5 + j - 5, 2*i*4*N)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1, d2) -> (0, d1, d0 * 2, 0)>
#map46 = affine_map<(i, j, k) -> (i*0, 1*j, i * 128 floordiv 64, j * 0 floordiv 64)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1, d2) -> (d0, d0 * 4, 0, 0, 0)>
#map47 = affine_map<(i, j, k) -> (i * 64 ceildiv 64, i * 512 ceildiv 128, 4 * j mod 4, 4*j*4 mod 8, k mod 1)>

// floordiv should resolve similarly to ceildiv and be unique'd out.
// CHECK-NOT: #map48{{[a-z]}}
#map48 = affine_map<(i, j, k) -> (i * 64 floordiv 64, i * 512 floordiv 128, 4 * j mod 4, 4*j*4 mod 8)>

// Simplifications for mod using known GCD's of the LHS expr.
// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0] -> (0, 0, 0, 1)>
#map49 = affine_map<(i, j)[s0] -> ( (i * 4 + 8) mod 4, 32 * j * s0 * 8 mod 256, (4*i + (j * (s0 * 2))) mod 2, (4*i + 3) mod 2)>

// Floordiv, ceildiv divide by one.
// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1)[s0] -> (d0 * 2 + 1, d1 + s0)>
#map50 = affine_map<(i, j)[s0] -> ( (i * 2 + 1) ceildiv 1, (j + s0) floordiv 1)>

// floordiv, ceildiv, and mod where LHS is negative.
// CHECK: #map{{[0-9]+}} = affine_map<(d0) -> (-2, 1, -1)>
#map51 = affine_map<(i) -> (-5 floordiv 3, -5 mod 3, -5 ceildiv 3)>

// Parenthesis elision.
// CHECK: #map{{[0-9]+}} = affine_map<(d0) -> (d0 * 16 - (d0 + 1) + 15)>
#map52 = affine_map<(d0) -> (16*d0 + ((d0 + 1) * -1) + 15)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0) -> (d0 - (d0 + 1))>
#map53 = affine_map<(d0) -> (d0 - (d0 + 1))>

// CHECK: #map{{[0-9]+}} = affine_map<(d0)[s0] -> ((-s0) floordiv 4, d0 floordiv -1)>
#map54 = affine_map<(d0)[s0] -> (-s0 floordiv 4, d0 floordiv -1)>

// CHECK: #map{{[0-9]+}} = affine_map<() -> ()>
#map55 = affine_map<() -> ()>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1) -> (d0, d0 * 2 + d1 * 4 + 2, 1, 2, (d0 * 4) mod 8)>
#map56 = affine_map<(d0, d1) -> ((4*d0 + 2) floordiv 4, (4*d0 + 8*d1 + 5) floordiv 2, (2*d0 + 4*d1 + 3) mod 2, (3*d0 - 4) mod 3, (4*d0 + 8*d1) mod 8)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1) -> (d1, d0, 0)>
#map57 = affine_map<(d0, d1) -> (d0 - d0 + d1, -d0 + d0 + d0, (1 + d0 + d1 floordiv 4) - (d0 + d1 floordiv 4 + 1))>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1) -> (d0 * 3, (d0 + d1) * 2, d0 mod 2)>
#map58 = affine_map<(d0, d1) -> (4*d0 - 2*d0 + d0, (d0 + d1) + (d0 + d1), 2 * (d0 mod 2) - d0 mod 2)>

// CHECK: #map{{[0-9]+}} = affine_map<(d0, d1) -> (d0 mod 5, (d1 mod 35) mod 4)>
#map59 = affine_map<(d0, d1) -> ((d0 mod 35) mod 5, (d1 mod 35) mod 4)>

// Single identity maps are removed.
// CHECK: @f0(memref<2x4xi8, 1>)
func.func private @f0(memref<2x4xi8, #map0, 1>)

// Single identity maps are removed.
// CHECK: @f1(memref<2x4xi8, 1>)
func.func private @f1(memref<2x4xi8, #map1, 1>)

// CHECK: @f2(memref<i8, #map{{[0-9]+}}, 1>)
func.func private @f2(memref<i8, #map2, 1>)

// CHECK: @f3(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f3(memref<2x4xi8, #map3, 1>)
// CHECK: @f3a(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f3a(memref<2x4xi8, #map3a, 1>)
// CHECK: @f3b(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f3b(memref<2x4xi8, #map3b, 1>)
// CHECK: @f3c(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f3c(memref<2x4xi8, #map3c, 1>)
// CHECK: @f3d(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f3d(memref<2x4xi8, #map3d, 1>)
// CHECK: @f3e(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f3e(memref<2x4xi8, #map3e, 1>)
// CHECK: @f3f(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f3f(memref<2x4xi8, #map3f, 1>)
// CHECK: @f3g(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f3g(memref<2x4xi8, #map3g, 1>)
// CHECK: @f3h(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f3h(memref<2x4xi8, #map3h, 1>)
// CHECK: @f3i(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f3i(memref<2x4xi8, #map3i, 1>)
// CHECK: @f3j(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f3j(memref<2x4xi8, #map3j, 1>)
// CHECK: @f3k(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f3k(memref<2x4xi8, #map3k, 1>)
// CHECK: @f3l(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f3l(memref<2x4xi8, #map3l, 1>)

// CHECK: @f4(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f4(memref<2x4xi8, #map4, 1>)

// CHECK: @f5(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f5(memref<2x4xi8, #map5, 1>)

// CHECK: @f6(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f6(memref<2x4xi8, #map6, 1>)

// CHECK: @f7(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f7(memref<2x4xi8, #map7, 1>)

// CHECK: @f8(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f8(memref<2x4xi8, #map8, 1>)

// CHECK: @f9(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f9(memref<2x4xi8, #map9, 1>)

// CHECK: @f10(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f10(memref<2x4xi8, #map10, 1>)

// CHECK: @f11(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f11(memref<2x4xi8, #map11, 1>)

// CHECK: @f12(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f12(memref<2x4xi8, #map12, 1>)

// CHECK: @f13(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f13(memref<2x4xi8, #map13, 1>)

// CHECK: @f14(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f14(memref<2x4xi8, #map14, 1>)

// CHECK: @f15(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f15(memref<2x4xi8, #map15, 1>)

// CHECK: @f16(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f16(memref<2x4xi8, #map16, 1>)

// CHECK: @f17(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f17(memref<2x4xi8, #map17, 1>)

// CHECK: @f19(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f19(memref<2x4xi8, #map19, 1>)

// CHECK: @f20(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f20(memref<2x4xi8, #map20, 1>)

// CHECK: @f18(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f18(memref<2x4xi8, #map18, 1>)

// CHECK: @f21(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f21(memref<2x4xi8, #map21, 1>)

// CHECK: @f22(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f22(memref<2x4xi8, #map22, 1>)

// CHECK: @f23(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f23(memref<2x4xi8, #map23, 1>)

// CHECK: @f24(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f24(memref<2x4xi8, #map24, 1>)

// CHECK: @f25(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f25(memref<2x4xi8, #map25, 1>)

// CHECK: @f26(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f26(memref<2x4xi8, #map26, 1>)

// CHECK: @f29(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f29(memref<2x4xi8, #map29, 1>)

// CHECK: @f30(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f30(memref<2x4xi8, #map30, 1>)

// CHECK: @f32(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f32(memref<2x4xi8, #map32, 1>)

// CHECK: @f33(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f33(memref<2x4xi8, #map33, 1>)

// CHECK: @f34(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f34(memref<2x4xi8, #map34, 1>)

// CHECK: @f35(memref<2x4x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f35(memref<2x4x4xi8, #map35, 1>)

// CHECK: @f36(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f36(memref<2x4xi8, #map36, 1>)

// CHECK: @f37(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f37(memref<2x4xi8, #map37, 1>)

// CHECK: @f38(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f38(memref<2x4xi8, #map38, 1>)

// CHECK: @f39(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func.func private @f39(memref<2x4xi8, #map39, 1>)

// CHECK: @f43(memref<2x4xi8, #map{{[0-9]+}}>)
func.func private @f43(memref<2x4xi8, #map43>)

// CHECK: @f44(memref<2x4xi8, #map{{[0-9]+}}>)
func.func private @f44(memref<2x4xi8, #map44>)

// CHECK: @f45(memref<100x100x100xi8, #map{{[0-9]+}}>)
func.func private @f45(memref<100x100x100xi8, #map45>)

// CHECK: @f46(memref<100x100x100xi8, #map{{[0-9]+}}>)
func.func private @f46(memref<100x100x100xi8, #map46>)

// CHECK: @f47(memref<100x100x100xi8, #map{{[0-9]+}}>)
func.func private @f47(memref<100x100x100xi8, #map47>)

// CHECK: @f48(memref<100x100x100xi8, #map{{[0-9]+}}>)
func.func private @f48(memref<100x100x100xi8, #map48>)

// CHECK: @f49(memref<100x100xi8, #map{{[0-9]+}}>)
func.func private @f49(memref<100x100xi8, #map49>)

// CHECK: @f50(memref<100x100xi8, #map{{[0-9]+}}>)
func.func private @f50(memref<100x100xi8, #map50>)

// CHECK: @f51(memref<1xi8, #map{{[0-9]+}}>)
func.func private @f51(memref<1xi8, #map51>)

// CHECK: @f52(memref<1xi8, #map{{[0-9]+}}>)
func.func private @f52(memref<1xi8, #map52>)

// CHECK: @f53(memref<1xi8, #map{{[0-9]+}}>)
func.func private @f53(memref<1xi8, #map53>)

// CHECK: @f54(memref<10xi32, #map{{[0-9]+}}>)
func.func private @f54(memref<10xi32, #map54>)

// CHECK: "foo.op"() {map = #map{{[0-9]+}}} : () -> ()
"foo.op"() {map = #map55} : () -> ()

// CHECK: @f56(memref<1x1xi8, #map{{[0-9]+}}>)
func.func private @f56(memref<1x1xi8, #map56>)

// CHECK: "f57"() {map = #map{{[0-9]+}}} : () -> ()
"f57"() {map = #map57} : () -> ()

// CHECK: "f58"() {map = #map{{[0-9]+}}} : () -> ()
"f58"() {map = #map58} : () -> ()

// CHECK: "f59"() {map = #map{{[0-9]+}}} : () -> ()
"f59"() {map = #map59} : () -> ()
