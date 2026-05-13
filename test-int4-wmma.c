// Test: INT4 WMMA instruction generation for gfx1200 (RX 9060 XT)
// Compile: clang -O2 -mcpu=gfx1200 -target amdgcn-amd-amdhsa -emit-llvm -S test-int4-wmma.c -o test.ll
// Check:   llc -mtriple=amdgcn -mcpu=gfx1200 -verify-machineinstrs test.ll -o test.s

typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef int int32_t;

// === GFX11-style WMMA256 INT4 (needs wmma-256b-insts, our new path) ===
// v_wmma_i32_16x16x16_iu4: A=1xi32(8x4bit), B=1xi32(8x4bit), D=v8i32
__attribute__((noinline))
void test_wmma256_int4(int32_t *C, uint32_t A, uint32_t B, int32_t *D) {
    // Packed INT4 values in 32-bit registers
    // v_wmma_i32_16x16x16_iu4: 8 INT4 values per matrix in each i32
    // neg_lo=0(unsigned), A, neg_hi=0(unsigned), B, accumulator, clamp=0
    typedef int32_t v8i32 __attribute__((ext_vector_type(8)));
    v8i32 acc = *(v8i32 *)C;
    v8i32 res = __builtin_amdgcn_wmma_i32_16x16x16_iu4(
        0, A, 0, B, acc, 0);
    *(v8i32 *)D = res;
}

// === GFX12-style WMMA128 INT4 (needs wmma-128b-insts, already existed) ===
// v_wmma_i32_16x16x32_iu4: A=v2i32(16x4bit), B=v2i32(16x4bit), D=v8i32
__attribute__((noinline))
void test_wmma128_int4(int32_t *C, uint32_t *A, uint32_t *B, int32_t *D) {
    typedef int32_t v2i32 __attribute__((ext_vector_type(2)));
    typedef int32_t v8i32 __attribute__((ext_vector_type(8)));
    v8i32 acc = *(v8i32 *)C;
    v2i32 va = *(v2i32 *)A;
    v2i32 vb = *(v2i32 *)B;
    v8i32 res = __builtin_amdgcn_wmma_i32_16x16x32_iu4(
        0, va, 0, vb, acc, 0);
    *(v8i32 *)D = res;
}

// === SWMMAC sparse INT4 (needs swmmac-gfx1200-insts) ===
// v_swmmac_i32_16x16x32_iu4: sparse WMMA with 2:4 structured sparsity
__attribute__((noinline))
void test_swmmac_int4_32(int32_t *C, uint32_t A_sparse, uint32_t *B, int32_t *D, uint16_t index) {
    typedef int32_t v2i32 __attribute__((ext_vector_type(2)));
    typedef int32_t v8i32 __attribute__((ext_vector_type(8)));
    v8i32 acc = *(v8i32 *)C;
    v2i32 vb = *(v2i32 *)B;
    v8i32 res = __builtin_amdgcn_swmmac_i32_16x16x32_iu4(
        0, A_sparse, 0, vb, acc, index);
    *(v8i32 *)D = res;
}

// v_swmmac_i32_16x16x64_iu4: larger sparse tile
__attribute__((noinline))
void test_swmmac_int4_64(int32_t *C, uint32_t *A, uint32_t *B, int32_t *D, uint32_t index) {
    typedef int32_t v2i32 __attribute__((ext_vector_type(2)));
    typedef int32_t v4i32 __attribute__((ext_vector_type(4)));
    typedef int32_t v8i32 __attribute__((ext_vector_type(8)));
    v8i32 acc = *(v8i32 *)C;
    v2i32 va = *(v2i32 *)A;
    v4i32 vb = *(v4i32 *)B;
    v8i32 res = __builtin_amdgcn_swmmac_i32_16x16x64_iu4(
        0, va, 0, vb, acc, index);
    *(v8i32 *)D = res;
}
