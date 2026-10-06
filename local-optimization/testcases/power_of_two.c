// Test case - Power of 2 operations as shifts

unsigned scale(unsigned x){
    unsigned a = x * 16;  // x << 4
    unsigned b = x / 8;   // x >> 3
    unsigned c = x % 4;   // x & 3
    int d = (int)x / 4;   // signed division: must stay a division

    return a + b + c + d;
}

// ---------------------------------------------------------------------------
// Expected results, checked by run_tests.sh with FileCheck.
//
// TEST-OPTS: instcombine
// InstCombine's power-of-2 rule covers multiplication only.
// ONLY-LABEL: define {{.*}} @scale(
// ONLY:       shl i32 {{%.*}}, 4
// ONLY:       udiv i32 {{%.*}}, 8
// ONLY:       urem i32 {{%.*}}, 4
// ONLY:       sdiv i32 {{%.*}}, 4
//
// All optimizations: strength reduction also handles unsigned / and %.
// ALL-LABEL:  define {{.*}} @scale(
// ALL-NEXT:   entry:
// ALL-NEXT:     [[A:%.*]] = shl i32 %x, 4
// ALL-NEXT:     [[B:%.*]] = lshr i32 %x, 3
// ALL-NEXT:     [[C:%.*]] = and i32 %x, 3
// ALL-NEXT:     [[D:%.*]] = sdiv i32 %x, 4
// ALL:          ret i32
