#include<stdio.h>

// Test case - Strength reduction by replacing
// multiplication by powers of 2 with left shift

int main(){
    int a=3,b=4,c,d,e,f;

    d = 2*c; // strength reduction c << 1
    e = f*8; // strength reduction f << 3

    return a;
}

// Expected (checked by run_tests.sh):
// CHECK: shl i32 {{%.*}}, 1
// CHECK: shl i32 {{%.*}}, 3
// CHECK: ret i32 3
