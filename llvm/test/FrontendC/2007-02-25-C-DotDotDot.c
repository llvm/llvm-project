// RUN: %llvmgcc -O0 -S -o - -emit-llvm -fno-inline -fno-unit-at-a-time %s | \
// RUN:   grep {call float @foo}

// Make sure the call to foo is compiled as:
//  call float @foo()
// not
//  call float (...)* bitcast (float ()* @foo to float (...)*)( )

static float foo() { return 0.0; }
float bar() { return foo()*10.0;}


