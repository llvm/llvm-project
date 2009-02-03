// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

typedef struct { } the_coolest_struct_in_the_world;
extern the_coolest_struct_in_the_world xyzzy;
void *foo() { return &xyzzy; }

