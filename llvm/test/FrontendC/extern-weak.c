// RUN: %llvmgcc -O3 -S -o - -emit-llvm %s | grep extern_weak
// RUN: %llvmgcc -O3 -S -o - -emit-llvm %s | llvm-as | llc

#if !defined(__linux__) && !defined(__FreeBSD__) && \
    !defined(__OpenBSD__) && !defined(__CYGWIN__) && !defined(__DragonFly__)
void foo() __attribute__((weak_import));
#else
void foo() __attribute__((weak));
#endif

void bar() { foo(); }

