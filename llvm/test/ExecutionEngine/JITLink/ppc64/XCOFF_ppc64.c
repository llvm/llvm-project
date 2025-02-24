// AIX's support for llvm-mc does not have enough support for directives like .csect
// so we can't use the tool. llvm-jitlink -check is not available as it requries
// implementation of registerXCOFFGraphInfo. Will revisit this testcase once support
// is more complete.

// REQUIRES: target=powerpc64-ibm-aix{{.*}}

// RUN: rm -rf %t && mkdir -p %t
// RUN: clang --target=powerpc64-ibm-aix -c -O3 -fPIC -o %t/xcoff_ppc64.o %s
// RUN: llvm-jitlink -triple=powerpc64-ibm-aix %t/xcoff_ppc64.o

int main(void) { return 0; }
