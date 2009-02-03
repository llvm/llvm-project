// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

/* This is apparently legal C.  
 */
extern __inline__ void test() { }

void test() {
}
