// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

extern int vfork(void);
test() {
  vfork();
}
