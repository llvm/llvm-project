// RUN: %llvmgcc -Werror -c %s -o /dev/null
#pragma mark LLVM's world
#ifdef DO_ERROR
#error LLVM's world
#endif
int i;
