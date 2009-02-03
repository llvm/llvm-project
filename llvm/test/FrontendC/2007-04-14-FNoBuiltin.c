// RUN: %llvmgcc -S %s -O2 -fno-builtin -o - | grep call.*printf
// Check that -fno-builtin is honored.

extern int printf(const char*, ...);
void foo(const char *msg) {
	printf("%s\n",msg);
}
