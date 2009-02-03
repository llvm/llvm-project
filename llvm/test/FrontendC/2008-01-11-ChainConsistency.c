// RUN: %llvmgcc -S %s -o - -fnested-functions | not grep nest

void n1(void) { void a(void) { a(); } a(); }
