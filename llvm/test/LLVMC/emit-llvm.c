// RUN: llvmc -c -emit-llvm -o - %s | llvm-dis | grep "@f0()" | count 1

int f0(void) {
}
