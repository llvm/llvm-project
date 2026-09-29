/*
 * This file is used to test dsymutil support for call site entries with tail
 * calls (DW_AT_call_pc).
 *
 * Instructions for regenerating binaries (on Darwin/x86_64):
 *
 * 1. Copy the source to a top-level directory to work around having absolute
 *    paths in the symtab's OSO entries.
 *
 *    mkdir -p /Inputs/ && cp tail-call.c /Inputs && cd /Inputs
 *
 * 2. Compile with call site info enabled. -O2 is used to get tail call
 *    promotion.
 *
 *    clang -g -O2 tail-call.c -c -o tail-call.macho.x86_64.o
 *    clang tail-call.macho.x86_64.o -o tail-call.macho.x86_64
 *
 * 3. Copy the binaries back into the repo's Inputs directory. You'll need
 *    -oso-prepend-path=%p to link.
 */

volatile int x;

__attribute__((disable_tail_calls, noinline)) void func2() { x++; }

__attribute__((noinline)) void func1() { func2(); /* tail */ }

__attribute__((disable_tail_calls)) int main() { func1(); /* regular */ }
