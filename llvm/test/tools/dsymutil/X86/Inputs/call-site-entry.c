/*
 * This file is used to test dsymutil support for call site entries
 * (DW_TAG_call_site / DW_AT_call_return_pc).
 *
 * Instructions for regenerating binaries (on Darwin/x86_64):
 *
 * 1. Copy the source to a top-level directory to work around having absolute
 *    paths in the symtab's OSO entries.
 *
 *    mkdir -p /Inputs/ && cp call-site-entry.c /Inputs && cd /Inputs
 *
 * 2. Compile with call site info enabled.
 *
 *    clang -g -O1 -Xclang -disable-llvm-passes call-site-entry.c -c -o call-site-entry.macho.x86_64.o
 *    clang call-site-entry.macho.x86_64.o -o call-site-entry.macho.x86_64
 *
 * 3. Copy the binaries back into the repo's Inputs directory. You'll need
 *    -oso-prepend-path=%p to link.
 */

__attribute__((optnone)) int f() {}

int main() {
  f();
}
