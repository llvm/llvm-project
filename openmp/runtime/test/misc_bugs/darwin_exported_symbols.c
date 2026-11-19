// REQUIRES: darwin, shared-libomp
// RUN: nm -gU %libomp-library | %not grep " __Z"

// Mach-O has no version script, so the linker is told to keep the C++ symbols
// of the runtime out of the export table, as exports_so.txt does on ELF. Some
// of them are template instantiations, i.e. weak definitions, which dyld
// coalesces across all images of a process. If they were exported, a second
// copy of libomp in the process would have its internal calls bound to the
// first copy.

int main(void) { return 0; }
