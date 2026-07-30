// Compile with "cl /c /Zi /GR- test-pdb.cpp"
// Link with "link test-pdb.obj /debug /nodefaultlib /entry:main
// /out:test-pdb.exe"

#include "test-pdb.h"

int __cdecl _purecall(void) { return 0; }

int main(int argc, char **argv) { return foo(argc) + bar(argc); }
