// Compile with "cl /c /Zi /GR- test-pdb-alt.cpp"
// Link with "link test-pdb.obj test-pdb-alt.obj /debug /nodefaultlib
// /entry:main /out:test-pdb.exe"

#include "test-pdb.h"

int bar(int n) { return n - 1; }
