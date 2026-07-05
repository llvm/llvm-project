#ifndef TEST_PDB_H
#define TEST_PDB_H

#include "test-pdb-nested.h"

int bar(int n);

inline int foo(int n) { return baz(n) + 1; }

#endif
