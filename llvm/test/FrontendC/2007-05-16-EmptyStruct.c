// PR 1417

// RUN: %llvmgcc -xc  %s -c -o - | llvm-dis | grep "struct.anon = type \{  \}"

struct { } *X;
