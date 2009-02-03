// PR 1332
// RUN: %llvmgcc %s -S -o /dev/null

struct Z { int a:1; int :0; int c:1; } z;
