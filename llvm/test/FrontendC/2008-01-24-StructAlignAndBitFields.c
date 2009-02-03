// RUN: %llvmgcc %s -S -o -

struct U { char a; short b; int c:25; char d; } u;

