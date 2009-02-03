// RUN: not %llvmgcc %s -S -o /dev/null |& not grep {internal compiler error}

struct A X[(927 - 37) / sizeof(struct A)];
