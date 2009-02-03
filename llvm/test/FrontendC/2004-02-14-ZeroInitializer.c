// RUN: %llvmgcc -xc %s -S -o - | grep zeroinitializer

int X[1000];

