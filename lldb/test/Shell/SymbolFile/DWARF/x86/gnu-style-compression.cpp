// REQUIRES: zlib

// RUN: %clang %s -target x86_64-pc-linux -g -gsplit-dwarf -c -o %t \
// RUN:   -Wa,--compress-debug-sections=zlib-gnu
// RUN: %lldb %t -o "target var s a" -b | FileCheck %s

// CHECK: (const short) s = 47
// CHECK: (const A) a = (a = 42)

struct A {
  long a = 42;
};
extern constexpr short s = 47;
extern constexpr A a{};
