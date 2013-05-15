//===-- b.c -----------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
static int __b_global = 2;

int b(int arg) {
    int result = arg + __b_global;
    return result;
}

int bb(int arg1) {
    int result2 = arg1 - __b_global;
    return result2;
}
