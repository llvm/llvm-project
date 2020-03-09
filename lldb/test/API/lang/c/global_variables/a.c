//===-- a.c -----------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
int g_a = 123;
struct Point {
    int x;
    int y;
};
struct Point g_marked_spot = { 20, 21 };

