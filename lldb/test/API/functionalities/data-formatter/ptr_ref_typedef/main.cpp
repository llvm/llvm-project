//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

typedef int Foo;

int main() {
	int lval = 1;
	Foo* x = &lval;
	Foo& y = lval;
	Foo&& z = 1;
	return 0; // Set breakpoint here
}

