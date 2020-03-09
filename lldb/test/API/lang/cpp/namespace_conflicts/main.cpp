//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace n {
    struct D {
        int i;
        static int anInt() { return 2; }
        int dump() { return i; }
    };
}

using namespace n;

int foo(D* D) {
    return D->dump(); //% self.expect("expression -- D->dump()", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["int", "2"])
}

int main (int argc, char const *argv[])
{
    D myD { D::anInt() };
    foo(&myD);
    return 0; 
}
