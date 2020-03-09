//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

int foo(int x, int y) {
    int z = 3 + x;
    return z + y; //% self.expect("frame variable -s", substrs=['ARG: (int) x = -3','ARG: (int) y = 0'])
     //% self.expect("frame variable -s x", substrs=['ARG: (int) x = -3'])
     //% self.expect("frame variable -s y", substrs=['ARG: (int) y = 0'])
     //% self.expect("frame variable -s z", substrs=['LOCAL: (int) z = 0'])
}

int main (int argc, char const *argv[])
{
    return foo(-3,0);  //% self.expect("frame variable -s argc argv", substrs=['ARG: (int) argc ='])
}
