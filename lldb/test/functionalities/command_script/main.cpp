//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <iostream>

int
product (int x, int y)
{
    int result = x * y;
    return result;
}

int
sum (int a, int b)
{
    int result = a + b;
    return result;
}

int
strange_max (int m, int n)
{
    if (m > n)
        return m;
    else if (n > m)
        return n;
    else
        return 0;
}

int
foo (int i, int j)
{
    if (strange_max (i, j) == i)
        return product (i, j);
    else if (strange_max  (i, j) == j)
        return sum (i, j);
    else
        return product (sum (i, i), sum (j, j));
}

int
main(int argc, char const *argv[])
{

    int array[9];
	memset(array,0,9*sizeof(int));

    array[0] = foo (1238, 78392);
    array[1] = foo (379265, 23674);
    array[2] = foo (872934, 234);
    array[3] = foo (1238, 78392);
    array[4] = foo (379265, 23674);
    array[5] = foo (872934, 234);
    array[6] = foo (1238, 78392);
    array[7] = foo (379265, 23674);
    array[8] = foo (872934, 234);

    return 0;
}
