//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

struct BagOfInts
{
    int x;
    int y;
    int z;
    BagOfInts(int X) :
    x(X),
    y(X+1),
    z(X+2) {}
};

struct BagOfFloats
{
    float x;
    float y;
    float z;
    BagOfFloats(float X) :
    x(X+0.334),
    y(X+0.500),
    z(X+0.667) {}
};

struct BagOfBags
{
    BagOfInts x;
    BagOfInts y;
    BagOfFloats z;
    BagOfFloats q;
    BagOfBags() : 
    x('E'),
    y('B'),
    z(1.1),
    q(20.11) {}
};

struct EmptyStruct {};

struct Plenty
{
    BagOfInts *some_values;
    int* array;
    int array_size;
    int bitfield;
    
    Plenty(int N, bool flagA, bool flagB) :
    some_values(new BagOfInts(N)),
    array(new int[N]),
    array_size(N),
    bitfield( (flagA ? 0x01 : 0x00) | (flagB ? 0x10 : 0x00) )
    {
        for (int j = 0; j < N; j++)
            array[j] = N-j;
    }
};

int main (int argc, const char * argv[])
{
    BagOfInts int_bag(6);
    BagOfFloats float_bag(2.71);
    
    BagOfBags bag_bag;
    EmptyStruct es;
    
    Plenty plenty_of_stuff(5,true,false);
    
    plenty_of_stuff.bitfield = 0x11; // Set break point at this line.
    
    bag_bag.x.z = 12;
    
    return 0;
    
}

