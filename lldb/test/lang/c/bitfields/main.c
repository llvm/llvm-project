//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdint.h>
#include <stdio.h>
int main (int argc, char const *argv[])
{
    struct Bits
    {
        uint32_t    : 1, // Unnamed bitfield
                    b1 : 1,
                    b2 : 2,
                    : 2, // Unnamed bitfield
                    b3 : 3,
                    : 2, // Unnamed bitfield (this will get removed)
                    b4 __attribute__ ((align(16))),
                    b5 : 5,
                    b6 : 6,
                    b7 : 7,
                    four : 4;
    };

    printf("%lu", sizeof(struct Bits));

    struct Bits bits;
    int i;
    for (i=0; i<(1<<1); i++)
        bits.b1 = i;        //// break $source:$line
    for (i=0; i<(1<<2); i++)
        bits.b2 = i;        //// break $source:$line
    for (i=0; i<(1<<3); i++)
        bits.b3 = i;        //// break $source:$line
    for (i=0; i<(1<<4); i++)
        bits.b4 = i;        //// break $source:$line
    for (i=0; i<(1<<5); i++)
        bits.b5 = i;        //// break $source:$line
    for (i=0; i<(1<<6); i++)
        bits.b6 = i;        //// break $source:$line
    for (i=0; i<(1<<7); i++)
        bits.b7 = i;        //// break $source:$line
    for (i=0; i<(1<<4); i++)
        bits.four = i;      //// break $source:$line
    return 0;               //// Set break point at this line.

}
