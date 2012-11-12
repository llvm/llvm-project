//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
struct Summarize
{
    int first;
    int second;
};

typedef Summarize summarize_t;
typedef summarize_t *summarize_ptr_t;

summarize_t global_mine = {30, 40};

int
main()
{
    summarize_t mine = {10, 20};
    summarize_ptr_t mine_ptr = &mine;
    printf ("Summarize: first: %d second: %d and address: 0x%p\n", mine.first, mine.second, mine_ptr); // Set break point at this line.
    printf ("Global summarize: first: %d second: %d.\n", global_mine.first, global_mine.second);
    return 0;
}
