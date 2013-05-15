//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
int main (int argc, char const *argv[])
{
    struct point_tag {
        int x;
        int y;
    };

    struct rect_tag {
        struct point_tag bottom_left;
        struct point_tag top_right;
    };
    char char_16[16] = "Hello World\n";
    char *strings[] = { "Hello", "Hola", "Bonjour", "Guten Tag" };
    char char_matrix[3][3] = {{'a', 'b', 'c' }, {'d', 'e', 'f' }, {'g', 'h', 'i' }};
    char char_matrix_matrix[3][2][3] =
    {   {{'a', 'b', 'c' }, {'d', 'e', 'f' }},
        {{'A', 'B', 'C' }, {'D', 'E', 'F' }},
        {{'1', '2', '3' }, {'4', '5', '6' }}};
    short short_4[4] = { 1,2,3,4 };
    short short_matrix[1][2] = { {1,2} };
    unsigned short ushort_4[4] = { 1,2,3,4 };
    unsigned short ushort_matrix[2][3] = {
        { 1, 2, 3},
        {11,22,33}
    };
    int int_2[2] = { 1, 2 };
    unsigned int uint_2[2] = { 1, 2 };
    long long_6[6] = { 1, 2, 3, 4, 5, 6 };
    unsigned long ulong_6[6] = { 1, 2, 3, 4, 5, 6 };
    struct point_tag points_2[2] = {
        {1,2},
        {3,4}
    };
    struct point_tag points_2_4_matrix[2][4] = { // Set break point at this line. 
        {{ 1, 2}, { 3, 4}, { 5, 6}, { 7, 8}},
        {{11,22}, {33,44}, {55,66}, {77,88}}
    };
    struct rect_tag rects_2[2] = {
        {{1,2}, {3,4}},
        {{5,6}, {7,8}}
    };
    return 0;
}
