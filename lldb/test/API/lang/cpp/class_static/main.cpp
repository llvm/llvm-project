// I made this example after noting that I was unable to display an unsized
// static class array. It turns out that gcc 4.2 will emit DWARF that correctly
// describes the PointType, but it will incorrectly emit debug info for the
// "g_points" array where the following things are wrong:
//  - the DW_TAG_array_type won't have a subrange info
//  - the DW_TAG_variable for "g_points" won't have a valid byte size, so even
//    though we know the size of PointType, we can't infer the actual size
//    of the array by dividing the size of the variable by the number of
//    elements.

#include <stdio.h>

typedef struct PointType
{
    int x, y;
} PointType;

class A
{
public:
    static PointType g_points[];
};

PointType A::g_points[] = 
{
    {    1,    2 },
    {   11,   22 }
};

static PointType g_points[] = 
{
    {    3,    4 },
    {   33,   44 }
};

int
main (int argc, char const *argv[])
{
    const char *hello_world = "Hello, world!";
    printf ("A::g_points[1].x = %i\n", A::g_points[1].x); // Set break point at this line.
    printf ("::g_points[1].x = %i\n", g_points[1].x);
    printf ("%s\n", hello_world);
    return 0;
}
