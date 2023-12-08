#include <stdio.h>

#include <stdio.h>

// Forward declare an enumeration (only works in C, not C++)
typedef enum ops ops;

struct foo {
	ops *op;
};

int main (int argc, char const *argv[])
{
    enum bitfield {
        None = 0,
        A = 1 << 0,
        B = 1 << 1,
        C = 1 << 2,
        AB = A | B,
        ALL = A | B | C,
    };

    enum non_bitfield {
        Alpha = 3,
        Beta = 4
    };

    enum enum_test_days {
        Monday = -3,
        Tuesday,
        Wednesday,
        Thursday,
        Friday,
        Saturday,
        Sunday,
        kNumDays
    };

    enum bitfield a = A, b = B, c = C, ab = AB, ac = A | C, all = ALL;
    int nonsense = a + b + c + ab + ac + all;
    enum non_bitfield omega = Alpha | Beta;

    enum enum_test_days day;
    struct foo f;
    f.op = NULL; // Breakpoint for bitfield
    for (day = Monday - 1; day <= kNumDays + 1; day++)
    {
        printf("day as int is %i\n", (int)day); // Set break point at this line.
    }
    return 0;
}
