#include <stdio.h>

int main() { int argc = 0; char **argv = (char **)0;

    enum days {
        Monday = 10,
        Tuesday,
        Wednesday,
        Thursday,
        Friday,
        Saturday,
        Sunday,
        kNumDays
    };
    enum days day;
    for (day = Monday - 1; day <= kNumDays + 1; day++)
    {
        printf("day as int is %i\n", (int)day);
    }
    return 0; // Set break point at this line.
}
