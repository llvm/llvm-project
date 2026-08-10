#include <stdio.h>

int main (int argc, char const *argv[])
{
    printf("Stop here\n"); //% self.runCmd("expression auto $add = [](int first, int second) { return first + second; }")
                           //% self.expect("expression $add(2,3)", substrs = ['= 5'])
    return 0;
}
