#include <stdio.h>

int main () {
    int a = 10;
    int b = 20;

    for (int i = 0; i < 10; i++){
        a = (a+b)/a;
    }

    if ( b == a ){
        return 1;
    }

    a = 5;

    return 0;
}