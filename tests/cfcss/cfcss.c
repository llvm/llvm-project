#include<stdio.h>
#include<stdlib.h>
void __cfcss_error() {
    printf(" Signatures do not match");
    exit(0);
}
int main() {
  for (int i = 0; i < 10; i++)
    printf(" Value is %d", i);
}
