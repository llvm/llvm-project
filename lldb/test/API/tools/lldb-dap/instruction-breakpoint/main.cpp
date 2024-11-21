#include <stdio.h>
#include <unistd.h>

int function(int x) {

  if (x == 0) // breakpoint 1
    return x;

  if ((x % 2) != 0)
    return x;
  else
    return function(x - 1) + x;
}

int main(int argc, char const *argv[]) {
  int n = function(2);
  return n;
}