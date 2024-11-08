#include <stdio.h>

int main() {
  int counter = 0;
  printf("I print one time: %d.\n", counter++);
  printf("I print two times: %d.\n", counter++);
  return counter;
}
