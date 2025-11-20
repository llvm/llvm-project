#include <stdio.h>

int bad_read(int index) {
  int array[] = {0, 1, 2};
  return array[index];
}

int main(int argc, char** argv) {
  bad_read(10);
  printf("Execution continued\n");
  return 0;
}
