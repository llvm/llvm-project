#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main() {
  int stack_int = 5;
  int *heap_int = (int*) malloc(sizeof (int));
  *heap_int = 10;

  char stack_str[] = "stack string";
  char *heap_str = (char*) malloc(80);
  strcpy (heap_str, "heap string");

  return stack_int; // break here;
}
