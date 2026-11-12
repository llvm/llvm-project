#include <stdlib.h>

int main(void) {
  char *str;
  int size = 9; //strlen("patatino") + 1
  str = (char *)malloc(sizeof(char)*size);
  *(str+0) = 'p';
  *(str+1) = 'a';
  *(str+2) = 't';
  *(str+3) = 'a';
  *(str+4) = 't';
  *(str+5) = 'i';
  *(str+6) = 'n';
  *(str+7) = 'o';
  *(str+8) = '\0';
  return 0;
}
