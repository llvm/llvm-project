#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern char **environ;

int main(int argc, char const *argv[]) {
  char **env_var_pointer = environ;
  for (char *env_variable = *env_var_pointer; env_variable;
       env_variable = *++env_var_pointer) {
    printf("%s\n", env_variable);
  }
  return 0;
}
