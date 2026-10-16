#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <direct.h>
#define environ _environ
#else
#include <unistd.h>
extern char **environ;
#endif

static int echo_cmd(const char *val) {
  puts(val);
  return EXIT_SUCCESS;
}

static int env_cmd(void) {
  for (int i = 0; environ[i]; ++i)
    printf("%s\n", environ[i]);
  return EXIT_SUCCESS;
}

static int handle_cmd(int argc, const char *argv[]) {
  if (argc < 2)
    return EXIT_FAILURE;

  const char *cmd_name = argv[1];
  if (strcmp(cmd_name, "echo") == 0) {
    if (argc < 3)
      return EXIT_FAILURE;
    return echo_cmd(argv[2]);
  }

  if (strcmp(cmd_name, "env") == 0) {
    return env_cmd();
  }
  return EXIT_FAILURE;
}

int main(int argc, const char *argv[]) {
  if (argc > 1 && strcmp(argv[1], "no_cmd") != 0)
    return handle_cmd(argc, argv);

  const char *foo = getenv("FOO");
  const char *nodebugheap = getenv("_NO_DEBUG_HEAP");
  int counter = 1;

  return EXIT_SUCCESS; // breakpoint
}
