#include <stdio.h>

int main(int argc, char const *argv[]) {
    printf("Hello world.\n"); // Set breakpoint here
    char line[100];
    if (fgets(line, sizeof(line), stdin)) {
        fprintf(stdout, "input line to stdout: %s", line);
        fprintf(stderr, "input line to stderr: %s", line);
    }
    if (fgets(line, sizeof(line), stdin)) {
        fprintf(stdout, "input line to stdout: %s", line);
        fprintf(stderr, "input line to stderr: %s", line);
    }
    if (fgets(line, sizeof(line), stdin)) {
        fprintf(stdout, "input line to stdout: %s", line);
        fprintf(stderr, "input line to stderr: %s", line);
    }
    printf("Exiting now\n");
}
