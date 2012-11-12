#include <stdio.h>

int main(int argc, char const *argv[]) {
    printf("Hello world.\n");
    char line[100];
    int count = 1;
    while (fgets(line, sizeof(line), stdin)) { // Reading from stdin...
        fprintf(stderr, "input line=>%d\n", count++);
        if (count > 3)
            break;
    }

    printf("Exiting now\n");
}
