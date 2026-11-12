#import <dispatch/dispatch.h>
#include <stdio.h>

void one() {
  printf("one...\n"); // breakpoint 1
}

void two() {
  printf("two...\n");
  one();
}

void three() {
  printf("three...\n");
  two();
}

int main(int argc, char *argv[]) {
  printf("main...\n");
  // Nest from main queue > global queue > main queue.
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
                 ^{
                   dispatch_async(dispatch_get_main_queue(), ^{
                     three();
                   });
                 });
  dispatch_main();
}
