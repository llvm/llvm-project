#include <Foundation/Foundation.h>

int main(int argc, char *argv[]) {
  @autoreleasepool {

    NSArray *crew = [NSArray arrayWithObjects:@"Jim", @"Jason", @"Jonas", @"Ismail", nil];

    // This will throw an exception.
    NSLog(@"%@", [crew objectAtIndex:10]);
  }

  return 0;
}
