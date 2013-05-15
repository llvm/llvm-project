#import <Foundation/Foundation.h>

// Tests to run:

// Breakpoint 1
// --
// (lldb) expr (int)[str compare:@"hello"]
// (int) $0 = 0
// (lldb) expr (int)[str compare:@"world"]
// (int) $1 = -1
// (lldb) expr (int)[@"" length]
// (int) $2 = 0

int main ()
{
  NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

  NSString *str = [NSString stringWithCString:"hello" encoding:NSASCIIStringEncoding];

  NSLog(@"String \"%@\" has length %lu", str, [str length]); // Set breakpoint here.

  [pool drain];
  return 0;
}
