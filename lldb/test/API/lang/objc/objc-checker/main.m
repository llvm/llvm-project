#import <Foundation/Foundation.h>

// This should be a big enough struct that it will force
// the struct return convention:
typedef struct BigStruct {
  float a, b, c, d, e, f, g, h, i, j, k, l;
} BigStruct;


@interface Simple : NSObject
{
  int _value;
}
- (int) value;
- (void) setValue: (int) newValue;
- (BigStruct) getBigStruct;
@end

@implementation Simple
- (int) value
{
  return _value;
}

- (void) setValue: (int) newValue
{
  _value = newValue;
}

- (BigStruct) getBigStruct
{
  BigStruct big_struct = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                          7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  return big_struct;
}
@end

int main ()
{
  NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
  Simple *my_simple = [[Simple alloc] init];
  my_simple.value = 20;
  // Set a breakpoint here.
  NSLog (@"Object has value: %d.", my_simple.value); 
  [pool drain];
  return 0;
}
