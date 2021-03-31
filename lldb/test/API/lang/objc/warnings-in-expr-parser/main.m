#include <Foundation/Foundation.h>

@interface MyClass : NSObject
@property int m;
@end

@implementation MyClass {
}
@end

int main() {
  MyClass *m = [[MyClass alloc] init];
  m.m;
  return 0; // break here
}
