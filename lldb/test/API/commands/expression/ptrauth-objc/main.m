#import <Foundation/Foundation.h>
#include <stdio.h>

@interface PtrAuthTestObj : NSObject
@property(nonatomic, assign) int value;
- (int)doubleValue;
- (int)addValue:(int)other;
@end

@implementation PtrAuthTestObj
- (int)doubleValue {
  return self.value * 2;
}
- (int)addValue:(int)other {
  return self.value + other;
}
@end

@interface PtrAuthDerived : PtrAuthTestObj
- (int)tripleValue;
@end

@implementation PtrAuthDerived
- (int)tripleValue {
  return self.value * 3;
}
@end

int main(int argc, const char *argv[]) {
  PtrAuthTestObj *obj = [[PtrAuthTestObj alloc] init];
  obj.value = 21;

  PtrAuthDerived *derived = [[PtrAuthDerived alloc] init];
  derived.value = 10;

  int result = [obj doubleValue]; // break here
  printf("%d %d\n", result, [derived tripleValue]);
  return 0;
}
