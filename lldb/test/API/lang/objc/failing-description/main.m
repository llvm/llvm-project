#import <Foundation/Foundation.h>

@interface Bad : NSObject
@end

@implementation Bad {
  BOOL _lookHere;
}

- (NSString *)description {
  int *i = NULL;
  *i = 0;
  return @"surprise";
}
@end

int main() {
  Bad *bad = [Bad new];
  printf("break here\n");
  return 0;
}
