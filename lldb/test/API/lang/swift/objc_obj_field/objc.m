#include "Header.h"

@implementation ObjcClass

- (instancetype)init {
  self = [super init];
  if (self) {
    self.someString = @"The objc string";
  }
  return self;
}

@end
