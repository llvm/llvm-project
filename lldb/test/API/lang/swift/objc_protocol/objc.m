#include "bridging-header.h"

@implementation ObjcClass

- (instancetype)init {
  self = [super init];
  if (self) {
    self.someString = @"The objc string";
  }
  return self;
}

+ (id<ObjcProtocol>)getP {
  return [ObjcClass new];
}
@end
