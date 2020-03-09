@import ObjCModule;
#include "Library-Swift.h"

@implementation ObjCClass {
  id mangled_swift_obj;
  id rawname_swift_obj;
}

- (instancetype)init {
  self = [super init];
  if (self) {
    mangled_swift_obj = [MangledSwiftClass new];
    rawname_swift_obj = [RawNameSwiftClass new];
  }
  return self;
}

- (id)getMangled {
  return mangled_swift_obj;
}

- (id)getRawname {
  return rawname_swift_obj;
}

- (NSString *)getString {
  return [mangled_swift_obj getString];
}

@end

