#import "objc/runtime.h"

@protocol NSCoding
@end

#ifdef __has_attribute
#if __has_attribute(objc_root_class)
__attribute__((objc_root_class))
#endif
#endif
@interface NSObject <NSCoding> {
  id isa;
  int refcount;
}
@end
@implementation NSObject
+ (id)new {
  return class_createInstance(self, 0);
}
@end

@interface Base : NSObject
@end
@implementation Base
@end

@interface Derived : Base
@end
@implementation Derived
@end

int main() {
  Base *object = [Derived new];
  Base *base = [Base new];
  return object != base; // break here
}
