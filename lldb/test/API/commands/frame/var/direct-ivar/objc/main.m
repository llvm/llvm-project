#include <objc/NSObject.h>

@interface Classic : NSObject {
@public
  int _ivar;
}
@end

@implementation Classic
- (void)fun {
  // check self
}

- (void)run {
  __weak Classic *weakSelf = self;
  ^{
    Classic *self = weakSelf;
    // check idiomatic self

    // Use `self` to extend its lifetime (for lldb to inspect the variable).
    [self copy];
  }();
}
@end

int main() {
  Classic *c = [Classic new];
  c->_ivar = 30;
  [c fun];
  [c run];
}
