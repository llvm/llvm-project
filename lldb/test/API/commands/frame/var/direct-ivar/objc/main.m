#include <objc/NSObject.h>

@interface Classic : NSObject {
@public
  int _ivar;
}
@end

@implementation Classic
- (int)fun {
  // check self
}
@end

int main() {
  Classic *c = [Classic new];
  c->_ivar = 30;
  [c fun];
}
