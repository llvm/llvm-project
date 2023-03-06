#include <objc/NSObject.h>

struct Structure {
  int m_field;
  int fun() { return this->m_field; }
};

@interface Classic : NSObject {
@public
  int _ivar;
}
@end

@implementation Classic
- (int)fun {
  return self->_ivar;
}
@end

int main() {
  Structure s;
  s.m_field = 30;
  s.fun();

  Classic *c = [Classic new];
  c->_ivar = 41;
  [c fun];
}
