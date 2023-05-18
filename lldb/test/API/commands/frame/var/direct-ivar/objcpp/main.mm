#import <objc/NSObject.h>
#include <stdio.h>

struct Structure {
  int m_field;
  void fun() {
    puts("check this\n");
  }
};

@interface Classic : NSObject {
@public
  int _ivar;
}
@end

@implementation Classic
- (void)fun {
  puts("check self\n");
}
@end

int main() {
  Structure s;
  s.m_field = 41;
  s.fun();

  Classic *c = [Classic new];
  c->_ivar = 30;
  [c fun];

  Classic *self = c;
  puts("check explicit self\n");
  (void)self;
}
