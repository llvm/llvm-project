#import "lib.h"
#include <stdio.h>

@interface Bar : Foo
@property int barProp;
- (id)init;
@end

@implementation Bar

- (id)init {
  self = [super init];
  self.barProp = 15;
  return self;
}
@end

int main() {
  Bar *bar = [Bar new];
  puts("break here");
  return 0;
}
