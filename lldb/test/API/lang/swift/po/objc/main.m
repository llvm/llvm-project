@import Foundation;
@import Base;
#import "Foo.h"

@implementation Base
@end

int main(int argc, char **argv) {
  Foo *foo = [[Foo alloc] init];
  Base *base = [foo getBase];
  return 0; // break here

}
