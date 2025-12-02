#import <Foundation/Foundation.h>

@interface Foo : NSObject
@property(readwrite) int fooProp;
@end

@implementation Foo
@end

int main() {
  Foo *f = [Foo new];
  [f setFooProp:10];
  return f.fooProp;
}
