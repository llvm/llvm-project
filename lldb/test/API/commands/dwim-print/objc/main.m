#import <Foundation/Foundation.h>

@interface Child : NSObject
@property(nonatomic, copy) NSString *name;
@end

@implementation Child
@end

@interface Parent : NSObject
@property(nonatomic, strong) Child *child;
@end

@implementation Parent
@end

@interface C : NSObject {
  int _customDeclaredIvar;
}
@property int declaredBacking;
@property int undeclaredBacking;
@property int implicitBacking;
@end

@implementation C
@synthesize declaredBacking = _customDeclaredIvar;
@synthesize undeclaredBacking = _customUndeclaredIvar;
// implicitBacking: no @synthesize -> compiler auto-creates _implicitBacking.
@end

int main(int argc, char **argv) {
  Child *child = [Child new];
  child.name = @"Seven";
  Parent *parent = [Parent new];
  parent.child = child;
  puts("break here");

  C *obj = [C new];
  obj.declaredBacking = 42;
  obj.undeclaredBacking = 7;
  obj.implicitBacking = 99;
  puts("break here for backing storage");
  return 0;
}
