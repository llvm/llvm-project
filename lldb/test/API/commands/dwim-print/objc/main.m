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

int main(int argc, char **argv) {
  Child *child = [Child new];
  child.name = @"Seven";
  Parent *parent = [Parent new];
  parent.child = child;
  puts("break here");
  return 0;
}
