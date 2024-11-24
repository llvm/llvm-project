#import <Foundation/Foundation.h>

@interface Object : NSObject
@property(nonatomic) int number;
@end

@implementation Object
@end

int main(int argc, char **argv) {
  Object *obj = [Object new];
  obj.number = 15;
  puts("break here");
  return 0;
}
