#import <Foundation/Foundation.h>
#import "Item.h"

@interface Builder : NSObject
+ (NSArray<Item *> *)makeArray;
@end

int main() {
  NSArray<Item *> *items = [Builder makeArray];
  NSLog(@"break here %@", items);
  return 0;
}
