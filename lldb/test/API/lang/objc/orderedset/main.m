#import <Foundation/Foundation.h>

int main() {
  NSOrderedSet *orderedSet =
      [NSOrderedSet orderedSetWithArray:@[@1,@2,@3,@1]];
  NSLog(@"%@",orderedSet);
  return 0; // break here
}
