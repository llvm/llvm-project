#include <Foundation/Foundation.h>

@class IncompleteClass;

@interface CompleteClass : NSObject
@end

@interface CompleteClassWithImpl : NSObject
@end
@implementation CompleteClassWithImpl
@end

IncompleteClass *incomplete = 0;
CompleteClass *complete = 0;
CompleteClassWithImpl *complete_impl = 0;

int main() {
  return 0; // break here
}
