#import <Foundation/Foundation.h>
#import "myclass.h"

@implementation MyClass
{
  IMP myImp;
}
- (id)init {
  if (self = [super init])
  {
    SEL theSelector = @selector(init);
    self->myImp = [self methodForSelector:theSelector]; 
  }
  return self;
}
@end
