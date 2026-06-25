#import "Item.h"

@implementation Item
- (instancetype)initWithName:(NSString *)name {
  if (self = [super init])
    _name = [name copy];
  return self;
}
@end
