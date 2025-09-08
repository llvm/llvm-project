#import "Foo.h"

@implementation Foo

- (id)init {
  return self;
}

- (id)initWithString:(nonnull NSString *)value {
  self->_values = @[value];
  return self;
}

- (nonnull id)initWithString:(nonnull NSString *)value andOtherString:(nonnull NSString *) otherValue {
  self->_values = @[value, otherValue];
  return self;
}

@end
