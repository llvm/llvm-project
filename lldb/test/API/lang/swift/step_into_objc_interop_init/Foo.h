#import <Foundation/Foundation.h>

@interface Foo : NSObject

@property (nonnull) NSArray<NSString *> *values;

- (nonnull id)init;
- (nonnull id)initWithString:(nonnull NSString *)value;
- (nonnull id)initWithString:(nonnull NSString *)value andOtherString:(nonnull NSString *) otherValue;

@end
