#import <Foundation/Foundation.h>

@interface Item : NSObject
@property (nonatomic, readonly) NSString *name;
- (instancetype)initWithName:(NSString *)name;
@end
