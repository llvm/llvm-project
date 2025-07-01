#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface Gadget : NSObject
@property(nonatomic, assign) NSInteger integer;
@property(nonatomic, assign) BOOL boolean;
@property(nonatomic, strong) NSObject *object;
@property(nonatomic, copy) NSString *string;
@property(nonatomic, copy) NSObject *stringObject;
@end

NS_ASSUME_NONNULL_END
