
#include <Foundation/Foundation.h>

@protocol ObjcProtocol <NSObject>
@end

@interface ObjcClass : NSObject <ObjcProtocol>
@property NSString *someString;
- (instancetype)init;
+ (id<ObjcProtocol>)getP;
@end
