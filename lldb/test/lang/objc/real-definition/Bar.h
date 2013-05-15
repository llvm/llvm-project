#import <Foundation/Foundation.h>

@class InternalClass;

@interface Bar : NSObject {
    @private
    InternalClass *storage;
}

- (NSString *)description;

@end