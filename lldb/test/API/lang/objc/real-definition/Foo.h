#import <Foundation/Foundation.h>

#import "Bar.h"

@interface Foo : NSObject {
    Bar *_bar;
}

- (NSString *)description;

@end
