#import <objc/NSObject.h>
#import <stdint.h>

@interface InternalDefiner : NSObject {
@public
    uintptr_t foo;
}

-(id)initWithFoo:(uintptr_t)f andBar:(uintptr_t)b;

@end
