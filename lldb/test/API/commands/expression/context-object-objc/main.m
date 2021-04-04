#import <objc/NSObject.h>

@interface ObjcClass : NSObject {
    int field;
}

@property int property;

+(ObjcClass*)createNew;

-(id)init;

-(int)method;

@end

@implementation ObjcClass

+(ObjcClass*)createNew {
    return [ObjcClass new];
}

-(id)init {
    self = [super init];
    if (self) {
        field = 1111;
        _property = 2222;
    }
    return self;
}

-(int)method {
    return 3333;
}

@end

int main()
{
    @autoreleasepool {
        ObjcClass* objcClass = [ObjcClass new];
        
        int field = 4444;
        
        return 0; // Break here
    }
}
