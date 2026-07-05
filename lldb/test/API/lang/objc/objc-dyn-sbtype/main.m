#import <Foundation/Foundation.h>

@interface MyBaseClass : NSObject
{}
-(id) init;
-(int) getInt;
@end

@implementation MyBaseClass
- (id) init {
	return (self = [super init]);
}

- (int) getInt {
	return 1;
}
@end

@interface MyDerivedClass : MyBaseClass
{
	int x;
	int y;
}
-(id) init;
-(int) getInt;
@end

@implementation MyDerivedClass
- (id) init {
	self = [super init];
	if (self) {
		self-> x = 0;
		self->y = 1;
	}
	return self;
}

- (int) getInt {
	y = x++;
	return x;
}
@end


int main (int argc, char const *argv[])
{
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];	
	NSObject* object = [[MyDerivedClass alloc] init];
	MyBaseClass* base = [[MyDerivedClass alloc] init];
    [pool release]; // Set breakpoint here.
    return 0;
}

