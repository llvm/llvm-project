#import <Foundation/Foundation.h>

@interface OverridesALot: NSObject

- (void)boring;

@end

@implementation OverridesALot

+ (id)alloc {
  NSLog(@"alloc");
  return [super alloc];
}

+ (id)allocWithZone: (NSZone *)z {
  NSLog(@"allocWithZone:");
  return [super allocWithZone: z];
}

+ (id)new {
  NSLog(@"new");
  return [super new];
}

- (id)init {
  NSLog(@"init");
  return [super init];
}

- (id)self {
  NSLog(@"self");
  return [super self];
}

+ (id)class {
  NSLog(@"class");
  return [super class];
}

- (BOOL)isKindOfClass: (Class)c {
  NSLog(@"isKindOfClass:");
  return [super isKindOfClass: c];
}

- (BOOL)respondsToSelector: (SEL)s {
  NSLog(@"respondsToSelector:");
  return [super respondsToSelector: s];
}

- (id)retain {
  NSLog(@"retain");
  return [super retain];
}

- (oneway void)release {
  NSLog(@"release");
  [super release];
}

- (id)autorelease { 
  NSLog(@"autorelease");
  return [super autorelease];
}

- (void)boring {
  NSLog(@"boring");
}

@end

@interface OverridesInit: NSObject

- (void)boring;

@end

@implementation OverridesInit

- (id)init {
  NSLog(@"init");
  return [super init];
}

@end

int main() {
  id obj;

  // First make an object of the class that overrides everything,
  // and make sure we step into all the methods:
  
  obj = [OverridesALot alloc]; // Stop here to start stepping
  [obj release]; // Stop Location 2
  
  obj = [OverridesALot allocWithZone: NULL]; // Stop Location 3
  [obj release]; // Stop Location 4
  
  obj = [OverridesALot new]; // Stop Location 5
  [obj release]; // Stop Location 6
  
  obj = [[OverridesALot alloc] init]; // Stop Location 7
  [obj self]; // Stop Location 8
  [obj isKindOfClass: [OverridesALot class]]; // Stop Location 9
  [obj respondsToSelector: @selector(hello)]; // Stop Location 10
  [obj retain];  // Stop Location 11
  [obj autorelease]; // Stop Location 12
  [obj boring]; // Stop Location 13
  [obj release]; // Stop Location 14

  // Now try a class that only overrides init but not alloc, to make
  // sure we get into the second method in a combined call:
  
  obj = [[OverridesInit alloc] init]; // Stop Location 15

  return 0; // Stop Location 15
}
