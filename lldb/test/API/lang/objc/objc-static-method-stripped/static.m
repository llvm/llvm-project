#import <Foundation/Foundation.h>

@interface Foo : NSObject
+(void) doSomethingWithString: (NSString *) string;
-(void) doSomethingWithNothing;
@end

@implementation Foo
+(void) doSomethingWithString: (NSString *) string
{
  NSLog (@"String is: %@.", string); // Set breakpoint here.
}

+(int) doSomethingElseWithString: (NSString *) string
{
  NSLog (@"String is still: %@.", string);
  return [string length];
}

-(void) doSomethingWithNothing
{
}
@end

int main()
{
  [Foo doSomethingWithString: @"Some string I have in mind."];
  return 0;
}
