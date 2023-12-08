#import <Foundation/Foundation.h>

@interface Foo : NSObject
+(int) doSomethingWithString: (NSString *) string;
-(int) doSomethingInstance: (NSString *) string;
@end

@implementation Foo
+(int) doSomethingWithString: (NSString *) string
{
  NSLog (@"String is: %@.", string);
  return [string length];
}

-(int) doSomethingInstance: (NSString *)string
{
  return [Foo doSomethingWithString:string];
}
@end

int main()
{
  return 0; // Set breakpoint here.
}
