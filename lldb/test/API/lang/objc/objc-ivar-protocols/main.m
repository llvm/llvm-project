#import <Foundation/Foundation.h>

@protocol MyProtocol
-(void)aMethod;
@end

@interface MyClass : NSObject {
  id <MyProtocol> myId;
  NSObject <MyProtocol> *myObject;
};

-(void)doSomething;

@end

@implementation MyClass

-(void)doSomething
{
  NSLog(@"Hello"); //% self.expect("expression -- myId", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["id"]);
                   //% self.expect("expression -- myObject", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["NSObject"]);
}

@end

int main ()
{
  @autoreleasepool
  {
    MyClass *c = [MyClass alloc];
    [c doSomething];
  }
}
