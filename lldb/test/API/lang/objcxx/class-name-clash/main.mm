#import <Foundation/Foundation.h>

namespace NS {
  class MyObject { int i = 42; };
  NS::MyObject globalObject;
}

@interface MyObject: NSObject
@end

int main ()
{
  @autoreleasepool
  {
    MyObject *o = [MyObject alloc];
    return 0; //% self.expect("fr var o", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["(MyObject"]);
              //% self.expect("fr var globalObject", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["42"]);
  }
}


