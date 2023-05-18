#import <Foundation/Foundation.h>

int main ()
{
  @autoreleasepool
  {
    NSLog(@"Hello"); //% self.expect("expression -- *((NSConcretePointerArray*)[NSPointerArray strongObjectsPointerArray])", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["count", "capacity", "options", "mutations"]);
                     //% self.expect("expression -- ((NSConcretePointerArray*)[NSPointerArray strongObjectsPointerArray])->count", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["unsigned"]);
  }
}
