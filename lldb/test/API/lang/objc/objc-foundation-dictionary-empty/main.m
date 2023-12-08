#import <Foundation/Foundation.h>

int main(void)
{
  NSDictionary *emptyDictionary = [[NSDictionary alloc] init];
  return 0; //% self.expect("frame var emptyDictionary", substrs = ["0 key/value pairs"]);
}
