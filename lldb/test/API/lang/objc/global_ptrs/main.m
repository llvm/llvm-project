#import <Foundation/Foundation.h>

id g_obj_ptr = nil;

int
main()
{
  g_obj_ptr = @"Some NSString";
  NSLog(@"My string was %@.", g_obj_ptr);
  return 0;
}
