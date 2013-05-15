#import "ivars-in-blocks.h"

int
main (int argc, char **argv)
{
  IAmBlocky *my_blocky = [[IAmBlocky alloc] init];
  int blocky_value;
  blocky_value = [my_blocky callABlock: 33];
  return 0;
}
