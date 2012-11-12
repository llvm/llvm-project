#import "ivars-in-blocks.h"

typedef int (^my_block_ptr_type) (int);

@interface IAmBlocky()
{
  int _hidden_ivar;
  my_block_ptr_type _block_ptr;
}

@end

@implementation IAmBlocky
 
- (void) makeBlockPtr;
{
  _block_ptr = ^(int inval)
  {
    _hidden_ivar += inval;
    return blocky_ivar * inval; // Break here inside the block.
  };
}

- (IAmBlocky *) init
{
  blocky_ivar = 10;
  _hidden_ivar = 20;
  // Interesting...  Apparently you can't make a block in your init method.  This crashes...
  // [self makeBlockPtr];
  return self;
}

- (int) callABlock: (int) block_value
{
  if (_block_ptr == NULL)
    [self makeBlockPtr];
  return _block_ptr (block_value);
}
@end

