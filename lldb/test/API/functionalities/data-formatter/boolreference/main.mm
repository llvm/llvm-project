#import <Foundation/Foundation.h>

typedef struct {
    BOOL fieldOne : 1;
    BOOL fieldTwo : 1;
    BOOL fieldThree : 1;
    BOOL fieldFour : 1;
    BOOL fieldFive : 1;
} BoolBitFields;

int main (int argc, const char * argv[])
{
  NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

	BOOL yes  = YES;
	BOOL no = NO;
  BOOL unset = 12;
	
	BOOL &yes_ref = yes;
	BOOL &no_ref = no;
	BOOL &unset_ref = unset;
	
	BOOL* yes_ptr = &yes;
	BOOL* no_ptr = &no;
	BOOL* unset_ptr = &unset;

  BoolBitFields myField = {0};
  myField.fieldTwo = YES;
  myField.fieldFive = YES;

    [pool drain];// Set break point at this line.
    return 0;
}

