#import <Foundation/Foundation.h>

NSMutableArray *
GetArray ()
{
  static NSMutableArray *the_array = NULL;
  if (the_array == NULL)
    the_array = [[NSMutableArray alloc] init];
  return the_array;
}

int 
AddElement (char *value)
{
  NSString *element = [NSString stringWithUTF8String: value];
  int cur_elem = [GetArray() count];
  [GetArray() addObject: element];
  return cur_elem;
}

const char *
GetElement (int idx)
{
  if (idx >= [GetArray() count])
    return NULL;
  else
    return [[GetArray() objectAtIndex: idx] UTF8String];
}
