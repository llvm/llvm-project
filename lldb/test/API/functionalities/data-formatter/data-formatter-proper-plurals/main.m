#import <Foundation/Foundation.h>

int main (int argc, const char * argv[])
{
  NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

	NSArray* key = [NSArray arrayWithObjects:@"foo",nil];
	NSArray* value = [NSArray arrayWithObjects:@"key",nil];
	NSDictionary *dict = [NSDictionary dictionaryWithObjects:value forKeys:key];

  NSMutableIndexSet *imset = [[NSMutableIndexSet alloc] init];
  [imset addIndex:4];

  CFBinaryHeapRef binheap_ref = CFBinaryHeapCreate(NULL, 15, &kCFStringBinaryHeapCallBacks, NULL);
  CFBinaryHeapAddValue(binheap_ref, CFSTR("Hello world"));

  NSData *immutableData = [[NSData alloc] initWithBytes:"HELLO" length:1];

  [pool drain];// Set break point at this line.
  return 0;
}

