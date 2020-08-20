#import <Foundation/Foundation.h>

int main (int argc, const char * argv[])
{

    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];


	NSMutableArray* arr = [[NSMutableArray alloc] init];
	[arr addObject:@"hello"];
	[arr addObject:@"world"];
	[arr addObject:@"this"];
	[arr addObject:@"is"];
	[arr addObject:@"me"];
	[arr addObject:[NSURL URLWithString:@"http://www.apple.com/"]];

	NSDate *aDate = [NSDate distantFuture];
	NSValue *aValue = [NSNumber numberWithInt:5];
	NSString *aString = @"a string";

	NSArray *other_arr = [NSArray arrayWithObjects:aDate, aValue, aString, arr, nil];
	NSArray *empty_arr = @[];

    [pool drain];// Set break point at this line.
    return 0;
}

