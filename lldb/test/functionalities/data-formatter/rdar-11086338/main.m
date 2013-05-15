//===-- main.m ------------------------------------------------*- ObjC -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

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

    [pool drain];// Set break point at this line.
    return 0;
}

