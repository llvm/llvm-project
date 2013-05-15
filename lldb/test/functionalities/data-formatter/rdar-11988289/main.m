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


	NSArray* keys = @[@"foo",@"bar",@"baz"];
	NSArray* values = @[@"hello",@[@"X",@"Y"],@{@1 : @"one",@2 : @"two"}];
	NSDictionary* dictionary = [NSDictionary dictionaryWithObjects:values forKeys:keys];
	NSMutableDictionary* mutabledict = [NSMutableDictionary dictionaryWithCapacity:5];
	[mutabledict setObject:@"123" forKey:@23];
	[mutabledict setObject:[NSURL URLWithString:@"http://www.apple.com"] forKey:@"foobar"];
	[mutabledict setObject:@[@"a",@12] forKey:@57];
	[mutabledict setObject:dictionary forKey:@"puartist"];

    [pool drain];// Set break point at this line.
    return 0;
}

