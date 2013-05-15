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
    

	NSArray* key = [NSArray arrayWithObjects:@"foo",nil];
	NSArray* value = [NSArray arrayWithObjects:@"key",nil];
	NSDictionary *dict = [NSDictionary dictionaryWithObjects:value forKeys:key];

    CFMutableBagRef mutable_bag_ref = CFBagCreateMutable(NULL, 15, NULL);
    CFBagSetValue(mutable_bag_ref, CFSTR("Hello world"));

    NSCountedSet *nscounted_set = [[NSCountedSet alloc] initWithCapacity:5];
    [nscounted_set addObject:@"foo"];

    NSMutableIndexSet *imset = [[NSMutableIndexSet alloc] init];
    [imset addIndex:4];

    CFBinaryHeapRef binheap_ref = CFBinaryHeapCreate(NULL, 15, &kCFStringBinaryHeapCallBacks, NULL);
    CFBinaryHeapAddValue(binheap_ref, CFSTR("Hello world"));

    NSSet* nsset = [[NSSet alloc] initWithObjects:@"foo",nil];

    NSData *immutableData = [[NSData alloc] initWithBytes:"HELLO" length:1];


    [pool drain];// Set break point at this line.
    return 0;
}

