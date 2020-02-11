//===-- main.m ------------------------------------------------*- ObjC -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>

int main (int argc, const char * argv[])
{
    
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

    NSArray* foo = [NSArray arrayWithObjects:@1,@2,@3,@4,@5, nil];
    NSDictionary *bar = @{@1 : @"one",@2 : @"two", @3 : @"three", @4 : @"four", @5 : @"five", @6 : @"six", @7 : @"seven"};
    id x = foo;
    x = bar; // Set break point at this line.

    [pool drain];
    return 0;
}

