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

	BOOL yes  = YES;
	BOOL no = NO;
    BOOL unset = 12;
	
	BOOL &yes_ref = yes;
	BOOL &no_ref = no;
	BOOL &unset_ref = unset;
	
	BOOL* yes_ptr = &yes;
	BOOL* no_ptr = &no;
	BOOL* unset_ptr = &unset;

    [pool drain];// Set break point at this line.
    return 0;
}

