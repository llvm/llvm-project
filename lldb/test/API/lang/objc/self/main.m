//===-- main.m ------------------------------------------*- Objective-C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>

@interface A : NSObject
{
    int m_a;
}
-(id)init;
-(void)accessMember:(int)a;
+(void)accessStaticMember:(int)a;
@end

static int s_a = 5;

@implementation A
-(id)init
{
    self = [super init];
    
    if (self)
        m_a = 2;

    return self;
}

-(void)accessMember:(int)a
{
    m_a = a; // breakpoint 1
}

+(void)accessStaticMember:(int)a
{
    s_a = a; // breakpoint 2
}
@end

int main()
{
    NSAutoreleasePool *pool = [NSAutoreleasePool alloc];
    A *my_a = [[A alloc] init];
    
    [my_a accessMember:3];
    [A accessStaticMember:5];
    
    [pool release];
}
