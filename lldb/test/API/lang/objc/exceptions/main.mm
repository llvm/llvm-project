//===-- main.m ------------------------------------------------*- ObjC -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>

#import <exception>
#import <stdexcept>

@interface MyCustomException: NSException
@end
@implementation MyCustomException
@end

void foo(int n)
{
    NSDictionary *info = [NSDictionary dictionaryWithObjectsAndKeys:@"some_value", @"some_key", nil];
    switch (n) {
        case 0:
            @throw [[NSException alloc] initWithName:@"ThrownException" reason:@"SomeReason" userInfo:info];
        case 1:
            @throw [[MyCustomException alloc] initWithName:@"ThrownException" reason:@"SomeReason" userInfo:info];
        case 2:
            throw std::runtime_error("C++ exception");
    }
}

void rethrow(int n)
{
    @try {
        foo(n);
    } @catch(NSException *e) {
        @throw;
    }
}

int main(int argc, const char * argv[])
{
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

    NSDictionary *info = [NSDictionary dictionaryWithObjectsAndKeys:@"some_value", @"some_key", nil];
    NSException *e1 = [[NSException alloc] initWithName:@"ExceptionName" reason:@"SomeReason" userInfo:info];
    NSException *e2;

    @try {
        foo(atoi(argv[1]));
    } @catch(NSException *e) {
        e2 = e;
    }

    NSLog(@"1"); // Set break point at this line.

    rethrow(atoi(argv[1]));

    [pool drain];
    return 0;
}

