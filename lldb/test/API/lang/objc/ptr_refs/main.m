//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>

@interface MyClass : NSObject {
};
-(void)test;
@end

@implementation MyClass
-(void)test {
    printf("%p\n", self); // break here
}
@end

@interface MyOwner : NSObject {
  @public id ownedThing; // should be id, to test <rdar://problem/31363513>
};
@end

@implementation MyOwner
@end

int main (int argc, char const *argv[]) {
    @autoreleasepool {
        MyOwner *owner = [[MyOwner alloc] init];
        owner->ownedThing = [[MyClass alloc] init];
        [(MyClass*)owner->ownedThing test];
    }
    return 0;
}

