//===-- main.m ------------------------------------------------*- ObjC -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>

int main(int argc, const char **argv)
{
    @autoreleasepool
    {
        const NSUInteger values[] = { 1, 2, 3, 4, 5 };
        
        NSIndexPath* indexPath1 = [NSIndexPath indexPathWithIndexes:values length:1];
        NSIndexPath* indexPath2 = [NSIndexPath indexPathWithIndexes:values length:2];
        NSIndexPath* indexPath3 = [NSIndexPath indexPathWithIndexes:values length:3];
        NSIndexPath* indexPath4 = [NSIndexPath indexPathWithIndexes:values length:4];
        NSIndexPath* indexPath5 = [NSIndexPath indexPathWithIndexes:values length:5];
        
        NSLog(@"%@", indexPath1); // break here
        NSLog(@"%@", indexPath2);
        NSLog(@"%@", indexPath3);
        NSLog(@"%@", indexPath4);
        NSLog(@"%@", indexPath5);
    }
    return 0;
}
