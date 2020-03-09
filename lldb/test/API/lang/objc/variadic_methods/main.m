//===-- main.m -------------------------------------------*- Objective-C-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>

@interface VarClass : NSObject
- (id) lottaArgs: (id) first, ...;
@end

@implementation VarClass
- (id) lottaArgs: (id) first, ...
{
  return first;
}
@end

int
main()
{
  VarClass *my_var = [[VarClass alloc] init];
  id something = [my_var lottaArgs: @"111", @"222", nil];
  NSLog (@"%@ - %@", my_var, something); //% self.expect("expression -O -- [my_var lottaArgs:@\"111\", @\"222\", nil]", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["111"])
  return 0;
}

