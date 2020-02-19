//===-- main.m --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#import <Foundation/Foundation.h>

@interface ThisClassTestsThings : NSObject
@end

@implementation ThisClassTestsThings
- (int)doSomething {

    id s = self;
    NSLog(@"%@",s); //% s = self.frame().FindVariable("s"); s.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
    //% s_type = s.GetType()
    //% typeClass = s_type.GetTypeClass()
    //% condition = (typeClass == lldb.eTypeClassClass) or (typeClass ==lldb.eTypeClassObjCObject) or (typeClass == lldb.eTypeClassObjCInterface) or (typeClass == lldb.eTypeClassObjCObjectPointer) or (typeClass == lldb.eTypeClassPointer)
    //% self.assertTrue(condition, "s has the wrong TypeClass")
    return 0;
}
- (id)init {
    return (self = [super init]);
}
@end


int main (int argc, char const *argv[])
{
    return [[[ThisClassTestsThings alloc] init] doSomething];
}
