//===-- main.m ------------------------------------------------*- ObjC -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>

@interface Foo: NSObject
{}
- (id) init;
@end

@interface Bar: Foo
{
	int _iVar;
}
- (id) init;
@end

@implementation Foo

- (id) init
{
	self = [super init];
	return self;
}

@end

@implementation Bar

- (id) init
{
	self = [super init];
	if (self)
		self->_iVar = 5;
	return self;
}

@end

int main()
{
	Bar* aBar = [Bar new];
	id nothing = [aBar noSuchSelector]; // Break at this line
	return 0;
}

