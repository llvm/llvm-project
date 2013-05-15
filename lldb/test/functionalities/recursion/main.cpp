//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

struct node;
struct node {
	int value;
	node* next;
	node () : value(1),next(NULL) {}
	node (int v) : value(v), next(NULL) {}
};

void make_tree(node* root, int count)
{
	int countdown=1;
	if (!root)
		return;
	root->value = countdown;
	while (count > 0)
	{
		root->next = new node(++countdown);
		root = root->next;
		count--;
	}
}

int main (int argc, const char * argv[])
{
	node root(1);
	make_tree(&root,25000);
	return 0; // Set break point at this line.
}
