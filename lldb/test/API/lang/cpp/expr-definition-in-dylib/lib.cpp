#include "lib.h"

#include <cstdio>

int Foo::method() { return -72; }

Foo::Foo() { std::puts(__func__); }

Foo::~Foo() { std::puts(__func__); }

Bar::Bar() { std::puts(__func__); }

Bar::~Bar() { std::puts(__func__); }
