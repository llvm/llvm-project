#include "base.h"

FooBase::FooBase() : x(12345) {}
ns::Foo2Base::Foo2Base() : x(23456) {}

void FooBase::bar() {}
void ns::Foo2Base::bar() {}
