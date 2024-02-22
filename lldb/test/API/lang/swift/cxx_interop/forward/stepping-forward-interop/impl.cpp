#include "cxx-functions.h"

void cxxFunction() {
    int a = 42;
}

void CxxClass::cxxMethod() {
    int a = 42;
}

ClassWithConstructor::ClassWithConstructor(int a, bool b, double c)
    : a(a), b(b), c(c) {}

int ClassWithExtension::definedInExtension() { return val; }

int ClassWithCallOperator::operator()() { return 42; }

