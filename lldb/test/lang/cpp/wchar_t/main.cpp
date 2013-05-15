//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

template <typename T>
class Foo
{
public:
    Foo () : object() {}
    Foo (T x) : object(x) {}
    T getObject() { return object; }
private:
    T object;
};


int main (int argc, char const *argv[])
{
    Foo<int> foo_x('a');
    Foo<wchar_t> foo_y(L'a');
    const wchar_t *mazeltov = L"מזל טוב";
    return 0; // Set break point at this line.
}
