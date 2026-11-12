#include <stdio.h>
#include <memory>

class BaseClass
{
public:
    BaseClass();
    virtual ~BaseClass() { }
};

class DerivedClass : public BaseClass
{
public:
    DerivedClass();
    virtual ~DerivedClass() { }
protected:
    int mem;
};

BaseClass::BaseClass()
{
}

DerivedClass::DerivedClass() : BaseClass()
{
    mem = 101;
}

int
main (int argc, char **argv)
{
  BaseClass *b = nullptr; // Break here and check b has 0 children
  b = new DerivedClass();  // Break here and check b still has 0 children
  b = nullptr;  // Break here and check b has one child now
  return 0; // Break here and check b has 0 children again
}
