#include <stdio.h>

namespace namesp
{
  class Virtual {
  public:
    virtual void doSomething() {
      printf ("namesp function did something.\n");
    }
  }; 
}

class Virtual {
  public:
  virtual void doSomething() {
    printf("Virtual function did something.\n");
  }
};

int
main()
{
  namesp::Virtual my_outer;
  Virtual my_virtual;

  // Break here to get started
  my_outer.doSomething();
  my_virtual.doSomething();

  return 0;
}
    
