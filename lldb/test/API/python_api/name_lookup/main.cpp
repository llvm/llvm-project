#include <stdio.h>

extern "C" int unique_function_name(int i)
{
  return puts(__PRETTY_FUNCTION__);
}

int unique_function_name()
{
  return puts(__PRETTY_FUNCTION__);
}

int unique_function_name(float f)
{
  return puts(__PRETTY_FUNCTION__);
}

namespace e
{
  int unique_function_name()
  {
    return puts(__PRETTY_FUNCTION__);
  }
  
  namespace g
  {
    int unique_function_name()
    {
      return puts(__PRETTY_FUNCTION__);
    }
  }
}

class g
{
public:
  int unique_function_name()
  {
    return puts(__PRETTY_FUNCTION__); 
  }
  
  int unique_function_name(int i)
  {
    return puts(__PRETTY_FUNCTION__); 
  }
};

int main (int argc, char const *argv[])
{
  g g;
  g.unique_function_name();
  g.unique_function_name(argc);
  return 0;
}
