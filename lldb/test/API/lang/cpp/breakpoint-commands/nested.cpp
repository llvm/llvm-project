#include <stdio.h>

namespace Foo
{
  namespace Bar
  {
    class Baz
    {
    public:
      Baz (int value):m_value(value) {}
      int Function () 
      {
        printf ("%s returning: %d.\n", __FUNCTION__, m_value);
        return m_value + 1;
      }
    private:
      int m_value;
    };

    class Baz2
    {
    public:
      Baz2 (int value):m_value(value) {}
      int Function () 
      {
        printf ("%s returning: %d.\n", __FUNCTION__, m_value);
        return m_value + 2;
      }
    private:
      int m_value;
    };

    static int bar_value = 20;
    int Function ()
    {
      printf ("%s returning: %d.\n", __FUNCTION__, bar_value);
      return bar_value + 3;
    }
  }
}

class Baz
{
public:
    Baz (int value):m_value(value) {}
    int Function () 
    {
        printf ("%s returning: %d.\n", __FUNCTION__, m_value);
        return m_value + 4;
    }
private:
    int m_value;
};

int
Function ()
{
    printf ("I am a global function, I return 333.\n");
    return 333;
}

int main ()
{
  Foo::Bar::Baz mine(200);
  Foo::Bar::Baz2 mine2(300);
  ::Baz bare_baz (500);

  printf ("Yup, got %d from Baz.\n", mine.Function());
  printf ("Yup, got %d from Baz.\n", mine2.Function());
  printf ("Yup, got %d from Baz.\n", bare_baz.Function());  
  printf ("And  got %d from Bar.\n", Foo::Bar::Function());
  printf ("And  got %d from ::.\n", ::Function());

  return 0;

}
