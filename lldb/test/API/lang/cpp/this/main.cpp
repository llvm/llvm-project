#include <stdio.h>

template <class T> class A
{
public:
  void accessMember(T a);
  T accessMemberConst() const;
  static int accessStaticMember();

  void accessMemberInline(T a) __attribute__ ((always_inline))
  {
    m_a = a; // breakpoint 4
  }

  T m_a;
  static int s_a;
};

template <class T> int A<T>::s_a = 5;

template <class T> void A<T>::accessMember(T a)
{
  m_a = a; // breakpoint 1
}

template <class T> T A<T>::accessMemberConst() const
{
  return m_a; // breakpoint 2
}

template <class T> int A<T>::accessStaticMember()
{
  return s_a; // breakpoint 3
} 

int main()
{
  A<int> my_a;

  my_a.accessMember(3);
  my_a.accessMemberConst();
  A<int>::accessStaticMember();
  my_a.accessMemberInline(5);
}
