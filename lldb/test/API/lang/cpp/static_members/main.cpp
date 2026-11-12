struct A {
  short m_a;
  static long s_b;
  static int s_c;

  long access() {
    return m_a + s_b + s_c; // stop in member function
  }
};

long A::s_b = 2;
int A::s_c = 3;

int main() {
  A my_a;
  my_a.m_a = 1;

  int arr[2]{0};

  my_a.access(); // stop in main
  return 0;
}
