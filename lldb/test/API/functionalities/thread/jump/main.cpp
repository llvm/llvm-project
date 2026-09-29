// This test verifies the correct handling of program counter jumps.

int otherfn();

template<typename T>
T min(T a, T b)
{
    if (a < b)
    {
        return a; // 1st marker
    } else {
        return b; // 2nd marker
    }
}

int jump_positive_offset() {
  int var_1 = 10;
  var_1 = 20; // breakpoint 1

  int var_2 = 40; // jump_offset 1
  return var_2;
}

int jump_negative_offset() {
  int var_3 = 10; // jump_offset 2
  var_3 = 99;

  return var_3; // breakpoint 2
}

int main ()
{
    int i;
    double j;
    int min_i_a = 4, min_i_b = 5;
    double min_j_a = 7.0, min_j_b = 8.0;
    i = min(min_i_a, min_i_b); // 3rd marker
    j = min(min_j_a, min_j_b); // 4th marker

    jump_positive_offset();
    jump_negative_offset();
    return 0;
}
