typedef int TTT;
typedef int &td_int_ref;

int main() {
  int i = 0;
  // references to typedefs
  TTT &l_ref = i;
  TTT &&r_ref = static_cast<TTT &&>(i);
  // typedef of a reference
  td_int_ref td_to_ref_type = i;

  return l_ref; // break here
}
