typedef int TTT;

int main() {
  int i = 0;
  TTT &l_ref = i;
  TTT &&r_ref = static_cast<TTT &&>(i);
  return l_ref; // break here
}
