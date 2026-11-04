int break_on_me() {
  int i = 10; // breakhere
  i++;
  return i; // secondbreak
}

int main() { return break_on_me(); }
