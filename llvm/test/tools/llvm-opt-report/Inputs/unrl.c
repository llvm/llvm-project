void bar();

void foo() {
  for (int i = 0; i < 5; ++i)
    bar();

  for (int i = 0; i < 11; ++i)
    bar();
}

