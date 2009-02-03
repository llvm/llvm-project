// RUN: %llvmgcc -xc %s -S -o - | grep {internal constant }

// The synthetic global made by the CFE for big initializer should be marked
// constant.

void bar();
void foo() {
  char Blah[] = "asdlfkajsdlfkajsd;lfkajds;lfkjasd;flkajsd;lkfja;sdlkfjasd";
  bar(Blah);
}
