// Test visualization of general branch constructs in C.





void simple_loops() {           // CHECK: @LINE|{{.*}}simple_loops()
  int i;
  for (i = 0; i < 100; ++i) {   // BRCOV: Branch ([[@LINE]]:15): [True: [[#min(C,100)]], False: 1]
  }
  while (i > 0)                 // BRCOV: Branch ([[@LINE]]:10): [True: [[#min(C,100)]], False: 1]
    i--;
  do {} while (i++ < 75);       // BRCOV: Branch ([[@LINE]]:16): [True: [[#min(C,75)]], False: 1]

}

void conditionals() {           // CHECK: @LINE|{{.*}}conditionals()
  for (int i = 0; i < 100; ++i) {//BRCOV: Branch ([[@LINE]]:19): [True: [[#min(C,100)]], False: 1]
    if (i % 2) {                // BRCOV: Branch ([[@LINE]]:9): [True: [[#min(C,50)]], False: [[#min(C,50)]]]
      if (i) {}                 // BRCOV: Branch ([[@LINE]]:11): [True: [[#min(C,50)]], False: 0]
    } else if (i % 3) {         // BRCOV: Branch ([[@LINE]]:16): [True: [[#min(C,33)]], False: [[#min(C,17)]]]
      if (i) {}                 // BRCOV: Branch ([[@LINE]]:11): [True: [[#min(C,33)]], False: 0]
    } else {
      if (i) {}                 // BRCOV: Branch ([[@LINE]]:11): [True: [[#min(C,16)]], False: 1]
    }
                                // BRCOV: Branch ([[@LINE+1]]:9): [True: [[#min(C,100)]], Folded]
    if (1 && i) {}              // BRCOV: Branch ([[@LINE]]:14): [True: [[#min(C,99)]], False: 1]
    if (0 || i) {}              // BRCOV: Branch ([[@LINE]]:9): [Folded, False: [[#min(C,100)]]]
  }                             // BRCOV: Branch ([[@LINE-1]]:14): [True: [[#min(C,99)]], False: 1]

}

void early_exits() {            // CHECK: @LINE|{{.*}}early_exits()
  int i = 0;

  if (i) {}                     // BRCOV: Branch ([[@LINE]]:7): [True: 0, False: 1]

  while (i < 100) {             // BRCOV: Branch ([[@LINE]]:10): [True: [[#min(C,51)]], False: 0]
    i++;
    if (i > 50)                 // BRCOV: Branch ([[@LINE]]:9): [True: 1, False: [[#min(C,50)]]]
      break;
    if (i % 2)                  // BRCOV: Branch ([[@LINE]]:9): [True: [[#min(C,25)]], False: [[#min(C,25)]]]
      continue;
  }

  if (i) {}                     // BRCOV: Branch ([[@LINE]]:7): [True: 1, False: 0]

  do {
    if (i > 75)                 // BRCOV: Branch ([[@LINE]]:9): [True: 1, False: [[#min(C,25)]]]
      return;
    else
      i++;
  } while (i < 100);            // BRCOV: Branch ([[@LINE]]:12): [True: [[#min(C,25)]], False: 0]

  if (i) {}                     // BRCOV: Branch ([[@LINE]]:7): [True: 0, False: 0]

}

void jumps() {                  // CHECK: @LINE|{{.*}}jumps()
  int i;

  for (i = 0; i < 2; ++i) {     // BRCOV: Branch ([[@LINE]]:15): [True: 1, False: 0]
    goto outofloop;
    // Never reached -> no weights
    if (i) {}                   // BRCOV: Branch ([[@LINE]]:9): [True: 0, False: 0]
  }

outofloop:
  if (i) {}                     // BRCOV: Branch ([[@LINE]]:7): [True: 0, False: 1]

  goto loop1;

  while (i) {                   // BRCOV: Branch ([[@LINE]]:10): [True: 0, False: 1]
  loop1:
    if (i) {}                   // BRCOV: Branch ([[@LINE]]:9): [True: 0, False: 1]
  }

  goto loop2;
first:
second:
third:
  i++;
  if (i < 3)                    // BRCOV: Branch ([[@LINE]]:7): [True: [[#min(C,2)]], False: 1]
    goto loop2;

  while (i < 3) {               // BRCOV: Branch ([[@LINE]]:10): [True: 0, False: 1]
  loop2:
    switch (i) {
    case 0:                     // BRCOV: Branch ([[@LINE]]:5): [True: 1, Folded]
      goto first;
    case 1:                     // BRCOV: Branch ([[@LINE]]:5): [True: 1, Folded]
      goto second;
    case 2:                     // BRCOV: Branch ([[@LINE]]:5): [True: 1, Folded]
      goto third;
    }
  }

  for (i = 0; i < 10; ++i) {    // BRCOV: Branch ([[@LINE]]:15): [True: [[#min(C,10)]], False: 1]
    goto withinloop;
                                // never reached -> no weights
    if (i) {}                   // BRCOV: Branch ([[@LINE]]:9): [True: 0, False: 0]
  withinloop:
    if (i) {}                   // BRCOV: Branch ([[@LINE]]:9): [True: [[#min(C,9)]], False: 1]
  }

}

void switches() {               // CHECK: @LINE|{{.*}}switches()
  static int weights[] = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5};

  // No cases -> no weights
  switch (weights[0]) {
  default:                      // BRCOV: Branch ([[@LINE]]:3): [True: 1, Folded]
    break;
  }
                                // BRCOV: Branch ([[@LINE+1]]:63): [True: [[#min(C,15)]], False: 0]
  for (int i = 0, len = sizeof(weights) / sizeof(weights[0]); i < len; ++i) {
    switch (i[weights]) {
    case 1:                     // BRCOV: Branch ([[@LINE]]:5): [True: 1, Folded]
      if (i) {}                 // BRCOV: Branch ([[@LINE]]:11): [True: 0, False: 1]
      // fallthrough
    case 2:                     // BRCOV: Branch ([[@LINE]]:5): [True: [[#min(C,2)]], Folded]
      if (i) {}                 // BRCOV: Branch ([[@LINE]]:11): [True: [[#min(C,2)]], False: 1]
      break;
    case 3:                     // BRCOV: Branch ([[@LINE]]:5): [True: [[#min(C,3)]], Folded]
      if (i) {}                 // BRCOV: Branch ([[@LINE]]:11): [True: [[#min(C,3)]], False: 0]
      continue;
    case 4:                     // BRCOV: Branch ([[@LINE]]:5): [True: [[#min(C,4)]], Folded]
      if (i) {}                 // BRCOV: Branch ([[@LINE]]:11): [True: [[#min(C,4)]], False: 0]
      switch (i) {
      case 6 ... 9:             // BRCOV: Branch ([[@LINE]]:7): [True: [[#min(C,4)]], Folded]
        if (i) {}               // BRCOV: Branch ([[@LINE]]:13): [True: [[#min(C,4)]], False: 0]
        continue;
      }

    default:                    // BRCOV: Branch ([[@LINE]]:5): [True: [[#min(C,5)]], Folded]
      if (i == len - 1)         // BRCOV: Branch ([[@LINE]]:11): [True: 1, False: [[#min(C,4)]]]
        return;
    }
  }

  // Never reached -> no weights
  if (weights[0]) {}            // BRCOV: Branch ([[@LINE]]:7): [True: 0, False: 0]

}

void big_switch() {             // CHECK: @LINE|{{.*}}big_switch()
  for (int i = 0; i < 32; ++i) {// BRCOV: Branch ([[@LINE]]:19): [True: [[#min(C,32)]], False: 1]
    switch (1 << i) {
    case (1 << 0):              // BRCOV: Branch ([[@LINE]]:5): [True: 1, Folded]
      if (i) {}                 // BRCOV: Branch ([[@LINE]]:11): [True: 0, False: 1]
      // fallthrough
    case (1 << 1):              // BRCOV: Branch ([[@LINE]]:5): [True: 1, Folded]
      if (i) {}                 // BRCOV: Branch ([[@LINE]]:11): [True: 1, False: 1]
      break;
    case (1 << 2) ... (1 << 12):// BRCOV: Branch ([[@LINE]]:5): [True: [[#min(C,11)]], Folded]
      if (i) {}                 // BRCOV: Branch ([[@LINE]]:11): [True: [[#min(C,11)]], False: 0]
      break;
      // The branch for the large case range above appears after the case body.

    case (1 << 13):             // BRCOV: Branch ([[@LINE]]:5): [True: 1, Folded]
      if (i) {}                 // BRCOV: Branch ([[@LINE]]:11): [True: 1, False: 0]
      break;
    case (1 << 14) ... (1 << 28)://BRCOV: Branch ([[@LINE]]:5): [True: [[#min(C,15)]], Folded]
      if (i) {}                 // BRCOV: Branch ([[@LINE]]:11): [True: [[#min(C,15)]], False: 0]
      break;
    // The branch for the large case range above appears after the case body.

    case (1 << 29) ... ((1 << 29) + 1):
      if (i) {}                 // BRCOV: Branch ([[@LINE]]:11): [True: 1, False: 0]
      break;
    default:                    // BRCOV: Branch ([[@LINE]]:5): [True: [[#min(C,2)]], Folded]
      if (i) {}                 // BRCOV: Branch ([[@LINE]]:11): [True: [[#min(C,2)]], False: 0]
      break;
    }
  }

}

void boolean_operators() {      // CHECK: @LINE|{{.*}}boolean_operators()
  int v;
  for (int i = 0; i < 100; ++i) {
    v = i % 3 || i;             // BRCOV: Branch ([[@LINE]]:9): [True: [[#min(C,66)]], False: [[#min(C,34)]]]
                                // BRCOV: Branch ([[@LINE-1]]:18): [True: [[#min(C,33)]], False: 1]
    v = i % 3 && i;             // BRCOV: Branch ([[@LINE]]:9): [True: [[#min(C,66)]], False: [[#min(C,34)]]]
                                // BRCOV: Branch ([[@LINE-1]]:18): [True: [[#min(C,66)]], False: 0]
    v = i % 3 || i % 2 || i;    // BRCOV: Branch ([[@LINE]]:9): [True: [[#min(C,66)]], False: [[#min(C,34)]]]
                                // BRCOV: Branch ([[@LINE-1]]:18): [True: [[#min(C,17)]], False: [[#min(C,17)]]]
    v = i % 2 && i % 3 && i;    // BRCOV: Branch ([[@LINE-2]]:27): [True: [[#min(C,16)]], False: 1]
  }                             // BRCOV: Branch ([[@LINE-1]]:9): [True: [[#min(C,50)]], False: [[#min(C,50)]]]
                                // BRCOV: Branch ([[@LINE-2]]:18): [True: [[#min(C,33)]], False: [[#min(C,17)]]]
}                               // BRCOV: Branch ([[@LINE-3]]:27): [True: [[#min(C,33)]], False: 0]

void boolop_loops() {           // CHECK: @LINE|{{.*}}boolop_loops()
  int i = 100;

  while (i && i > 50)           // BRCOV: Branch ([[@LINE]]:10): [True: [[#min(C,51)]], False: 0]
    i--;                        // BRCOV: Branch ([[@LINE-1]]:15): [True: [[#min(C,50)]], False: 1]

  while ((i % 2) || (i > 0))    // BRCOV: Branch ([[@LINE]]:10): [True: [[#min(C,25)]], False: [[#min(C,26)]]]
    i--;                        // BRCOV: Branch ([[@LINE-1]]:21): [True: [[#min(C,25)]], False: 1]

  for (i = 100; i && i > 50; --i);  // BRCOV: Branch ([[@LINE]]:17): [True: [[#min(C,51)]], False: 0]
                                    // BRCOV: Branch ([[@LINE-1]]:22): [True: [[#min(C,50)]], False: 1]
  for (; (i % 2) || (i > 0); --i);  // BRCOV: Branch ([[@LINE]]:10): [True: [[#min(C,25)]], False: [[#min(C,26)]]]
                                    // BRCOV: Branch ([[@LINE-1]]:21): [True: [[#min(C,25)]], False: 1]
}

void conditional_operator() {   // CHECK: @LINE|{{.*}}conditional_operator()
  int i = 100;

  int j = i < 50 ? i : 1;       // BRCOV: Branch ([[@LINE]]:11): [True: 0, False: 1]

  int k = i ?: 0;               // BRCOV: Branch ([[@LINE]]:11): [True: 1, False: 0]

}

void do_fallthrough() {         // CHECK: @LINE|{{.*}}do_fallthrough()
  for (int i = 0; i < 10; ++i) {// BRCOV: Branch ([[@LINE]]:19): [True: [[#min(C,10)]], False: 1]
    int j = 0;
    do {
      // The number of exits out of this do-loop via the break statement
      // exceeds the counter value for the loop (which does not include the
      // fallthrough count). Make sure that does not violate any assertions.
      if (i < 8) break;
      j++;
    } while (j < 2);            // BRCOV: Branch ([[@LINE]]:14): [True: [[#min(C,2)]], False: [[#min(C,2)]]]
  }
}

static void static_func() {     // CHECK: @LINE|{{.*}}static_func()
  for (int i = 0; i < 10; ++i) {// BRCOV: Branch ([[@LINE]]:19): [True: [[#min(C,10)]], False: 1]
  }
}










int main(int argc, const char *argv[]) {
  simple_loops();
  conditionals();
  early_exits();
  jumps();
  switches();
  big_switch();
  boolean_operators();
  boolop_loops();
  conditional_operator();
  do_fallthrough();
  static_func();
  (void)0;
  (void)0;
  return 0;
}
