// RUN: llvm-cov show %S/Inputs/deferred-regions.covmapping -instr-profile %S/Inputs/deferred-regions.profdata -show-line-counts-or-regions -dump -path-equivalence=/Users/vk/src/llvm.org-coverage-braces/llvm/test/tools,%S/.. %s 2>%t.markers > %t.out && FileCheck %s -input-file %t.out && FileCheck %s -input-file %t.markers -check-prefix=MARKER

void foo(int x) {
  if (x == 0) { // CHECK: [[@LINE]]|{{ +}}2|
    return; // CHECK: [[@LINE]]|{{ +}}1|
  }

} // CHECK: [[@LINE]]|{{ +}}1|

void bar() {
  return;

} // CHECK: [[@LINE]]|{{ +}}1|

void for_loop() {
  if (false)
    return; // CHECK: [[@LINE]]|{{ +}}0|

  for (int i = 0; i < 10; ++i) { // CHECK: [[@LINE]]|{{ +}}2|
    if (i % 2 == 0) // CHECK: [[@LINE]]|{{ +}}2|
      continue; // CHECK: [[@LINE]]|{{ +}}1|

    if (i % 5 == 0) // CHECK: [[@LINE]]|{{ +}}1|
      break; // CHECK: [[@LINE]]|{{ +}}0|

    int x = i;
    return; // CHECK: [[@LINE]]|{{ +}}1|

  } // CHECK: [[@LINE]]|{{ +}}1|
}

struct Error {};

void while_loop() {
  if (false)
    return; // CHECK: [[@LINE]]|{{ +}}0|

  int x = 0;
  while (++x < 10) { // CHECK: [[@LINE]]|{{ +}}3|
    if (x == 1) // CHECK: [[@LINE]]|{{ +}}2|
      continue; // CHECK: [[@LINE]]|{{ +}}1|

    while (++x < 4) { // CHECK: [[@LINE]]|{{ +}}1|
      if (x == 3) // CHECK: [[@LINE]]|{{ +}}1|
        break; // CHECK: [[@LINE]]|{{ +}}1|
               // CHECK: [[@LINE]]|{{ +}}0|
      while (++x < 5) {} // CHECK: [[@LINE]]|{{ +}}0|
    } // CHECK: [[@LINE]]|{{ +}}1|

    if (x == 0) // CHECK: [[@LINE]]|{{ +}}1|
      throw Error(); // CHECK: [[@LINE]]|{{ +}}0|
                // CHECK: [[@LINE]]|{{ +}}1|
    while (++x < 9) { // CHECK: [[@LINE]]|{{ +}}6|
      if (x == 0) // CHECK: [[@LINE]]|{{ +}}5|
        break; // CHECK: [[@LINE]]|{{ +}}0|

    }
  }
}

void gotos() {
  if (false) // CHECK: [[@LINE]]|{{ +}}1|
    goto out; // CHECK: [[@LINE]]|{{ +}}0|
          // CHECK: [[@LINE]]|{{ +}}1|
  return; // CHECK: [[@LINE]]|{{ +}}1|

out: // CHECK: [[@LINE]]|{{ +}}0|
	return;
}

void if_else(bool flag) {
  if (flag) { // CHECK: [[@LINE]]|{{ +}}2|
    return;   // CHECK: [[@LINE]]|{{ +}}1|
  } else {    // CHECK: [[@LINE]]|{{ +}}1|
    return;   // CHECK: [[@LINE]]|{{ +}}1|
  }           // CHECK: [[@LINE]]|{{ +}}1|
}

int main() {
  foo(0);
  foo(1);
  bar();
  for_loop();
  while_loop();
  gotos();
  if_else(true);
  if_else(false);
  return 0;
}

// MARKER: Highlighted line 17, 5 -> 11
// MARKER-NEXT: Marker at 19:3 = 1
// MARKER-NEXT: Marker at 19:27 = 1
// MARKER-NEXT: Highlighted line 24, 7 -> 12
// MARKER-NEXT: Highlighted line 36, 5 -> 11
// MARKER-NEXT: Highlighted line 46, 1 -> ?
// MARKER-NEXT: Highlighted line 47, 1 -> 7
// MARKER-NEXT: Highlighted line 47, 7 -> 14
// MARKER-NEXT: Highlighted line 47, 14 -> 21
// MARKER-NEXT: Highlighted line 47, 21 -> 23
// MARKER-NEXT: Highlighted line 47, 23 -> 25
// MARKER-NEXT: Highlighted line 51, 7 -> 20
// MARKER-NEXT: Marker at 53:5 = 1
// MARKER-NEXT: Highlighted line 55, 9 -> 14
// MARKER-NEXT: Highlighted line 63, 5 -> 13
// MARKER-NEXT: Highlighted line 67, 1 -> ?
// MARKER-NEXT: Highlighted line 68, 1 -> 8
// MARKER-NEXT: Highlighted line 68, 8 -> ?
// MARKER-NEXT: Highlighted line 69, 1 -> 2
// MARKER-NEXT: Highlighted line 77, 1 -> 2
