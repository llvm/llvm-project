// RUN: llvm-cov show %S/Inputs/deferred-regions.covmapping -instr-profile %S/Inputs/deferred-regions.profdata -show-line-counts-or-regions -dump -path-equivalence=/Users/vk/src/llvm.org-coverage-braces/llvm/test/tools,%S/.. %s 2>&1 | FileCheck %s

void foo(int x) {
  if (x == 0) {
    return; // CHECK: [[@LINE]]|{{ +}}1|
  }

} // CHECK: [[@LINE]]|{{ +}}2|

void bar() {
  return;

} // CHECK: [[@LINE]]|{{ +}}1|

void for_loop() {
  if (false)
    return; // CHECK: [[@LINE]]|{{ +}}0|

  for (int i = 0; i < 10; ++i) { // CHECK: [[@LINE]]|{{ +}}2|
    if (i % 2 == 0)
      continue; // CHECK: [[@LINE]]|{{ +}}1|

    if (i % 5 == 0)
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
    if (x == 1)
      continue; // CHECK: [[@LINE]]|{{ +}}1|

    while (++x < 4) { // CHECK: [[@LINE]]|{{ +}}1|
      if (x == 3)
        break; // CHECK: [[@LINE]]|{{ +}}1|
               // CHECK: [[@LINE]]|{{ +}}0|
      while (++x < 5) {} // CHECK: [[@LINE]]|{{ +}}0|
    } // CHECK: [[@LINE]]|{{ +}}1|

    if (x == 0)
      throw Error(); // CHECK: [[@LINE]]|{{ +}}0|

    while (++x < 9) { // CHECK: [[@LINE]]|{{ +}}6|
      if (x == 0) // CHECK: [[@LINE]]|{{ +}}5|
        break; // CHECK: [[@LINE]]|{{ +}}0|

    }
  }
}

void gotos() {
  if (false)
    goto out; // CHECK: [[@LINE]]|{{ +}}0|

  return;

out: // CHECK: [[@LINE]]|{{ +}}1|
	return;
}

int main() {
  foo(0);
  foo(1);
  bar();
  for_loop();
  while_loop();
  gotos();
  return 0;
}
