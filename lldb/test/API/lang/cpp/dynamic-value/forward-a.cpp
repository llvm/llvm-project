class A;

A *take_A(A *a) {
  return a; // Break here for forward declared A
}
