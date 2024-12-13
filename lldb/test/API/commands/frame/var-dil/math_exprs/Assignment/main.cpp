int main(int argc, char** argv) {
  int i = 10;
  float f = 1.5f;
  float* p = &f;

  enum Enum { ONE, TWO };
  Enum eOne = ONE;
  Enum eTwo = TWO;

  // BREAK(TestAssignment)
  // BREAK(TestCompositeAssignmentInvalid)
  // BREAK(TestCompositeAssignmentAdd)
  // BREAK(TestCompositeAssignmentSub)
  // BREAK(TestCompositeAssignmentMul)
  // BREAK(TestCompositeAssignmentDiv)
  // BREAK(TestCompositeAssignmentRem)
  // BREAK(TestCompositeAssignmentBitwise)
  return 0;  // Set a breakpoint here
}
