// RUN: %libomptarget-compilexx-run-and-check-generic

#include <assert.h>
#include <omp.h>
#include <stdio.h>

#pragma omp declare target

class Parent1 {
public:
  virtual int Parent1Foo(int x) { return x; }
};

class Parent2 {
public:
  virtual int Parent2Foo(int x) { return 2 * x; }
};

class Parent3 {
public:
  virtual int Parent3Foo(int x) { return 3 * x; }
};

class Parent4 {
public:
  virtual int Parent4Foo(int x) { return 4 * x; }
};

class Parent5 {
public:
  virtual int Parent5Foo(int x) { return 5 * x; }
};

class Child : public Parent1,
              public Parent2,
              public Parent3,
              public Parent4,
              public Parent5 {
public:
  int Parent1Foo(int x) override { return 6 * x; }
  int Parent2Foo(int x) override { return 7 * x; }
  int Parent3Foo(int x) override { return 8 * x; }

  // parent 4 stays the same

  int Parent5Foo(int x) override { return 10 * x; }
};

#pragma omp end declare target

int test_five_parent_inheritance() {
  Parent1 parent1;
  Parent2 parent2;
  Parent3 parent3;
  Parent4 parent4;
  Parent5 parent5;
  Child child;

  // map results back to host
  int result_parent1, result_parent2, result_parent3, result_parent4,
      result_parent5;
  int result_child_parent1, result_child_parent2, result_child_parent3,
      result_child_parent4, result_child_parent5;
  int result_child_as_parent1, result_child_as_parent2, result_child_as_parent3,
      result_child_as_parent4, result_child_as_parent5;

  // Add reference-based results
  int ref_result_parent1, ref_result_parent2, ref_result_parent3,
      ref_result_parent4, ref_result_parent5;
  int ref_result_child_parent1, ref_result_child_parent2,
      ref_result_child_parent3, ref_result_child_parent4,
      ref_result_child_parent5;
  int ref_result_child_as_parent1, ref_result_child_as_parent2,
      ref_result_child_as_parent3, ref_result_child_as_parent4,
      ref_result_child_as_parent5;

#pragma omp target data map(parent1, parent2, parent3, parent4, parent5, child)
  {
    // Base class pointers
    Parent1 *ptr_parent1 = &parent1;
    Parent2 *ptr_parent2 = &parent2;
    Parent3 *ptr_parent3 = &parent3;
    Parent4 *ptr_parent4 = &parent4;
    Parent5 *ptr_parent5 = &parent5;

    // Base class references
    Parent1 &ref_parent1 = parent1;
    Parent2 &ref_parent2 = parent2;
    Parent3 &ref_parent3 = parent3;
    Parent4 &ref_parent4 = parent4;
    Parent5 &ref_parent5 = parent5;

    // Child pointers
    Child *ptr_child = &child;
    Parent1 *ptr_child_cast_parent1 = &child;
    Parent2 *ptr_child_cast_parent2 = &child;
    Parent3 *ptr_child_cast_parent3 = &child;
    Parent4 *ptr_child_cast_parent4 = &child;
    Parent5 *ptr_child_cast_parent5 = &child;

    // Child references
    Child &ref_child = child;
    Parent1 &ref_child_cast_parent1 = child;
    Parent2 &ref_child_cast_parent2 = child;
    Parent3 &ref_child_cast_parent3 = child;
    Parent4 &ref_child_cast_parent4 = child;
    Parent5 &ref_child_cast_parent5 = child;

#pragma omp target map(                                                        \
        from : result_parent1, result_parent2, result_parent3, result_parent4, \
            result_parent5, result_child_parent1, result_child_parent2,        \
            result_child_parent3, result_child_parent4, result_child_parent5,  \
            result_child_as_parent1, result_child_as_parent2,                  \
            result_child_as_parent3, result_child_as_parent4,                  \
            result_child_as_parent5, ref_result_parent1, ref_result_parent2,   \
            ref_result_parent3, ref_result_parent4, ref_result_parent5,        \
            ref_result_child_parent1, ref_result_child_parent2,                \
            ref_result_child_parent3, ref_result_child_parent4,                \
            ref_result_child_parent5, ref_result_child_as_parent1,             \
            ref_result_child_as_parent2, ref_result_child_as_parent3,          \
            ref_result_child_as_parent4, ref_result_child_as_parent5)          \
    map(ptr_parent1[0 : 0], ptr_parent2[0 : 0], ptr_parent3[0 : 0],            \
            ptr_parent4[0 : 0], ptr_parent5[0 : 0], ptr_child[0 : 0],          \
            ptr_child_cast_parent1[0 : 0], ptr_child_cast_parent2[0 : 0],      \
            ptr_child_cast_parent3[0 : 0], ptr_child_cast_parent4[0 : 0],      \
            ptr_child_cast_parent5[0 : 0], ref_parent1, ref_parent2,           \
            ref_parent3, ref_parent4, ref_parent5, ref_child,                  \
            ref_child_cast_parent1, ref_child_cast_parent2,                    \
            ref_child_cast_parent3, ref_child_cast_parent4,                    \
            ref_child_cast_parent5)
    {
      // Base class calls using pointers
      result_parent1 = ptr_parent1->Parent1Foo(1);
      result_parent2 = ptr_parent2->Parent2Foo(1);
      result_parent3 = ptr_parent3->Parent3Foo(1);
      result_parent4 = ptr_parent4->Parent4Foo(1);
      result_parent5 = ptr_parent5->Parent5Foo(1);

      // Direct child calls using pointers
      result_child_parent1 = ptr_child->Parent1Foo(1);
      result_child_parent2 = ptr_child->Parent2Foo(1);
      result_child_parent3 = ptr_child->Parent3Foo(1);
      result_child_parent4 = ptr_child->Parent4Foo(1);
      result_child_parent5 = ptr_child->Parent5Foo(1);

      // Polymorphic calls through parent pointers
      result_child_as_parent1 = ptr_child_cast_parent1->Parent1Foo(1);
      result_child_as_parent2 = ptr_child_cast_parent2->Parent2Foo(1);
      result_child_as_parent3 = ptr_child_cast_parent3->Parent3Foo(1);
      result_child_as_parent4 = ptr_child_cast_parent4->Parent4Foo(1);
      result_child_as_parent5 = ptr_child_cast_parent5->Parent5Foo(1);

      // Base class calls using references
      ref_result_parent1 = ref_parent1.Parent1Foo(1);
      ref_result_parent2 = ref_parent2.Parent2Foo(1);
      ref_result_parent3 = ref_parent3.Parent3Foo(1);
      ref_result_parent4 = ref_parent4.Parent4Foo(1);
      ref_result_parent5 = ref_parent5.Parent5Foo(1);

      // Direct child calls using references
      ref_result_child_parent1 = ref_child.Parent1Foo(1);
      ref_result_child_parent2 = ref_child.Parent2Foo(1);
      ref_result_child_parent3 = ref_child.Parent3Foo(1);
      ref_result_child_parent4 = ref_child.Parent4Foo(1);
      ref_result_child_parent5 = ref_child.Parent5Foo(1);

      // Polymorphic calls through parent references
      ref_result_child_as_parent1 = ref_child_cast_parent1.Parent1Foo(1);
      ref_result_child_as_parent2 = ref_child_cast_parent2.Parent2Foo(1);
      ref_result_child_as_parent3 = ref_child_cast_parent3.Parent3Foo(1);
      ref_result_child_as_parent4 = ref_child_cast_parent4.Parent4Foo(1);
      ref_result_child_as_parent5 = ref_child_cast_parent5.Parent5Foo(1);
    }
  }

  // Verify pointer-based results
  assert(result_parent1 == 1 && "Parent1 Foo failed");
  assert(result_parent2 == 2 && "Parent2 Foo failed");
  assert(result_parent3 == 3 && "Parent3 Foo failed");
  assert(result_parent4 == 4 && "Parent4 Foo failed");
  assert(result_parent5 == 5 && "Parent5 Foo failed");

  assert(result_child_parent1 == 6 && "Child Parent1 Foo failed");
  assert(result_child_parent2 == 7 && "Child Parent2 Foo failed");
  assert(result_child_parent3 == 8 && "Child Parent3 Foo failed");
  assert(result_child_parent4 == 4 && "Child Parent4 Foo failed");
  assert(result_child_parent5 == 10 && "Child Parent5 Foo failed");

  assert(result_child_as_parent1 == 6 && "Child Parent1 Cast Foo failed");
  assert(result_child_as_parent2 == 7 && "Child Parent2 Cast Foo failed");
  assert(result_child_as_parent3 == 8 && "Child Parent3 Cast Foo failed");
  assert(result_child_as_parent4 == 4 && "Child Parent4 Cast Foo failed");
  assert(result_child_as_parent5 == 10 && "Child Parent5 Cast Foo failed");

  // Verify reference-based results
  assert(ref_result_parent1 == 1 && "Reference Parent1 Foo failed");
  assert(ref_result_parent2 == 2 && "Reference Parent2 Foo failed");
  assert(ref_result_parent3 == 3 && "Reference Parent3 Foo failed");
  assert(ref_result_parent4 == 4 && "Reference Parent4 Foo failed");
  assert(ref_result_parent5 == 5 && "Reference Parent5 Foo failed");

  assert(ref_result_child_parent1 == 6 && "Reference Child Parent1 Foo failed");
  assert(ref_result_child_parent2 == 7 && "Reference Child Parent2 Foo failed");
  assert(ref_result_child_parent3 == 8 && "Reference Child Parent3 Foo failed");
  assert(ref_result_child_parent4 == 4 && "Reference Child Parent4 Foo failed");
  assert(ref_result_child_parent5 == 10 &&
         "Reference Child Parent5 Foo failed");

  assert(ref_result_child_as_parent1 == 6 &&
         "Reference Child Parent1 Cast Foo failed");
  assert(ref_result_child_as_parent2 == 7 &&
         "Reference Child Parent2 Cast Foo failed");
  assert(ref_result_child_as_parent3 == 8 &&
         "Reference Child Parent3 Cast Foo failed");
  assert(ref_result_child_as_parent4 == 4 &&
         "Reference Child Parent4 Cast Foo failed");
  assert(ref_result_child_as_parent5 == 10 &&
         "Reference Child Parent5 Cast Foo failed");

  return 0;
}

int test_five_parent_inheritance_implicit() {
  Parent1 parent1;
  Parent2 parent2;
  Parent3 parent3;
  Parent4 parent4;
  Parent5 parent5;
  Child child;

  // map results back to host
  int result_parent1, result_parent2, result_parent3, result_parent4,
      result_parent5;
  int result_child_parent1, result_child_parent2, result_child_parent3,
      result_child_parent4, result_child_parent5;
  int result_child_as_parent1, result_child_as_parent2, result_child_as_parent3,
      result_child_as_parent4, result_child_as_parent5;

  // Add reference-based results
  int ref_result_parent1, ref_result_parent2, ref_result_parent3,
      ref_result_parent4, ref_result_parent5;
  int ref_result_child_parent1, ref_result_child_parent2,
      ref_result_child_parent3, ref_result_child_parent4,
      ref_result_child_parent5;
  int ref_result_child_as_parent1, ref_result_child_as_parent2,
      ref_result_child_as_parent3, ref_result_child_as_parent4,
      ref_result_child_as_parent5;

#pragma omp target data map(parent1, parent2, parent3, parent4, parent5, child)
  {
    // Base class pointers
    Parent1 *ptr_parent1 = &parent1;
    Parent2 *ptr_parent2 = &parent2;
    Parent3 *ptr_parent3 = &parent3;
    Parent4 *ptr_parent4 = &parent4;
    Parent5 *ptr_parent5 = &parent5;

    // Base class references
    Parent1 &ref_parent1 = parent1;
    Parent2 &ref_parent2 = parent2;
    Parent3 &ref_parent3 = parent3;
    Parent4 &ref_parent4 = parent4;
    Parent5 &ref_parent5 = parent5;

    // Child pointers
    Child *ptr_child = &child;
    Parent1 *ptr_child_cast_parent1 = &child;
    Parent2 *ptr_child_cast_parent2 = &child;
    Parent3 *ptr_child_cast_parent3 = &child;
    Parent4 *ptr_child_cast_parent4 = &child;
    Parent5 *ptr_child_cast_parent5 = &child;

    // Child references
    Child &ref_child = child;
    Parent1 &ref_child_cast_parent1 = child;
    Parent2 &ref_child_cast_parent2 = child;
    Parent3 &ref_child_cast_parent3 = child;
    Parent4 &ref_child_cast_parent4 = child;
    Parent5 &ref_child_cast_parent5 = child;

#pragma omp target map(                                                        \
        from : result_parent1, result_parent2, result_parent3, result_parent4, \
            result_parent5, result_child_parent1, result_child_parent2,        \
            result_child_parent3, result_child_parent4, result_child_parent5,  \
            result_child_as_parent1, result_child_as_parent2,                  \
            result_child_as_parent3, result_child_as_parent4,                  \
            result_child_as_parent5, ref_result_parent1, ref_result_parent2,   \
            ref_result_parent3, ref_result_parent4, ref_result_parent5,        \
            ref_result_child_parent1, ref_result_child_parent2,                \
            ref_result_child_parent3, ref_result_child_parent4,                \
            ref_result_child_parent5, ref_result_child_as_parent1,             \
            ref_result_child_as_parent2, ref_result_child_as_parent3,          \
            ref_result_child_as_parent4, ref_result_child_as_parent5)
    {
      // Base class calls using pointers
      result_parent1 = ptr_parent1->Parent1Foo(1);
      result_parent2 = ptr_parent2->Parent2Foo(1);
      result_parent3 = ptr_parent3->Parent3Foo(1);
      result_parent4 = ptr_parent4->Parent4Foo(1);
      result_parent5 = ptr_parent5->Parent5Foo(1);

      // Direct child calls using pointers
      result_child_parent1 = ptr_child->Parent1Foo(1);
      result_child_parent2 = ptr_child->Parent2Foo(1);
      result_child_parent3 = ptr_child->Parent3Foo(1);
      result_child_parent4 = ptr_child->Parent4Foo(1);
      result_child_parent5 = ptr_child->Parent5Foo(1);

      // Polymorphic calls through parent pointers
      result_child_as_parent1 = ptr_child_cast_parent1->Parent1Foo(1);
      result_child_as_parent2 = ptr_child_cast_parent2->Parent2Foo(1);
      result_child_as_parent3 = ptr_child_cast_parent3->Parent3Foo(1);
      result_child_as_parent4 = ptr_child_cast_parent4->Parent4Foo(1);
      result_child_as_parent5 = ptr_child_cast_parent5->Parent5Foo(1);

      // Base class calls using references
      ref_result_parent1 = ref_parent1.Parent1Foo(1);
      ref_result_parent2 = ref_parent2.Parent2Foo(1);
      ref_result_parent3 = ref_parent3.Parent3Foo(1);
      ref_result_parent4 = ref_parent4.Parent4Foo(1);
      ref_result_parent5 = ref_parent5.Parent5Foo(1);

      // Direct child calls using references
      ref_result_child_parent1 = ref_child.Parent1Foo(1);
      ref_result_child_parent2 = ref_child.Parent2Foo(1);
      ref_result_child_parent3 = ref_child.Parent3Foo(1);
      ref_result_child_parent4 = ref_child.Parent4Foo(1);
      ref_result_child_parent5 = ref_child.Parent5Foo(1);

      // Polymorphic calls through parent references
      ref_result_child_as_parent1 = ref_child_cast_parent1.Parent1Foo(1);
      ref_result_child_as_parent2 = ref_child_cast_parent2.Parent2Foo(1);
      ref_result_child_as_parent3 = ref_child_cast_parent3.Parent3Foo(1);
      ref_result_child_as_parent4 = ref_child_cast_parent4.Parent4Foo(1);
      ref_result_child_as_parent5 = ref_child_cast_parent5.Parent5Foo(1);
    }
  }
  // Verify pointer-based results
  assert(result_parent1 == 1 && "Implicit Parent1 Foo failed");
  assert(result_parent2 == 2 && "Implicit Parent2 Foo failed");
  assert(result_parent3 == 3 && "Implicit Parent3 Foo failed");
  assert(result_parent4 == 4 && "Implicit Parent4 Foo failed");
  assert(result_parent5 == 5 && "Implicit Parent5 Foo failed");

  assert(result_child_parent1 == 6 && "Implicit Child Parent1 Foo failed");
  assert(result_child_parent2 == 7 && "Implicit Child Parent2 Foo failed");
  assert(result_child_parent3 == 8 && "Implicit Child Parent3 Foo failed");
  assert(result_child_parent4 == 4 && "Implicit Child Parent4 Foo failed");
  assert(result_child_parent5 == 10 && "Implicit Child Parent5 Foo failed");

  assert(result_child_as_parent1 == 6 &&
         "Implicit Child Parent1 Cast Foo failed");
  assert(result_child_as_parent2 == 7 &&
         "Implicit Child Parent2 Cast Foo failed");
  assert(result_child_as_parent3 == 8 &&
         "Implicit Child Parent3 Cast Foo failed");
  assert(result_child_as_parent4 == 4 &&
         "Implicit Child Parent4 Cast Foo failed");
  assert(result_child_as_parent5 == 10 &&
         "Implicit Child Parent5 Cast Foo failed");

  // Verify reference-based results
  assert(ref_result_parent1 == 1 && "Implicit Reference Parent1 Foo failed");
  assert(ref_result_parent2 == 2 && "Implicit Reference Parent2 Foo failed");
  assert(ref_result_parent3 == 3 && "Implicit Reference Parent3 Foo failed");
  assert(ref_result_parent4 == 4 && "Implicit Reference Parent4 Foo failed");
  assert(ref_result_parent5 == 5 && "Implicit Reference Parent5 Foo failed");

  assert(ref_result_child_parent1 == 6 &&
         "Implicit Reference Child Parent1 Foo failed");
  assert(ref_result_child_parent2 == 7 &&
         "Implicit Reference Child Parent2 Foo failed");
  assert(ref_result_child_parent3 == 8 &&
         "Implicit Reference Child Parent3 Foo failed");
  assert(ref_result_child_parent4 == 4 &&
         "Implicit Reference Child Parent4 Foo failed");
  assert(ref_result_child_parent5 == 10 &&
         "Implicit Reference Child Parent5 Foo failed");

  assert(ref_result_child_as_parent1 == 6 &&
         "Implicit Reference Child Parent1 Cast Foo failed");
  assert(ref_result_child_as_parent2 == 7 &&
         "Implicit Reference Child Parent2 Cast Foo failed");
  assert(ref_result_child_as_parent3 == 8 &&
         "Implicit Reference Child Parent3 Cast Foo failed");
  assert(ref_result_child_as_parent4 == 4 &&
         "Implicit Reference Child Parent4 Cast Foo failed");
  assert(ref_result_child_as_parent5 == 10 &&
         "Implicit Reference Child Parent5 Cast Foo failed");

  return 0;
}

int main() {
  test_five_parent_inheritance();
  test_five_parent_inheritance_implicit();

  // CHECK: PASS
  printf("PASS\n");
  return 0;
}
