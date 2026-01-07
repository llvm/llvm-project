// RUN: %libomptarget-compilexx-run-and-check-generic

#include <assert.h>
#include <omp.h>
#include <stdio.h>

#pragma omp declare target

class Mother {
public:
  virtual int MotherFoo(int x) { return x; }
};

class Father {
public:
  virtual int FatherFoo(int x) { return x * 2; }
};

class Child_1 : public Mother, public Father {
public:
  virtual int FatherFoo(int x) { return x * 3; }
};

class Child_2 : public Mother, public Father {
public:
  virtual int MotherFoo(int x) { return x * 4; }
};

class Child_3 : public Mother, public Father {
public:
  virtual int MotherFoo(int x) { return x * 5; }
  virtual int FatherFoo(int x) { return x * 6; }
};

#pragma omp end declare target

int test_multiple_inheritance() {
  Mother mother;
  Father father;
  Child_1 child_1;
  Child_2 child_2;
  Child_3 child_3;

  // map results back to host
  int result_mother, result_father;
  int result_child1_father, result_child1_mother, result_child1_as_mother,
      result_child1_as_father;
  int result_child2_mother, result_child2_father, result_child2_as_mother,
      result_child2_as_father;
  int result_child3_mother, result_child3_father, result_child3_as_mother,
      result_child3_as_father;

  // Add reference-based results
  int ref_result_mother, ref_result_father;
  int ref_result_child1_father, ref_result_child1_mother,
      ref_result_child1_as_mother, ref_result_child1_as_father;
  int ref_result_child2_mother, ref_result_child2_father,
      ref_result_child2_as_mother, ref_result_child2_as_father;
  int ref_result_child3_mother, ref_result_child3_father,
      ref_result_child3_as_mother, ref_result_child3_as_father;

#pragma omp target data map(father, mother, child_1, child_2, child_3)
  {
    // Base class pointers and references
    Mother *ptr_mother = &mother;
    Father *ptr_father = &father;
    Mother &ref_mother = mother;
    Father &ref_father = father;

    // Child_1 pointers, references and casts
    Child_1 *ptr_child_1 = &child_1;
    Mother *ptr_child_1_cast_mother = &child_1;
    Father *ptr_child_1_cast_father = &child_1;
    Child_1 &ref_child_1 = child_1;
    Mother &ref_child_1_cast_mother = child_1;
    Father &ref_child_1_cast_father = child_1;

    // Child_2 pointers, references and casts
    Child_2 *ptr_child_2 = &child_2;
    Mother *ptr_child_2_cast_mother = &child_2;
    Father *ptr_child_2_cast_father = &child_2;
    Child_2 &ref_child_2 = child_2;
    Mother &ref_child_2_cast_mother = child_2;
    Father &ref_child_2_cast_father = child_2;

    // Child_3 pointers and casts
    Child_3 *ptr_child_3 = &child_3;
    Mother *ptr_child_3_cast_mother = &child_3;
    Father *ptr_child_3_cast_father = &child_3;
    Child_3 &ref_child_3 = child_3;
    Mother &ref_child_3_cast_mother = child_3;
    Father &ref_child_3_cast_father = child_3;

#pragma omp target map(                                                        \
        from : result_mother, result_father, result_child1_father,             \
            result_child1_mother, result_child1_as_mother,                     \
            result_child1_as_father, result_child2_mother,                     \
            result_child2_father, result_child2_as_mother,                     \
            result_child2_as_father, result_child3_mother,                     \
            result_child3_father, result_child3_as_mother,                     \
            result_child3_as_father, ref_result_mother, ref_result_father,     \
            ref_result_child1_father, ref_result_child1_mother,                \
            ref_result_child1_as_mother, ref_result_child1_as_father,          \
            ref_result_child2_mother, ref_result_child2_father,                \
            ref_result_child2_as_mother, ref_result_child2_as_father,          \
            ref_result_child3_mother, ref_result_child3_father,                \
            ref_result_child3_as_mother, ref_result_child3_as_father)          \
    map(ptr_mother[0 : 0], ptr_father[0 : 0], ptr_child_1[0 : 0],              \
            ptr_child_1_cast_mother[0 : 0], ptr_child_1_cast_father[0 : 0],    \
            ptr_child_2[0 : 0], ptr_child_2_cast_mother[0 : 0],                \
            ptr_child_2_cast_father[0 : 0], ptr_child_3[0 : 0],                \
            ptr_child_3_cast_mother[0 : 0], ptr_child_3_cast_father[0 : 0],    \
            ref_mother, ref_father, ref_child_1, ref_child_1_cast_mother,      \
            ref_child_1_cast_father, ref_child_2, ref_child_2_cast_mother,     \
            ref_child_2_cast_father, ref_child_3, ref_child_3_cast_mother,     \
            ref_child_3_cast_father)
    {
      // These calls will fail if Clang does not
      // translate/attach the vtable pointer in each object

      // Pointer-based calls
      // Mother
      result_mother = ptr_mother->MotherFoo(1);
      // Father
      result_father = ptr_father->FatherFoo(1);
      // Child_1
      result_child1_father = ptr_child_1->FatherFoo(1);
      result_child1_mother = ptr_child_1->MotherFoo(1);
      result_child1_as_mother = ptr_child_1_cast_mother->MotherFoo(1);
      result_child1_as_father = ptr_child_1_cast_father->FatherFoo(1);
      // Child_2
      result_child2_mother = ptr_child_2->MotherFoo(1);
      result_child2_father = ptr_child_2->FatherFoo(1);
      result_child2_as_mother = ptr_child_2_cast_mother->MotherFoo(1);
      result_child2_as_father = ptr_child_2_cast_father->FatherFoo(1);
      // Child_3
      result_child3_mother = ptr_child_3->MotherFoo(1);
      result_child3_father = ptr_child_3->FatherFoo(1);
      result_child3_as_mother = ptr_child_3_cast_mother->MotherFoo(1);
      result_child3_as_father = ptr_child_3_cast_father->FatherFoo(1);

      // Reference-based calls
      // Mother
      ref_result_mother = ref_mother.MotherFoo(1);
      // Father
      ref_result_father = ref_father.FatherFoo(1);
      // Child_1
      ref_result_child1_father = ref_child_1.FatherFoo(1);
      ref_result_child1_mother = ref_child_1.MotherFoo(1);
      ref_result_child1_as_mother = ref_child_1_cast_mother.MotherFoo(1);
      ref_result_child1_as_father = ref_child_1_cast_father.FatherFoo(1);
      // Child_2
      ref_result_child2_mother = ref_child_2.MotherFoo(1);
      ref_result_child2_father = ref_child_2.FatherFoo(1);
      ref_result_child2_as_mother = ref_child_2_cast_mother.MotherFoo(1);
      ref_result_child2_as_father = ref_child_2_cast_father.FatherFoo(1);
      // Child_3
      ref_result_child3_mother = ref_child_3.MotherFoo(1);
      ref_result_child3_father = ref_child_3.FatherFoo(1);
      ref_result_child3_as_mother = ref_child_3_cast_mother.MotherFoo(1);
      ref_result_child3_as_father = ref_child_3_cast_father.FatherFoo(1);
    }
  }

  // Check pointer-based results
  assert(result_mother == 1 && "Mother Foo failed");
  assert(result_father == 2 && "Father Foo failed");
  assert(result_child1_father == 3 && "Child_1 Father Foo failed");
  assert(result_child1_mother == 1 && "Child_1 Mother Foo failed");
  assert(result_child1_as_mother == 1 &&
         "Child_1 Mother Parent Cast Foo failed");
  assert(result_child1_as_father == 3 &&
         "Child_1 Father Parent Cast Foo failed");
  assert(result_child2_mother == 4 && "Child_2 Mother Foo failed");
  assert(result_child2_father == 2 && "Child_2 Father Foo failed");
  assert(result_child2_as_mother == 4 &&
         "Child_2 Mother Parent Cast Foo failed");
  assert(result_child2_as_father == 2 &&
         "Child_2 Father Parent Cast Foo failed");
  assert(result_child3_mother == 5 && "Child_3 Mother Foo failed");
  assert(result_child3_father == 6 && "Child_3 Father Foo failed");
  assert(result_child3_as_mother == 5 &&
         "Child_3 Mother Parent Cast Foo failed");
  assert(result_child3_as_father == 6 &&
         "Child_3 Father Parent Cast Foo failed");

  // Check reference-based results
  assert(ref_result_mother == 1 && "Reference Mother Foo failed");
  assert(ref_result_father == 2 && "Reference Father Foo failed");
  assert(ref_result_child1_father == 3 &&
         "Reference Child_1 Father Foo failed");
  assert(ref_result_child1_mother == 1 &&
         "Reference Child_1 Mother Foo failed");
  assert(ref_result_child1_as_mother == 1 &&
         "Reference Child_1 Mother Parent Cast Foo failed");
  assert(ref_result_child1_as_father == 3 &&
         "Reference Child_1 Father Parent Cast Foo failed");
  assert(ref_result_child2_mother == 4 &&
         "Reference Child_2 Mother Foo failed");
  assert(ref_result_child2_father == 2 &&
         "Reference Child_2 Father Foo failed");
  assert(ref_result_child2_as_mother == 4 &&
         "Reference Child_2 Mother Parent Cast Foo failed");
  assert(ref_result_child2_as_father == 2 &&
         "Reference Child_2 Father Parent Cast Foo failed");
  assert(ref_result_child3_mother == 5 &&
         "Reference Child_3 Mother Foo failed");
  assert(ref_result_child3_father == 6 &&
         "Reference Child_3 Father Foo failed");
  assert(ref_result_child3_as_mother == 5 &&
         "Reference Child_3 Mother Parent Cast Foo failed");
  assert(ref_result_child3_as_father == 6 &&
         "Reference Child_3 Father Parent Cast Foo failed");

  return 0;
}

int test_multiple_inheritance_implicit() {
  Mother mother;
  Father father;
  Child_1 child_1;
  Child_2 child_2;
  Child_3 child_3;

  // map results back to host
  int result_mother, result_father;
  int result_child1_father, result_child1_mother, result_child1_as_mother,
      result_child1_as_father;
  int result_child2_mother, result_child2_father, result_child2_as_mother,
      result_child2_as_father;
  int result_child3_mother, result_child3_father, result_child3_as_mother,
      result_child3_as_father;

  // Add reference-based results
  int ref_result_mother, ref_result_father;
  int ref_result_child1_father, ref_result_child1_mother,
      ref_result_child1_as_mother, ref_result_child1_as_father;
  int ref_result_child2_mother, ref_result_child2_father,
      ref_result_child2_as_mother, ref_result_child2_as_father;
  int ref_result_child3_mother, ref_result_child3_father,
      ref_result_child3_as_mother, ref_result_child3_as_father;

#pragma omp target data map(father, mother, child_1, child_2, child_3)
  {
    // Base class pointers and references
    Mother *ptr_mother = &mother;
    Father *ptr_father = &father;
    Mother &ref_mother = mother;
    Father &ref_father = father;

    // Child_1 pointers, references and casts
    Child_1 *ptr_child_1 = &child_1;
    Mother *ptr_child_1_cast_mother = &child_1;
    Father *ptr_child_1_cast_father = &child_1;
    Child_1 &ref_child_1 = child_1;
    Mother &ref_child_1_cast_mother = child_1;
    Father &ref_child_1_cast_father = child_1;

    // Child_2 pointers, references and casts
    Child_2 *ptr_child_2 = &child_2;
    Mother *ptr_child_2_cast_mother = &child_2;
    Father *ptr_child_2_cast_father = &child_2;
    Child_2 &ref_child_2 = child_2;
    Mother &ref_child_2_cast_mother = child_2;
    Father &ref_child_2_cast_father = child_2;

    // Child_3 pointers and casts
    Child_3 *ptr_child_3 = &child_3;
    Mother *ptr_child_3_cast_mother = &child_3;
    Father *ptr_child_3_cast_father = &child_3;
    Child_3 &ref_child_3 = child_3;
    Mother &ref_child_3_cast_mother = child_3;
    Father &ref_child_3_cast_father = child_3;

    // Implicit mapping test - no explicit map clauses for pointers/references
#pragma omp target map(                                                        \
        from : result_mother, result_father, result_child1_father,             \
            result_child1_mother, result_child1_as_mother,                     \
            result_child1_as_father, result_child2_mother,                     \
            result_child2_father, result_child2_as_mother,                     \
            result_child2_as_father, result_child3_mother,                     \
            result_child3_father, result_child3_as_mother,                     \
            result_child3_as_father, ref_result_mother, ref_result_father,     \
            ref_result_child1_father, ref_result_child1_mother,                \
            ref_result_child1_as_mother, ref_result_child1_as_father,          \
            ref_result_child2_mother, ref_result_child2_father,                \
            ref_result_child2_as_mother, ref_result_child2_as_father,          \
            ref_result_child3_mother, ref_result_child3_father,                \
            ref_result_child3_as_mother, ref_result_child3_as_father)
    {
      // These calls will fail if Clang does not
      // translate/attach the vtable pointer in each object

      // Pointer-based calls
      // Mother
      result_mother = ptr_mother->MotherFoo(1);
      // Father
      result_father = ptr_father->FatherFoo(1);
      // Child_1
      result_child1_father = ptr_child_1->FatherFoo(1);
      result_child1_mother = ptr_child_1->MotherFoo(1);
      result_child1_as_mother = ptr_child_1_cast_mother->MotherFoo(1);
      result_child1_as_father = ptr_child_1_cast_father->FatherFoo(1);
      // Child_2
      result_child2_mother = ptr_child_2->MotherFoo(1);
      result_child2_father = ptr_child_2->FatherFoo(1);
      result_child2_as_mother = ptr_child_2_cast_mother->MotherFoo(1);
      result_child2_as_father = ptr_child_2_cast_father->FatherFoo(1);
      // Child_3
      result_child3_mother = ptr_child_3->MotherFoo(1);
      result_child3_father = ptr_child_3->FatherFoo(1);
      result_child3_as_mother = ptr_child_3_cast_mother->MotherFoo(1);
      result_child3_as_father = ptr_child_3_cast_father->FatherFoo(1);

      // Reference-based calls
      // Mother
      ref_result_mother = ref_mother.MotherFoo(1);
      // Father
      ref_result_father = ref_father.FatherFoo(1);
      // Child_1
      ref_result_child1_father = ref_child_1.FatherFoo(1);
      ref_result_child1_mother = ref_child_1.MotherFoo(1);
      ref_result_child1_as_mother = ref_child_1_cast_mother.MotherFoo(1);
      ref_result_child1_as_father = ref_child_1_cast_father.FatherFoo(1);
      // Child_2
      ref_result_child2_mother = ref_child_2.MotherFoo(1);
      ref_result_child2_father = ref_child_2.FatherFoo(1);
      ref_result_child2_as_mother = ref_child_2_cast_mother.MotherFoo(1);
      ref_result_child2_as_father = ref_child_2_cast_father.FatherFoo(1);
      // Child_3
      ref_result_child3_mother = ref_child_3.MotherFoo(1);
      ref_result_child3_father = ref_child_3.FatherFoo(1);
      ref_result_child3_as_mother = ref_child_3_cast_mother.MotherFoo(1);
      ref_result_child3_as_father = ref_child_3_cast_father.FatherFoo(1);
    }
  }

  // Check pointer-based results
  assert(result_mother == 1 && "Implicit Mother Foo failed");
  assert(result_father == 2 && "Implicit Father Foo failed");
  assert(result_child1_father == 3 && "Implicit Child_1 Father Foo failed");
  assert(result_child1_mother == 1 && "Implicit Child_1 Mother Foo failed");
  assert(result_child1_as_mother == 1 &&
         "Implicit Child_1 Mother Parent Cast Foo failed");
  assert(result_child1_as_father == 3 &&
         "Implicit Child_1 Father Parent Cast Foo failed");
  assert(result_child2_mother == 4 && "Implicit Child_2 Mother Foo failed");
  assert(result_child2_father == 2 && "Implicit Child_2 Father Foo failed");
  assert(result_child2_as_mother == 4 &&
         "Implicit Child_2 Mother Parent Cast Foo failed");
  assert(result_child2_as_father == 2 &&
         "Implicit Child_2 Father Parent Cast Foo failed");
  assert(result_child3_mother == 5 && "Implicit Child_3 Mother Foo failed");
  assert(result_child3_father == 6 && "Implicit Child_3 Father Foo failed");
  assert(result_child3_as_mother == 5 &&
         "Implicit Child_3 Mother Parent Cast Foo failed");
  assert(result_child3_as_father == 6 &&
         "Implicit Child_3 Father Parent Cast Foo failed");

  // Check reference-based results
  assert(ref_result_mother == 1 && "Implicit Reference Mother Foo failed");
  assert(ref_result_father == 2 && "Implicit Reference Father Foo failed");
  assert(ref_result_child1_father == 3 &&
         "Implicit Reference Child_1 Father Foo failed");
  assert(ref_result_child1_mother == 1 &&
         "Implicit Reference Child_1 Mother Foo failed");
  assert(ref_result_child1_as_mother == 1 &&
         "Implicit Reference Child_1 Mother Parent Cast Foo failed");
  assert(ref_result_child1_as_father == 3 &&
         "Implicit Reference Child_1 Father Parent Cast Foo failed");
  assert(ref_result_child2_mother == 4 &&
         "Implicit Reference Child_2 Mother Foo failed");
  assert(ref_result_child2_father == 2 &&
         "Implicit Reference Child_2 Father Foo failed");
  assert(ref_result_child2_as_mother == 4 &&
         "Implicit Reference Child_2 Mother Parent Cast Foo failed");
  assert(ref_result_child2_as_father == 2 &&
         "Implicit Reference Child_2 Father Parent Cast Foo failed");
  assert(ref_result_child3_mother == 5 &&
         "Implicit Reference Child_3 Mother Foo failed");
  assert(ref_result_child3_father == 6 &&
         "Implicit Reference Child_3 Father Foo failed");
  assert(ref_result_child3_as_mother == 5 &&
         "Implicit Reference Child_3 Mother Parent Cast Foo failed");
  assert(ref_result_child3_as_father == 6 &&
         "Implicit Reference Child_3 Father Parent Cast Foo failed");

  return 0;
}

int main() {
  test_multiple_inheritance();
  test_multiple_inheritance_implicit();

  // CHECK: PASS
  printf("PASS\n");
  return 0;
}
