// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

   struct fu;
   void foo(struct fu);
   void bar() {
      foo;
   }
