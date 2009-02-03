// RUN: %llvmgcc -std=c99 %s -S -o - | llvm-as | \
// RUN:    opt -std-compile-opts -disable-output
// PR580

int X, Y;
int foo() {
  int i;
        for (i=0; i<100; i++ )
        {
                break;
                i = ( X || Y ) ;
        }
}

