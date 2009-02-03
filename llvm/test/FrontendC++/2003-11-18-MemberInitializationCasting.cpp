// RUN: %llvmgcc -xc++ -S -o - %s | llvm-as | opt -die | llvm-dis |  notcast

struct A {
        A() : i(0) {}
        int getI() {return i;}
        int i;
};

int f(int j)
{
        A a;
        return j+a.getI();
}
