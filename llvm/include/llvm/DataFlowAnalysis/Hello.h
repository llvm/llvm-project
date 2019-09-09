#ifndef LLVM_DFA_HELLO_H
#define LLVM_DFA_HELLO_H

#include <iostream>

namespace llvm{

namespace Intrinsic{
enum ID : unsigned;
}

class Hello{
    int n;
public:
    Hello(int);
    int data();
};

}   // end namespace LLVM

#endif // LLVM_DFA_HELLO_H
