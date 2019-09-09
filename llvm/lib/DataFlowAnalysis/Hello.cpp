#include <iostream>
#include "llvm/DataFlowAnalysis/Hello.h"

using namespace llvm;

Hello::Hello(int d){
    n = d;
}

int Hello::data(){
    return n;
}

