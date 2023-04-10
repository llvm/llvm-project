//
// Created by tanmay on 10/19/22.
//

#ifndef LLVM_FUNCTIONMATCHERS_H
#define LLVM_FUNCTIONMATCHERS_H

#include <string>

using namespace std;

namespace atomiccondition {

bool isASinFunction(const string FunctionName);
bool isACosFunction(const string FunctionName);
bool isATanFunction(const string FunctionName);
bool isSinFunction(const string FunctionName);
bool isCosFunction(const string FunctionName);
bool isTanFunction(const string FunctionName);
bool isSinhFunction(const string FunctionName);
bool isCoshFunction(const string FunctionName);
bool isTanhFunction(const string FunctionName);
bool isExpFunction(const string FunctionName);
bool isLogFunction(const string FunctionName);
bool isSqrtFunction(const string FunctionName);

bool isAddFunction(const string FunctionName);
bool isSubFunction(const string FunctionName);
bool isMulFunction(const string FunctionName);
bool isDivFunction(const string FunctionName);

bool isFMAFuncton(const string FunctionName);
bool isPrintFunction(const string FunctionName);
bool isCPFloatFunction(const string FunctionName);

} // namespace atomiccondition

#endif // LLVM_FUNCTIONMATCHERS_H
