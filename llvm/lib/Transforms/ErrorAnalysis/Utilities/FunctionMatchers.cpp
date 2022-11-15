#include "llvm/Transforms/ErrorAnalysis/Utilities/FunctionMatchers.h"

namespace atomiccondition {

bool isASinFunction(const string FunctionName) {
  if(FunctionName.find("asin") != std::string::npos) {
    return FunctionName.compare("asin") == 0;
  }

  return false;
}

bool isACosFunction(const string FunctionName) {
  if(FunctionName.find("acos") != std::string::npos) {
    return FunctionName.compare("acos") == 0;
  }

  return false;
}

bool isATanFunction(const string FunctionName) {
  if(FunctionName.find("atan") != std::string::npos) {
    return FunctionName.compare("atan") == 0;
  }

  return false;
}

bool isSinFunction(const string FunctionName) {
  if(FunctionName.find("sin") != std::string::npos) {
    return FunctionName.compare("sin") == 0 ||
           FunctionName.find("llvm.sin.")  != std::string::npos;
  }

  return false;
}

bool isCosFunction(const string FunctionName) {
  if(FunctionName.find("cos") != std::string::npos) {
    return FunctionName.compare("cos") == 0 ||
           FunctionName.find("llvm.cos.")  != std::string::npos;
  }

  return false;
}

bool isTanFunction(const string FunctionName) {
  if(FunctionName.find("tan") != std::string::npos) {
    return FunctionName.compare("tan") == 0 ||
           FunctionName.find("llvm.tan.")  != std::string::npos;
  }

  return false;
}

bool isSinhFunction(const string FunctionName) {
  if(FunctionName.find("sinh") != std::string::npos) {
    return FunctionName.compare("sinh") == 0;
  }

  return false;
}

bool isCoshFunction(const string FunctionName) {
  if(FunctionName.find("cosh") != std::string::npos) {
    return FunctionName.compare("cosh") == 0;
  }

  return false;
}

bool isTanhFunction(const string FunctionName) {
  if(FunctionName.find("tanh") != std::string::npos) {
    return FunctionName.compare("tanh") == 0;
  }

  return false;
}

bool isExpFunction(const string FunctionName) {
  if(FunctionName.find("exp") != std::string::npos) {
    return FunctionName.compare("tan") == 0 ||
           FunctionName.find("llvm.exp.")  != std::string::npos;
  }

  return false;
}

bool isLogFunction(const string FunctionName) {
  if(FunctionName.find("log") != std::string::npos) {
    return FunctionName.compare("log") == 0 ||
           FunctionName.find("llvm.log.")  != std::string::npos;
  }

  return false;
}

bool isSqrtFunction(const string FunctionName) {
  if(FunctionName.find("sqrt") != std::string::npos) {
    return FunctionName.compare("sqrt") == 0 ||
           FunctionName.find("llvm.sqrt.")  != std::string::npos;
  }

  return false;
}

bool isFMAFuncton(const string FunctionName) {
  return FunctionName.find("llvm.fmuladd.") == 0 ||
         FunctionName.find("llvm.fma.")  != std::string::npos;
}

bool isPrintFunction(const string FunctionName) {
  if(FunctionName.find("print") != std::string::npos) {
    return FunctionName.compare("printf") == 0;
  }

  return false;
}

} // namespace atomiccondition