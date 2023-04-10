#include "llvm/Transforms/ErrorAnalysis/Utilities/FunctionMatchers.h"

namespace atomiccondition {

bool isASinFunction(const string FunctionName) {
  if(FunctionName.find("asin") != std::string::npos) {
    return FunctionName.compare("asin") == 0 ||
           FunctionName.compare("asinf") == 0 ||
           FunctionName.compare("cpf_asin") == 0 ||
           FunctionName.compare("cpf_asinf") == 0;
  }

  return false;
}

bool isACosFunction(const string FunctionName) {
  if(FunctionName.find("acos") != std::string::npos) {
    return FunctionName.compare("acos") == 0 ||
           FunctionName.compare("acosf") == 0 ||
           FunctionName.compare("cpf_acos") == 0 ||
           FunctionName.compare("cpf_acosf") == 0;
  }

  return false;
}

bool isATanFunction(const string FunctionName) {
  if(FunctionName.find("atan") != std::string::npos) {
    return FunctionName.compare("atan") == 0 ||
           FunctionName.compare("atanf") == 0 ||
           FunctionName.compare("cpf_atan") == 0 ||
           FunctionName.compare("cpf_atanf") == 0;
  }

  return false;
}

bool isSinFunction(const string FunctionName) {
  if(FunctionName.find("sin") != std::string::npos) {
    return FunctionName.compare("sin") == 0 ||
           FunctionName.compare("sinf") == 0 ||
           FunctionName.compare("cpf_sin") == 0 ||
           FunctionName.compare("cpf_sinf") == 0 ||
           FunctionName.compare("llvm.sin.f32") == 0 ||
           FunctionName.compare("llvm.sin.f64") == 0;
  }

  return false;
}

bool isCosFunction(const string FunctionName) {
  if(FunctionName.find("cos") != std::string::npos) {
    return FunctionName.compare("cos") == 0 ||
           FunctionName.compare("cosf") == 0 ||
           FunctionName.compare("cpf_cos") == 0 ||
           FunctionName.compare("cpf_cosf") == 0 ||
           FunctionName.compare("llvm.cos.f32") == 0 ||
           FunctionName.compare("llvm.cos.f64") == 0;
  }

  return false;
}

bool isTanFunction(const string FunctionName) {
  if(FunctionName.find("tan") != std::string::npos) {
    return FunctionName.compare("tan") == 0 ||
           FunctionName.compare("tanf") == 0 ||
           FunctionName.compare("cpf_tan") == 0 ||
           FunctionName.compare("cpf_tanf") == 0 ||
           FunctionName.compare("llvm.tan.f32") == 0 ||
           FunctionName.compare("llvm.tan.f64") == 0;
  }

  return false;
}

bool isSinhFunction(const string FunctionName) {
  if(FunctionName.find("sinh") != std::string::npos) {
    return FunctionName.compare("sinh") == 0 ||
           FunctionName.compare("sinhf") == 0 ||
           FunctionName.compare("cpf_sinh") == 0 ||
           FunctionName.compare("cpf_sinhf") == 0 ||
           FunctionName.compare("llvm.sinh.f32") == 0 ||
           FunctionName.compare("llvm.sinh.f64") == 0;
  }

  return false;
}

bool isCoshFunction(const string FunctionName) {
  if(FunctionName.find("cosh") != std::string::npos) {
    return FunctionName.compare("cosh") == 0 ||
           FunctionName.compare("coshf") == 0 ||
           FunctionName.compare("cpf_cosh") == 0 ||
           FunctionName.compare("cpf_coshf") == 0 ||
           FunctionName.compare("llvm.cosh.f32") == 0 ||
           FunctionName.compare("llvm.cosh.f64") == 0;
  }

  return false;
}

bool isTanhFunction(const string FunctionName) {
  if(FunctionName.find("tanh") != std::string::npos) {
    return FunctionName.compare("tanh") == 0 ||
           FunctionName.compare("tanhf") == 0 ||
           FunctionName.compare("cpf_tanh") == 0 ||
           FunctionName.compare("cpf_tanhf") == 0 ||
           FunctionName.compare("llvm.tanh.f32") == 0 ||
           FunctionName.compare("llvm.tanh.f64") == 0;
  }

  return false;
}

bool isExpFunction(const string FunctionName) {
  if(FunctionName.find("exp") != std::string::npos) {
    return FunctionName.compare("exp") == 0 ||
           FunctionName.compare("expf") == 0 ||
           FunctionName.compare("cpf_exp") == 0 ||
           FunctionName.compare("cpf_expf") == 0 ||
           FunctionName.compare("llvm.exp.f32") == 0 ||
           FunctionName.compare("llvm.exp.f64") == 0;
  }

  return false;
}

bool isLogFunction(const string FunctionName) {
  if(FunctionName.find("log") != std::string::npos) {
    return FunctionName.compare("log") == 0 ||
           FunctionName.compare("logf") == 0 ||
           FunctionName.compare("cpf_log") == 0 ||
           FunctionName.compare("cpf_logf") == 0 ||
           FunctionName.compare("llvm.log.f32") == 0 ||
           FunctionName.compare("llvm.log.f64") == 0;
  }

  return false;
}

bool isSqrtFunction(const string FunctionName) {
  if(FunctionName.find("sqrt") != std::string::npos) {
    return FunctionName.compare("sqrt") == 0 ||
           FunctionName.compare("sqrtf") == 0 ||
           FunctionName.compare("cpf_sqrt") == 0 ||
           FunctionName.compare("cpf_sqrtf") == 0 ||
           FunctionName.compare("llvm.sqrt.f32") == 0 ||
           FunctionName.compare("llvm.sqrt.f64") == 0;
  }

  return false;
}

bool isAddFunction(const string FunctionName) {
  return FunctionName.compare("cpf_add") == 0 ||
         FunctionName.compare("cpf_addf") == 0;
}

bool isSubFunction(const string FunctionName) {
  return FunctionName.compare("cpf_sub") == 0 ||
         FunctionName.compare("cpf_subf") == 0;
}

bool isMulFunction(const string FunctionName) {
  return FunctionName.compare("cpf_mul") == 0 ||
         FunctionName.compare("cpf_mulf") == 0;
}

bool isDivFunction(const string FunctionName) {
  return FunctionName.compare("cpf_div") == 0 ||
         FunctionName.compare("cpf_divf") == 0;
}

bool isFMAFuncton(const string FunctionName) {
  return FunctionName.compare("llvm.fmuladd.f32") == 0 ||
         FunctionName.compare("llvm.fmuladd.f64") == 0 ||
         FunctionName.compare("llvm.fma.f32") == 0 ||
         FunctionName.compare("llvm.fma.f64") == 0 ||
         FunctionName.compare("cpf_fma") == 0 ||
         FunctionName.compare("cpf_fmaf") == 0;
}

bool isPrintFunction(const string FunctionName) {
  if(FunctionName.find("print") != std::string::npos) {
    return FunctionName.compare("printf") == 0;
  }

  return false;
}

bool isCPFloatFunction(const string FunctionName) {
  return FunctionName.find("cpf_") != std::string::npos;
}

} // namespace atomiccondition