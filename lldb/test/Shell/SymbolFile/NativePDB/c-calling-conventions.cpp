// clang-format off
// REQUIRES: lld, (target-x86 || target-x86_64)

// RUN: %build --compiler=clang-cl --arch=32 --nodefaultlib --output=%t-32.exe %s
// RUN: lldb-test symbols %t-32.exe | FileCheck --check-prefixes CHECK-32,CHECK-BOTH %s
// RUN: %build --compiler=clang-cl --arch=64 --nodefaultlib --output=%t-64.exe %s
// RUN: lldb-test symbols %t-64.exe | FileCheck --check-prefixes CHECK-64,CHECK-BOTH %s

extern "C" {
int FuncCCall() { return 0; }
int __stdcall FuncStdCall() { return 0; }
int __fastcall FuncFastCall() { return 0; }
int __vectorcall FuncVectorCall() { return 0; }

int __cdecl _underscoreCdecl() { return 0; }
int __stdcall _underscoreStdcall() { return 0; }
int __fastcall _underscoreFastcall() { return 0; }
int __vectorcall _underscoreVectorcall() { return 0; }
}

int main() {
  FuncCCall();
  FuncStdCall();
  FuncFastCall();
  FuncVectorCall();
  _underscoreCdecl();
  _underscoreStdcall();
  _underscoreFastcall();
  _underscoreVectorcall();
  return 0;
}

// CHECK-BOTH-DAG: Function{{.*}}, demangled = FuncCCall,
// CHECK-BOTH-DAG: Function{{.*}}, demangled = FuncVectorCall@@0,
// CHECK-BOTH-DAG: Function{{.*}}, demangled = _underscoreCdecl,
// CHECK-BOTH-DAG: Function{{.*}}, demangled = _underscoreVectorcall@@0,
// CHECK-BOTH-DAG: Function{{.*}}, demangled = main,

// __stdcall and __fastcall aren't available on 64 bit

// CHECK-32-DAG: Function{{.*}}, demangled = _FuncStdCall@0,
// CHECK-64-DAG: Function{{.*}}, demangled = FuncStdCall,

// CHECK-32-DAG: Function{{.*}}, demangled = @FuncFastCall@0,
// CHECK-64-DAG: Function{{.*}}, demangled = FuncFastCall,

// CHECK-32-DAG: Function{{.*}}, demangled = __underscoreStdcall@0,
// CHECK-64-DAG: Function{{.*}}, demangled = _underscoreStdcall,

// CHECK-32-DAG: Function{{.*}}, demangled = @_underscoreFastcall@0,
// CHECK-64-DAG: Function{{.*}}, demangled = _underscoreFastcall,
